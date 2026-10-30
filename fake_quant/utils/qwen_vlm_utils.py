import os
import re
import inspect

import torch
from llava.constants import IGNORE_INDEX


def ensure_image_url(image: str) -> str:
    prefixes = ["http://", "https://", "file://", "data:image;"]
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return "file://" + os.path.abspath(image)
    raise ValueError(f"Invalid image path/url: {image}")


def resolve_qwen_vlm_family(args=None, model=None, tokenizer=None):
    explicit_model_name = str(getattr(args, "model", "") if args is not None else "").lower()
    model_name_or_path = ""
    if model is not None:
        model_name_or_path = str(
            getattr(model, "name_or_path", "")
            or getattr(getattr(model, "config", None), "_name_or_path", "")
            or getattr(getattr(model, "config", None), "name_or_path", "")
        ).lower()
    model_cls = type(model).__name__.lower() if model is not None else ""
    tokenizer_repr = str(tokenizer).lower() if tokenizer is not None else ""
    joined = " ".join([explicit_model_name, model_name_or_path, model_cls, tokenizer_repr])

    if "qwen3-vl" in joined or "qwen3vl" in joined:
        return "qwen3vl"
    if (
        "qwen2-vl" in joined
        or "qwen2vl" in joined
        or "qwen2.5-vl" in joined
        or "qwen2.5vl" in joined
        or "qwen2_5_vl" in joined
    ):
        return "qwen2vl"
    if "qwen-vl" in joined or "qwenvl" in joined:
        return "qwenvl"
    return None


def is_qwen_vlm_family(args=None, model=None, tokenizer=None):
    return resolve_qwen_vlm_family(args=args, model=model, tokenizer=tokenizer) is not None


QWEN3_VLMEVALKIT_PROMPT_VERSION = "vlmevalkit_prompt_v1"
QWEN3_FINAL_ANSWER_INSTRUCTION_VERSION = "final_answer_v1"


def qwen3_vlmevalkit_aligned_message(message):
    """Return the dataset prompt unchanged to match VLMEvalKit default eval."""
    return message


def qwen3_final_answer_instruction(is_multi_choice: bool = True) -> str:
    if is_multi_choice:
        return (
            "Answer the following multiple-choice question.\n"
            "End your response with exactly one final line in this format:\n"
            "Answer: $LETTER\n"
            "where LETTER is one of the options."
        )
    return (
        "Answer the question.\n"
        "Put only your final answer in the last line."
    )


def _message_text(message) -> str:
    if not isinstance(message, list):
        return ""
    return "\n".join(
        str(item.get("value", ""))
        for item in message
        if isinstance(item, dict) and item.get("type") == "text"
    )


def has_qwen3_final_answer_instruction(message) -> bool:
    text = _message_text(message)
    return "Answer: $LETTER" in text or "Put only your final answer in the last line." in text


def infer_qwen3_is_multi_choice(message) -> bool:
    text = _message_text(message)
    if re.search(r"correct answer.*options", text, flags=re.IGNORECASE | re.DOTALL):
        return True
    if re.search(r"\b(options|choices)\s*:", text, flags=re.IGNORECASE):
        return True
    if re.search(r"(?m)^\s*[A-Z][\.\):]\s+\S+", text):
        return True
    return False


def append_qwen3_final_answer_instruction(message, is_multi_choice: bool = True):
    if not isinstance(message, list) or has_qwen3_final_answer_instruction(message):
        return message
    copied = [dict(item) if isinstance(item, dict) else item for item in message]
    copied.append(
        {
            "type": "text",
            "value": qwen3_final_answer_instruction(is_multi_choice=is_multi_choice),
        }
    )
    return copied


def _extract_message_parts(message):
    prompt_texts = []
    answer_texts = []
    image_paths = []
    for msg in message:
        msg_type = msg["type"]
        if msg_type == "text":
            prompt_texts.append(msg["value"])
        elif msg_type == "textanw":
            answer_texts.append(msg["value"])
        elif msg_type == "image":
            image_paths.append(msg["value"])
        else:
            raise ValueError(f"Invalid message type: {msg_type}")
    return prompt_texts, answer_texts, image_paths


def _strip_thinking_content(text):
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _build_qwen23_content(message, family):
    content = []
    prompt_texts, answer_texts, _ = _extract_message_parts(message)
    for msg in message:
        msg_type = msg["type"]
        if msg_type == "image":
            item = {"type": "image", "image": ensure_image_url(msg["value"])}
            if family == "qwen2vl":
                item["max_pixels"] = 16384 * 28 * 28
                item["min_pixels"] = 1280 * 28 * 28
            content.append(item)
        elif msg_type == "text":
            content.append({"type": "text", "text": msg["value"]})
    return content, "".join(answer_texts), prompt_texts


def _qwen3_image_patch_size(image_processor):
    processor = getattr(image_processor, "image_processor", None)
    return getattr(processor, "patch_size", None)


def _process_qwen_vision_info(
    messages,
    image_processor=None,
    family=None,
    return_video_kwargs=False,
):
    from qwen_vl_utils import process_vision_info

    params = inspect.signature(process_vision_info).parameters
    kwargs = {}
    if family == "qwen3vl":
        if "image_patch_size" not in params:
            raise RuntimeError(
                "qwen-vl-utils is too old for Qwen3-VL. "
                "process_vision_info must support image_patch_size."
            )
        patch_size = _qwen3_image_patch_size(image_processor)
        if patch_size is None:
            raise ValueError("Qwen3-VL processor does not expose image_processor.patch_size.")
        kwargs["image_patch_size"] = patch_size

    if return_video_kwargs:
        missing = [
            name
            for name in ("return_video_kwargs", "return_video_metadata")
            if name not in params
        ]
        if missing:
            raise RuntimeError(
                "qwen-vl-utils is too old for Qwen3-VL vLLM video inputs. "
                f"Missing process_vision_info args: {missing}"
            )
        kwargs["return_video_kwargs"] = True
        kwargs["return_video_metadata"] = True

    outputs = process_vision_info(messages, **kwargs)
    if return_video_kwargs:
        if not isinstance(outputs, tuple) or len(outputs) != 3:
            raise RuntimeError("Expected process_vision_info to return images, videos, video kwargs.")
        return outputs

    if isinstance(outputs, tuple) and len(outputs) == 3:
        return outputs[0], outputs[1]
    return outputs


def _has_mm_data(value):
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return True


def build_qwen1_eval_inputs(message, image_processor, device):
    from transformers.image_utils import load_image

    replace_mapping = {
        "\nOptions:": "\nChoices:",
        "Please select the correct answer from the options above.": "Answer with the letter.",
    }
    prompt = "<|im_start|>User:"
    images = []
    for msg in message:
        if msg["type"] == "image":
            images.append(load_image(msg["value"]))
            prompt += "<image>"
        elif msg["type"] == "text":
            instruction = msg["value"].strip()
            for src, dst in replace_mapping.items():
                instruction = instruction.replace(src, dst)
            prompt += instruction
        else:
            raise ValueError(f"Invalid message type for Qwen-VL eval: {msg['type']}")
    prompt += "<end_of_utterance>\nAssistant: Answer:"
    inputs = image_processor(text=prompt, images=images, return_tensors="pt")
    return inputs.to(device)


def build_qwen1_train_inputs(message, image_processor, device, supervision_mode="answer_only"):
    from transformers.image_utils import load_image

    if supervision_mode != "answer_only":
        raise ValueError(f"Unsupported supervision_mode for qwenvl: {supervision_mode}")

    replace_mapping = {
        "\nOptions:": "\nChoices:",
        "Please select the correct answer from the options above.": "Answer with the letter.",
    }
    prompt = "<|im_start|>User:"
    answer_text = ""
    images = []
    for msg in message:
        if msg["type"] == "image":
            images.append(load_image(msg["value"]))
            prompt += "<image>"
        elif msg["type"] == "text":
            instruction = msg["value"].strip()
            for src, dst in replace_mapping.items():
                instruction = instruction.replace(src, dst)
            prompt += instruction
        elif msg["type"] == "textanw":
            answer_text += msg["value"]
        else:
            raise ValueError(f"Invalid message type for Qwen-VL train: {msg['type']}")
    prompt += "<end_of_utterance>\nAssistant: Answer:"
    full_text = prompt + answer_text

    prompt_inputs = image_processor(text=prompt, images=images, return_tensors="pt")
    full_inputs = image_processor(text=full_text, images=images, return_tensors="pt")

    labels = full_inputs["input_ids"].clone()
    q_len = prompt_inputs["input_ids"].size(1)
    labels[:, :q_len] = IGNORE_INDEX
    pad_id = getattr(image_processor, "tokenizer", image_processor).pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = IGNORE_INDEX

    meta = {
        "family": "qwenvl",
        "prompt_text": prompt,
        "full_text": full_text,
        "prompt_token_length": int(q_len),
        "answer_text": answer_text,
    }
    return full_inputs.to(device), labels.to(device), meta


def build_qwen2_eval_inputs(message, image_processor, device):
    from qwen_vl_utils import process_vision_info

    content, _, _ = _build_qwen23_content(message, "qwen2vl")
    messages = [{"role": "user", "content": content}]
    prompt = image_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, videos = process_vision_info(messages)
    inputs = image_processor(
        text=[prompt],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def build_qwen3_eval_inputs(message, image_processor, device):
    content, _, _ = _build_qwen23_content(message, "qwen3vl")
    messages = [{"role": "user", "content": content}]
    prompt = image_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, videos = _process_qwen_vision_info(
        messages,
        image_processor=image_processor,
        family="qwen3vl",
    )
    inputs = image_processor(
        text=[prompt],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def build_qwen3_vllm_prompt(message, image_processor):
    content, _, _ = _build_qwen23_content(message, "qwen3vl")
    messages = [{"role": "user", "content": content}]
    prompt = image_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, videos, mm_processor_kwargs = _process_qwen_vision_info(
        messages,
        image_processor=image_processor,
        family="qwen3vl",
        return_video_kwargs=True,
    )
    multi_modal_data = {}
    if _has_mm_data(images):
        multi_modal_data["image"] = images
    if _has_mm_data(videos):
        multi_modal_data["video"] = videos
    prompt_meta = {
        "messages": messages,
        "mm_processor_kwargs": mm_processor_kwargs or {},
    }
    return prompt, multi_modal_data, prompt_meta


def _build_qwen23_train_inputs(message, image_processor, device, family, supervision_mode="answer_only", strip_thinking=False):
    if supervision_mode != "answer_only":
        raise ValueError(f"Unsupported supervision_mode for {family}: {supervision_mode}")

    content, answer_text, _ = _build_qwen23_content(message, family)
    if strip_thinking:
        answer_text = _strip_thinking_content(answer_text)

    prompt_messages = [{"role": "user", "content": content}]
    full_messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
    ]

    prompt_text = image_processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = image_processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    images, videos = _process_qwen_vision_info(
        prompt_messages,
        image_processor=image_processor,
        family=family,
    )
    prompt_inputs = image_processor(
        text=[prompt_text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )
    full_inputs = image_processor(
        text=[full_text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )

    labels = full_inputs["input_ids"].clone()
    q_len = prompt_inputs["input_ids"].size(1)
    labels[:, :q_len] = IGNORE_INDEX
    pad_id = image_processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = IGNORE_INDEX

    non_ignored = torch.nonzero(labels[0] != IGNORE_INDEX, as_tuple=False).flatten()
    meta = {
        "family": family,
        "prompt_text": prompt_text,
        "full_text": full_text,
        "prompt_token_length": int(q_len),
        "answer_text": answer_text,
        "answer_token_start": int(non_ignored[0].item()) if len(non_ignored) > 0 else None,
        "answer_token_end": int(non_ignored[-1].item()) if len(non_ignored) > 0 else None,
    }
    return full_inputs.to(device), labels.to(device), meta


def build_qwen2_train_inputs(message, image_processor, device, supervision_mode="answer_only", strip_thinking=False):
    return _build_qwen23_train_inputs(
        message,
        image_processor,
        device,
        "qwen2vl",
        supervision_mode=supervision_mode,
        strip_thinking=strip_thinking,
    )


def build_qwen3_train_inputs(message, image_processor, device, supervision_mode="answer_only", strip_thinking=True):
    return _build_qwen23_train_inputs(
        message,
        image_processor,
        device,
        "qwen3vl",
        supervision_mode=supervision_mode,
        strip_thinking=strip_thinking,
    )


def build_qwen_vlm_eval_inputs(message, image_processor, device, args=None, model=None, tokenizer=None):
    family = resolve_qwen_vlm_family(args=args, model=model, tokenizer=tokenizer)
    if family == "qwenvl":
        return build_qwen1_eval_inputs(message, image_processor, device)
    if family == "qwen2vl":
        return build_qwen2_eval_inputs(message, image_processor, device)
    if family == "qwen3vl":
        return build_qwen3_eval_inputs(message, image_processor, device)
    raise ValueError("Unsupported Qwen VLM family for eval inputs")


def build_qwen_vlm_train_inputs(
    message,
    image_processor,
    device,
    args=None,
    model=None,
    tokenizer=None,
    supervision_mode="answer_only",
):
    family = resolve_qwen_vlm_family(args=args, model=model, tokenizer=tokenizer)
    if family == "qwenvl":
        return build_qwen1_train_inputs(
            message,
            image_processor,
            device,
            supervision_mode=supervision_mode,
        )
    if family == "qwen2vl":
        return build_qwen2_train_inputs(
            message,
            image_processor,
            device,
            supervision_mode=supervision_mode,
            strip_thinking=False,
        )
    if family == "qwen3vl":
        default_strip = True
        if args is not None:
            default_strip = bool(getattr(args, "mask_thinking_tokens", True))
        return build_qwen3_train_inputs(
            message,
            image_processor,
            device,
            supervision_mode=supervision_mode,
            strip_thinking=default_strip,
        )
    raise ValueError("Unsupported Qwen VLM family for train inputs")
