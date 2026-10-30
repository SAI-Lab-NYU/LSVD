import logging
from typing import Dict, List, Tuple

import torch
from PIL import Image


IGNORE_INDEX = -100


def is_molmo_model(args=None, model=None, tokenizer=None) -> bool:
    if tokenizer == "molmo":
        return True
    model_name = str(getattr(args, "model", "") if args is not None else "").lower()
    if "molmo" in model_name:
        return True
    if model is None:
        return False
    class_name = model.__class__.__name__.lower()
    config = getattr(model, "config", None)
    config_name = str(getattr(config, "_name_or_path", "")).lower()
    model_type = str(getattr(config, "model_type", "")).lower()
    inner = getattr(model, "model", None)
    inner_name = inner.__class__.__name__.lower() if inner is not None else ""
    return "molmo" in class_name or "molmo" in inner_name or "molmo" in config_name or "molmo" in model_type


def _model_dtype(model):
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return dtype
    try:
        return next(iter(model.parameters())).dtype
    except StopIteration:
        return torch.bfloat16


def _load_image(path: str):
    return Image.open(path).convert("RGB")


def _message_parts(message) -> Tuple[str, str, List[Image.Image]]:
    replace_mapping = {
        "\nOptions:": "\nChoices:",
        "Please select the correct answer from the options above.": "Answer with the letter.",
    }
    prompt_parts = []
    answer_parts = []
    images = []
    for msg in message:
        msg_type = msg.get("type")
        if msg_type == "image":
            images.append(_load_image(msg["value"]))
        elif msg_type == "text":
            text = str(msg.get("value", "")).strip()
            for src, dst in replace_mapping.items():
                text = text.replace(src, dst)
            prompt_parts.append(text)
        elif msg_type == "textanw":
            answer_parts.append(str(msg.get("value", "")).strip())
    prompt = "\n".join(part for part in prompt_parts if part).strip()
    answer = "\n".join(part for part in answer_parts if part).strip()
    return prompt, answer, images


def _processor_process(processor, prompt: str, images: List[Image.Image]):
    kwargs = {"text": prompt}
    if images:
        kwargs["images"] = images
    return processor.process(**kwargs)


def _ensure_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _batch_and_move_inputs(inputs: Dict, device, image_dtype=None):
    batch = {}
    for key, value in inputs.items():
        value = _ensure_tensor(value)
        if value.dim() == 0:
            value = value.unsqueeze(0)
        if key == "input_ids" and value.dim() == 1:
            value = value.unsqueeze(0)
        elif key != "input_ids":
            value = value.unsqueeze(0)
        if key == "images" and image_dtype is not None and value.is_floating_point():
            value = value.to(device=device, dtype=image_dtype)
        else:
            value = value.to(device)
        batch[key] = value
    return batch


def build_molmo_eval_inputs(message, processor, device, model=None):
    prompt, _, images = _message_parts(message)
    inputs = _processor_process(processor, prompt, images)
    return _batch_and_move_inputs(inputs, device, image_dtype=_model_dtype(model) if model is not None else None)


def build_molmo_train_inputs(message, processor, device, model=None, label_mode="qa-qa", label_shift=True):
    prompt, answer, images = _message_parts(message)
    if not answer:
        logging.warning("Molmo train input has no textanw answer; all labels will be ignored.")
    if label_mode not in {"qa-qa", "q-a"}:
        raise NotImplementedError(f"Molmo label_mode={label_mode!r} is not supported.")

    question_inputs = _processor_process(processor, prompt, images)
    full_prompt = f"{prompt}{answer}" if not answer else f"{prompt}\n{answer}"
    full_inputs = _processor_process(processor, full_prompt, images)

    inputs = _batch_and_move_inputs(full_inputs, device, image_dtype=_model_dtype(model) if model is not None else None)
    question = _batch_and_move_inputs(question_inputs, device, image_dtype=_model_dtype(model) if model is not None else None)

    labels = inputs["input_ids"].clone()
    question_len = int(question["input_ids"].shape[1])
    labels[:, :question_len] = IGNORE_INDEX

    pad_token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
    if pad_token_id is not None:
        labels[labels == pad_token_id] = IGNORE_INDEX

    if label_shift:
        inputs["input_ids"] = inputs["input_ids"][:, :-1]
        labels = labels[:, 1:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, :-1]

    return inputs, labels


def forward_molmo_prefill(model, inputs, **forward_kwargs):
    """Run a Molmo prefill forward pass without using HF generation.

    Calibration hooks only need the prompt/image forward pass to execute; calling
    Molmo's generate_from_batch can depend on transformers generation internals
    that vary across versions.
    """
    return model(**inputs, use_cache=False, **forward_kwargs)


def generate_molmo(model, processor, inputs, max_new_tokens=16, **generation_kwargs):
    from transformers import GenerationConfig
    import transformers

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        use_cache=True,
        stop_strings="<|endoftext|>",
        pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
        **generation_kwargs,
    )
    try:
        output = model.generate_from_batch(inputs, gen_config, tokenizer=processor.tokenizer)
    except AttributeError as exc:
        if "_extract_past_from_model_output" not in str(exc):
            raise
        raise RuntimeError(
            "Molmo generate_from_batch is incompatible with transformers "
            f"{transformers.__version__}: GenerationMixin has no "
            f"_extract_past_from_model_output. Original error: {exc}"
        ) from exc
    start = inputs["input_ids"].shape[1]
    generated = output[:, start:]
    return processor.tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
