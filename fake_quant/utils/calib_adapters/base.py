import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


IGNORE_INDEX = -100
ROLE_PROMPT = 0
ROLE_REASONING = 1
ROLE_FINAL_ANSWER = 2
ROLE_OTHER_ASSISTANT = 3


@dataclass
class ReasoningTrace:
    sample_index: int
    adapter_name: str
    raw_generation: str
    reasoning_text: str = ""
    final_text: str = ""
    prompt_text: str = ""
    generated_token_ids: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_index": int(self.sample_index),
            "adapter_name": self.adapter_name,
            "raw_generation": self.raw_generation,
            "reasoning_text": self.reasoning_text,
            "final_text": self.final_text,
            "prompt_text": self.prompt_text,
            "generated_token_ids": self.generated_token_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningTrace":
        return cls(
            sample_index=int(data.get("sample_index", 0)),
            adapter_name=str(data.get("adapter_name", "")),
            raw_generation=str(data.get("raw_generation", "")),
            reasoning_text=str(data.get("reasoning_text", "")),
            final_text=str(data.get("final_text", "")),
            prompt_text=str(data.get("prompt_text", "")),
            generated_token_ids=data.get("generated_token_ids", None),
        )


@dataclass
class ReasoningBatch:
    inputs: Dict[str, Any]
    labels: torch.Tensor
    role_mask: torch.Tensor
    meta: Dict[str, Any]


class ReasoningAdapter:
    name = "base"

    @classmethod
    def supports(cls, args=None, model=None, tokenizer=None, image_processor=None) -> bool:
        raise NotImplementedError

    def __init__(self, args=None, model=None, tokenizer=None, image_processor=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def build_prompt_inputs(self, sample, device):
        raise NotImplementedError

    def decode_generated(self, generated_ids) -> str:
        raise NotImplementedError

    def build_train_batch(self, sample, trace: ReasoningTrace, device) -> ReasoningBatch:
        raise NotImplementedError

    def cache_tag(self) -> str:
        return ""

    def split_reasoning_final(self, raw_text: str) -> Tuple[str, str]:
        return split_reasoning_final(raw_text)

    def generate_trace(self, model, sample, sample_index: int, device, args) -> ReasoningTrace:
        prompt_inputs = self.build_prompt_inputs(sample, device)
        max_new_tokens = int(getattr(args, "reasoning_max_new_tokens", 4096))
        temperature = float(getattr(args, "reasoning_temperature", 0.0))
        do_sample = temperature > 0.0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": do_sample,
        }
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            generation_config = copy.deepcopy(generation_config)
            generation_config.do_sample = do_sample
            if do_sample:
                generation_config.temperature = temperature
            else:
                # Qwen3-VL Thinking checkpoints may ship sampling defaults in
                # generation_config. Greedy trace generation ignores them, but
                # Transformers warns unless they are reset to neutral values.
                neutral_sampling = {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 50,
                    "typical_p": 1.0,
                    "epsilon_cutoff": 0.0,
                    "eta_cutoff": 0.0,
                    "min_p": None,
                }
                for key, value in neutral_sampling.items():
                    if hasattr(generation_config, key):
                        setattr(generation_config, key, value)
            generation_kwargs["generation_config"] = generation_config
        elif temperature > 0.0:
            generation_kwargs["temperature"] = temperature
        with torch.no_grad():
            generated = model.generate(**prompt_inputs, **generation_kwargs)
        input_ids = prompt_inputs["input_ids"]
        generated_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated)
        ]
        raw_text = self.decode_generated(generated_ids)
        reasoning_text, final_text = self.split_reasoning_final(raw_text)
        prompt_text = self.prompt_text_from_sample(sample)
        return ReasoningTrace(
            sample_index=sample_index,
            adapter_name=self.name,
            raw_generation=raw_text,
            reasoning_text=reasoning_text,
            final_text=final_text,
            prompt_text=prompt_text,
            generated_token_ids=generated_ids[0].detach().cpu().tolist() if generated_ids else None,
        )

    def prompt_text_from_sample(self, sample) -> str:
        texts = []
        if isinstance(sample, list):
            for item in sample:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item.get("value", "")))
        return "\n".join(texts)


def split_reasoning_final(raw_text: str) -> Tuple[str, str]:
    text = raw_text or ""
    reasoning = ""
    for pattern in [
        r"<think>(.*?)</think>",
        r"<thinking>(.*?)</thinking>",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            final = (text[: match.start()] + " " + text[match.end() :]).strip()
            return reasoning, _strip_special(final)

    close_match = re.search(r"</think>|</thinking>", text, flags=re.IGNORECASE)
    if close_match:
        reasoning = text[: close_match.start()].strip()
        final = text[close_match.end() :].strip()
        return _strip_special(reasoning), _strip_special(final)

    return "", _strip_special(text)


def _strip_special(text: str) -> str:
    text = text.replace("<s>", "").replace("</s>", "").strip()
    text = re.sub(r"<\|[^|]+?\|>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sample_with_answer(sample, answer_text: str):
    if not isinstance(sample, list):
        raise NotImplementedError("Reasoning calibration expects list-of-message samples.")
    copied = []
    for item in sample:
        if isinstance(item, dict) and item.get("type") == "textanw":
            continue
        copied.append(dict(item) if isinstance(item, dict) else item)
    copied.append({"type": "textanw", "value": answer_text})
    return copied


def find_final_start_prefix(raw_generation: str, final_text: str) -> Optional[str]:
    if not raw_generation or not final_text:
        return None
    idx = raw_generation.rfind(final_text)
    if idx < 0:
        return None
    return raw_generation[:idx]


def clean_generation_for_replay(text: str) -> str:
    text = (text or "").strip()
    stop_tokens = [
        "<|im_end|>",
        "<|endoftext|>",
        "<|eot_id|>",
        "<|end|>",
        "</s>",
    ]
    changed = True
    while changed:
        changed = False
        for token in stop_tokens:
            if text.endswith(token):
                text = text[: -len(token)].strip()
                changed = True
    return text


def first_token_divergence(full_input_ids: torch.Tensor, prefix_input_ids: torch.Tensor) -> Optional[int]:
    if full_input_ids is None or prefix_input_ids is None:
        return None
    full = full_input_ids[0] if full_input_ids.ndim > 1 else full_input_ids
    prefix = prefix_input_ids[0] if prefix_input_ids.ndim > 1 else prefix_input_ids
    limit = min(int(full.numel()), int(prefix.numel()))
    if limit <= 0:
        return 0
    diff = torch.nonzero(full[:limit] != prefix[:limit], as_tuple=False).flatten()
    if len(diff) > 0:
        return int(diff[0].item())
    return limit


def find_subsequence(sequence: List[int], subsequence: List[int], from_end: bool = True) -> Optional[int]:
    if not sequence or not subsequence or len(subsequence) > len(sequence):
        return None
    starts = range(len(sequence) - len(subsequence), -1, -1) if from_end else range(0, len(sequence) - len(subsequence) + 1)
    for start in starts:
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return None


def encode_text_no_special(tokenizer, text: str) -> List[int]:
    if tokenizer is None or not text:
        return []
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def replay_batch_from_prompt_inputs(
    prompt_inputs,
    generated_token_ids: List[int],
    tokenizer=None,
    final_text: str = "",
    default_role: int = ROLE_FINAL_ANSWER,
) -> ReasoningBatch:
    inputs = dict(prompt_inputs)
    prompt_ids = inputs["input_ids"]
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise NotImplementedError("Reasoning calibration exact replay currently expects batch size 1.")
    gen_ids = torch.tensor(
        [int(x) for x in generated_token_ids],
        device=prompt_ids.device,
        dtype=prompt_ids.dtype,
    ).view(1, -1)
    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
    prompt_len = int(prompt_ids.shape[1])
    inputs["input_ids"] = full_ids
    if "attention_mask" in inputs and isinstance(inputs["attention_mask"], torch.Tensor):
        mask = inputs["attention_mask"]
        gen_mask = torch.ones((mask.shape[0], gen_ids.shape[1]), device=mask.device, dtype=mask.dtype)
        inputs["attention_mask"] = torch.cat([mask, gen_mask], dim=1)

    labels = full_ids.clone()
    labels[:, :prompt_len] = IGNORE_INDEX
    role_mask = torch.zeros_like(labels, dtype=torch.long)
    if gen_ids.shape[1] > 0:
        role_mask[:, prompt_len:] = int(default_role)

    final_ids = encode_text_no_special(tokenizer, final_text)
    final_offset = find_subsequence([int(x) for x in generated_token_ids], final_ids, from_end=True)
    if final_offset is not None:
        final_start = prompt_len + int(final_offset)
        positions = torch.arange(full_ids.shape[1], device=full_ids.device).view(1, -1)
        assistant_mask = labels != IGNORE_INDEX
        role_mask[assistant_mask & (positions < final_start)] = ROLE_REASONING
        role_mask[assistant_mask & (positions >= final_start)] = ROLE_FINAL_ANSWER

    return ReasoningBatch(
        inputs=inputs,
        labels=labels,
        role_mask=role_mask,
        meta={
            "prompt_token_length": prompt_len,
            "exact_generated_token_replay": True,
            "generated_token_count": int(gen_ids.shape[1]),
            "final_token_offset": final_offset,
        },
    )


def role_mask_from_labels(
    labels: torch.Tensor,
    default_role: int,
    prompt_len: Optional[int] = None,
    final_start: Optional[int] = None,
) -> torch.Tensor:
    role_mask = torch.zeros_like(labels, dtype=torch.long)
    assistant_mask = labels != IGNORE_INDEX
    role_mask[assistant_mask] = int(default_role)
    if final_start is None:
        return role_mask

    seq_len = labels.shape[1]
    final_start = max(0, min(int(final_start), seq_len))
    prompt_len = 0 if prompt_len is None else max(0, min(int(prompt_len), seq_len))
    positions = torch.arange(seq_len, device=labels.device).view(1, -1)
    reasoning_mask = assistant_mask & (positions >= prompt_len) & (positions < final_start)
    final_mask = assistant_mask & (positions >= final_start)
    role_mask[reasoning_mask] = ROLE_REASONING
    role_mask[final_mask] = ROLE_FINAL_ANSWER
    return role_mask
