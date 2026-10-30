import hashlib
import json
import logging
import os
from typing import Iterable, List

import torch
import torch.nn.functional as F

from .calib_adapters import get_reasoning_adapter
from .calib_adapters.base import (
    IGNORE_INDEX,
    ROLE_FINAL_ANSWER,
    ROLE_OTHER_ASSISTANT,
    ROLE_REASONING,
    ReasoningBatch,
    ReasoningTrace,
)


def enabled(args) -> bool:
    return bool(getattr(args, "reasoning_calib", False))


def _safe_token(value) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def rollout_engine(args) -> str:
    return str(getattr(args, "reasoning_rollout_engine", "vllm")).lower()


def trace_cache_path(args, adapter_name: str, adapter_cache_tag: str = "") -> str:
    return _trace_cache_path(args, adapter_name, adapter_cache_tag)


def _trace_cache_path(args, adapter_name: str, adapter_cache_tag: str = "") -> str:
    cache_dir = getattr(args, "reasoning_trace_cache_dir", None)
    if cache_dir is None:
        cache_dir = os.path.join(getattr(args, "act_cache_dir", "."), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    mode = getattr(args, "reasoning_calib_mode", "rollout_teacher_forcing")
    engine = rollout_engine(args)
    model = _safe_token(getattr(args, "model", "model"))
    dataset = _safe_token(getattr(args, "cal_dataset", "data"))
    tokens = [
        model,
        dataset,
        f"n{getattr(args, 'nsamples', 'na')}",
        f"s{getattr(args, 'seed', 'na')}",
        adapter_name,
        _safe_token(adapter_cache_tag) if adapter_cache_tag else "",
        _safe_token(engine),
        _safe_token(mode),
        f"mt{getattr(args, 'reasoning_max_new_tokens', 4096)}",
        f"temp{getattr(args, 'reasoning_temperature', 0.0)}",
    ]
    tokens = [token for token in tokens if token]
    digest = hashlib.sha1("_".join(tokens).encode("utf-8")).hexdigest()[:10]
    return os.path.join(cache_dir, "_".join(tokens) + f"_{digest}_reasoning_traces.pt")


def get_adapter(args, model, tokenizer, image_processor):
    return get_reasoning_adapter(
        args=args,
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )


def cache_tag(args, model=None, tokenizer=None, image_processor=None) -> str:
    if not enabled(args):
        return ""
    adapter_name = str(getattr(args, "reasoning_adapter_name", "adapter_unknown"))
    if model is not None:
        adapter = get_adapter(args, model, tokenizer, image_processor)
        adapter_name = adapter.name
        adapter_cache_tag = adapter.cache_tag()
        try:
            setattr(args, "reasoning_adapter_name", adapter_name)
            setattr(args, "reasoning_adapter_cache_tag", adapter_cache_tag)
        except Exception:
            pass
    else:
        adapter_cache_tag = str(getattr(args, "reasoning_adapter_cache_tag", ""))
    temp = str(getattr(args, "reasoning_temperature", 0.0)).replace(".", "p")
    rw = str(getattr(args, "reasoning_weight", 1.0)).replace(".", "p")
    aw = str(getattr(args, "answer_weight", 1.0)).replace(".", "p")
    engine = rollout_engine(args)
    adapter_cache_part = f"_{adapter_cache_tag}" if adapter_cache_tag else ""
    return (
        f"_reasoning_{adapter_name}{adapter_cache_part}_{engine}_{getattr(args, 'reasoning_calib_mode', 'rollout_teacher_forcing')}"
        f"_mt{getattr(args, 'reasoning_max_new_tokens', 4096)}_temp{temp}_rw{rw}_aw{aw}"
    )


def _as_list(dataloader: Iterable):
    if isinstance(dataloader, list):
        return dataloader
    return list(dataloader)


def sample_fingerprint(sample) -> str:
    payload = json.dumps(sample, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _expected_prompt_text(adapter, sample) -> str:
    try:
        return adapter.prompt_text_from_sample(sample)
    except Exception:
        return ""


def _validate_trace_samples(payload, traces, samples, adapter):
    if samples is None:
        return None
    expected_count = len(samples)
    payload_fingerprints = payload.get("sample_fingerprints", None)
    if payload_fingerprints is not None:
        expected_fingerprints = [sample_fingerprint(sample) for sample in samples]
        if list(payload_fingerprints[:expected_count]) != expected_fingerprints:
            return "Reasoning trace cache sample fingerprints do not match current calibration samples."
        return None

    prompt_checked = 0
    for sample_index, (sample, trace) in enumerate(zip(samples, traces)):
        if trace.sample_index != sample_index:
            return (
                f"Reasoning trace sample_index mismatch at position {sample_index}: "
                f"cache has {trace.sample_index}."
            )
        if not trace.prompt_text:
            continue
        prompt_checked += 1
        expected_prompt = _expected_prompt_text(adapter, sample)
        if expected_prompt and trace.prompt_text != expected_prompt:
            return (
                f"Reasoning trace prompt_text mismatch at sample {sample_index}; "
                "cache samples do not match the current calibration dataloader."
            )

    if prompt_checked == expected_count:
        return None
    return (
        "Reasoning trace cache has no sample_fingerprints and not all traces have "
        "prompt_text for sample alignment validation."
    )


def _load_valid_trace_cache(cache_file: str, nsamples: int, samples=None, adapter=None):
    payload = torch.load(cache_file, map_location="cpu")
    traces = [ReasoningTrace.from_dict(item) for item in payload.get("traces", [])]
    if len(traces) < nsamples:
        return None, (
            f"Reasoning trace cache has {len(traces)} traces but {nsamples} samples are needed."
        )
    if not all(trace.generated_token_ids is not None for trace in traces[:nsamples]):
        return None, "Reasoning trace cache has no exact generated token ids."
    samples = samples[:nsamples] if samples is not None else None
    sample_problem = _validate_trace_samples(payload, traces[:nsamples], samples, adapter)
    if sample_problem is not None:
        return None, sample_problem
    return traces[:nsamples], None


def _raise_missing_trace_cache(cache_file: str, reason: str, args):
    engine = rollout_engine(args)
    raise FileNotFoundError(
        f"Reasoning trace cache is required for rollout_engine={engine}, but it is not usable.\n"
        f"Cache path: {cache_file}\n"
        f"Reason: {reason}\n"
        "Generate it offline first with fake_quant/tools/generate_reasoning_traces_vllm.py, "
        "or explicitly use --reasoning_rollout_engine hf for legacy online HF generation."
    )


def load_or_generate_traces(model, dataloader, tokenizer, image_processor, args, device=None) -> List[ReasoningTrace]:
    if not enabled(args):
        return []
    device = device if device is not None else next(model.parameters()).device
    adapter = get_adapter(args, model, tokenizer, image_processor)
    adapter_cache_tag = adapter.cache_tag()
    try:
        setattr(args, "reasoning_adapter_name", adapter.name)
        setattr(args, "reasoning_adapter_cache_tag", adapter_cache_tag)
    except Exception:
        pass
    samples = _as_list(dataloader)
    cache_file = _trace_cache_path(args, adapter.name, adapter_cache_tag)
    engine = rollout_engine(args)
    should_read_cache = engine != "hf" or bool(getattr(args, "use_cache", True))
    cache_problem = "cache file does not exist"
    if should_read_cache and os.path.exists(cache_file):
        traces, cache_problem = _load_valid_trace_cache(
            cache_file,
            len(samples),
            samples=samples,
            adapter=adapter,
        )
        if traces is not None:
            logging.info("Loaded reasoning trace cache from %s", cache_file)
            return traces
        logging.info("Found unusable reasoning trace cache at %s: %s", cache_file, cache_problem)

    if engine != "hf":
        _raise_missing_trace_cache(cache_file, cache_problem, args)

    logging.info("Generating reasoning calibration traces with adapter=%s", adapter.name)
    was_training = model.training
    model.eval()
    traces = []
    for sample_index, sample in enumerate(samples):
        traces.append(adapter.generate_trace(model, sample, sample_index, device, args))
    if was_training:
        model.train()
    torch.save(
        {
            "adapter": adapter.name,
            "adapter_cache_tag": adapter_cache_tag,
            "model": getattr(args, "model", None),
            "rollout_engine": "hf",
            "sample_fingerprints": [sample_fingerprint(sample) for sample in samples],
            "traces": [trace.to_dict() for trace in traces],
        },
        cache_file,
    )
    logging.info("Saved reasoning trace cache to %s", cache_file)
    return traces


def materialize_reasoning_batches(model, dataloader, tokenizer, image_processor, args, device=None) -> List[ReasoningBatch]:
    if not enabled(args):
        return []
    device = device if device is not None else next(model.parameters()).device
    adapter = get_adapter(args, model, tokenizer, image_processor)
    samples = _as_list(dataloader)
    traces = load_or_generate_traces(
        model=model,
        dataloader=samples,
        tokenizer=tokenizer,
        image_processor=image_processor,
        args=args,
        device=device,
    )
    batches = []
    for sample, trace in zip(samples, traces):
        batches.append(adapter.build_train_batch(sample, trace, device))
    return batches


def forward_reasoning_batch(model, batch: ReasoningBatch):
    try:
        return model(**batch.inputs, use_cache=False)
    except TypeError:
        return model(**batch.inputs)


def _get_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    raise TypeError("Cannot locate logits in model outputs for reasoning calibration loss.")


def compute_reasoning_loss(outputs, batch: ReasoningBatch, args):
    logits = _get_logits(outputs)
    labels = batch.labels.to(logits.device)
    role_mask = batch.role_mask.to(logits.device)
    if logits.size(1) < 2 or labels.size(1) < 2:
        raise ValueError("Reasoning calibration batch is too short to compute shifted CE loss.")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_roles = role_mask[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view_as(shift_labels)

    valid = shift_labels != IGNORE_INDEX
    weights = torch.zeros_like(loss, dtype=torch.float32)
    reasoning_weight = float(getattr(args, "reasoning_weight", 1.0))
    answer_weight = float(getattr(args, "answer_weight", 1.0))
    weights[shift_roles == ROLE_REASONING] = reasoning_weight
    weights[shift_roles == ROLE_FINAL_ANSWER] = answer_weight
    weights[shift_roles == ROLE_OTHER_ASSISTANT] = answer_weight
    weights = weights * valid.float()
    denom = weights.sum()
    if denom <= 0:
        weights = valid.float()
        denom = weights.sum().clamp_min(1.0)
    return (loss * weights).sum() / denom
