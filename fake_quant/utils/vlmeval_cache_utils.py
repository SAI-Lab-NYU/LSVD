"""Identity-checked cache handling for VLMEvalKit generation and scoring."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


CACHE_SCHEMA_VERSION = 1
PREDICTION_MANIFEST = "vlm_prediction_cache_manifest.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def identity_digest(identity: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_cell(value: Any) -> str:
    if value is None:
        return "<none>"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "<nan>"
    except TypeError:
        pass
    return str(value)


def dataset_fingerprint(dataset: Any) -> str:
    """Hash prompt/scoring metadata without rehashing large base64 image blobs."""

    data = dataset.data
    columns = sorted(column for column in data.columns if column != "image")
    digest = hashlib.sha256()
    digest.update(str(getattr(dataset, "dataset_name", "")).encode("utf-8"))
    for column in columns:
        digest.update(b"\0column\0")
        digest.update(column.encode("utf-8"))
    for row in data[columns].itertuples(index=False, name=None):
        digest.update(b"\0row\0")
        for value in row:
            digest.update(_normalise_cell(value).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _sample_tensor(tensor: torch.Tensor, sample_count: int = 16) -> bytes:
    tensor = tensor.detach()
    if tensor.numel() == 0:
        return b""
    if tensor.is_contiguous():
        flat = tensor.view(-1)
        count = min(sample_count, flat.numel())
        if count == 1:
            indices = torch.zeros(1, device=flat.device, dtype=torch.long)
        else:
            indices = torch.linspace(
                0,
                flat.numel() - 1,
                steps=count,
                device=flat.device,
                dtype=torch.float64,
            ).round().to(torch.long)
        sample = flat.index_select(0, indices)
    else:
        # Avoid materialising a full contiguous copy of a large tensor.
        coordinates = []
        for fraction in (0.0, 0.5, 1.0):
            coordinates.append(
                tuple(
                    min(size - 1, int(round((size - 1) * fraction)))
                    for size in tensor.shape
                )
            )
        sample = torch.stack([tensor[coordinate] for coordinate in coordinates])
    if sample.is_floating_point() or sample.is_complex():
        sample = sample.to(torch.float32)
    elif sample.dtype == torch.bool:
        sample = sample.to(torch.uint8)
    else:
        sample = sample.to(torch.int64)
    return sample.cpu().contiguous().numpy().tobytes()


def model_state_fingerprint(model: torch.nn.Module) -> str:
    """Hash structure plus deterministic samples from every parameter and buffer."""

    digest = hashlib.sha256()
    seen: set[str] = set()
    tensors: list[tuple[str, torch.Tensor]] = []
    tensors.extend(model.named_parameters())
    tensors.extend(model.named_buffers())
    for name, tensor in sorted(tensors, key=lambda item: item[0]):
        if name in seen:
            continue
        seen.add(name)
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_sample_tensor(tensor))
    return digest.hexdigest()


def _generation_config(args: Any, max_new_tokens: int) -> dict[str, Any]:
    keys = (
        "model",
        "seed",
        "w_bits",
        "a_bits",
        "k_bits",
        "v_bits",
        "rank_ratio",
        "svd_mode",
        "svd_modules",
        "calib_method",
        "taylor_order",
        "nsamples",
        "localft_iters",
        "localft_lr",
        "qat_lr_R",
        "qat_lr_UV",
        "qat_start_iter",
        "dosample",
        "case_study",
    )
    config = {
        key: _normalise_cell(getattr(args, key, None))
        for key in keys
    }
    config["max_new_tokens"] = max_new_tokens
    return config


def prediction_cache_identity(
    dataset: Any,
    args: Any,
    model_fingerprint: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    source_file = Path(__file__).with_name("eval_utilsdistllava.py")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "vlm_prediction_cache",
        "dataset": getattr(dataset, "dataset_name", None),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "model_state_fingerprint": model_fingerprint,
        "generation": _generation_config(args, max_new_tokens),
        "evaluator_sha256": sha256_file(source_file),
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _quarantine(
    paths: Iterable[Path],
    root: Path,
    reason: str,
) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    target = root / ".vlm_cache_quarantine" / f"{timestamp}_{reason}_{uuid.uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=False)
    for path in existing:
        destination = target / path.name
        shutil.move(str(path), str(destination))
    return target


def prepare_prediction_cache(
    save_path: str | Path,
    identity: dict[str, Any],
) -> tuple[bool, Path | None]:
    """Prepare generation cache and quarantine artifacts on any identity mismatch."""

    root = Path(save_path)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / PREDICTION_MANIFEST
    expected_digest = identity_digest(identity)
    current = _read_manifest(manifest_path)
    if (
        current is not None
        and current.get("identity_digest") == expected_digest
        and current.get("identity") == identity
    ):
        return True, None

    artifacts = list(root.glob("vlm_eval_rank*.pkl"))
    artifacts.extend(root.glob("vlm_eval.pkl"))
    artifacts.extend(root.glob("vlm_result*"))
    artifacts.append(manifest_path)
    quarantine = _quarantine(artifacts, root, "prediction_identity_mismatch")
    _write_manifest(
        manifest_path,
        {
            "identity": identity,
            "identity_digest": expected_digest,
            "status": "in_progress",
        },
    )
    return False, quarantine


def safe_load_rank_cache(path: str | Path, loader: Any) -> dict[Any, Any]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    try:
        value = loader(str(cache_path))
        if not isinstance(value, dict):
            raise TypeError(f"expected dict, got {type(value).__name__}")
        return value
    except Exception as error:
        quarantine = cache_path.with_name(
            f"{cache_path.name}.corrupt_{uuid.uuid4().hex[:8]}"
        )
        cache_path.replace(quarantine)
        print(
            f"[WARN] quarantined unreadable rank cache {cache_path}: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        return {}


def mark_prediction_cache_complete(
    save_path: str | Path,
    identity: dict[str, Any],
    result_file: str | Path,
) -> None:
    manifest_path = Path(save_path) / PREDICTION_MANIFEST
    _write_manifest(
        manifest_path,
        {
            "identity": identity,
            "identity_digest": identity_digest(identity),
            "result_sha256": sha256_file(result_file),
            "status": "complete",
        },
    )


def _safe_tag(value: str | None) -> str:
    if not value:
        return "exact"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def scoring_cache_artifacts(
    dataset: str,
    result_file: str | Path,
    judge_model: str | None,
) -> tuple[list[Path], Path]:
    from vlmeval.smp import get_intermediate_file_path

    result = str(result_file)
    dataset_lower = dataset.lower()
    artifacts: list[Path] = []
    if dataset_lower in {"hallusionbench", "pope"}:
        artifacts.extend(
            [
                Path(get_intermediate_file_path(result, "_auxmatch")),
                Path(get_intermediate_file_path(result, "_tmp", "pkl")),
                Path(get_intermediate_file_path(result, "_score", "csv")),
            ]
        )
    elif dataset_lower == "ocrbench":
        artifacts.append(Path(get_intermediate_file_path(result, "_score", "json")))
    elif "mmvet" in dataset_lower:
        if not judge_model:
            raise ValueError("MMVet scoring requires an explicit judge model")
        storage = get_intermediate_file_path(result, f"_{judge_model}")
        artifacts.extend(
            [
                Path(storage),
                Path(get_intermediate_file_path(result, f"_{judge_model}", "pkl")),
                Path(get_intermediate_file_path(storage, "_score", "csv")),
                Path(get_intermediate_file_path(storage, "_score_fine", "csv")),
            ]
        )
    manifest = Path(result_file).with_name(
        f"{Path(result_file).stem}_{_safe_tag(dataset)}_"
        f"{_safe_tag(judge_model)}_score_manifest.json"
    )
    return artifacts, manifest


def expected_capability_score_file(
    dataset: str,
    result_file: str | Path,
    judge_model: str | None,
) -> Path | None:
    from vlmeval.smp import get_intermediate_file_path

    result = str(result_file)
    dataset_lower = dataset.lower()
    if dataset_lower in {"hallusionbench", "pope"}:
        return Path(get_intermediate_file_path(result, "_score", "csv"))
    if dataset_lower == "ocrbench":
        return Path(get_intermediate_file_path(result, "_score", "json"))
    if "mmvet" in dataset_lower:
        if not judge_model:
            raise ValueError("MMVet scoring requires an explicit judge model")
        storage = get_intermediate_file_path(result, f"_{judge_model}")
        return Path(get_intermediate_file_path(storage, "_score", "csv"))
    return None


def prepare_scoring_cache(
    dataset: str,
    result_file: str | Path,
    judge_model: str | None,
) -> tuple[dict[str, Any], Path, Path | None]:
    result_path = Path(result_file)
    artifacts, manifest_path = scoring_cache_artifacts(
        dataset, result_path, judge_model
    )
    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "vlm_scoring_cache",
        "dataset": dataset,
        "judge_model": judge_model or "exact_matching",
        "result_sha256": sha256_file(result_path),
    }
    expected_digest = identity_digest(identity)
    current = _read_manifest(manifest_path)
    if (
        current is not None
        and current.get("identity_digest") == expected_digest
        and current.get("identity") == identity
    ):
        return identity, manifest_path, None

    quarantine = _quarantine(
        [*artifacts, manifest_path],
        result_path.parent,
        "scoring_identity_mismatch",
    )
    _write_manifest(
        manifest_path,
        {
            "identity": identity,
            "identity_digest": expected_digest,
            "status": "in_progress",
        },
    )
    return identity, manifest_path, quarantine


def mark_scoring_cache_complete(
    manifest_path: str | Path,
    identity: dict[str, Any],
    score_file: str | Path,
) -> None:
    _write_manifest(
        Path(manifest_path),
        {
            "identity": identity,
            "identity_digest": identity_digest(identity),
            "score_file": str(score_file),
            "score_sha256": sha256_file(score_file),
            "status": "complete",
        },
    )
