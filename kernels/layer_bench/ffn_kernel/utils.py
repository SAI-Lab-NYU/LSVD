import gc
import logging
import math
import pickle
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import SDPBackend

KERNELS_DIR = Path(__file__).resolve().parents[2]
FD_KERNEL_DIR = KERNELS_DIR / "layer_bench" / "fd_kernel"
FFN_DATA_DIR = KERNELS_DIR / "autotune_cache"
if str(FD_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(FD_KERNEL_DIR))

from flash_int8_latent_per_head_decoding_rope_stage1 import _fwd_kernel_flash_int8_latent_decode_rope_stage1


LOGGER = logging.getLogger(__name__)


# ---- Triton autotune configs used by MixFFN kernels ----
def get_rmsnorm_quant_autotune_config():
    configs = []
    for block_m in [1, 2, 4, 8, 16, 32]:
        for block_k in [64, 128, 256, 512]:
            for stages in [2, 3, 4]:
                for warps in [2, 4, 8]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_K": block_k},
                            num_stages=stages,
                            num_warps=warps,
                        )
                    )
    return configs


def get_dequant_autotune_config():
    configs = []
    for block_m in [16, 32, 64, 128]:
        for block_n in [64, 128, 256, 512]:
            for stages in [2, 3, 4]:
                for warps in [2, 4, 8]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n},
                            num_stages=stages,
                            num_warps=warps,
                        )
                    )
    return configs


def get_swiglu_a2_autotune_config():
    configs = []
    for block_m in [16, 32, 64, 128]:
        for block_n in [64, 128, 256, 512]:
            for group_n in [2, 4, 8]:
                for stages in [2, 3, 4]:
                    for warps in [2, 4, 8]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "GROUP_N": group_n},
                                num_stages=stages,
                                num_warps=warps,
                            )
                        )
    return configs


def get_swiglu_a2_vec_autotune_config():
    configs = []
    for block_n in [64, 128, 256, 512, 1024]:
        for stages in [2, 3, 4]:
            for warps in [2, 4, 8]:
                configs.append(
                    triton.Config(
                        {"BLOCK_N": block_n},
                        num_stages=stages,
                        num_warps=warps,
                    )
                )
    return configs


def get_token_quant_autotune_config():
    configs = []
    for block_m in [1, 2, 4, 8, 16, 32]:
        for block_k in [64, 128, 256, 512]:
            for stages in [2, 3, 4]:
                for warps in [2, 4, 8]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_K": block_k},
                            num_stages=stages,
                            num_warps=warps,
                        )
                    )
    return configs


def get_swiglu_a2_vec_pack_autotune_config():
    configs = []
    for block_n in [64, 128, 256, 512, 1024]:
        for stages in [2, 3, 4]:
            for warps in [2, 4, 8]:
                configs.append(
                    triton.Config(
                        {"BLOCK_N": block_n, "PACK_N": block_n // 2},
                        num_stages=stages,
                        num_warps=warps,
                    )
                )
    return configs


def get_token_quant_pack_autotune_config():
    configs = []
    for block_m in [1, 2, 4, 8, 16, 32]:
        for block_k in [64, 128, 256, 512]:
            for stages in [2, 3, 4]:
                for warps in [2, 4, 8]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_K": block_k, "PACK_K": block_k // 2},
                            num_stages=stages,
                            num_warps=warps,
                        )
                    )
    return configs


def get_rmsnorm_quant_pack_autotune_config():
    configs = []
    for block_m in [1, 2, 4, 8, 16, 32]:
        for block_k in [64, 128, 256, 512]:
            for stages in [2, 3, 4]:
                for warps in [2, 4, 8]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_K": block_k, "PACK_K": block_k // 2},
                            num_stages=stages,
                            num_warps=warps,
                        )
                    )
    return configs


def get_w8a8_vec_matmul_autotune_config():
    configs = []
    for block_m in [1, 2, 4, 8, 16]:
        for block_n in [16, 32, 64]:
            for block_k in [32, 64]:
                for warps in [2, 4, 8]:
                    for stages in [2, 3, 4]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                                num_warps=warps,
                                num_stages=stages,
                            )
                        )
    return configs


def get_w8a8_vec_up_matmul_autotune_config():
    configs = []
    for block_m in [1, 2, 4, 8, 16]:
        for block_n in [32, 64, 128]:
            for block_k in [32, 64, 128]:
                for warps in [2, 4, 8]:
                    for stages in [2, 3, 4]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                                num_warps=warps,
                                num_stages=stages,
                            )
                        )
    return configs


def get_w8a8_vec_down_matmul_autotune_config():
    configs = []
    for block_m in [1, 2, 4, 8, 16]:
        for block_n in [16, 32, 64]:
            for block_k in [64, 128, 256]:
                for warps in [2, 4, 8]:
                    for stages in [2, 3, 4]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                                num_warps=warps,
                                num_stages=stages,
                            )
                        )
    return configs


def get_w8a8_dot_matmul_autotune_config():
    configs = []
    # tl.dot requires compile-time M/N/K tile dimensions of at least 16.
    for block_m in [16, 32, 64]:
        for block_n in [32, 64, 128]:
            for block_k in [64, 128, 256]:
                for warps in [2, 4, 8]:
                    for stages in [2, 3, 4]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                                num_warps=warps,
                                num_stages=stages,
                            )
                        )
    return configs


def get_w8a8_dot_up_matmul_autotune_config():
    configs = []
    for block_m in [16, 32]:
        for block_n in [64, 128, 256]:
            for block_k in [64, 128, 256]:
                for warps in [4, 8]:
                    for stages in [3, 4]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                                num_warps=warps,
                                num_stages=stages,
                            )
                        )
    return configs


def get_w8a8_dot_down_matmul_autotune_config():
    configs = []
    for block_m in [16, 32]:
        for block_n in [32, 64, 128]:
            for block_k in [128, 256, 512, 1024]:
                for warps in [4, 8]:
                    for stages in [3, 4]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                                num_warps=warps,
                                num_stages=stages,
                            )
                        )
    return configs


def get_swiglu_a2_quant_block_autotune_config():
    configs = []
    for block_m in [1, 2, 4, 8, 16]:
        for block_n in [256, 512, 1024]:
            for warps in [2, 4, 8]:
                for stages in [2, 3, 4]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n},
                            num_warps=warps,
                            num_stages=stages,
                        )
                    )
    return configs


def get_w8a8_svd_up_a_matmul_autotune_config():
    configs = []
    for block_m in [16, 32]:
        for block_n in [32, 64, 128]:
            for block_k in [128, 256, 512]:
                for warps in [4, 8]:
                    for stages in [3, 4]:
                        configs.append(triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k}, num_warps=warps, num_stages=stages))
    return configs


def get_w8a8_svd_up_b_matmul_autotune_config():
    configs = []
    for block_m in [16, 32]:
        for block_n in [64, 128, 256]:
            for block_k in [64, 128, 256, 512]:
                for warps in [4, 8]:
                    for stages in [3, 4]:
                        configs.append(triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k}, num_warps=warps, num_stages=stages))
    return configs


def get_w8a8_svd_down_a_matmul_autotune_config():
    configs = []
    for block_m in [16, 32]:
        for block_n in [32, 64, 128]:
            for block_k in [256, 512, 1024]:
                for warps in [4, 8]:
                    for stages in [3, 4]:
                        configs.append(triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k}, num_warps=warps, num_stages=stages))
    return configs


def get_w8a8_svd_down_b_matmul_autotune_config():
    configs = []
    for block_m in [16, 32]:
        for block_n in [32, 64, 128]:
            for block_k in [64, 128, 256, 512]:
                for warps in [4, 8]:
                    for stages in [3, 4]:
                        configs.append(triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k}, num_warps=warps, num_stages=stages))
    return configs


def get_svd_rank_quant_autotune_config():
    configs = []
    for block_r in [64, 128, 256, 512, 1024]:
        for warps in [2, 4, 8]:
            for stages in [2, 3, 4]:
                configs.append(triton.Config({"BLOCK_R": block_r}, num_warps=warps, num_stages=stages))
    return configs


def get_svd_swiglu_quant_autotune_config():
    configs = []
    for block_n in [256, 512, 1024]:
        for warps in [2, 4, 8]:
            for stages in [2, 3, 4]:
                configs.append(triton.Config({"BLOCK_N": block_n}, num_warps=warps, num_stages=stages))
    return configs


def get_swiglu_quant_autotune_config():
    # Compatibility shim for the shared autotune summary code. The real v4
    # SwiGLU kernel is row-wise, so BLOCK_M is fixed to one logical row.
    configs = []
    for cfg in get_svd_swiglu_quant_autotune_config():
        kwargs = cfg.all_kwargs()
        configs.append(
            triton.Config(
                {"BLOCK_M": 1, "BLOCK_N": kwargs["BLOCK_N"]},
                num_warps=kwargs["num_warps"],
                num_stages=kwargs["num_stages"],
            )
        )
    return configs


def get_two_branch_autotune_config(vec_input: bool = False):
    configs = []
    if vec_input:
        for bo in [16, 32, 64, 128]:
            for bh in [16, 32, 64, 128]:
                for stages in [2, 3, 4, 5]:
                    for warps in [2, 4, 8]:
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_O": bo,
                                    "BLOCK_H": bh,
                                },
                                num_stages=stages,
                                num_warps=warps,
                            )
                        )
    else:
        for bm in [16, 32, 64, 128]:
            for bo in [16, 32, 64, 128]:
                for bh in [16, 32, 64, 128]:
                    for stages in [2, 3, 4, 5]:
                        for warps in [2, 4, 8]:
                            configs.append(
                                triton.Config(
                                    {
                                        "BLOCK_M": bm,
                                        "BLOCK_O": bo,
                                        "BLOCK_H": bh,
                                    },
                                    num_stages=stages,
                                    num_warps=warps,
                                )
                            )
    return configs


def get_dual_quantize_vec_autotune_config():
    configs = []
    for block_n in [64, 128, 256, 512, 1024]:
        for stages in [2, 3, 4]:
            for warps in [2, 4, 8]:
                configs.append(triton.Config({"BLOCK_N": block_n}, num_warps=warps, num_stages=stages))
    return configs


def get_dense_gateup_swiglu_autotune_config():
    configs = []
    # tl.dot-backed kernels need MMA-friendly tile sizes; tiny BLOCK_M values
    # such as 1/2/4/8 fail Triton lowering for these shapes.
    for block_m in [16, 32, 64]:
        for block_n in [64, 128, 256]:
            for block_k in [64, 128]:
                for stages in [2, 3, 4]:
                    for warps in [2, 4, 8]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                                num_warps=warps,
                                num_stages=stages,
                            )
                        )
    return configs


def get_dual_svd_up_a_dequant_autotune_config():
    configs = []
    # tl.dot-backed kernels need MMA-friendly tile sizes; tiny BLOCK_M values
    # such as 1/2/4/8 fail Triton lowering for these shapes.
    for block_m in [16, 32, 64]:
        for block_r in [32, 64, 128, 256]:
            for block_k in [64, 128]:
                for stages in [2, 3, 4]:
                    for warps in [2, 4, 8]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_R": block_r, "BLOCK_K": block_k},
                                num_warps=warps,
                                num_stages=stages,
                            )
                        )
    return configs


def get_svd_down_a_quant_autotune_config():
    configs = []
    # Down projection flips the FFN shape: K is the large intermediate
    # dimension and R/N are the smaller hidden-side tiles.
    for block_m in [16, 32]:
        for block_r in [32, 64, 128]:
            for block_k in [128, 256, 512, 1024]:
                for stages in [3, 4]:
                    for warps in [4, 8]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_R": block_r, "BLOCK_K": block_k},
                                num_warps=warps,
                                num_stages=stages,
                            )
                        )
    return configs


def get_svd_down_b_dequant_autotune_config():
    configs = []
    for block_m in [16, 32]:
        for block_n in [32, 64, 128]:
            for block_r in [64, 128, 256, 512]:
                for stages in [3, 4]:
                    for warps in [4, 8]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_R": block_r},
                                num_warps=warps,
                                num_stages=stages,
                            )
                        )
    return configs


def get_dense_down_dequant_autotune_config():
    configs = []
    for block_m in [16, 32]:
        for block_n in [32, 64, 128]:
            for block_k in [128, 256, 512, 1024]:
                for stages in [3, 4]:
                    for warps in [4, 8]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                                num_warps=warps,
                                num_stages=stages,
                            )
                        )
    return configs


def get_mixed_gateup_swiglu_from_latents_autotune_config():
    configs = []
    # tl.dot-backed kernels need MMA-friendly tile sizes; tiny BLOCK_M values
    # such as 1/2/4/8 fail Triton lowering for these shapes.
    for block_m in [16, 32, 64]:
        for block_n in [64, 128, 256]:
            for block_r in [32, 64, 128, 256]:
                for stages in [2, 3, 4]:
                    for warps in [2, 4, 8]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_R": block_r},
                                num_warps=warps,
                                num_stages=stages,
                            )
                        )
    return configs

ATTENTION_KERNELS = (_fwd_kernel_flash_int8_latent_decode_rope_stage1,)
MIX_ATTENTION_KERNEL_NAME = f"{_fwd_kernel_flash_int8_latent_decode_rope_stage1.base_fn.__name__}_mix"
DEFAULT_PROVIDER_RECORD_NAMES = {"mixffn"}
DEFAULT_PROVIDERS = ["mixffn"]
FFN_STAGE_BENCH_PROVIDERS = {"mixffn"}
FAKE_FP16_CORRECTNESS_PROVIDERS = FFN_STAGE_BENCH_PROVIDERS
FAKE_FP16_STAGE_ORDER = ("x_norm", "up", "gate", "mid", "down", "out")


def _mix():
    from . import mix_ffn_kernel

    return mix_ffn_kernel


class _MixMapping(Mapping):
    def __init__(self, attr: str):
        self.attr = attr

    @property
    def _data(self):
        return getattr(_mix(), self.attr)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


class _LayerKernelMapping(Mapping):
    def __getitem__(self, key):
        if key == "__attention__":
            return ATTENTION_KERNELS
        return FFN_KERNELS[key]

    def __iter__(self):
        yield "__attention__"
        yield from FFN_KERNELS

    def __len__(self):
        return 1 + len(FFN_KERNELS)


FFN_NAME_MAP = _MixMapping("NAME")
FFN_KERNELS = _MixMapping("KERNELS")
LAYER_KERNELS = _LayerKernelMapping()


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    original_dtype = x.dtype
    x_float = x.to(torch.float32)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_float * torch.rsqrt(variance + eps)
    return (x_norm.to(weight.dtype) * weight).to(original_dtype)


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_attention_kernel_record_name(provider: str) -> str:
    return MIX_ATTENTION_KERNEL_NAME


def get_kernel_record_name(autotuner, provider: str | None = None) -> str:
    if autotuner in ATTENTION_KERNELS and provider is not None:
        return get_attention_kernel_record_name(provider)
    return autotuner.base_fn.__name__


def _iter_cache_records(records: Dict, path: Tuple[str, ...] = ()) -> Iterator[Tuple[Tuple[str, ...], str, Dict]]:
    for key, value in records.items():
        if not isinstance(value, dict):
            continue
        if "cache" in value and "keys" in value:
            yield path, key, value
            continue
        yield from _iter_cache_records(value, (*path, key))


def apply_cache_records(best_config_pkl_path: Path, kernels_map: Dict[str, tuple], provider_filter: str | None = None) -> None:
    if not best_config_pkl_path.exists():
        LOGGER.warning("Cache file not found: %s", best_config_pkl_path)
        return

    LOGGER.info("Loading autotune cache: %s", best_config_pkl_path)
    with best_config_pkl_path.open("rb") as f:
        records = pickle.load(f)

    cache_records = {}
    for path, key, value in _iter_cache_records(records):
        if provider_filter is not None:
            if provider_filter not in path:
                continue
        elif any(segment in DEFAULT_PROVIDER_RECORD_NAMES for segment in path):
            continue
        cache_records[key] = value
    if not cache_records:
        LOGGER.warning("No compatible cache records found in %s", best_config_pkl_path)
        return

    if provider_filter is not None and provider_filter in kernels_map:
        kernel_groups = (kernels_map["__attention__"], kernels_map[provider_filter])
    else:
        kernel_groups = kernels_map.values()

    seen = set()
    for kernels in kernel_groups:
        for autotuner in kernels:
            if not isinstance(autotuner, triton.runtime.Autotuner):
                continue
            record_name = get_kernel_record_name(autotuner, provider_filter)
            if record_name in seen:
                continue
            seen.add(record_name)
            if record_name not in cache_records:
                continue
            record = cache_records[record_name]
            if autotuner.keys != record["keys"]:
                LOGGER.warning("Skipping %s due to key mismatch", record_name)
                continue
            autotuner.cache.update(record["cache"])
            LOGGER.info("Loaded cache for %s", record_name)


def clear_cache_records(kernels_map: Dict[str, tuple]) -> None:
    seen = set()
    for kernels in kernels_map.values():
        for autotuner in kernels:
            if not isinstance(autotuner, triton.runtime.Autotuner):
                continue
            fn_name = autotuner.base_fn.__name__
            if fn_name in seen:
                continue
            seen.add(fn_name)
            autotuner.cache.clear()
            if hasattr(autotuner, "best_config"):
                autotuner.best_config = None


def resolve_best_config_paths(names: Iterable[str]) -> List[Path]:
    return [FFN_DATA_DIR / f"{name}.pkl" for name in names]


def resolve_best_config_paths_by_batch(names: Iterable[str], batch_sizes: Iterable[int]) -> tuple[Dict[int, List[Path]], List[Path]]:
    batch_sizes = list(batch_sizes)
    per_batch: Dict[int, List[Path]] = {batch_size: [] for batch_size in batch_sizes}
    shared: List[Path] = []
    pattern = re.compile(r"-bsz(\d+)$")
    for name in names:
        match = pattern.search(name)
        if match:
            batch_size = int(match.group(1))
            if batch_size in per_batch:
                per_batch[batch_size].append(FFN_DATA_DIR / f"{name}.pkl")
                continue
        shared.append(FFN_DATA_DIR / f"{name}.pkl")
    return per_batch, shared


def _make_bench_runner(fn, bench_mode: str):
    if bench_mode == "eager":
        return fn
    if bench_mode != "cuda_graph":
        raise ValueError(f"Unsupported bench_mode: {bench_mode}")

    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    captured = {}
    with torch.cuda.graph(graph):
        captured["out"] = fn()

    def replay():
        graph.replay()
        return captured["out"]

    return replay


def bench_ms(fn, repeat_time: int, bench_mode: str) -> Tuple[float, float, float]:
    quantiles = [0.5, 0.2, 0.8]
    runner = _make_bench_runner(fn, bench_mode)
    return triton.testing.do_bench(runner, quantiles=quantiles, rep=repeat_time)


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        text = str(exc).lower()
        return "out of memory" in text or "cuda error: out of memory" in text
    return False


def clear_cuda_state() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def sdpa_backend_supported(backend: SDPBackend, q_len: int, kv_len: int, is_causal: bool) -> tuple[bool, str | None]:
    if backend == SDPBackend.FLASH_ATTENTION and is_causal and q_len != kv_len:
        return (
            False,
            "Flash SDPA does not support is_causal=True when query_len != kv_len "
            f"(query_len={q_len}, kv_len={kv_len}).",
        )
    return True, None


def _fake_fp16_dense_linear(x: torch.Tensor, branch) -> torch.Tensor:
    weight = (branch.q.to(torch.float32) * branch.scale.to(torch.float32)[:, None]).to(x.dtype)
    return x @ weight.t()


def _fake_fp16_svd_linear(x: torch.Tensor, branch) -> torch.Tensor:
    a_weight = (branch.a_q.to(torch.float32) * branch.a_scale.to(torch.float32)[:, None]).to(x.dtype)
    b_weight = (branch.b_q.to(torch.float32) * branch.b_scale.to(torch.float32)[:, None]).to(x.dtype)
    latent = x @ a_weight.t()
    return latent @ b_weight.t()


def _fake_fp16_mixed_linear(x: torch.Tensor, proj) -> torch.Tensor:
    parts = []
    if proj.svd_branch is not None:
        parts.append(_fake_fp16_svd_linear(x, proj.svd_branch))
    if proj.dense_branch is not None:
        parts.append(_fake_fp16_dense_linear(x, proj.dense_branch))
    if not parts:
        raise ValueError("Mixed projection has neither SVD nor dense branch")
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


def _tensor_stats_text(name: str, tensor: torch.Tensor) -> str:
    with torch.no_grad():
        t = tensor.detach()
        finite = torch.isfinite(t)
        finite_count = int(finite.sum().item())
        total = int(t.numel())
        nan_count = int(torch.isnan(t).sum().item()) if t.is_floating_point() or t.is_complex() else 0
        inf_count = int(torch.isinf(t).sum().item()) if t.is_floating_point() or t.is_complex() else 0
        if finite_count:
            finite_values = t[finite].to(torch.float32)
            min_value = float(finite_values.min().item())
            max_value = float(finite_values.max().item())
        else:
            min_value = float("nan")
            max_value = float("nan")
        return (
            f"{name}: dtype={t.dtype} shape={tuple(t.shape)} "
            f"finite={finite_count}/{total} nan={nan_count} inf={inf_count} "
            f"min={min_value:.4g} max={max_value:.4g}"
        )


def _first_bad_tensor_name(stages: Dict[str, torch.Tensor]) -> str:
    for name, tensor in stages.items():
        if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all().item():
            return name
    return "none"


def _bad_tensor_stats_lines(stages: Dict[str, torch.Tensor], indent: str = "    ", max_items: int = 8) -> str:
    bad = []
    for name, tensor in stages.items():
        if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all().item():
            bad.append(f"{indent}{_tensor_stats_text(name, tensor)}")
            if len(bad) >= max_items:
                break
    return "\n".join(bad) if bad else f"{indent}all finite"


def _dequantize_per_token(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale.to(torch.float32)[:, None]


def _kernel_logical_fake_fp16_stages(stages: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    logical: Dict[str, torch.Tensor] = {}
    if "x_q" in stages and "x_scale" in stages:
        logical["x_norm"] = _dequantize_per_token(stages["x_q"], stages["x_scale"])
    if isinstance(stages.get("up_fp"), torch.Tensor):
        logical["up"] = stages["up_fp"]
    if isinstance(stages.get("gate_fp"), torch.Tensor):
        logical["gate"] = stages["gate_fp"]
    if isinstance(stages.get("mid_fp"), torch.Tensor):
        logical["mid"] = stages["mid_fp"]
    elif "mid_q" in stages and "mid_scale" in stages:
        logical["mid"] = _dequantize_per_token(stages["mid_q"], stages["mid_scale"])
    if isinstance(stages.get("down_fp"), torch.Tensor):
        logical["down"] = stages["down_fp"]
    if isinstance(stages.get("out"), torch.Tensor):
        logical["out"] = stages["out"]
    return logical


def _stage_diff_metrics(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> Dict[str, object]:
    actual_f = actual.reshape(-1).to(torch.float32)
    expected_f = expected.reshape(-1).to(torch.float32)
    if actual_f.shape != expected_f.shape:
        return {
            "stage": name,
            "ok": False,
            "shape": f"{tuple(actual.shape)} != {tuple(expected.shape)}",
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "max_rel": float("inf"),
            "finite": 0,
            "total": max(int(actual_f.numel()), int(expected_f.numel())),
        }
    finite_mask = torch.isfinite(actual_f) & torch.isfinite(expected_f)
    diff = torch.abs(actual_f - expected_f)
    finite_diff = diff[finite_mask]
    rel = diff / torch.clamp(torch.abs(expected_f), min=1e-6)
    finite_rel = rel[finite_mask]
    return {
        "stage": name,
        "ok": bool(torch.allclose(actual_f, expected_f, atol=atol, rtol=rtol, equal_nan=False)),
        "shape": str(tuple(actual.shape)),
        "max_abs": float(finite_diff.max().item()) if finite_diff.numel() else float("nan"),
        "mean_abs": float(finite_diff.mean().item()) if finite_diff.numel() else float("nan"),
        "max_rel": float(finite_rel.max().item()) if finite_rel.numel() else float("nan"),
        "finite": int(finite_mask.sum().item()),
        "total": int(finite_mask.numel()),
    }


def _fake_fp16_stage_diff_metrics(kernel_stages: Dict[str, torch.Tensor], ref_stages: Dict[str, torch.Tensor], atol: float, rtol: float) -> List[Dict[str, object]]:
    kernel_logical = _kernel_logical_fake_fp16_stages(kernel_stages)
    metrics = []
    for name in FAKE_FP16_STAGE_ORDER:
        if name in kernel_logical and name in ref_stages:
            metrics.append(_stage_diff_metrics(name, kernel_logical[name], ref_stages[name], atol, rtol))
    return metrics


def _dequantize_like(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    return (q.to(torch.float32) * scale.to(torch.float32)[:, None]).to(dtype)


def _concat_available_tensors(*tensors: torch.Tensor | None) -> torch.Tensor | None:
    parts = [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
    if not parts:
        return None
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


def _fake_fp16_isolated_stage_diff_metrics(
    x: torch.Tensor,
    kernel_stages: Dict[str, torch.Tensor],
    hybrid: torch.nn.Module,
    atol: float,
    rtol: float,
) -> List[Dict[str, object]]:
    metrics: List[Dict[str, object]] = []

    if "x_q" in kernel_stages and "x_scale" in kernel_stages:
        x_norm_actual = _dequantize_like(kernel_stages["x_q"], kernel_stages["x_scale"])
        x_norm_expected = _rms_norm(x, hybrid.norm_weight, hybrid.eps).reshape(-1, x.shape[-1])
        metrics.append(_stage_diff_metrics("x_norm", x_norm_actual, x_norm_expected, atol, rtol))

        up = _fake_fp16_mixed_linear(x_norm_actual, hybrid.up_proj)
        gate = _fake_fp16_mixed_linear(x_norm_actual, hybrid.gate_proj)
        gateup_expected = up * F.silu(gate)
        gateup_actual = _concat_available_tensors(kernel_stages.get("svd_mid_fp"), kernel_stages.get("dense_mid_fp"))
        if gateup_actual is not None:
            metrics.append(_stage_diff_metrics("gateup_swiglu", gateup_actual, gateup_expected, atol, rtol))

    if "mid_q" in kernel_stages and "mid_scale" in kernel_stages:
        mid_actual = _dequantize_like(kernel_stages["mid_q"], kernel_stages["mid_scale"])
        mid_expected = _concat_available_tensors(kernel_stages.get("svd_mid_fp"), kernel_stages.get("dense_mid_fp"))
        if mid_expected is not None:
            metrics.append(_stage_diff_metrics("mid_quant", mid_actual, mid_expected, atol, rtol))

        down_expected = _fake_fp16_mixed_linear(mid_actual, hybrid.down_proj)
        down_actual = kernel_stages.get("down_fp")
        if isinstance(down_actual, torch.Tensor):
            metrics.append(_stage_diff_metrics("down", down_actual, down_expected, atol, rtol))

            out_actual = kernel_stages.get("out")
            if isinstance(out_actual, torch.Tensor):
                out_expected = (down_actual + x.reshape(-1, x.shape[-1])).reshape_as(x)
                metrics.append(_stage_diff_metrics("residual_add", out_actual, out_expected, atol, rtol))

    return metrics


def _worst_stage_name(stage_metrics: List[Dict[str, object]], key: str) -> str:
    finite_metrics = [m for m in stage_metrics if math.isfinite(float(m.get(key, float("nan"))))]
    if not finite_metrics:
        return "none"
    return str(max(finite_metrics, key=lambda m: float(m[key]))["stage"])


def _first_failing_stage_name(stage_metrics: List[Dict[str, object]]) -> str:
    for metrics in stage_metrics:
        if not bool(metrics.get("ok", False)):
            return str(metrics["stage"])
    return "none"


def _format_stage_metrics(stage_metrics: List[Dict[str, object]], indent: str = "    ") -> str:
    if not stage_metrics:
        return f"{indent}<no comparable stages>"
    return "\n".join(
        f"{indent}{metrics['stage']}: ok={metrics['ok']} "
        f"max_abs={float(metrics['max_abs']):.4g} mean_abs={float(metrics['mean_abs']):.4g} "
        f"max_rel={float(metrics['max_rel']):.4g} finite={metrics['finite']}/{metrics['total']} "
        f"shape={metrics['shape']}"
        for metrics in stage_metrics
    )


def _compact_stage_metrics(stage_metrics: List[Dict[str, object]]) -> str:
    return "; ".join(
        f"{m['stage']}:ok={m['ok']},max_abs={float(m['max_abs']):.4g},max_rel={float(m['max_rel']):.4g}"
        for m in stage_metrics
    )


def fake_fp16_ffn_reference_stages(x: torch.Tensor, hybrid: torch.nn.Module) -> Dict[str, torch.Tensor]:
    x_norm = _rms_norm(x, hybrid.norm_weight, hybrid.eps).reshape(-1, x.shape[-1])
    up = _fake_fp16_mixed_linear(x_norm, hybrid.up_proj)
    gate = _fake_fp16_mixed_linear(x_norm, hybrid.gate_proj)
    mid = up * F.silu(gate)
    down = _fake_fp16_mixed_linear(mid, hybrid.down_proj)
    out = (down + x.reshape(-1, x.shape[-1])).reshape_as(x)
    return {"input": x, "x_norm": x_norm, "up": up, "gate": gate, "mid": mid, "down": down, "out": out}


def fake_fp16_ffn_reference(x: torch.Tensor, hybrid: torch.nn.Module) -> torch.Tensor:
    return fake_fp16_ffn_reference_stages(x, hybrid)["out"]


def run_fake_fp16_correctness(x: torch.Tensor, module: torch.nn.Module, provider: str, atol: float, rtol: float) -> Dict[str, object]:
    if provider not in FAKE_FP16_CORRECTNESS_PROVIDERS:
        return {}
    with torch.inference_mode():
        hybrid = _mix().get_mixed_gateup_v8_ffn(module)
        check_x = torch.randn_like(x)
        check_kernel_stages = hybrid.forward_stages(check_x)
        ref_stages = fake_fp16_ffn_reference_stages(check_x, hybrid)
        check_kernel_out = check_kernel_stages["out"].to(torch.float32)
        ref_out = ref_stages["out"].to(torch.float32)
        kernel_finite = torch.isfinite(check_kernel_out)
        ref_finite = torch.isfinite(ref_out)
        kernel_nan = torch.isnan(check_kernel_out)
        ref_nan = torch.isnan(ref_out)
        kernel_inf = torch.isinf(check_kernel_out)
        ref_inf = torch.isinf(ref_out)
        check_kernel_first_bad = _first_bad_tensor_name(check_kernel_stages)
        ref_first_bad = _first_bad_tensor_name(ref_stages)
        finite_mask = torch.isfinite(check_kernel_out) & torch.isfinite(ref_out)
        diff = torch.abs(check_kernel_out - ref_out)
        finite_diff = diff[finite_mask]
        max_abs = float(finite_diff.max().item()) if finite_diff.numel() else float("nan")
        mean_abs = float(finite_diff.mean().item()) if finite_diff.numel() else float("nan")
        rel = diff / torch.clamp(torch.abs(ref_out), min=1e-6)
        finite_rel = rel[finite_mask]
        max_rel = float(finite_rel.max().item()) if finite_rel.numel() else float("nan")
        ok = bool(torch.allclose(check_kernel_out, ref_out, atol=atol, rtol=rtol, equal_nan=False))
        stage_metrics = _fake_fp16_stage_diff_metrics(check_kernel_stages, ref_stages, atol, rtol)
        isolated_stage_metrics = _fake_fp16_isolated_stage_diff_metrics(check_x, check_kernel_stages, hybrid, atol, rtol)
        first_fail_stage = _first_failing_stage_name(stage_metrics)
        worst_abs_stage = _worst_stage_name(stage_metrics, "max_abs")
        worst_rel_stage = _worst_stage_name(stage_metrics, "max_rel")
        isolated_ok = all(bool(metrics.get("ok", False)) for metrics in isolated_stage_metrics) if isolated_stage_metrics else ok
        isolated_first_fail_stage = _first_failing_stage_name(isolated_stage_metrics)
        isolated_worst_abs_stage = _worst_stage_name(isolated_stage_metrics, "max_abs")
        isolated_worst_rel_stage = _worst_stage_name(isolated_stage_metrics, "max_rel")
        if check_kernel_first_bad != "none" or ref_first_bad != "none":
            LOGGER.warning(
                "\nfake_fp16_correctness FFN-only stage stats\n  provider: %s\n  check_kernel_first_bad: %s\n"
                "  ref_first_bad: %s\n  check_kernel_bad:\n%s\n  ref_bad:\n%s",
                provider,
                check_kernel_first_bad,
                ref_first_bad,
                _bad_tensor_stats_lines(check_kernel_stages, indent="    "),
                _bad_tensor_stats_lines(ref_stages, indent="    "),
            )
        if not ok:
            LOGGER.warning(
                "\nfake_fp16_correctness FFN-only stage-wise diff\n  provider: %s\n  first_fail_stage: %s\n"
                "  worst_abs_stage: %s\n  worst_rel_stage: %s\n  propagated_stage_metrics:\n%s\n"
                "  isolated_first_fail_stage: %s\n  isolated_worst_abs_stage: %s\n  isolated_worst_rel_stage: %s\n"
                "  isolated_stage_metrics:\n%s",
                provider,
                first_fail_stage,
                worst_abs_stage,
                worst_rel_stage,
                _format_stage_metrics(stage_metrics, indent="    "),
                isolated_first_fail_stage,
                isolated_worst_abs_stage,
                isolated_worst_rel_stage,
                _format_stage_metrics(isolated_stage_metrics, indent="    "),
            )
        return {
            "fake_fp16_ok": ok,
            "fake_fp16_max_abs": max_abs,
            "fake_fp16_mean_abs": mean_abs,
            "fake_fp16_max_rel": max_rel,
            "fake_fp16_finite": int(finite_mask.sum().item()),
            "fake_fp16_total": int(finite_mask.numel()),
            "fake_fp16_kernel_finite": int(kernel_finite.sum().item()),
            "fake_fp16_ref_finite": int(ref_finite.sum().item()),
            "fake_fp16_kernel_nan": int(kernel_nan.sum().item()),
            "fake_fp16_ref_nan": int(ref_nan.sum().item()),
            "fake_fp16_kernel_inf": int(kernel_inf.sum().item()),
            "fake_fp16_ref_inf": int(ref_inf.sum().item()),
            "fake_fp16_kernel_first_bad": check_kernel_first_bad,
            "fake_fp16_ref_first_bad": ref_first_bad,
            "fake_fp16_first_fail_stage": first_fail_stage,
            "fake_fp16_worst_abs_stage": worst_abs_stage,
            "fake_fp16_worst_rel_stage": worst_rel_stage,
            "fake_fp16_stage_metrics": _compact_stage_metrics(stage_metrics),
            "fake_fp16_isolated_ok": isolated_ok,
            "fake_fp16_isolated_first_fail_stage": isolated_first_fail_stage,
            "fake_fp16_isolated_worst_abs_stage": isolated_worst_abs_stage,
            "fake_fp16_isolated_worst_rel_stage": isolated_worst_rel_stage,
            "fake_fp16_isolated_stage_metrics": _compact_stage_metrics(isolated_stage_metrics),
        }


def benchmark_stage_breakdown(x: torch.Tensor, module: torch.nn.Module, provider: str, repeat_time: int = 100, stage_mode: str = "cached"):
    if provider != "mixffn":
        raise ValueError(f"Stage timing only supports mixffn, got {provider}")
    return _mix().benchmark_stage_breakdown(x, module, provider, repeat_time=repeat_time, stage_mode=stage_mode)


def make_ffn_module(args, batch_size: int, rank_ratio: float, svd_ratio: float) -> torch.nn.Module:
    _, module = _mix().make_ffn_inputs(
        batch_size=batch_size,
        seq_len=args.query_len,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        seed=args.seed,
        dtype=torch.float16,
        device="cuda",
        svd_ratio=svd_ratio,
        rank_ratio=rank_ratio,
        round_to=args.round_to,
    )
    return module.eval()


def build_module_bank(args) -> Dict[Tuple[str, float | None, float | None], torch.nn.Module]:
    module_bank: Dict[Tuple[str, float | None, float | None], torch.nn.Module] = {}
    for provider in args.providers:
        if provider == "mixffn":
            provider_settings = [(rank_ratio, svd_ratio) for rank_ratio in args.ffn_rank_ratios for svd_ratio in args.ffn_svd_ratios]
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        for rank_ratio, svd_ratio in provider_settings:
            effective_rank_ratio = 0.5 if rank_ratio is None else rank_ratio
            effective_svd_ratio = 0.5 if svd_ratio is None else svd_ratio
            key = (provider, rank_ratio, svd_ratio)
            if key in module_bank:
                continue
            LOGGER.info("Preparing module cache for provider=%s rank_ratio=%.2f svd_ratio=%.2f", provider, effective_rank_ratio, effective_svd_ratio)
            module_bank[key] = make_ffn_module(args, batch_size=1, rank_ratio=effective_rank_ratio, svd_ratio=effective_svd_ratio)
    clear_cuda_state()
    return module_bank


def get_effective_ffn_shapes(
    provider: str,
    hidden_size: int,
    intermediate_size: int,
    rank_ratio: float | None,
    svd_ratio: float | None,
    round_to: int,
) -> Dict[str, int | None]:
    if provider == "fp16":
        return {"rounded_rank": None, "svd_out": None, "dense_out": None, "m_tokens": None}
    if rank_ratio is None:
        raise ValueError(f"rank_ratio is required for provider={provider}")
    rounded_rank = _mix()._rounded_rank(hidden_size, intermediate_size, rank_ratio, round_to)
    if provider == "mixffn":
        if svd_ratio is None:
            raise ValueError(f"svd_ratio is required for {provider}")
        svd_out = _mix().svd_out_from_ratio(intermediate_size, svd_ratio, round_to=round_to)
        dense_out = intermediate_size - svd_out
        return {"rounded_rank": rounded_rank, "svd_out": svd_out, "dense_out": dense_out, "m_tokens": None}
    raise ValueError(f"Unsupported provider: {provider}")
