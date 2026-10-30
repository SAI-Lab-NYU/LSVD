import argparse
import gc
import logging
import pickle
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from benchmark_triton_int8_layer import (
    Int8AttentionBlock,
    make_latent_kv_cache,
)
from ffn_kernel.utils import (
    ATTENTION_KERNELS,
    DEFAULT_PROVIDERS,
    FFN_DATA_DIR,
    FFN_KERNELS,
    FFN_NAME_MAP,
    LAYER_KERNELS,
    apply_cache_records,
    get_kernel_record_name,
    make_ffn_module,
    parse_float_list,
    parse_int_list,
    resolve_best_config_paths,
    setup_logging,
)


LOGGER = logging.getLogger(__name__)


def normalize_providers(providers: Iterable[str]) -> List[str]:
    providers = list(providers)
    invalid = [provider for provider in providers if provider not in FFN_NAME_MAP]
    if invalid:
        raise ValueError(f"Unknown providers: {invalid}. Valid providers are {sorted(FFN_NAME_MAP)}.")
    return providers


def parse_args():
    parser = argparse.ArgumentParser(description="Autotune Triton FFN kernels used by the int8 layer benchmark.")
    parser.add_argument("--providers", nargs="+", default=DEFAULT_PROVIDERS, choices=sorted(FFN_NAME_MAP))
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 16, 32, 64])
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--context-lens", type=parse_int_list, default=[1, 256, 1024, 8192])
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--attention-rank-ratio", type=float, default=0.5)
    parser.add_argument("--ffn-rank-ratios", type=parse_float_list, default=[0.5, 0.7, 0.9])
    parser.add_argument("--ffn-svd-ratios", type=parse_float_list, default=[0.1, 0.5, 0.9])
    parser.add_argument("--round-to", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--best-config-path", type=str, default="triton-int8-layer-autotune")
    parser.add_argument("--base-best-config-paths", nargs="*", default=[])
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def init_record_files(best_config_path: str, args_dict: Dict) -> tuple[Path, Path, Dict]:
    FFN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = FFN_DATA_DIR / f"{best_config_path}.yaml"
    pkl_path = FFN_DATA_DIR / f"{best_config_path}.pkl"
    yaml_path.write_text("", encoding="utf-8")
    records = {"args": args_dict}
    with pkl_path.open("wb") as f:
        pickle.dump(records, f)
    return yaml_path, pkl_path, records


def clear_cuda_state() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def append_yaml_result(yaml_path: Path, batch_size: int, context_len: int, provider: str, rank_ratio: float, svd_ratio: float, best_configs: Dict) -> None:
    import yaml

    current = {}
    if yaml_path.exists() and yaml_path.read_text(encoding="utf-8").strip():
        current = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    batch_key = f"batch-{batch_size}"
    context_key = f"context-{context_len}"
    rank_key = f"rank-{rank_ratio:.3f}"
    svd_key = f"svd-{svd_ratio:.3f}"
    current.setdefault(batch_key, {}).setdefault(context_key, {}).setdefault(rank_key, {}).setdefault(svd_key, {})[provider] = best_configs
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(current, f, sort_keys=False)


def append_pickle_result(pkl_path: Path, records: Dict, batch_size: int, context_len: int, provider: str, rank_ratio: float, svd_ratio: float, kernels) -> None:
    batch_key = f"batch-{batch_size}"
    context_key = f"context-{context_len}"
    rank_key = f"rank-{rank_ratio:.3f}"
    svd_key = f"svd-{svd_ratio:.3f}"
    records.setdefault(batch_key, {}).setdefault(context_key, {}).setdefault(rank_key, {}).setdefault(svd_key, {}).setdefault(provider, {})
    for kernel in kernels:
        if not hasattr(kernel, "best_config"):
            continue
        record_name = get_kernel_record_name(kernel, provider)
        record = {
            "cache": kernel.cache,
            "keys": kernel.keys,
            "best_config": kernel.best_config.all_kwargs(),
        }
        records[batch_key][context_key][rank_key][svd_key][provider][record_name] = record
        records[record_name] = record
    with pkl_path.open("wb") as f:
        pickle.dump(records, f)


def provider_settings(provider: str, rank_ratios: List[float], svd_ratios: List[float]) -> List[tuple[float, float]]:
    if provider == "mixffn":
        return [(rank_ratio, svd_ratio) for rank_ratio in rank_ratios for svd_ratio in svd_ratios]
    raise ValueError(f"Unsupported provider: {provider}")


def main():
    args = parse_args()
    setup_logging(args.log_level)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for autotune.")
    if args.hidden_size != args.num_heads * args.head_dim:
        raise ValueError("--hidden-size must equal --num-heads * --head-dim")

    providers = normalize_providers(args.providers)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    for best_config_path in resolve_best_config_paths(args.base_best_config_paths):
        apply_cache_records(best_config_path, LAYER_KERNELS)

    attention_block = Int8AttentionBlock(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        rank_ratio=args.attention_rank_ratio,
        device="cuda",
    ).eval()

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    with torch.inference_mode():
        for batch_size in args.batch_sizes:
            batch_best_config_path = f"{args.best_config_path}-bsz{batch_size}"
            batch_args_dict = {
                "providers": providers,
                "batch_sizes": [batch_size],
                "query_len": args.query_len,
                "context_lens": args.context_lens,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "num_heads": args.num_heads,
                "head_dim": args.head_dim,
                "attention_rank_ratio": args.attention_rank_ratio,
                "ffn_rank_ratios": args.ffn_rank_ratios,
                "ffn_svd_ratios": args.ffn_svd_ratios,
                "round_to": args.round_to,
                "seed": args.seed,
            }
            yaml_path, pkl_path, records = init_record_files(batch_best_config_path, batch_args_dict)
            LOGGER.info("Starting autotune batch=%d cache_prefix=%s", batch_size, batch_best_config_path)
            LOGGER.info("Output YAML: %s", yaml_path)
            LOGGER.info("Output PKL: %s", pkl_path)

            def run_context_autotune(context_len: int) -> None:
                x = torch.randn(
                    (batch_size, args.query_len, args.hidden_size),
                    device="cuda",
                    dtype=torch.float16,
                    generator=generator,
                ).contiguous()
                if args.query_len != 1 or context_len <= 0:
                    LOGGER.warning(
                        "Skipping autotune batch=%d context=%d because latent int8 attention requires decode query_len=1 and context_len>0.",
                        batch_size,
                        context_len,
                    )
                    del x
                    clear_cuda_state()
                    return
                latent_inputs = make_latent_kv_cache(
                    batch_size=batch_size,
                    num_heads=args.num_heads,
                    context_len=context_len,
                    head_dim=args.head_dim,
                    rank_ratio=args.attention_rank_ratio,
                    device="cuda",
                    generator=generator,
                )
                for provider in providers:
                    cache_latent_k, cache_latent_k_scale, cache_latent_v, cache_latent_v_scale, infer_state = latent_inputs
                    attn_input = attention_block(
                        x,
                        cache_latent_k,
                        cache_latent_k_scale,
                        cache_latent_v,
                        cache_latent_v_scale,
                        infer_state,
                    )
                    for rank_ratio, svd_ratio in provider_settings(provider, args.ffn_rank_ratios, args.ffn_svd_ratios):
                        try:
                            LOGGER.info(
                                "Autotuning provider=%s batch=%d context=%d rank_ratio=%.2f svd_ratio=%.2f",
                                provider,
                                batch_size,
                                context_len,
                                rank_ratio,
                                svd_ratio,
                            )
                            ffn_module = make_ffn_module(args, batch_size, rank_ratio, svd_ratio)
                            _ = FFN_NAME_MAP[provider](attn_input, ffn_module)
                            kernels = ATTENTION_KERNELS + FFN_KERNELS[provider]
                            best_configs = {
                                get_kernel_record_name(kernel, provider): kernel.best_config.all_kwargs()
                                for kernel in kernels
                                if hasattr(kernel, "best_config")
                            }
                            append_yaml_result(yaml_path, batch_size, context_len, provider, rank_ratio, svd_ratio, best_configs)
                            append_pickle_result(pkl_path, records, batch_size, context_len, provider, rank_ratio, svd_ratio, kernels)
                        except Exception:
                            LOGGER.exception(
                                "Autotuning failed for provider=%s batch=%d context=%d rank_ratio=%.2f svd_ratio=%.2f",
                                provider,
                                batch_size,
                                context_len,
                                rank_ratio,
                                svd_ratio,
                            )
                            continue
                    del attn_input
                del x
                del latent_inputs
                clear_cuda_state()

            for context_len in args.context_lens:
                try:
                    run_context_autotune(context_len)
                except Exception:
                    LOGGER.exception(
                        "Context autotune failed for batch=%d context=%d cache_prefix=%s",
                        batch_size,
                        context_len,
                        batch_best_config_path,
                    )
                    clear_cuda_state()
                    continue


if __name__ == "__main__":
    main()
