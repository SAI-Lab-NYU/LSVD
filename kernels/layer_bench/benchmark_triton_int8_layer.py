import argparse
import csv
import logging
import math
from pathlib import Path
from typing import Dict, List, Mapping

import torch
from torch.nn.attention import SDPBackend

from ffn_kernel.baseline_kernel import (
    FP16LayerBaseline,
    Int8AttentionBlock,
    MockInferState,
)
from ffn_kernel.utils import (
    ATTENTION_KERNELS,
    DEFAULT_PROVIDERS,
    FAKE_FP16_CORRECTNESS_PROVIDERS,
    FFN_KERNELS,
    FFN_NAME_MAP,
    FFN_STAGE_BENCH_PROVIDERS,
    apply_cache_records,
    bench_ms,
    benchmark_stage_breakdown,
    build_module_bank,
    clear_cache_records,
    clear_cuda_state,
    get_effective_ffn_shapes,
    is_oom_error,
    parse_float_list,
    parse_int_list,
    resolve_best_config_paths_by_batch,
    run_fake_fp16_correctness,
    sdpa_backend_supported,
    setup_logging,
)

LOGGER = logging.getLogger(__name__)
LAYER_KERNELS = {
    "__attention__": ATTENTION_KERNELS,
    **FFN_KERNELS,
}
FFN_STAGE_COLUMNS = (
    "rmsnorm_quant",
    "gateup_svd",
    "gateup_dense",
    "swiglu_quant",
    "down_proj",
    "residual_add",
    "ffn_stage_sum_ms",
    "sum_ms",
)
FP16_EAGER_ATTENTION = "fp16_eager"
FP16_PROVIDER = "fp16_flash"
INT8_ATTENTION = "int8_latent"
PROVIDER_DISPLAY_NAMES = {"mixffn": "lsvd"}


def make_kv_cache(
    batch_size: int,
    num_heads: int,
    context_len: int,
    head_dim: int,
    device: str,
    generator: torch.Generator,
):
    k_cache = torch.randint(
        -128,
        128,
        (batch_size, num_heads, context_len, head_dim),
        device=device,
        dtype=torch.int8,
        generator=generator,
    )
    v_cache = torch.randint(
        -128,
        128,
        (batch_size, num_heads, context_len, head_dim),
        device=device,
        dtype=torch.int8,
        generator=generator,
    )
    k_cache_scale = (
        torch.rand((batch_size, num_heads, context_len, 1), device=device, dtype=torch.float16, generator=generator) * 0.05
        + 1e-4
    )
    v_cache_scale = (
        torch.rand((batch_size, num_heads, context_len, 1), device=device, dtype=torch.float16, generator=generator) * 0.05
        + 1e-4
    )
    return k_cache.contiguous(), v_cache.contiguous(), k_cache_scale.contiguous(), v_cache_scale.contiguous()


def make_fp16_kv_cache(cache: torch.Tensor, query_len: int) -> torch.Tensor:
    # FP16 copy of the cache with query_len free slots at the end for the new tokens (written in place by the FP16 layer).
    bsz, num_heads, context_len, head_dim = cache.shape
    fp16_cache = cache.new_empty((bsz, num_heads, context_len + query_len, head_dim), dtype=torch.float16)
    fp16_cache[:, :, :context_len].copy_(cache)
    return fp16_cache


def quantize_latent_tensor(
    x: torch.Tensor,
    return_dequantized: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    scale = x.abs().amax(dim=-1) / 128
    scale = scale.clamp_min(1e-4).to(torch.float16)
    q = torch.clamp(torch.round(x / scale[:, :, None]), -128, 127).to(torch.int8)
    dequantized = None
    if return_dequantized:
        dequantized = (q.to(torch.float16) * scale[:, :, None]).contiguous()
    return q.contiguous(), scale.contiguous(), dequantized


def make_latent_kv_cache(
    batch_size: int,
    num_heads: int,
    context_len: int,
    head_dim: int,
    rank_ratio: float,
    device: str,
    generator: torch.Generator,
):
    # context_len cached tokens plus one slot per request for the decoding token, written by Int8AttentionBlock
    rank_head = max(1, int(rank_ratio * head_dim))
    seq_len = context_len + 1
    cache_latent_k = torch.randn(
        (batch_size * seq_len, num_heads, rank_head),
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    cache_latent_v = torch.randn(
        (batch_size * seq_len, num_heads, rank_head),
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    cache_latent_k_q, cache_latent_k_scale, _ = quantize_latent_tensor(cache_latent_k, return_dequantized=False)
    cache_latent_v_q, cache_latent_v_scale, _ = quantize_latent_tensor(cache_latent_v, return_dequantized=False)
    infer_state = MockInferState(batch_size, seq_len, [seq_len] * batch_size)
    return (
        cache_latent_k_q,
        cache_latent_k_scale,
        cache_latent_v_q,
        cache_latent_v_scale,
        infer_state,
    )


def get_stage_shapes(stage: str, configured_query_len: int, context_len: int) -> tuple[int, int]:
    if stage == "decode":
        return configured_query_len, context_len
    if stage == "prefill":
        return context_len, 0
    raise ValueError(f"Unsupported stage: {stage}")


def print_results(rows: List[Dict[str, float]]) -> None:
    provider_width = max(14, *(len(row["provider"]) for row in rows))
    attention_width = max(10, *(len(row["attention_backend"]) for row in rows))
    print("-" * (provider_width + attention_width + 169))
    print(
        f"{'attn':>{attention_width}} {'provider':>{provider_width}} {'batch':>8} {'rank':>8} {'rank_eff':>10} {'svd':>8} {'svd_out':>9} {'dense':>8} {'ctx_len':>8} "
        f"{'attn(ms)':>12} {'ffn(ms)':>12} {'layer(ms)':>12} {'p20(ms)':>12} {'p80(ms)':>12} {'spdup fp16':>12}"
    )
    print("-" * (provider_width + attention_width + 169))
    for row in rows:
        rank_ratio = "-" if row["ffn_rank_ratio"] is None else f"{row['ffn_rank_ratio']:.2f}"
        rounded_rank = "-" if row["rounded_rank"] is None else str(row["rounded_rank"])
        svd_ratio = "-" if row["ffn_svd_ratio"] is None else f"{row['ffn_svd_ratio']:.2f}"
        svd_out = "-" if row["svd_out"] is None else str(row["svd_out"])
        dense_out = "-" if row["dense_out"] is None else str(row["dense_out"])
        print(
            f"{row['attention_backend']:>{attention_width}} "
            f"{row['provider']:>{provider_width}} "
            f"{row['batch_size']:>8} "
            f"{rank_ratio:>8} "
            f"{rounded_rank:>10} "
            f"{svd_ratio:>8} "
            f"{svd_out:>9} "
            f"{dense_out:>8} "
            f"{row['context_len']:>8} "
            f"{row['attention_ms']:>12.3f} "
            f"{row['ffn_ms']:>12.3f} "
            f"{row['layer_ms']:>12.3f} "
            f"{row['layer_p20_ms']:>12.3f} "
            f"{row['layer_p80_ms']:>12.3f} "
            f"{row['speedup_vs_fp16']:>12.3f}"
        )


def print_stage_results(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    provider_width = max(14, *(len(str(row["provider"])) for row in rows))
    print("-" * (provider_width + 143))
    print(
        f"{'provider':>{provider_width}} {'batch':>8} {'ctx_len':>8} {'q_len':>6} {'rank':>8} {'rank_eff':>10} "
        f"{'svd':>8} {'svd_out':>9} {'dense':>8} {'ffn_e2e':>10} {'stage_sum':>10} "
        f"{'gate_svd':>10} {'gate_dense':>11} {'swiglu':>10} {'down':>10}"
    )
    print("-" * (provider_width + 143))
    for row in rows:
        print(
            f"{str(row['provider']):>{provider_width}} "
            f"{int(row['batch_size']):>8} "
            f"{int(row['context_len']):>8} "
            f"{int(row['query_len']):>6} "
            f"{float(row['ffn_rank_ratio']):>8.2f} "
            f"{int(row['rounded_rank']):>10} "
            f"{float(row['ffn_svd_ratio']):>8.2f} "
            f"{int(row['svd_out']):>9} "
            f"{int(row['dense_out']):>8} "
            f"{float(row['ffn_ms']):>10.3f} "
            f"{float(row.get('ffn_stage_sum_ms', float('nan'))):>10.3f} "
            f"{float(row.get('gateup_svd', float('nan'))):>10.3f} "
            f"{float(row.get('gateup_dense', float('nan'))):>11.3f} "
            f"{float(row.get('swiglu_quant', float('nan'))):>10.3f} "
            f"{float(row.get('down_proj', float('nan'))):>10.3f}"
        )


def write_stage_results(rows: List[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "provider",
        "batch_size",
        "context_len",
        "query_len",
        "hidden_size",
        "intermediate_size",
        "attention_rank_ratio",
        "ffn_rank_ratio",
        "ffn_svd_ratio",
        "rounded_rank",
        "svd_out",
        "dense_out",
        "round_to",
        "attention_ms",
        "ffn_ms",
        "layer_ms",
        *FFN_STAGE_COLUMNS,
        "fake_fp16_ok",
        "fake_fp16_max_abs",
        "fake_fp16_mean_abs",
        "fake_fp16_max_rel",
        "fake_fp16_finite",
        "fake_fp16_total",
        "fake_fp16_kernel_finite",
        "fake_fp16_ref_finite",
        "fake_fp16_kernel_nan",
        "fake_fp16_ref_nan",
        "fake_fp16_kernel_inf",
        "fake_fp16_ref_inf",
        "fake_fp16_kernel_first_bad",
        "fake_fp16_ref_first_bad",
        "fake_fp16_first_fail_stage",
        "fake_fp16_worst_abs_stage",
        "fake_fp16_worst_rel_stage",
        "fake_fp16_stage_metrics",
        "fake_fp16_isolated_ok",
        "fake_fp16_isolated_first_fail_stage",
        "fake_fp16_isolated_worst_abs_stage",
        "fake_fp16_isolated_worst_rel_stage",
        "fake_fp16_isolated_stage_metrics",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote stage bench TSV: {output_path}")


def import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to write the speedup plot.") from exc
    return plt


def format_series_label(label: str, provider_display_names: Mapping[str, str] | None = None) -> str:
    for provider, display_name in (provider_display_names or {}).items():
        if label == provider:
            return display_name
        if label.startswith(f"{provider} |"):
            return f"{display_name}{label[len(provider):]}"
    return label


def plot_speedups(
    rows: List[Dict[str, float]],
    output_path: Path,
    provider_display_names: Mapping[str, str] | None = None,
) -> None:
    plt = import_matplotlib()
    from matplotlib.lines import Line2D

    batch_sizes = sorted({row["batch_size"] for row in rows})
    series_labels = [label for label in sorted({row["series_label"] for row in rows}) if label != FP16_EAGER_ATTENTION]
    display_labels = [format_series_label(label, provider_display_names) for label in series_labels]
    palette = ["#8DA5F7", "#A8BE89", "#E9B85D", "#DF625C", "#7E70B8", "#66B6B0", "#C77CBB", "#777777"]
    label_colors = {label: palette[idx % len(palette)] for idx, label in enumerate(series_labels)}

    for batch_size in batch_sizes:
        batch_rows = [row for row in rows if row["batch_size"] == batch_size]
        context_lens = sorted({row["context_len"] for row in batch_rows if math.isfinite(row["speedup_vs_fp16"])})
        if not context_lens:
            LOGGER.warning("Skipping plot for batch=%d because no valid speedup rows remain.", batch_size)
            continue

        context_positions = {context_len: idx for idx, context_len in enumerate(context_lens)}
        fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=180)
        for label in series_labels:
            provider_rows = sorted(
                [
                    row
                    for row in batch_rows
                    if row["series_label"] == label and math.isfinite(row["speedup_vs_fp16"])
                ],
                key=lambda row: row["context_len"],
            )
            if not provider_rows:
                continue
            color = label_colors[label]
            xs = [context_positions[row["context_len"]] for row in provider_rows]
            ys = [row["speedup_vs_fp16"] for row in provider_rows]
            ax.plot(
                xs,
                ys,
                color=color,
                marker="o",
                markersize=9.5,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=1.4,
                linewidth=3.2,
                solid_capstyle="round",
                label=label,
                zorder=3,
            )
            for x_pos, y_val in zip(xs, ys):
                ax.annotate(
                    f"{y_val:.2f}x",
                    (x_pos, y_val),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                    },
                    zorder=8,
                )
            ax.plot(
                [context_positions[row["context_len"]] for row in provider_rows],
                [provider_rows[0]["speedup_vs_fp16"]] * len(provider_rows),
                color=color,
                linestyle="--",
                linewidth=2.5,
                alpha=0.42,
                zorder=2,
            )

        ax.axhline(1.0, color="#222222", linestyle="--", linewidth=2.2, alpha=0.75)
        ax.set_xticks(range(len(context_lens)))
        ax.set_xticklabels([str(v) for v in context_lens], fontsize=12)
        ax.set_xlabel("Context Length", fontsize=15)
        ax.set_ylabel("Speedup vs FP16 Eager", fontsize=15)
        ax.set_title(f"Layer Speedup (batch={batch_size})", fontsize=18, fontweight="bold", pad=10)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, color="#B8B8B8", alpha=0.35, linewidth=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#222222")
        if series_labels:
            handles = [
                Line2D(
                    [0],
                    [0],
                    color=label_colors[label],
                    marker="o",
                    markersize=9,
                    linewidth=3,
                    markerfacecolor=label_colors[label],
                    markeredgecolor="white",
                    markeredgewidth=1.2,
                    label=label,
                )
                for label in series_labels
            ]
            legend_ncols = min(max(1, len(handles)), 4)
            fig.legend(
                handles=handles,
                labels=display_labels,
                loc="lower center",
                ncols=legend_ncols,
                frameon=True,
                framealpha=0.96,
                edgecolor="#D0D0D0",
                title="Provider",
                title_fontsize=13,
                fontsize=11,
                bbox_to_anchor=(0.5, -0.02),
            )
        fig.tight_layout(rect=(0, 0.10, 1, 1))
        batch_output_path = output_path.with_name(f"{output_path.stem}_batch_{batch_size}{output_path.suffix}")
        fig.savefig(batch_output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote plot: {batch_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark an LLM decode layer with int8 Triton attention plus interchangeable Triton FFN kernels."
    )
    parser.add_argument("--providers", nargs="+", default=DEFAULT_PROVIDERS, choices=sorted(FFN_NAME_MAP))
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 16, 32, 64])
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--context-lens", type=parse_int_list, default=[1, 256, 1024, 8192])
    parser.add_argument("--stage", type=str, default="decode", choices=["decode", "prefill"])
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--attention-rank-ratio", type=float, default=0.5)
    parser.add_argument("--ffn-rank-ratios", type=parse_float_list, default=[0.5, 0.7, 0.9])
    parser.add_argument("--ffn-svd-ratios", type=parse_float_list, default=[0.1, 0.5, 0.9])
    parser.add_argument("--round-to", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-time", type=int, default=100)
    parser.add_argument("--bench-mode", type=str, default="cuda_graph", choices=["eager", "cuda_graph"])
    parser.add_argument(
        "--layer-mode",
        type=str,
        default="sum",
        choices=["sum", "e2e"],
        help="Use sum(attention, ffn) for stable layer latency, or benchmark the composed end-to-end layer directly.",
    )
    parser.add_argument(
        "--stage-bench",
        action="store_true",
        help="Also benchmark v6/v7 FFN stages on the exact attention output used by the layer bench.",
    )
    parser.add_argument(
        "--stage-bench-mode",
        type=str,
        default="cached",
        choices=["cached", "cuda_graph"],
        help="Timing mode for --stage-bench FFN stage callables.",
    )
    parser.add_argument(
        "--fake-fp16-correctness",
        action="store_true",
        help="Run FFN-only correctness: compare v6/v7 int8 FFN output on synthetic finite input with a PyTorch fake-fp16 reference built from the quantized int8 branch weights.",
    )
    parser.add_argument("--fake-fp16-atol", type=float, default=1e-1)
    parser.add_argument("--fake-fp16-rtol", type=float, default=1e-1)
    parser.add_argument(
        "--fake-fp16-fail-fast",
        action="store_true",
        help="Raise immediately when --fake-fp16-correctness fails allclose.",
    )
    parser.add_argument("--best-config-paths", nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-name", type=str, default="triton_int8_layer_speedup_vs_fp16")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.hidden_size != args.num_heads * args.head_dim:
        raise ValueError("--hidden-size must equal --num-heads * --head-dim")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    batch_cache_paths: Dict[int, List[Path]] = {}
    shared_cache_paths: List[Path] = []
    if args.best_config_paths:
        LOGGER.info("Requested autotune cache names: %s", ", ".join(args.best_config_paths))
        batch_cache_paths, shared_cache_paths = resolve_best_config_paths_by_batch(args.best_config_paths, args.batch_sizes)
    else:
        LOGGER.warning(
            "No --best-config-paths provided. Triton kernels will autotune online on first use, and that tuning time is outside the do_bench replay path but will still happen during warmup/capture."
        )

    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    attention_block = Int8AttentionBlock(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        rank_ratio=args.attention_rank_ratio,
        device="cuda",
    ).eval()
    fp16_baseline = FP16LayerBaseline(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        device="cuda",
    ).eval()
    module_bank = build_module_bank(args)
    rows = []
    stage_rows = []
    with torch.inference_mode():
        for batch_size in args.batch_sizes:
            LOGGER.info("Starting batch size sweep: batch=%d", batch_size)
            cache_paths = []
            if args.best_config_paths:
                clear_cache_records(LAYER_KERNELS)
                cache_paths = [*shared_cache_paths, *batch_cache_paths.get(batch_size, [])]
                if cache_paths:
                    LOGGER.info("Resolved autotune cache for batch=%d: %s", batch_size, ", ".join(str(path) for path in cache_paths))
                else:
                    LOGGER.warning("No batch-specific autotune cache matched batch=%d", batch_size)
            try:
                for context_len in args.context_lens:
                    try:
                        effective_query_len, effective_cache_len = get_stage_shapes(args.stage, args.query_len, context_len)
                        x = torch.randn(
                            (batch_size, effective_query_len, args.hidden_size),
                            device="cuda",
                            dtype=torch.float16,
                            generator=generator,
                        ).contiguous()
                        k_cache, v_cache, k_cache_scale, v_cache_scale = make_kv_cache(
                            batch_size=batch_size,
                            num_heads=args.num_heads,
                            context_len=effective_cache_len,
                            head_dim=args.head_dim,
                            device="cuda",
                            generator=generator,
                        )
                        latent_attention_supported = args.stage == "decode" and effective_query_len == 1 and effective_cache_len > 0
                        latent_inputs = None
                        if latent_attention_supported:
                            latent_inputs = make_latent_kv_cache(
                                batch_size=batch_size,
                                num_heads=args.num_heads,
                                context_len=effective_cache_len,
                                head_dim=args.head_dim,
                                rank_ratio=args.attention_rank_ratio,
                                device="cuda",
                                generator=generator,
                            )

                        fp16_k_cache = None
                        fp16_v_cache = None
                        fp16_cache_ok = False
                        try:
                            fp16_k_cache = make_fp16_kv_cache(k_cache, effective_query_len)
                            fp16_v_cache = make_fp16_kv_cache(v_cache, effective_query_len)
                            fp16_cache_ok = True
                        except Exception as exc:
                            if not is_oom_error(exc):
                                raise
                            LOGGER.warning(
                                "OOM: skipping fp16 cache conversion for batch=%d context=%d; continuing with int8 providers. error=%s",
                                batch_size,
                                context_len,
                                exc,
                                exc_info=True,
                            )
                            clear_cuda_state()

                        fp16_eager_attn_input = None
                        fp16_eager_ok = False
                        if fp16_cache_ok:
                            try:
                                fp16_eager_attention_ms, fp16_eager_attention_p20_ms, fp16_eager_attention_p80_ms = bench_ms(
                                    lambda: fp16_baseline.attention_only(x, fp16_k_cache, fp16_v_cache),
                                    repeat_time=args.repeat_time,
                                    bench_mode=args.bench_mode,
                                )
                                fp16_eager_attn_input = fp16_baseline.attention_only(x, fp16_k_cache, fp16_v_cache)
                                fp16_eager_ffn_ms, fp16_eager_ffn_p20_ms, fp16_eager_ffn_p80_ms = bench_ms(
                                    lambda: fp16_baseline.ffn_only(fp16_eager_attn_input),
                                    repeat_time=args.repeat_time,
                                    bench_mode=args.bench_mode,
                                )
                                fp16_eager_layer_ms, fp16_eager_layer_p20_ms, fp16_eager_layer_p80_ms = bench_ms(
                                    lambda: fp16_baseline(x, fp16_k_cache, fp16_v_cache),
                                    repeat_time=args.repeat_time,
                                    bench_mode=args.bench_mode,
                                )
                                fp16_eager_ok = True
                            except Exception as exc:
                                if not is_oom_error(exc):
                                    raise
                                LOGGER.warning(
                                    "OOM: skipping fp16 eager SDPA benchmark for batch=%d context=%d; continuing with other backends/providers. error=%s",
                                    batch_size,
                                    context_len,
                                    exc,
                                    exc_info=True,
                                )
                                clear_cuda_state()
                        fp16_flash_attn_input = None
                        fp16_flash_ok = False
                        flash_supported, flash_reason = sdpa_backend_supported(
                            SDPBackend.FLASH_ATTENTION,
                            q_len=effective_query_len,
                            kv_len=effective_cache_len + effective_query_len,
                            is_causal=args.stage == "prefill",
                        )
                        if flash_supported and fp16_cache_ok:
                            try:
                                fp16_flash_attention_ms, fp16_flash_attention_p20_ms, fp16_flash_attention_p80_ms = bench_ms(
                                    lambda: fp16_baseline.attention_only_flash(x, fp16_k_cache, fp16_v_cache),
                                    repeat_time=args.repeat_time,
                                    bench_mode=args.bench_mode,
                                )
                                fp16_flash_attn_input = fp16_baseline.attention_only_flash(x, fp16_k_cache, fp16_v_cache)
                                fp16_flash_ffn_ms, fp16_flash_ffn_p20_ms, fp16_flash_ffn_p80_ms = bench_ms(
                                    lambda: fp16_baseline.ffn_only(fp16_flash_attn_input),
                                    repeat_time=args.repeat_time,
                                    bench_mode=args.bench_mode,
                                )
                                if args.layer_mode == "e2e":
                                    fp16_flash_layer_ms, fp16_flash_layer_p20_ms, fp16_flash_layer_p80_ms = bench_ms(
                                        lambda: fp16_baseline.ffn_only(fp16_baseline.attention_only_flash(x, fp16_k_cache, fp16_v_cache)),
                                        repeat_time=args.repeat_time,
                                        bench_mode=args.bench_mode,
                                    )
                                else:
                                    fp16_flash_layer_ms = fp16_flash_attention_ms + fp16_flash_ffn_ms
                                    fp16_flash_layer_p20_ms = fp16_flash_attention_p20_ms + fp16_flash_ffn_p20_ms
                                    fp16_flash_layer_p80_ms = fp16_flash_attention_p80_ms + fp16_flash_ffn_p80_ms
                                fp16_flash_ok = True
                            except Exception as exc:
                                if not is_oom_error(exc):
                                    raise
                                LOGGER.warning(
                                    "OOM: skipping fp16 flash SDPA benchmark for batch=%d context=%d; continuing with other backends/providers. error=%s",
                                    batch_size,
                                    context_len,
                                    exc,
                                    exc_info=True,
                                )
                                clear_cuda_state()
                        else:
                            LOGGER.warning(
                                "Skipping fp16 flash SDPA benchmark for batch=%d context=%d: %s",
                                batch_size,
                                context_len,
                                flash_reason,
                            )
                        if fp16_eager_ok:
                            rows.append(
                                {
                                    "provider": FP16_PROVIDER,
                                    "attention_backend": FP16_EAGER_ATTENTION,
                                    "series_label": FP16_EAGER_ATTENTION,
                                    "context_len": context_len,
                                    "batch_size": batch_size,
                                    "query_len": effective_query_len,
                                    "hidden_size": args.hidden_size,
                                    "intermediate_size": args.intermediate_size,
                                    "attention_rank_ratio": args.attention_rank_ratio,
                                    "ffn_rank_ratio": None,
                                    "ffn_svd_ratio": None,
                                    "rounded_rank": None,
                                    "svd_out": None,
                                    "dense_out": None,
                                    "round_to": args.round_to,
                                    "attention_ms": float(fp16_eager_attention_ms),
                                    "attention_p20_ms": float(fp16_eager_attention_p20_ms),
                                    "attention_p80_ms": float(fp16_eager_attention_p80_ms),
                                    "ffn_ms": float(fp16_eager_ffn_ms),
                                    "ffn_p20_ms": float(fp16_eager_ffn_p20_ms),
                                    "ffn_p80_ms": float(fp16_eager_ffn_p80_ms),
                                    "layer_ms": float(fp16_eager_layer_ms),
                                    "layer_p20_ms": float(fp16_eager_layer_p20_ms),
                                    "layer_p80_ms": float(fp16_eager_layer_p80_ms),
                                }
                            )
                        if fp16_flash_ok:
                            rows.append(
                                {
                                    "provider": FP16_PROVIDER,
                                    "attention_backend": FP16_PROVIDER,
                                    "series_label": FP16_PROVIDER,
                                    "context_len": context_len,
                                    "batch_size": batch_size,
                                    "query_len": effective_query_len,
                                    "hidden_size": args.hidden_size,
                                    "intermediate_size": args.intermediate_size,
                                    "attention_rank_ratio": args.attention_rank_ratio,
                                    "ffn_rank_ratio": None,
                                    "ffn_svd_ratio": None,
                                    "rounded_rank": None,
                                    "svd_out": None,
                                    "dense_out": None,
                                    "round_to": args.round_to,
                                    "attention_ms": float(fp16_flash_attention_ms),
                                    "attention_p20_ms": float(fp16_flash_attention_p20_ms),
                                    "attention_p80_ms": float(fp16_flash_attention_p80_ms),
                                    "ffn_ms": float(fp16_flash_ffn_ms),
                                    "ffn_p20_ms": float(fp16_flash_ffn_p20_ms),
                                    "ffn_p80_ms": float(fp16_flash_ffn_p80_ms),
                                    "layer_ms": float(fp16_flash_layer_ms),
                                    "layer_p20_ms": float(fp16_flash_layer_p20_ms),
                                    "layer_p80_ms": float(fp16_flash_layer_p80_ms),
                                }
                            )
                        if fp16_eager_ok:
                            print(
                                f"batch={batch_size:>4} context={context_len:>5} attn={FP16_EAGER_ATTENTION:<10} provider={FP16_PROVIDER:<22} "
                                f"attn={fp16_eager_attention_ms:.3f} ms ffn={fp16_eager_ffn_ms:.3f} ms layer={fp16_eager_layer_ms:.3f} ms"
                            )
                        if fp16_flash_ok:
                            print(
                                f"batch={batch_size:>4} context={context_len:>5} attn={FP16_PROVIDER:<10} provider={FP16_PROVIDER:<22} "
                                f"attn={fp16_flash_attention_ms:.3f} ms ffn={fp16_flash_ffn_ms:.3f} ms layer={fp16_flash_layer_ms:.3f} ms"
                            )

                        if not latent_attention_supported:
                            LOGGER.warning(
                                "Skipping int8 latent attention benchmark for batch=%d context=%d stage=%s query_len=%d cache_len=%d; decode with query_len=1 and cache_len>0 is required.",
                                batch_size,
                                context_len,
                                args.stage,
                                effective_query_len,
                                effective_cache_len,
                            )
                            continue
                        for provider in args.providers:
                            if cache_paths:
                                clear_cache_records(LAYER_KERNELS)
                                for best_config_path in cache_paths:
                                    apply_cache_records(best_config_path, LAYER_KERNELS, provider_filter=provider)
                            if provider == "mixffn":
                                provider_settings = [
                                    (rank_ratio, svd_ratio)
                                    for rank_ratio in args.ffn_rank_ratios
                                    for svd_ratio in args.ffn_svd_ratios
                                ]
                            else:
                                raise ValueError(f"Unsupported provider: {provider}")

                            (
                                cache_latent_k,
                                cache_latent_k_scale,
                                cache_latent_v,
                                cache_latent_v_scale,
                                infer_state,
                            ) = latent_inputs
                            provider_attention_ms, provider_attention_p20_ms, provider_attention_p80_ms = bench_ms(
                                lambda block=attention_block: block(
                                    x,
                                    cache_latent_k,
                                    cache_latent_k_scale,
                                    cache_latent_v,
                                    cache_latent_v_scale,
                                    infer_state,
                                ),
                                repeat_time=args.repeat_time,
                                bench_mode=args.bench_mode,
                            )
                            provider_attn_input = attention_block(
                                x,
                                cache_latent_k,
                                cache_latent_k_scale,
                                cache_latent_v,
                                cache_latent_v_scale,
                                infer_state,
                            )

                            for rank_ratio, svd_ratio in provider_settings:
                                effective_rank_ratio = 0.5 if rank_ratio is None else rank_ratio
                                effective_svd_ratio = 0.5 if svd_ratio is None else svd_ratio
                                effective_shapes = get_effective_ffn_shapes(
                                    provider=provider,
                                    hidden_size=args.hidden_size,
                                    intermediate_size=args.intermediate_size,
                                    rank_ratio=rank_ratio,
                                    svd_ratio=svd_ratio,
                                    round_to=args.round_to,
                                )
                                LOGGER.info(
                                    "Benchmarking provider=%s batch=%d context=%d m_tokens=%d rank_ratio=%.2f rounded_rank=%s svd_ratio=%.2f svd_out=%s dense_out=%s",
                                    provider,
                                    batch_size,
                                    context_len,
                                    batch_size * effective_query_len,
                                    effective_rank_ratio,
                                    effective_shapes["rounded_rank"],
                                    effective_svd_ratio,
                                    effective_shapes["svd_out"],
                                    effective_shapes["dense_out"],
                                )
                                try:
                                    ffn_module = module_bank[(provider, rank_ratio, svd_ratio)]

                                    series_label = f"{provider} | rank={rank_ratio:.2f} svd={svd_ratio:.2f}"

                                    ffn_ms, ffn_p20_ms, ffn_p80_ms = bench_ms(
                                        lambda provider_name=provider, module=ffn_module, attn_input=provider_attn_input: FFN_NAME_MAP[provider_name](attn_input, module),
                                        repeat_time=args.repeat_time,
                                        bench_mode=args.bench_mode,
                                    )
                                    if args.layer_mode == "e2e":
                                        layer_ms, layer_p20_ms, layer_p80_ms = bench_ms(
                                            lambda provider_name=provider, module=ffn_module, block=attention_block: FFN_NAME_MAP[provider_name](
                                                block(
                                                    x,
                                                    cache_latent_k,
                                                    cache_latent_k_scale,
                                                    cache_latent_v,
                                                    cache_latent_v_scale,
                                                    infer_state,
                                                ),
                                                module,
                                            ),
                                            repeat_time=args.repeat_time,
                                            bench_mode=args.bench_mode,
                                        )
                                    else:
                                        layer_ms = provider_attention_ms + ffn_ms
                                        layer_p20_ms = provider_attention_p20_ms + ffn_p20_ms
                                        layer_p80_ms = provider_attention_p80_ms + ffn_p80_ms
                                    fake_fp16_metrics = {}
                                    if args.fake_fp16_correctness and provider in FAKE_FP16_CORRECTNESS_PROVIDERS:
                                        fake_fp16_metrics = run_fake_fp16_correctness(
                                            provider_attn_input,
                                            ffn_module,
                                            provider,
                                            atol=args.fake_fp16_atol,
                                            rtol=args.fake_fp16_rtol,
                                        )
                                        correctness_fail = not fake_fp16_metrics.get(
                                            "fake_fp16_isolated_ok",
                                            fake_fp16_metrics.get("fake_fp16_ok", False),
                                        )
                                        if args.fake_fp16_fail_fast and correctness_fail:
                                            raise RuntimeError(
                                                "fake_fp16_correctness failed: "
                                                f"provider={provider} batch={batch_size} context={context_len} "
                                                f"rank={effective_rank_ratio:.2f} svd={effective_svd_ratio:.2f} "
                                                f"max_abs={fake_fp16_metrics.get('fake_fp16_max_abs')} "
                                                f"max_rel={fake_fp16_metrics.get('fake_fp16_max_rel')} "
                                                f"first_fail_stage={fake_fp16_metrics.get('fake_fp16_first_fail_stage')} "
                                                f"worst_abs_stage={fake_fp16_metrics.get('fake_fp16_worst_abs_stage')} "
                                                f"isolated_ok={fake_fp16_metrics.get('fake_fp16_isolated_ok')} "
                                                f"isolated_first_fail_stage={fake_fp16_metrics.get('fake_fp16_isolated_first_fail_stage')}"
                                            )
                                    stage_breakdown = None
                                    if args.stage_bench and provider in FFN_STAGE_BENCH_PROVIDERS:
                                        stage_breakdown = benchmark_stage_breakdown(
                                            provider_attn_input,
                                            ffn_module,
                                            provider,
                                            repeat_time=args.repeat_time,
                                            stage_mode=args.stage_bench_mode,
                                        )
                                        stage_row = {
                                            "provider": provider,
                                            "batch_size": batch_size,
                                            "context_len": context_len,
                                            "query_len": effective_query_len,
                                            "hidden_size": args.hidden_size,
                                            "intermediate_size": args.intermediate_size,
                                            "attention_rank_ratio": args.attention_rank_ratio,
                                            "ffn_rank_ratio": rank_ratio,
                                            "ffn_svd_ratio": svd_ratio,
                                            "rounded_rank": effective_shapes["rounded_rank"],
                                            "svd_out": effective_shapes["svd_out"],
                                            "dense_out": effective_shapes["dense_out"],
                                            "round_to": args.round_to,
                                            "attention_ms": float(provider_attention_ms),
                                            "ffn_ms": float(ffn_ms),
                                            "layer_ms": float(layer_ms),
                                            **{name: float(stage_breakdown[name]) for name in FFN_STAGE_COLUMNS if name in stage_breakdown},
                                            **fake_fp16_metrics,
                                        }
                                        stage_rows.append(stage_row)
                                    row = {
                                        "provider": provider,
                                        "attention_backend": INT8_ATTENTION,
                                        "series_label": series_label,
                                        "context_len": context_len,
                                        "batch_size": batch_size,
                                        "query_len": effective_query_len,
                                        "hidden_size": args.hidden_size,
                                        "intermediate_size": args.intermediate_size,
                                        "attention_rank_ratio": args.attention_rank_ratio,
                                        "ffn_rank_ratio": rank_ratio,
                                        "ffn_svd_ratio": svd_ratio,
                                        "rounded_rank": effective_shapes["rounded_rank"],
                                        "svd_out": effective_shapes["svd_out"],
                                        "dense_out": effective_shapes["dense_out"],
                                        "round_to": args.round_to,
                                        "attention_ms": float(provider_attention_ms),
                                        "attention_p20_ms": float(provider_attention_p20_ms),
                                        "attention_p80_ms": float(provider_attention_p80_ms),
                                        "ffn_ms": float(ffn_ms),
                                        "ffn_p20_ms": float(ffn_p20_ms),
                                        "ffn_p80_ms": float(ffn_p80_ms),
                                        "layer_ms": float(layer_ms),
                                        "layer_p20_ms": float(layer_p20_ms),
                                        "layer_p80_ms": float(layer_p80_ms),
                                        **fake_fp16_metrics,
                                    }
                                    rows.append(row)
                                    correctness_text = ""
                                    if fake_fp16_metrics:
                                        correctness_text = (
                                            " | ffn_fake_fp16_correctness "
                                            f"ok={fake_fp16_metrics['fake_fp16_ok']} "
                                            f"max_abs={fake_fp16_metrics['fake_fp16_max_abs']:.4g} "
                                            f"mean_abs={fake_fp16_metrics['fake_fp16_mean_abs']:.4g} "
                                            f"max_rel={fake_fp16_metrics['fake_fp16_max_rel']:.4g} "
                                            f"finite={fake_fp16_metrics['fake_fp16_finite']}/{fake_fp16_metrics['fake_fp16_total']} "
                                            f"bad=(kernel:{fake_fp16_metrics['fake_fp16_kernel_first_bad']}, "
                                            f"ref:{fake_fp16_metrics['fake_fp16_ref_first_bad']}) "
                                            f"stage=(first:{fake_fp16_metrics['fake_fp16_first_fail_stage']}, "
                                            f"worst_abs:{fake_fp16_metrics['fake_fp16_worst_abs_stage']}, "
                                            f"worst_rel:{fake_fp16_metrics['fake_fp16_worst_rel_stage']}) "
                                            f"isolated=(ok:{fake_fp16_metrics['fake_fp16_isolated_ok']}, "
                                            f"first:{fake_fp16_metrics['fake_fp16_isolated_first_fail_stage']}, "
                                            f"worst_abs:{fake_fp16_metrics['fake_fp16_isolated_worst_abs_stage']}, "
                                            f"worst_rel:{fake_fp16_metrics['fake_fp16_isolated_worst_rel_stage']}) "
                                            f"nan=(kernel:{fake_fp16_metrics['fake_fp16_kernel_nan']}, "
                                            f"ref:{fake_fp16_metrics['fake_fp16_ref_nan']}) "
                                            f"inf=(kernel:{fake_fp16_metrics['fake_fp16_kernel_inf']}, "
                                            f"ref:{fake_fp16_metrics['fake_fp16_ref_inf']})"
                                        )
                                    print(
                                        f"batch={batch_size:>4} context={context_len:>5} attn={INT8_ATTENTION:<10} provider={provider:<22} "
                                        f"rank={effective_rank_ratio:.2f} rank_eff={effective_shapes['rounded_rank']} "
                                        f"svd={effective_svd_ratio:.2f} svd_out={effective_shapes['svd_out']} dense={effective_shapes['dense_out']} "
                                        f"attn_ms={provider_attention_ms:.3f} ffn={ffn_ms:.3f} ms layer={layer_ms:.3f} ms"
                                        f"{correctness_text}"
                                    )
                                except Exception as exc:
                                    if not is_oom_error(exc):
                                        raise
                                    LOGGER.error(
                                        "OOM: skipping provider=%s batch=%d context=%d rank_ratio=%.2f rounded_rank=%s svd_ratio=%.2f svd_out=%s dense_out=%s error=%s",
                                        provider,
                                        batch_size,
                                        context_len,
                                        effective_rank_ratio,
                                        effective_shapes["rounded_rank"],
                                        effective_svd_ratio,
                                        effective_shapes["svd_out"],
                                        effective_shapes["dense_out"],
                                        exc,
                                    )
                                    print(
                                        f"OOM skip: batch={batch_size} context={context_len} attn={INT8_ATTENTION} provider={provider} "
                                        f"rank={effective_rank_ratio:.2f} rank_eff={effective_shapes['rounded_rank']} "
                                        f"svd={effective_svd_ratio:.2f} svd_out={effective_shapes['svd_out']} dense={effective_shapes['dense_out']} "
                                        f"error={exc}"
                                    )
                                    clear_cuda_state()
                                    continue
                            del provider_attn_input
                            del cache_latent_k
                            del cache_latent_k_scale
                            del cache_latent_v
                            del cache_latent_v_scale
                            del infer_state
                        del fp16_eager_attn_input
                        if fp16_flash_attn_input is not None:
                            del fp16_flash_attn_input
                        del fp16_k_cache
                        del fp16_v_cache
                        del x
                        del k_cache
                        del v_cache
                        del k_cache_scale
                        del v_cache_scale
                        del latent_inputs
                        clear_cuda_state()
                    except Exception as exc:
                        if not is_oom_error(exc):
                            raise
                        try:
                            del fp16_flash_attn_input
                        except UnboundLocalError:
                            pass
                        try:
                            del fp16_eager_attn_input
                        except UnboundLocalError:
                            pass
                        try:
                            del fp16_k_cache
                        except UnboundLocalError:
                            pass
                        try:
                            del fp16_v_cache
                        except UnboundLocalError:
                            pass
                        try:
                            del latent_inputs
                        except UnboundLocalError:
                            pass
                        try:
                            del k_cache
                        except UnboundLocalError:
                            pass
                        try:
                            del v_cache
                        except UnboundLocalError:
                            pass
                        try:
                            del k_cache_scale
                        except UnboundLocalError:
                            pass
                        try:
                            del v_cache_scale
                        except UnboundLocalError:
                            pass
                        try:
                            del x
                        except UnboundLocalError:
                            pass
                        LOGGER.error(
                            "OOM: skipping batch=%d context=%d during attention/fp16 baseline setup error=%s",
                            batch_size,
                            context_len,
                            exc,
                        )
                        print(
                            f"OOM skip: batch={batch_size} context={context_len} provider=attention_or_fp16_baseline error={exc}"
                        )
                        clear_cuda_state()
                        continue
            except Exception as exc:
                LOGGER.exception("Batch sweep failed for batch=%d error=%s", batch_size, exc)
                print(f"ERROR skip: batch={batch_size} error={exc}")
                clear_cuda_state()
                continue

    baselines = {
        (row["batch_size"], row["context_len"]): row["layer_ms"]
        for row in rows
        if row["provider"] == FP16_PROVIDER and row["attention_backend"] == FP16_EAGER_ATTENTION
    }
    speedup_rows = []
    dropped_contexts = set()
    for row in rows:
        baseline_key = (row["batch_size"], row["context_len"])
        baseline = baselines.get(baseline_key)
        if baseline is None:
            dropped_contexts.add(baseline_key)
            continue
        row["speedup_vs_fp16"] = baseline / row["layer_ms"]
        speedup_rows.append(row)
    if dropped_contexts:
        dropped_text = ", ".join(f"batch={batch} context={context}" for batch, context in sorted(dropped_contexts))
        LOGGER.warning("Dropping rows with no FP16 eager speedup baseline: %s", dropped_text)
        print(f"Dropping rows with no FP16 eager speedup baseline: {dropped_text}")
    rows = speedup_rows

    rows.sort(
        key=lambda row: (
            row["batch_size"],
            row["context_len"],
            row["provider"],
            -1.0 if row["ffn_rank_ratio"] is None else row["ffn_rank_ratio"],
            -1.0 if row["ffn_svd_ratio"] is None else row["ffn_svd_ratio"],
        )
    )
    print_results(rows)
    if stage_rows:
        stage_rows.sort(
            key=lambda row: (
                row["batch_size"],
                row["context_len"],
                row["provider"],
                row["ffn_rank_ratio"],
                row["ffn_svd_ratio"],
            )
        )
        print_stage_results(stage_rows)
        write_stage_results(stage_rows, args.output_dir / f"{args.output_name}_stage_bench.tsv")
    plot_path = args.output_dir / f"{args.output_name}.png"
    try:
        plot_speedups(rows, plot_path, PROVIDER_DISPLAY_NAMES)
    except RuntimeError as exc:
        print(exc)


if __name__ == "__main__":
    main()
