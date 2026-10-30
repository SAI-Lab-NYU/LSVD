#!/usr/bin/env bash
# Decoding latency of one transformer layer (Fig. 5 setting): Eager attention, Flash Decoding and LSVD at batch 32
# and context 4096 / 6144 / 8192, on the current GPU.
#   1. autotunes the Triton kernels of the LSVD layer (mixffn) for this GPU and shape,
#      writing kernels/autotune_cache/<CACHE_NAME>-bsz<B>.{pkl,yaml} (skip with SKIP_AUTOTUNE=1);
#   2. times every method with CUDA-graph replay and writes a table and a plot to OUT_DIR.
#
#   bash kernels/run_fig5.sh
#   CONTEXT_LENS=4096 SKIP_AUTOTUNE=1 bash kernels/run_fig5.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
GPU_TAG="${GPU_TAG:-$("${PYTHON}" -c 'import torch; print(torch.cuda.get_device_name(0).replace(" ", "_"))')}"

BATCH_SIZE="${BATCH_SIZE:-32}"
CONTEXT_LENS="${CONTEXT_LENS:-4096,6144,8192}"
LSVD_RHO1="${LSVD_RHO1:-0.5}"            # LSVD attention parameter ratio
FFN_SVD_RATIO="${FFN_SVD_RATIO:-0.5}"    # fraction of FFN channels in the SVD branch
FFN_RANK_RATIO="${FFN_RANK_RATIO:-0.9}"  # rank ratio of the FFN SVD branch
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-14336}"
REPEAT_TIME="${REPEAT_TIME:-300}"
CACHE_NAME="${CACHE_NAME:-fig5-${GPU_TAG}}"
OUT_DIR="${OUT_DIR:-results/fig5_${GPU_TAG}}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
SHAPE=(--hidden-size 4096 --intermediate-size "${INTERMEDIATE_SIZE}" --num-heads 32 --head-dim 128)

if [ "${SKIP_AUTOTUNE:-0}" != "1" ]; then
    "${PYTHON}" "${HERE}/layer_bench/run_autotune_triton_int8_layer.py" \
        --providers mixffn --batch-sizes "${BATCH_SIZE}" --query-len 1 --context-lens "${CONTEXT_LENS}" \
        "${SHAPE[@]}" --attention-rank-ratio "${LSVD_RHO1}" \
        --ffn-rank-ratios "${FFN_RANK_RATIO}" --ffn-svd-ratios "${FFN_SVD_RATIO}" --round-to 64 \
        --best-config-path "${CACHE_NAME}" --log-level INFO 2>&1 | tee "${OUT_DIR}/autotune.log"
fi

"${PYTHON}" "${HERE}/layer_bench/benchmark_triton_int8_layer.py" \
    --providers mixffn --batch-sizes "${BATCH_SIZE}" --context-lens "${CONTEXT_LENS}" --query-len 1 --stage decode \
    "${SHAPE[@]}" --attention-rank-ratio "${LSVD_RHO1}" \
    --ffn-rank-ratios "${FFN_RANK_RATIO}" --ffn-svd-ratios "${FFN_SVD_RATIO}" --round-to 64 \
    --repeat-time "${REPEAT_TIME}" --bench-mode cuda_graph --layer-mode e2e \
    --best-config-paths "${CACHE_NAME}-bsz${BATCH_SIZE}" \
    --output-dir "${OUT_DIR}" --output-name "fig5_${GPU_TAG}" --log-level INFO 2>&1 | tee "${OUT_DIR}/bench.log"
echo "Outputs in ${OUT_DIR}"
