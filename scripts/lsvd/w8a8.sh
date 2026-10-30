#!/usr/bin/env bash
# LSVD W8A8 (paper Tab. 2 / 7): loss-aware per-head Q/K/V SVD + importance-score rank allocation,
# quantization-aware whitening, QuaRot rotation, RTN W8A8, and the hybrid SVD + RTN FFN with precomputed
# SVD-aware channel groups (FFN SVD channel ratio 0.5, FFN SVD retention 0.9, last 5 layers keep a plain RTN FFN).
# Quantization-aware whitening (QAW, paper Sec. 3.4): --act_calib_fake_quant collects the Q/K/V input-side
# calibration statistics from A8-fake-quantized activations (--a_bits_calib defaults to --a_bits). Its calibration
# caches carry an "_afq_ab8_..." tag, so they are separate from the FP16 (fp16.sh) caches. Pass
# --no-act_calib_fake_quant to run without QAW (the "W/o QAW" ablation).
#
#   MODEL=llavanext_7b EVAL_DATASET=SEEDBench_IMG bash scripts/lsvd/w8a8.sh
#
# Needs the channel-group JSONs in $LSVD_GROUPS_DIR (download, or build with scripts/lsvd/ffn_groups.sh).
# With LSVD_CALIB_CACHE_DIR=<dir> the released calibration caches in <dir>/<MODEL>/ are read instead of
# computing the statistics (README.md, "Precomputed artifacts").
# Parameters: MODEL, SEED (default 0), RATIO (default 0.5), EVAL_DATASET (default ScienceQA_TEST);
# see scripts/lsvd/common.sh. Results: fake_quant/utils/experiments/<model id>/W8A8K16V16lsvd/w8a8/<tag>/ratio<r>/seed<s>/
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
lsvd_parse_args "$@"

# model         entry point                      HF model id                          groups JSON (in $LSVD_GROUPS_DIR)                     FFN skip layers       model-specific flags          seed of the released caches
case "${MODEL:-}" in
  llava15_7b)    ENTRY=main_llava_hybrid_ffn.py;      MODEL_ID=liuhaotian/llava-v1.5-7b;          GROUPS_FILE=hybrid_groups_llava15_7b.json;    SKIP=(27 28 29 30 31); MODEL_ARGS=();                  CALIB_SEED=3 ;;
  llavanext_7b)  ENTRY=main_llava_next_hybrid_ffn.py; MODEL_ID=llava-hf/llava-v1.6-vicuna-7b-hf;  GROUPS_FILE=hybrid_groups_llavanext_7b.json;  SKIP=(27 28 29 30 31); MODEL_ARGS=(--label_mode qa-qa); CALIB_SEED=0 ;;
  llavanext_13b) ENTRY=main_llava_next_hybrid_ffn.py; MODEL_ID=llava-hf/llava-v1.6-vicuna-13b-hf; GROUPS_FILE=hybrid_groups_llavanext_13b.json; SKIP=(35 36 37 38 39); MODEL_ARGS=(--label_mode qa-qa); CALIB_SEED=0 ;;
  *) lsvd_usage llava15_7b llavanext_7b llavanext_13b ;;
esac
# The released caches were computed with the seed CALIB_SEED, which also sets the random Hadamard rotation and
# the hybrid-FFN calibration samples, so runs that read them use that seed.
CALIB_ARGS=()
if [ -n "${LSVD_CALIB_CACHE_DIR:-}" ]; then
    if [ "${LSVD_SEED_GIVEN}" = 1 ] && [ "${SEED}" != "${CALIB_SEED}" ]; then
        echo "[lsvd] the released ${MODEL} caches go with seed ${CALIB_SEED}; do not set SEED with LSVD_CALIB_CACHE_DIR" >&2
        exit 1
    fi
    SEED="${CALIB_SEED}"
    CALIB_ARGS=(--calib_cache_dir "${LSVD_CALIB_CACHE_DIR}/${MODEL}")
fi
TAG="$(lsvd_dataset_tag "${EVAL_DATASET}")"
GROUPS_JSON="${LSVD_GROUPS_DIR}/${GROUPS_FILE}"
if [ "${LSVD_DRY_RUN:-0}" != "1" ] && [ ! -f "${GROUPS_JSON}" ]; then
    echo "[lsvd] missing ${GROUPS_JSON}: download the groups JSONs (README.md) or run scripts/lsvd/ffn_groups.sh" >&2
    exit 1
fi

for seed in ${SEED}; do
  for ratio in ${RATIO}; do
    python "${ENTRY}" \
      --model "${MODEL_ID}" \
      --w_bits 8 --a_bits 8 --k_bits 16 --v_bits 16 \
      --w_rtn --w_clip \
      --cal_dataset ScienceQA_Train --nsamples 256 --bs 256 --seed "${seed}" \
      --eval_dataset "${EVAL_DATASET}" --tasks None \
      --svd_lm --svd_modules qkv --svd_mode U --is_per_head_svd \
      --act_aware --act_alpha 0.5 --calib_method cholesky --act_calib_fake_quant \
      --loss_aware --grad_info --taylor_order 2 \
      --rank_ratio "${ratio}" --use_true_param_ratio \
      --rotate \
      --use_hybrid_ffn --svd_ffn_ratio 0.5 --hybrid_ffn_rank_ratio 0.9 \
      --hybrid_ffn_groups_path "${GROUPS_JSON}" \
      --hybrid_ffn_skip_layers "${SKIP[@]}" \
      --hybrid_ffn_act_aware --hybrid_ffn_act_aware_mode chol \
      --hybrid_ffn_svd_latent_rotate --hybrid_ffn_svd_latent_rotate_mode hadamard \
      --hybrid_ffn_had \
      "${MODEL_ARGS[@]}" "${CALIB_ARGS[@]}" \
      --setting "lsvd/w8a8/${TAG}/ratio${ratio}/$([ ${#CALIB_ARGS[@]} -gt 0 ] && echo released || echo "seed${seed}")" \
      --cache_in_log \
      "${LSVD_EXTRA_ARGS[@]}"
  done
done
