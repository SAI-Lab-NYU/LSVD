#!/usr/bin/env bash
# Phase 1 of the hybrid FFN: SVD-aware channel-group analysis. Writes the channel-group JSON that
# scripts/lsvd/w8a8.sh loads with --hybrid_ffn_groups_path, under the same file name, into
# $LSVD_GROUPS_DIR. Refuses to overwrite an existing JSON unless LSVD_OVERWRITE=1.
#
#   MODEL=llavanext_7b bash scripts/lsvd/ffn_groups.sh
#
# The released groups JSONs were built with 64 calibration samples (--nsamples 64), not the 256
# used by the LSVD runs themselves. The commands below use the flag sets of those Phase-1 runs:
#   llavanext_7b / llavanext_13b: W8A8, QuaRot rotation + online Hadamard, Cholesky-whitened FFN
#       SVD, svd_ffn_ratio 0.2.
#   llava15_7b (a different recipe): W8 with 16-bit activations, no rotation / Hadamard, QKV
#       rank_ratio 0.5 as a plain rank ratio (no --use_true_param_ratio), svd_ffn_ratio 0.1; the
#       0.2 JSON is then sliced from the greedy acceptance order with
#       fake_quant/utils/derive_ratio_groups.py.
# Notes:
#   - Phase 1 does not modify the weights; the entry point afterwards still evaluates the plain
#     model (ScienceQA_TEST). That result is not used; append --skip_evaluation to skip it (the
#     paper runs did not).
#   - The LLaVA-Next Phase-1 runs used the default --label_mode (q-a); w8a8.sh uses qa-qa for
#     LLaVA-Next.
#   - The Phase-1 code evolved after these runs, so regenerated JSONs are not guaranteed to be
#     identical to the released ones.
# Parameters: MODEL, SEED (default 0); RATIO / EVAL_DATASET are fixed by the recipe.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
lsvd_parse_args "$@"
[ "${LSVD_DRY_RUN:-0}" = "1" ] || mkdir -p "${LSVD_GROUPS_DIR}"

check_out() {
    if [ "${LSVD_DRY_RUN:-0}" != "1" ] && [ -e "$1" ] && [ "${LSVD_OVERWRITE:-0}" != "1" ]; then
        echo "[lsvd] $1 exists; set LSVD_OVERWRITE=1 to regenerate it" >&2
        exit 1
    fi
}

# model         HF model id                          output JSON (file name read by w8a8.sh)
case "${MODEL:-}" in
  llavanext_7b)  MODEL_ID=llava-hf/llava-v1.6-vicuna-7b-hf;  OUT=hybrid_groups_llavanext_7b.json ;;
  llavanext_13b) MODEL_ID=llava-hf/llava-v1.6-vicuna-13b-hf; OUT=hybrid_groups_llavanext_13b.json ;;
  llava15_7b)    MODEL_ID=liuhaotian/llava-v1.5-7b;          OUT=hybrid_groups_llava15_7b.json ;;
  *) lsvd_usage llava15_7b llavanext_7b llavanext_13b ;;
esac

for seed in ${SEED}; do
  if [ "${MODEL}" = llava15_7b ]; then
    # Recipe B (LLaVA-v1.5 7B): analysis at svd_ffn_ratio 0.1 (kept as ..._r10.json), then slice to 0.2.
    OUT01="${LSVD_GROUPS_DIR}/hybrid_groups_llava15_7b_r10.json"
    check_out "${OUT01}"; check_out "${LSVD_GROUPS_DIR}/${OUT}"
    python main_llava_hybrid_ffn.py \
      --model "${MODEL_ID}" \
      --w_bits 8 --a_bits 16 --k_bits 16 --v_bits 16 \
      --w_rtn --w_clip \
      --cal_dataset ScienceQA_Train --nsamples 64 --bs 256 --seed "${seed}" \
      --eval_dataset ScienceQA_TEST --vlmtasks ScienceQA_TEST --tasks None \
      --svd_lm --svd_modules qkv --svd_mode U --is_per_head_svd \
      --act_aware --act_alpha 0.5 --calib_method cholesky \
      --loss_aware --grad_info --taylor_order 2 \
      --rank_ratio 0.5 \
      --use_hybrid_ffn --svd_ffn_ratio 0.1 --svd_ffn_calib_method svd_aware \
      --hybrid_ffn_analyze_path "${OUT01}" \
      --setting "lsvd/ffn_groups/${MODEL}/seed${seed}" \
      --cache_in_log --verbose \
      "${LSVD_EXTRA_ARGS[@]}" || exit 1
    python utils/derive_ratio_groups.py \
      --src "${OUT01}" --out "${LSVD_GROUPS_DIR}" --ratios 0.2 \
      --prefix hybrid_groups_llava15_7b || exit 1
    SLICED="${LSVD_GROUPS_DIR}/hybrid_groups_llava15_7b_r20_svd_aware.json"  # name written by derive_ratio_groups.py
    if [ "${LSVD_DRY_RUN:-0}" = "1" ]; then printf 'mv %q %q\n' "${SLICED}" "${LSVD_GROUPS_DIR}/${OUT}"
    else mv "${SLICED}" "${LSVD_GROUPS_DIR}/${OUT}" || exit 1; fi
  else
    check_out "${LSVD_GROUPS_DIR}/${OUT}"
    python main_llava_next_hybrid_ffn.py \
      --model "${MODEL_ID}" \
      --w_bits 8 --a_bits 8 --k_bits 16 --v_bits 16 \
      --w_rtn --w_clip \
      --cal_dataset ScienceQA_Train --nsamples 64 --bs 256 --seed "${seed}" \
      --eval_dataset ScienceQA_TEST --vlmtasks ScienceQA_TEST --tasks None \
      --svd_lm --svd_modules qkv --svd_mode U --is_per_head_svd \
      --act_aware --act_alpha 0.5 --calib_method cholesky \
      --loss_aware --grad_info --taylor_order 2 \
      --rank_ratio 0.5 --use_true_param_ratio \
      --rotate \
      --use_hybrid_ffn --svd_ffn_ratio 0.2 --svd_ffn_calib_method svd_aware --hybrid_ffn_rank_ratio 0.9 \
      --hybrid_ffn_analyze_path "${LSVD_GROUPS_DIR}/${OUT}" \
      --hybrid_ffn_act_aware --hybrid_ffn_act_aware_mode chol \
      --hybrid_ffn_had \
      --setting "lsvd/ffn_groups/${MODEL}/seed${seed}" \
      --cache_in_log --verbose \
      "${LSVD_EXTRA_ARGS[@]}" || exit 1
  fi
done
