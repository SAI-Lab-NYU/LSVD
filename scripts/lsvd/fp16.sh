#!/usr/bin/env bash
# LSVD-noQ (FP16): loss-aware per-head Q/K/V SVD + importance-score cross-layer rank allocation,
# no quantization (paper Tab. 1 / 5 / 6, LSVD-noQ rows; Tab. 3 "LSVD" row at RATIO=0.5).
#
#   MODEL=llava15_7b RATIO="0.9 0.8 0.7 0.6 0.5" EVAL_DATASET=SEEDBench_IMG bash scripts/lsvd/fp16.sh
#
# Parameters: MODEL, SEED (default 0), RATIO (default 0.5), EVAL_DATASET (default ScienceQA_TEST);
# see scripts/lsvd/common.sh. Results: fake_quant/utils/experiments/<model id>/W16A16K16V16lsvd/fp16/<tag>/ratio<r>/seed<s>/
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
lsvd_parse_args "$@"

# model         entry point             HF model id                           model-specific flags
case "${MODEL:-}" in
  smolvlm)       ENTRY=main_smolvlm.py;    MODEL_ID=HuggingFaceTB/SmolVLM-Instruct;     MODEL_ARGS=(--label_mode qa-qa) ;;
  qwen2vl_7b)    ENTRY=main_qwen2vl.py;    MODEL_ID=Qwen/Qwen2-VL-7B-Instruct;          MODEL_ARGS=(--label_mode qa-qa) ;;
  llava15_7b)    ENTRY=main_llava.py;      MODEL_ID=liuhaotian/llava-v1.5-7b;           MODEL_ARGS=() ;;
  llavanext_7b)  ENTRY=main_llava_next.py; MODEL_ID=llava-hf/llava-v1.6-vicuna-7b-hf;   MODEL_ARGS=(--label_mode qa-qa) ;;
  llavanext_13b) ENTRY=main_llava_next.py; MODEL_ID=llava-hf/llava-v1.6-vicuna-13b-hf;  MODEL_ARGS=(--label_mode qa-qa) ;;
  *) lsvd_usage smolvlm qwen2vl_7b llava15_7b llavanext_7b llavanext_13b ;;
esac
TAG="$(lsvd_dataset_tag "${EVAL_DATASET}")"

for seed in ${SEED}; do
  for ratio in ${RATIO}; do
    python "${ENTRY}" \
      --model "${MODEL_ID}" \
      --w_bits 16 --a_bits 16 --k_bits 16 --v_bits 16 \
      --w_clip \
      --cal_dataset ScienceQA_Train --nsamples 256 --bs 256 --seed "${seed}" \
      --eval_dataset "${EVAL_DATASET}" --tasks None \
      --svd_lm --svd_modules qkv --svd_mode U --is_per_head_svd \
      --act_aware --act_alpha 0.5 --calib_method cholesky \
      --loss_aware --grad_info --taylor_order 2 \
      --rank_ratio "${ratio}" --use_true_param_ratio \
      "${MODEL_ARGS[@]}" \
      --setting "lsvd/fp16/${TAG}/ratio${ratio}/seed${seed}" \
      --cache_in_log \
      "${LSVD_EXTRA_ARGS[@]}"
  done
done
