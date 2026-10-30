#!/usr/bin/env bash
# Optional SLURM wrapper for scripts/lsvd/<script>.sh. Adjust the resource lines to your cluster.
# Usage:
#   SCRIPT=w8a8 MODEL=llava15_7b SEED=0 RATIO=0.5 EVAL_DATASET=ScienceQA_TEST sbatch --export=ALL scripts/slurm_template.sbatch
#
#SBATCH --job-name=lsvd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --account=<your-account>
##SBATCH --partition=<your-partition>

set -eo pipefail
: "${SCRIPT:?set SCRIPT to fp16, w8a8 or ffn_groups}"
: "${MODEL:?set MODEL (see the model table in scripts/lsvd/${SCRIPT}.sh)}"

# Activate your environment here, e.g.:
#   source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate lsvd
# and export HF_HOME / LMUData / LSVD_GROUPS_DIR / LSVD_CACHE_DIR as described in README.md.

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
echo "[lsvd] $(date) host=$(hostname) script=${SCRIPT} model=${MODEL}"
bash "${REPO}/scripts/lsvd/${SCRIPT}.sh" "$@"
echo "[lsvd] $(date) done"
