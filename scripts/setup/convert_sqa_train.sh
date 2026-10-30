#!/usr/bin/env bash
# Convert ScienceQA train to the LLaVA QCM-LEA conversation format used for calibration.
# Produces ${LSVD_DATA_ROOT:-<repo>/data}/scienceqa/llava_train_QCM-LEA.json (6218 samples).
# Requires: pip install fire
set -euo pipefail
LSVD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${LSVD_DATA_ROOT:-${LSVD_ROOT}/data}"

python "${LSVD_ROOT}/scripts/setup/convert_sqa_to_llava.py" \
    convert_to_llava \
    --base-dir "${DATA_ROOT}/scienceqa" \
    --prompt-format "QCM-LEA" \
    --split train

expected=0d3eef744571645b9835ae6283fb23ab
actual=$(md5sum "${DATA_ROOT}/scienceqa/llava_train_QCM-LEA.json" | cut -d' ' -f1)
if [ "${actual}" = "${expected}" ]; then
    echo "llava_train_QCM-LEA.json md5 OK (${actual})"
else
    echo "warning: llava_train_QCM-LEA.json md5 ${actual} != ${expected} (file used in the paper)" >&2
fi
