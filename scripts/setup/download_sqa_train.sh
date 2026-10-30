#!/usr/bin/env bash
# Download ScienceQA (problems, splits, train images), following https://github.com/lupantech/ScienceQA.
#
# Data is stored under ${LSVD_DATA_ROOT:-<repo>/data}/scienceqa. The calibration loader
# (fake_quant/utils/data_utils.py) reads <repo>/data/scienceqa relative to fake_quant/, so if
# LSVD_DATA_ROOT points elsewhere a symlink <repo>/data/scienceqa -> $LSVD_DATA_ROOT/scienceqa is created.
set -euo pipefail
LSVD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${LSVD_DATA_ROOT:-${LSVD_ROOT}/data}"

mkdir -p "${DATA_ROOT}/scienceqa/images"
cd "${DATA_ROOT}/scienceqa"
wget https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/problems.json
wget https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/pid_splits.json

cd images
wget https://scienceqa.s3.us-west-1.amazonaws.com/images/train.zip
# wget https://scienceqa.s3.us-west-1.amazonaws.com/images/val.zip
# wget https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip

unzip -q train.zip
# unzip -q val.zip
# unzip -q test.zip

rm train.zip
# rm val.zip
# rm test.zip

if [ "$(readlink -f "${DATA_ROOT}")" != "$(readlink -f "${LSVD_ROOT}/data" 2>/dev/null || echo "${LSVD_ROOT}/data")" ]; then
    mkdir -p "${LSVD_ROOT}/data"
    ln -sfn "${DATA_ROOT}/scienceqa" "${LSVD_ROOT}/data/scienceqa"
    echo "linked ${LSVD_ROOT}/data/scienceqa -> ${DATA_ROOT}/scienceqa"
fi
