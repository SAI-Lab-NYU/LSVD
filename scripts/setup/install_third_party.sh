#!/usr/bin/env bash
# Install the third-party packages at the exact commits used for the paper (see third_party/README.md).
# Run inside the Python environment that already has requirements.txt installed.
# Clones into ${LSVD_THIRD_PARTY_DIR:-<repo>/third_party/src}.
set -euo pipefail
LSVD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TP="${LSVD_THIRD_PARTY_DIR:-${LSVD_ROOT}/third_party/src}"
mkdir -p "${TP}"

clone_at() {  # url dir commit
    if [ ! -d "$2/.git" ]; then git clone "$1" "$2"; fi
    git -C "$2" checkout --detach "$3"
}

# LLaVA (provides the `llava` package for LLaVA-v1.5 models) + 1-line cache_position patch.
# --no-deps: LLaVA pins torch==2.1.2 / transformers==4.37.2, which must not be installed.
clone_at https://github.com/haotian-liu/LLaVA.git "${TP}/LLaVA" c121f0432da27facab705978f83c4ada465e46fd
git -C "${TP}/LLaVA" apply --check "${LSVD_ROOT}/third_party/llava_cache_position.patch" 2>/dev/null \
    && git -C "${TP}/LLaVA" apply "${LSVD_ROOT}/third_party/llava_cache_position.patch" \
    || echo "[lsvd] LLaVA patch already applied or does not apply cleanly; check ${TP}/LLaVA"
pip install --no-deps -e "${TP}/LLaVA"

# VLMEvalKit (evaluation) + local edits (see third_party/README.md).
clone_at https://github.com/open-compass/VLMEvalKit.git "${TP}/VLMEvalKit" 00804217f868058f871f5ff252a7b9623c3475d9
git -C "${TP}/VLMEvalKit" apply --check "${LSVD_ROOT}/third_party/vlmevalkit_local_changes.patch" 2>/dev/null \
    && git -C "${TP}/VLMEvalKit" apply "${LSVD_ROOT}/third_party/vlmevalkit_local_changes.patch" \
    || echo "[lsvd] VLMEvalKit patch already applied or does not apply cleanly; check ${TP}/VLMEvalKit"
pip install -e "${TP}/VLMEvalKit"

# fast-hadamard-transform (CUDA extension used by the QuaRot-style online Hadamard transforms).
clone_at https://github.com/Dao-AILab/fast-hadamard-transform.git "${TP}/fast-hadamard-transform" e7706faf8d1c3b9f241e36860640ad1dac644ede
pip install --no-build-isolation "${TP}/fast-hadamard-transform"
