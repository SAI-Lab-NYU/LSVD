# shellcheck shell=bash
# Shared prologue for scripts/lsvd/*.sh (sourced, not executed).
#
# Environment variables (see README.md):
#   HF_HOME           HuggingFace cache; models are downloaded on first use.
#   LMUData           VLMEvalKit dataset directory (ScienceQA_TEST.tsv, SEEDBench_IMG.tsv,
#                     MMBench_TEST_EN_V11.tsv). If VLMEvalKit's own .env overrides it at import
#                     time, also set WSVD_LMUDATA_ROOT to the same directory.
#   LSVD_GROUPS_DIR   Hybrid-FFN channel-group JSONs (default: <repo>/groups); read by w8a8.sh,
#                     written by ffn_groups.sh.
#   LSVD_CACHE_DIR    Optional. Calibration caches and results are written to
#                     fake_quant/utils/experiments/; if LSVD_CACHE_DIR is set and that path does
#                     not exist yet, it is created as a symlink to LSVD_CACHE_DIR.
#   LSVD_CALIB_CACHE_DIR  Optional (w8a8.sh). Directory of the released calibration caches, one
#                     subdirectory per model; read instead of computing the statistics.
#   LSVD_DRY_RUN=1    Print the python command line(s) instead of running them.
#
# Run parameters (environment variables or KEY=VALUE arguments):
#   MODEL         required
#   SEED          default 0; a space-separated list runs one job per value
#   RATIO         default 0.5; parameter ratio rho_1, space-separated list allowed
#   EVAL_DATASET  default ScienceQA_TEST (also SEEDBench_IMG, MMBench_TEST_EN_V11)
# Any other argument is appended verbatim to the python command line.

LSVD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LSVD_ROOT
: "${LSVD_GROUPS_DIR:=${LSVD_ROOT}/groups}"
export LSVD_GROUPS_DIR

if [ "${LSVD_DRY_RUN:-0}" = "1" ]; then
    python() {
        printf '%q ' python "$@"
        printf '\n'
    }
else
    if [ -n "${LSVD_CACHE_DIR:-}" ]; then
        _lsvd_exp="${LSVD_ROOT}/fake_quant/utils/experiments"
        mkdir -p "${LSVD_CACHE_DIR}"
        if [ ! -e "${_lsvd_exp}" ] && [ ! -L "${_lsvd_exp}" ]; then
            ln -s "${LSVD_CACHE_DIR}" "${_lsvd_exp}"
        elif [ "$(readlink -f "${_lsvd_exp}")" != "$(readlink -f "${LSVD_CACHE_DIR}")" ]; then
            echo "[lsvd] warning: ${_lsvd_exp} already exists and is not LSVD_CACHE_DIR; using it as is." >&2
        fi
    fi
    [ -n "${HF_HOME:-}" ] || echo "[lsvd] note: HF_HOME is not set; using the HuggingFace default cache." >&2
    [ -n "${LMUData:-}" ] || echo "[lsvd] note: LMUData is not set; VLMEvalKit falls back to its default (~/LMUData)." >&2
fi

cd "${LSVD_ROOT}/fake_quant" || exit 1

# lsvd_parse_args "$@": KEY=VALUE for MODEL/SEED/RATIO/EVAL_DATASET, everything else -> LSVD_EXTRA_ARGS
lsvd_parse_args() {
    LSVD_EXTRA_ARGS=()
    local a
    for a in "$@"; do
        case "${a}" in
            MODEL=*|SEED=*|RATIO=*|EVAL_DATASET=*) export "${a?}" ;;
            *) LSVD_EXTRA_ARGS+=("${a}") ;;
        esac
    done
    LSVD_SEED_GIVEN="${SEED:+1}"
    : "${SEED:=0}"
    : "${RATIO:=0.5}"
    : "${EVAL_DATASET:=ScienceQA_TEST}"
}

# lsvd_dataset_tag <eval dataset>: short name used in the output path
lsvd_dataset_tag() {
    case "$1" in
        ScienceQA_TEST) echo sqa ;;
        SEEDBench_IMG) echo seed ;;
        MMBench_TEST_EN_V11) echo mmbench ;;
        *) echo "$1" | tr '[:upper:]' '[:lower:]' ;;
    esac
}

lsvd_usage() {
    echo "usage: MODEL=<model> [SEED=0] [RATIO=0.5] [EVAL_DATASET=ScienceQA_TEST] bash $0 [extra python args]" >&2
    echo "  models: $*" >&2
    exit 2
}
