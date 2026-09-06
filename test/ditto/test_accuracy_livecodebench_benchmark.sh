#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_info() {
    echo "[INFO] $*"
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

pick_cuda_visible_devices() {
    local need_gpus="$1"
    local min_free_mb="$2"
    local -a candidates=()

    command -v nvidia-smi >/dev/null 2>&1 || return 1

    mapfile -t candidates < <(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | awk -F',' -v min_free_mb="${min_free_mb}" '
                {
                    gsub(/ /, "", $1);
                    gsub(/ /, "", $2);
                    if (($2 + 0) >= min_free_mb) {
                        print $1 "," $2;
                    }
                }
            ' \
            | sort -t',' -k2,2nr -k1,1n \
            | cut -d',' -f1
    )

    if (( ${#candidates[@]} < need_gpus )); then
        return 1
    fi

    local -a picked=("${candidates[@]:0:need_gpus}")
    local joined
    joined="$(IFS=,; echo "${picked[*]}")"
    echo "${joined}"
}

have_livecodebench_install() {
    [[ -x "${PYTHON_BIN}" ]] || return 1

    "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib.util
import datasets

mods = ["lcb_runner", "pebble", "datasets"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
major = int(datasets.__version__.split(".", 1)[0])
raise SystemExit(0 if not missing and major < 4 else 1)
PY
}

ensure_livecodebench_repo() {
    if [[ -f "${LIVECODEBENCH_ROOT}/pyproject.toml" ]]; then
        return
    fi

    [[ "${AUTO_SETUP}" == "1" ]] || die "missing LiveCodeBench repo: ${LIVECODEBENCH_ROOT}"
    command -v git >/dev/null 2>&1 || die "git is required to clone LiveCodeBench"

    log_info "cloning LiveCodeBench into ${LIVECODEBENCH_ROOT}"
    git clone --depth 1 "${LIVECODEBENCH_REPO_URL}" "${LIVECODEBENCH_ROOT}"
}

ensure_livecodebench_venv() {
    if [[ "${USE_VENV}" != "1" ]]; then
        return
    fi

    if [[ -x "${VENV_DIR}/bin/python" ]]; then
        return
    fi

    log_info "creating LiveCodeBench venv at ${VENV_DIR}"
    "${BASE_PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
}

ensure_livecodebench_install() {
    ensure_livecodebench_venv

    if have_livecodebench_install; then
        return
    fi

    [[ "${INSTALL_DEPS}" == "1" ]] || die "missing LiveCodeBench Python package dependencies; rerun with INSTALL_DEPS=1"

    log_info "installing LiveCodeBench Python package"
    "${PYTHON_BIN}" -m pip install "datasets==${DATASETS_VERSION}"
    "${PYTHON_BIN}" -m pip install pebble
    "${PYTHON_BIN}" -m pip install -e "${LIVECODEBENCH_ROOT}" --no-deps
}

RUN_PRED_PY="${RUN_PRED_PY:-${SCRIPT_DIR}/run_pred.py}"
EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/eval_livecodebench.py}"

DITTO_ROOT="${DITTO_ROOT:-${SCRIPT_DIR}}"
LIVECODEBENCH_ROOT="${LIVECODEBENCH_ROOT:-/datasets/LiveCodeBench}"
LIVECODEBENCH_REPO_URL="${LIVECODEBENCH_REPO_URL:-https://github.com/LiveCodeBench/LiveCodeBench.git}"

BASE_PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-${LIVECODEBENCH_ROOT}/.venv}"
DATASETS_VERSION="${DATASETS_VERSION:-3.6.0}"
PYTHON_BIN="${BASE_PYTHON_BIN}"
if [[ "${USE_VENV}" == "1" ]]; then
    PYTHON_BIN="${VENV_DIR}/bin/python"
fi

MODEL_PATH="${MODEL_PATH:-/models/Llama-3-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"

DATASET_NAME="${DATASET_NAME:-livecodebench}"
TASKS="${TASKS:-livecodebench}"
SCENARIO="${SCENARIO:-codegeneration}"
RELEASE_VERSION="${RELEASE_VERSION:-release_latest}"
START_DATE="${START_DATE:-}"
END_DATE="${END_DATE:-}"
NOT_FAST="${NOT_FAST:-0}"

METHOD="${METHOD:-offloading-hash}"
TOPK="${TOPK:-0.10}"
SELECTIVE_START_LEN="${SELECTIVE_START_LEN:-512}"
case "${METHOD}" in
    flash-attn)
        METHOD="flashattn"
        ;;
esac

MAX_SEQ_LEN="${MAX_SEQ_LEN:-131072}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MP_NUM="${MP_NUM:-1}"
PP_NUM="${PP_NUM:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
AUTO_SELECT_GPUS="${AUTO_SELECT_GPUS:-0}"
MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-20000}"

HEARTBEAT_SEC="${HEARTBEAT_SEC:-10}"
DECODE_LOG_INTERVAL="${DECODE_LOG_INTERVAL:-20}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
DITTO_DEBUG_STALL="${DITTO_DEBUG_STALL:-0}"
DITTO_DEBUG_STALL_MIN_MS="${DITTO_DEBUG_STALL_MIN_MS:-0}"
DITTO_DEBUG_CPUGATHER="${DITTO_DEBUG_CPUGATHER:-0}"

MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-65536}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-}"
DATASET_LIMIT="${DATASET_LIMIT:-0}"

SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-0.2}"
SAMPLING_TOP_P="${SAMPLING_TOP_P:-0.95}"
LCB_MAX_NEW_TOKENS="${LCB_MAX_NEW_TOKENS:-2000}"
LCB_MODEL_STYLE="${LCB_MODEL_STYLE:-LLaMa3}"
NUM_PROCESS_EVALUATE="${NUM_PROCESS_EVALUATE:-12}"
TIMEOUT="${TIMEOUT:-6}"

AUTO_SETUP="${AUTO_SETUP:-1}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
DRY_RUN="${DRY_RUN:-0}"

CONFIG_FILE="${CONFIG_FILE:-}"
if [[ -z "${CONFIG_FILE}" ]]; then
    case "${METHOD}" in
        offloading)
            CONFIG_FILE="${SCRIPT_DIR}/config/hata_offloading/${MODEL_NAME}-64K.yaml"
            ;;
        flashattn)
            CONFIG_FILE="${SCRIPT_DIR}/config/full_attn/${MODEL_NAME}-64K.yaml"
            ;;
        *)
            CONFIG_FILE="${SCRIPT_DIR}/config/${MODEL_NAME}-64K-top${TOPK}.json"
            ;;
    esac
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/preds}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}")}"
RUN_TAG="${RUN_TAG:-${MODEL_TAG}-${DATASET_NAME}-${RELEASE_VERSION}-top${TOPK}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD}/${RUN_TAG}"

[[ "${SCENARIO}" == "codegeneration" ]] || die "Only SCENARIO=codegeneration is currently supported."
[[ -f "${RUN_PRED_PY}" ]] || die "missing ${RUN_PRED_PY}"
[[ -f "${EVAL_PY}" ]] || die "missing ${EVAL_PY}"
[[ -e "${MODEL_PATH}" ]] || die "missing model: ${MODEL_PATH}"
[[ -f "${CONFIG_FILE}" ]] || die "missing config: ${CONFIG_FILE}"

if [[ "${METHOD}" == offloading* || "${METHOD}" == *-offloading ]]; then
    [[ "${PP_NUM}" == "1" ]] || die "Ditto offloading supports tensor parallelism, but pipeline parallelism is not supported yet."
fi

REQUESTED_GPU_COUNT=$(( MP_NUM * PP_NUM ))
if (( REQUESTED_GPU_COUNT <= 0 )); then
    die "invalid GPU count derived from MP_NUM=${MP_NUM} and PP_NUM=${PP_NUM}"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES}" && "${AUTO_SELECT_GPUS}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="$(pick_cuda_visible_devices "${REQUESTED_GPU_COUNT}" "${MIN_FREE_GPU_MEMORY_MB}")" \
        || die "failed to auto-select ${REQUESTED_GPU_COUNT} GPU(s) with at least ${MIN_FREE_GPU_MEMORY_MB} MiB free; set CUDA_VISIBLE_DEVICES manually"
fi

mkdir -p "${OUTPUT_DIR}"

RUN_CMD=(
    "${PYTHON_BIN}" "${RUN_PRED_PY}"
    --model "${MODEL_PATH}"
    --dataset_name "${DATASET_NAME}"
    --dataset_path "${LIVECODEBENCH_ROOT}"
    --tasks "${TASKS}"
    --output_dir "${OUTPUT_DIR}"
    --method "${METHOD}"
    --config_file "${CONFIG_FILE}"
    --write_in_time
    --mp_num "${MP_NUM}"
    --pp_num "${PP_NUM}"
    --max_seq_len "${MAX_SEQ_LEN}"
    --batch_size "${BATCH_SIZE}"
    --topk "${TOPK}"
    --selective-start-len "${SELECTIVE_START_LEN}"
    --heartbeat-sec "${HEARTBEAT_SEC}"
    --decode-log-interval "${DECODE_LOG_INTERVAL}"
    --sampling-temperature "${SAMPLING_TEMPERATURE}"
    --sampling-top-p "${SAMPLING_TOP_P}"
    --release-version "${RELEASE_VERSION}"
    --lcb-max-new-tokens "${LCB_MAX_NEW_TOKENS}"
)

if [[ "${DATASET_LIMIT}" != "0" ]]; then
    RUN_CMD+=(--dataset_limit "${DATASET_LIMIT}")
fi
if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
    RUN_CMD+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi
if [[ -n "${MEM_FRACTION_STATIC}" ]]; then
    RUN_CMD+=(--mem-fraction-static "${MEM_FRACTION_STATIC}")
fi
if [[ "${NOT_FAST}" == "1" ]]; then
    RUN_CMD+=(--not-fast)
fi
if [[ -n "${START_DATE}" ]]; then
    RUN_CMD+=(--start-date "${START_DATE}")
fi
if [[ -n "${END_DATE}" ]]; then
    RUN_CMD+=(--end-date "${END_DATE}")
fi

EVAL_CMD=(
    "${PYTHON_BIN}" "${EVAL_PY}"
    --pred-file "${OUTPUT_DIR}/${DATASET_NAME}.jsonl"
    --output-dir "${OUTPUT_DIR}"
    --livecodebench-root "${LIVECODEBENCH_ROOT}"
    --release-version "${RELEASE_VERSION}"
    --scenario "${SCENARIO}"
    --model-style "${LCB_MODEL_STYLE}"
    --num-process-evaluate "${NUM_PROCESS_EVALUATE}"
    --timeout "${TIMEOUT}"
)
if [[ "${NOT_FAST}" == "1" ]]; then
    EVAL_CMD+=(--not-fast)
fi
if [[ -n "${START_DATE}" ]]; then
    EVAL_CMD+=(--start-date "${START_DATE}")
fi
if [[ -n "${END_DATE}" ]]; then
    EVAL_CMD+=(--end-date "${END_DATE}")
fi

log_info "dataset=${DATASET_NAME} scenario=${SCENARIO} release_version=${RELEASE_VERSION}"
log_info "output_dir=${OUTPUT_DIR}"
log_info "model=$(basename "${MODEL_PATH}") method=${METHOD} topk=${TOPK} selective_start_len=${SELECTIVE_START_LEN}"
log_info "config_file=${CONFIG_FILE}"
log_info "dataset_path=${LIVECODEBENCH_ROOT} gpus=${CUDA_VISIBLE_DEVICES} mp=${MP_NUM} pp=${PP_NUM}"
log_info "sampling_temperature=${SAMPLING_TEMPERATURE} sampling_top_p=${SAMPLING_TOP_P}"
log_info "heartbeat_sec=${HEARTBEAT_SEC} decode_log_interval=${DECODE_LOG_INTERVAL}"
if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
    log_info "max_total_tokens=${MAX_TOTAL_TOKENS}"
fi
if [[ -n "${START_DATE}" || -n "${END_DATE}" ]]; then
    log_info "date_window=${START_DATE:-unset}..${END_DATE:-unset}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ ! -f "${LIVECODEBENCH_ROOT}/pyproject.toml" ]]; then
        echo "[DRY_RUN] git clone --depth 1 ${LIVECODEBENCH_REPO_URL} ${LIVECODEBENCH_ROOT}"
    fi
    if [[ "${USE_VENV}" == "1" && ! -x "${VENV_DIR}/bin/python" ]]; then
        echo "[DRY_RUN] ${BASE_PYTHON_BIN} -m venv --system-site-packages ${VENV_DIR}"
    fi
    if ! have_livecodebench_install; then
        echo "[DRY_RUN] ${PYTHON_BIN} -m pip install datasets==${DATASETS_VERSION}"
        echo "[DRY_RUN] ${PYTHON_BIN} -m pip install pebble"
        echo "[DRY_RUN] ${PYTHON_BIN} -m pip install -e ${LIVECODEBENCH_ROOT} --no-deps"
    fi
    echo "[DRY_RUN] ${RUN_CMD[*]}"
    echo "[DRY_RUN] ${EVAL_CMD[*]}"
    exit 0
fi

ensure_livecodebench_repo
ensure_livecodebench_install

DITTO_ROOT="${DITTO_ROOT}" \
PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
DITTO_DEBUG_STALL="${DITTO_DEBUG_STALL}" \
DITTO_DEBUG_STALL_MIN_MS="${DITTO_DEBUG_STALL_MIN_MS}" \
DITTO_DEBUG_CPUGATHER="${DITTO_DEBUG_CPUGATHER}" \
    "${RUN_CMD[@]}" | tee -a "${OUTPUT_DIR}/reuse_ratio.log"

"${EVAL_CMD[@]}" | tee -a "${OUTPUT_DIR}/eval.log"

RESULT_JSON="${OUTPUT_DIR}/result.json"
[[ -s "${RESULT_JSON}" ]] || die "missing result json: ${RESULT_JSON}"
