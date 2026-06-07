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

offloading_effectively_disabled_from_cfg() {
    python3 - "${MODEL_PATH}" "${CONFIG_FILE}" <<'PY'
import json
import os
import sys

import yaml

model_path, cfg_path = sys.argv[1:3]

num_layers = 0
cfg_json = os.path.join(model_path, "config.json")
if os.path.isfile(cfg_json):
    try:
        with open(cfg_json, "r", encoding="utf-8") as f:
            mc = json.load(f)
        for k in ("num_hidden_layers", "n_layer", "num_layers"):
            if k in mc:
                num_layers = int(mc[k])
                break
    except Exception:
        pass

if not os.path.isfile(cfg_path):
    print(0)
    sys.exit(0)

with open(cfg_path, "r", encoding="utf-8") as f:
    text = f.read()

ext = os.path.splitext(cfg_path)[1].lower()
if ext in {".yaml", ".yml"}:
    cfg = yaml.safe_load(text)
else:
    cfg = json.loads(text)

max_skip = -1

def walk(node):
    global max_skip
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "num_skip_layers":
                try:
                    max_skip = max(max_skip, int(v))
                except Exception:
                    pass
            walk(v)
    elif isinstance(node, list):
        for x in node:
            walk(x)

walk(cfg)
print(1 if (num_layers > 0 and max_skip >= num_layers) else 0)
PY
}

RUN_PRED_PY="${RUN_PRED_PY:-${SCRIPT_DIR}/run_pred.py}"
EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/eval_longbench_infinitebench.py}"
SUMMARIZE_PY="${SUMMARIZE_PY:-${SCRIPT_DIR}/summarize_accuracy.py}"

DITTO_ROOT="${DITTO_ROOT:-${SCRIPT_DIR}}"
MODEL_PATH="${MODEL_PATH:-/models/Llama-3-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"

DATASET_NAME="${DATASET_NAME:-}"
DATASET_PATH="${DATASET_PATH:-}"
TASKS="${TASKS:-}"
DATASET_E_MODE="${DATASET_E_MODE:-0}"

[[ -n "${DATASET_NAME}" ]] || die "DATASET_NAME is required"
[[ -n "${DATASET_PATH}" ]] || die "DATASET_PATH is required"
if [[ -z "${TASKS}" ]]; then
    TASKS="${DATASET_NAME}"
fi

METHOD="${METHOD:-offloading-hash}"
TOPK="${TOPK:-0.10}"
SELECTIVE_START_LEN="${SELECTIVE_START_LEN:-512}"

case "${METHOD}" in
    flash-attn)
        METHOD="flashattn"
        ;;
esac

MAX_SEQ_LEN="${MAX_SEQ_LEN:-131072}"
ENGINE_CONTEXT_LENGTH="${ENGINE_CONTEXT_LENGTH:-65536}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MP_NUM="${MP_NUM:-1}"
PP_NUM="${PP_NUM:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
AUTO_SELECT_GPUS="${AUTO_SELECT_GPUS:-1}"
MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-20000}"
DATASET_LIMIT="${DATASET_LIMIT:-0}"
RESUME="${RESUME:-0}"

HEARTBEAT_SEC="${HEARTBEAT_SEC:-10}"
DECODE_LOG_INTERVAL="${DECODE_LOG_INTERVAL:-20}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
DITTO_DEBUG_STALL="${DITTO_DEBUG_STALL:-0}"
DITTO_DEBUG_STALL_MIN_MS="${DITTO_DEBUG_STALL_MIN_MS:-0}"
DITTO_DEBUG_CPUGATHER="${DITTO_DEBUG_CPUGATHER:-0}"
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE="${SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE:-0}"

MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-65536}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"

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
RUN_TAG="${RUN_TAG:-${MODEL_TAG}-${DATASET_NAME}-top${TOPK}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD}/${RUN_TAG}"

DRY_RUN="${DRY_RUN:-0}"
BASELINE_RESULT="${BASELINE_RESULT:-}"
MAX_AVG_DROP="${MAX_AVG_DROP:-3.0}"
MAX_SINGLE_DROP="${MAX_SINGLE_DROP:-8.0}"

[[ -f "${RUN_PRED_PY}" ]] || die "missing ${RUN_PRED_PY}"
[[ -f "${EVAL_PY}" ]] || die "missing ${EVAL_PY}"
[[ -f "${SUMMARIZE_PY}" ]] || die "missing ${SUMMARIZE_PY}"
[[ -e "${MODEL_PATH}" ]] || die "missing model: ${MODEL_PATH}"
[[ -d "${DATASET_PATH}" || -f "${DATASET_PATH}" ]] || die "missing dataset path: ${DATASET_PATH}"
[[ -f "${CONFIG_FILE}" ]] || die "missing config: ${CONFIG_FILE}"

if [[ "${METHOD}" == offloading* || "${METHOD}" == *-offloading ]]; then
    OFFLOADING_DISABLED="${OFFLOADING_DISABLED:-auto}"
    OFFLOADING_TP_FALLBACK="${OFFLOADING_TP_FALLBACK:-1}"
    offloading_disabled_resolved=0
    case "${OFFLOADING_DISABLED}" in
        1|true|TRUE|yes|YES)
            offloading_disabled_resolved=1
            ;;
        0|false|FALSE|no|NO)
            offloading_disabled_resolved=0
            ;;
        auto|AUTO)
            offloading_disabled_resolved="$(offloading_effectively_disabled_from_cfg)"
            ;;
        *)
            die "OFFLOADING_DISABLED must be one of: auto, 0/1, true/false, yes/no. Got: ${OFFLOADING_DISABLED}"
            ;;
    esac

    if [[ "${offloading_disabled_resolved}" == "1" ]]; then
        log_info "offloading is effectively disabled; allow tensor parallelism (mp=${MP_NUM}, pp=${PP_NUM})"
        if (( PP_NUM > 1 )) && [[ "${OFFLOADING_TP_FALLBACK}" == "1" ]]; then
            new_mp="$(( MP_NUM * PP_NUM ))"
            log_info "remap PP to TP for compatibility: (mp=${MP_NUM}, pp=${PP_NUM}) -> (mp=${new_mp}, pp=1)"
            MP_NUM="${new_mp}"
            PP_NUM=1
        fi
    else
        [[ "${MP_NUM}" == "1" ]] || die "Ditto offloading currently requires MP_NUM=1 (unless offloading is fully disabled)."
    fi
fi

REQUESTED_GPU_COUNT=$(( MP_NUM * PP_NUM ))
if (( REQUESTED_GPU_COUNT <= 0 )); then
    die "invalid GPU count derived from MP_NUM=${MP_NUM} and PP_NUM=${PP_NUM}"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES}" && "${AUTO_SELECT_GPUS}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="$(pick_cuda_visible_devices "${REQUESTED_GPU_COUNT}" "${MIN_FREE_GPU_MEMORY_MB}")" \
        || die "failed to auto-select ${REQUESTED_GPU_COUNT} GPU(s) with at least ${MIN_FREE_GPU_MEMORY_MB} MiB free; set CUDA_VISIBLE_DEVICES manually"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    CUDA_VISIBLE_DEVICES="0"
fi

mkdir -p "${OUTPUT_DIR}"

RUN_CMD=(
    python3 "${RUN_PRED_PY}"
    --model "${MODEL_PATH}"
    --dataset_name "${DATASET_NAME}"
    --dataset_path "${DATASET_PATH}"
    --tasks "${TASKS}"
    --output_dir "${OUTPUT_DIR}"
    --method "${METHOD}"
    --config_file "${CONFIG_FILE}"
    --write_in_time
    --mp_num "${MP_NUM}"
    --pp_num "${PP_NUM}"
    --max_seq_len "${MAX_SEQ_LEN}"
    --context-length "${ENGINE_CONTEXT_LENGTH}"
    --batch_size "${BATCH_SIZE}"
    --topk "${TOPK}"
    --selective-start-len "${SELECTIVE_START_LEN}"
    --heartbeat-sec "${HEARTBEAT_SEC}"
    --decode-log-interval "${DECODE_LOG_INTERVAL}"
)

if [[ "${DATASET_E_MODE}" == "1" ]]; then
    RUN_CMD+=(--e)
fi
if [[ "${DATASET_LIMIT}" != "0" ]]; then
    RUN_CMD+=(--dataset_limit "${DATASET_LIMIT}")
fi
if [[ "${RESUME}" == "1" ]]; then
    RUN_CMD+=(--resume)
fi
if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
    RUN_CMD+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi
if [[ -n "${MEM_FRACTION_STATIC}" ]]; then
    RUN_CMD+=(--mem-fraction-static "${MEM_FRACTION_STATIC}")
fi
if [[ -n "${ATTENTION_BACKEND}" ]]; then
    RUN_CMD+=(--attention-backend "${ATTENTION_BACKEND}")
fi

EVAL_CMD=(python3 "${EVAL_PY}" --model "${OUTPUT_DIR}")
if [[ "${DATASET_E_MODE}" == "1" ]]; then
    EVAL_CMD+=(--e)
fi

log_info "dataset=${DATASET_NAME} tasks=${TASKS}"
log_info "output_dir=${OUTPUT_DIR}"
log_info "model=$(basename "${MODEL_PATH}") method=${METHOD} topk=${TOPK} selective_start_len=${SELECTIVE_START_LEN}"
log_info "config_file=${CONFIG_FILE}"
log_info "dataset_path=${DATASET_PATH} gpus=${CUDA_VISIBLE_DEVICES} mp=${MP_NUM} pp=${PP_NUM}"
log_info "max_seq_len=${MAX_SEQ_LEN} engine_context_length=${ENGINE_CONTEXT_LENGTH}"
if [[ "${MP_NUM}" == "1" ]]; then
    log_info "single-rank run will use first visible GPU=${CUDA_VISIBLE_DEVICES%%,*}"
fi
log_info "heartbeat_sec=${HEARTBEAT_SEC} decode_log_interval=${DECODE_LOG_INTERVAL}"
if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
    log_info "max_total_tokens=${MAX_TOTAL_TOKENS}"
fi
log_info "strict_mem_check_during_idle=${SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE}"
if [[ -n "${ATTENTION_BACKEND}" ]]; then
    log_info "attention_backend=${ATTENTION_BACKEND}"
fi
if [[ "${RESUME}" == "1" ]]; then
    log_info "resume=1"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] ${RUN_CMD[*]}"
    echo "[DRY_RUN] ${EVAL_CMD[*]}"
    exit 0
fi

DITTO_ROOT="${DITTO_ROOT}" \
PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
DITTO_DEBUG_STALL="${DITTO_DEBUG_STALL}" \
DITTO_DEBUG_STALL_MIN_MS="${DITTO_DEBUG_STALL_MIN_MS}" \
DITTO_DEBUG_CPUGATHER="${DITTO_DEBUG_CPUGATHER}" \
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE="${SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE}" \
    "${RUN_CMD[@]}" | tee -a "${OUTPUT_DIR}/reuse_ratio.log"

"${EVAL_CMD[@]}" | tee -a "${OUTPUT_DIR}/eval.log"

RESULT_JSON="${OUTPUT_DIR}/result.json"
[[ -f "${RESULT_JSON}" ]] || die "missing result json: ${RESULT_JSON}"

python3 "${SUMMARIZE_PY}" \
    "${RESULT_JSON}" \
    "${BASELINE_RESULT}" \
    "${MAX_AVG_DROP}" \
    "${MAX_SINGLE_DROP}"
