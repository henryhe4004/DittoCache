#!/usr/bin/env bash

set -euo pipefail

# When run_pred sets Engine context_length > model max_position_embeddings (e.g. LongBench >32K on Qwen2.5).
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_info() {
    echo "[INFO] $*"
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

is_code_completion_task_list() {
    local tasks_csv="$1"
    local task
    IFS=',' read -r -a _tasks <<< "${tasks_csv}"
    for task in "${_tasks[@]}"; do
        case "${task}" in
            lcc|lcc_e|repobench-p|repobench-p_e)
                ;;
            *)
                return 1
                ;;
        esac
    done
    return 0
}

is_longbench_e_task_list() {
    local tasks_csv="$1"
    local task
    IFS=',' read -r -a _tasks <<< "${tasks_csv}"
    for task in "${_tasks[@]}"; do
        [[ "${task}" == *_e ]] || return 1
    done
    return 0
}

normalize_method_name() {
    local method="${1:-}"
    case "${method}" in
        flash_attn|flash-attn)
            echo "flashattn"
            ;;
        *)
            echo "${method}"
            ;;
    esac
}

tag_value() {
    local value="${1:-}"
    if [[ -z "${value}" ]]; then
        echo "keep"
    else
        echo "${value//./p}"
    fi
}

normalize_longbench_task_csv() {
    local tasks_csv="${1:-}"
    local out=()
    local task
    IFS=',' read -r -a _tasks <<< "${tasks_csv}"
    for task in "${_tasks[@]}"; do
        case "${task}" in
            multinews) out+=("multi_news") ;;
            mulitinews) out+=("multi_news") ;;
            multinews_e) out+=("multi_news_e") ;;
            mulitinews_e) out+=("multi_news_e") ;;
            *) out+=("${task}") ;;
        esac
    done
    (IFS=','; echo "${out[*]}")
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

# =========================
# Paths
# =========================
RUN_PRED_PY="${RUN_PRED_PY:-${SCRIPT_DIR}/run_pred.py}"
EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/eval_longbench_infinitebench.py}"
SUMMARIZE_PY="${SUMMARIZE_PY:-${SCRIPT_DIR}/summarize_accuracy.py}"

DITTO_ROOT="${DITTO_ROOT:-${SCRIPT_DIR}}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-14B-Instruct-1M}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
DATASET_PATH="${DATASET_PATH:-/datasets/LongBench}"
if [[ ! -d "${DATASET_PATH}" && -d "/datasets/LongBench" ]]; then
    DATASET_PATH="/datasets/LongBench"
fi

# =========================
# Runtime Settings
# =========================
# offloading branch methods: offloading-hash / offloading-loki / offloading-quest / offloading-infinigen
METHOD="$(normalize_method_name "${METHOD:-offloading}")"
TOPK="${TOPK:-0.10}"
SELECTIVE_START_LEN="${SELECTIVE_START_LEN:-2048}"
DATASET_NAME="${DATASET_NAME:-longbench}"
FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER:-}"
FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_LOWER:-}"
FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS:-${FIXED_NUM_SKIP_LAYER:-}}"
FIXED_DECAY_P="${FIXED_DECAY_P:-${DECAY_P:-}}"
ENABLE_MAX_REUSE_COUNT="${ENABLE_MAX_REUSE_COUNT:-}"
MAX_REUSE_COUNT_INPUT="${FIXED_MAX_REUSE_COUNT:-${MAX_REUSE_COUNT:-}}"
if [[ "${ENABLE_MAX_REUSE_COUNT}" == "0" ]]; then
    FIXED_MAX_REUSE_COUNT=""
elif [[ "${ENABLE_MAX_REUSE_COUNT}" == "1" || -n "${MAX_REUSE_COUNT_INPUT}" ]]; then
    FIXED_MAX_REUSE_COUNT="${MAX_REUSE_COUNT_INPUT}"
else
    FIXED_MAX_REUSE_COUNT=""
fi
if [[ -n "${FIXED_REUSE_THRESHOLD_UPPER}" && -z "${FIXED_REUSE_THRESHOLD_LOWER}" ]]; then
    FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_UPPER}"
fi
# -lcc_e,repobench-p_e,qasper_e,multifieldqa_en_e,hotpotqa_e,2wikimqa_e,trec_e,triviaqa_e,samsum_e,passage_count_e,passage_retrieval_en_e,gov_report_e,
TASKS="${TASKS:-multi_news_e}"
if [[ "${DATASET_NAME}" == "longbench" ]]; then
    TASKS="$(normalize_longbench_task_csv "${TASKS}")"
fi

# YAML/json config suffix (must match files under config/). Set this alone to align budgets.
CONFIG_SUFFIX="${CONFIG_SUFFIX:-64K}"

# When MAX_SEQ_LEN / ENGINE_CONTEXT_LENGTH / MAX_TOTAL_TOKENS are unset, derive from CONFIG_SUFFIX.
ditto_budget_from_suffix() {
    case "${CONFIG_SUFFIX}" in
        8[Kk]) echo 8192 ;;
        16[Kk]) echo 16384 ;;
        32[Kk]) echo 32768 ;;
        64[Kk]) echo 65536 ;;
        128[Kk]) echo 131072 ;;
        256[Kk]) echo 262144 ;;
        *) echo "" ;;
    esac
}
_LITE_BUDGET="$(ditto_budget_from_suffix)"
if [[ -n "${_LITE_BUDGET}" ]]; then
    : "${MAX_SEQ_LEN:=${_LITE_BUDGET}}"
    : "${ENGINE_CONTEXT_LENGTH:=${_LITE_BUDGET}}"
    : "${MAX_TOTAL_TOKENS:=${_LITE_BUDGET}}"
else
    : "${MAX_SEQ_LEN:=131072}"
    : "${ENGINE_CONTEXT_LENGTH:=65536}"
    : "${MAX_TOTAL_TOKENS:=65536}"
fi

BATCH_SIZE="${BATCH_SIZE:-1}"
MP_NUM="${MP_NUM:-1}"
PP_NUM="${PP_NUM:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
AUTO_SELECT_GPUS="${AUTO_SELECT_GPUS:-1}"
MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-20000}"
DATASET_LIMIT="${DATASET_LIMIT:-0}"

# Debug / observability
HEARTBEAT_SEC="${HEARTBEAT_SEC:-10}"
DECODE_LOG_INTERVAL="${DECODE_LOG_INTERVAL:-20}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
DITTO_DEBUG_STALL="${DITTO_DEBUG_STALL:-0}"
DITTO_DEBUG_STALL_MIN_MS="${DITTO_DEBUG_STALL_MIN_MS:-0}"
DITTO_DEBUG_CPUGATHER="${DITTO_DEBUG_CPUGATHER:-0}"
DITTO_RECORD_TRANSFER_STATS="${DITTO_RECORD_TRANSFER_STATS:-}"
DITTO_RECORD_OVERLAP_STATS="${DITTO_RECORD_OVERLAP_STATS:-}"
DITTO_TRANSFER_STATS_FILE="${DITTO_TRANSFER_STATS_FILE:-}"

# SGLang quantization (e.g. awq = plain AWQ, skip awq_marlin). Empty or "auto" = let SGLang decide.
ENGINE_QUANTIZATION="${ENGINE_QUANTIZATION:-}"

# Engine memory knobs (MAX_TOTAL_TOKENS may already be set from CONFIG_SUFFIX above)
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-}"

CONFIG_FILE="${CONFIG_FILE:-}"
if [[ -z "${CONFIG_FILE}" ]]; then
    case "${METHOD}" in
        offloading)
            CONFIG_FILE="${SCRIPT_DIR}/config/hata_offloading/${MODEL_NAME}-${CONFIG_SUFFIX}.yaml"
            ;;
        flashattn)
            CONFIG_FILE="${SCRIPT_DIR}/config/full_attn/${MODEL_NAME}-${CONFIG_SUFFIX}.yaml"
            ;;
        *)
            CONFIG_FILE="${SCRIPT_DIR}/config/${MODEL_NAME}-${CONFIG_SUFFIX}-top${TOPK}.json"
            ;;
    esac
fi

if [[ -n "${FIXED_REUSE_THRESHOLD_UPPER}" || -n "${FIXED_REUSE_THRESHOLD_LOWER}" || -n "${FIXED_NUM_SKIP_LAYERS}" || -n "${FIXED_DECAY_P}" || -n "${FIXED_MAX_REUSE_COUNT}" ]]; then
    GENERATED_CONFIG_DIR="${SCRIPT_DIR}/config/.generated"
    mkdir -p "${GENERATED_CONFIG_DIR}"
    CONFIG_EXT="${CONFIG_FILE##*.}"
    UPPER_TAG="$(tag_value "${FIXED_REUSE_THRESHOLD_UPPER}")"
    LOWER_TAG="$(tag_value "${FIXED_REUSE_THRESHOLD_LOWER}")"
    SKIP_TAG="$(tag_value "${FIXED_NUM_SKIP_LAYERS}")"
    DECAY_TAG="$(tag_value "${FIXED_DECAY_P}")"
    MAX_REUSE_TAG="$(tag_value "${FIXED_MAX_REUSE_COUNT}")"
    GENERATED_CONFIG_FILE="${GENERATED_CONFIG_DIR}/$(basename "${CONFIG_FILE%.*}")-${DATASET_NAME}-thup${UPPER_TAG}-thlo${LOWER_TAG}-skip${SKIP_TAG}-decay${DECAY_TAG}-reuse${MAX_REUSE_TAG}.${CONFIG_EXT}"

    python3 - "${CONFIG_FILE}" "${GENERATED_CONFIG_FILE}" "${FIXED_REUSE_THRESHOLD_UPPER}" "${FIXED_REUSE_THRESHOLD_LOWER}" "${FIXED_NUM_SKIP_LAYERS}" "${FIXED_DECAY_P}" "${FIXED_MAX_REUSE_COUNT}" <<'PY'
import json
import os
import sys

import yaml

src, dst, upper_raw, lower_raw, skip_raw, decay_raw, max_reuse_raw = sys.argv[1:8]
upper = None if upper_raw == "" else float(upper_raw)
lower = None if lower_raw == "" else float(lower_raw)
skip_layers = None if skip_raw == "" else int(skip_raw)
decay_p = None if decay_raw == "" else float(decay_raw)
max_reuse_count = None if max_reuse_raw == "" else int(max_reuse_raw)

with open(src, "r", encoding="utf-8") as f:
    text = f.read()

ext = os.path.splitext(src)[1].lower()
if ext in {".yaml", ".yml"}:
    cfg = yaml.safe_load(text)
else:
    cfg = json.loads(text)

def patch(node):
    if isinstance(node, dict):
        if upper is not None and "reuse_threshold_upper" in node:
            node["reuse_threshold_upper"] = upper
        if lower is not None and "reuse_threshold_lower" in node:
            node["reuse_threshold_lower"] = lower
        if skip_layers is not None and "num_skip_layers" in node:
            node["num_skip_layers"] = skip_layers
        if max_reuse_count is not None and "max_reuse_count" in node:
            node["max_reuse_count"] = max_reuse_count
        if decay_p is not None:
            if "decay_p" in node:
                node["decay_p"] = decay_p
            if "deacy_p" in node:
                node["deacy_p"] = decay_p
        for value in node.values():
            patch(value)
    elif isinstance(node, list):
        for item in node:
            patch(item)

patch(cfg)

with open(dst, "w", encoding="utf-8") as f:
    if ext in {".yaml", ".yml"}:
        yaml.safe_dump(cfg, f, sort_keys=False)
    else:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
PY
    CONFIG_FILE="${GENERATED_CONFIG_FILE}"
fi

# Output + compare
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/preds}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}")}"
RUN_TAG="${RUN_TAG:-${MODEL_TAG}-longbench-top${TOPK}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD}/${RUN_TAG}"

DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
BASELINE_RESULT="${BASELINE_RESULT:-}"
MAX_AVG_DROP="${MAX_AVG_DROP:-3.0}"
MAX_SINGLE_DROP="${MAX_SINGLE_DROP:-8.0}"

# =========================
# Validate Inputs
# =========================
[[ -f "${RUN_PRED_PY}" ]] || die "missing ${RUN_PRED_PY}"
[[ -f "${EVAL_PY}" ]] || die "missing ${EVAL_PY}"
[[ -f "${SUMMARIZE_PY}" ]] || die "missing ${SUMMARIZE_PY}"
[[ -e "${MODEL_PATH}" ]] || die "missing model: ${MODEL_PATH}"
[[ -d "${DATASET_PATH}" ]] || die "missing dataset dir: ${DATASET_PATH}"
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
        [[ "${PP_NUM}" == "1" ]] || die "Ditto offloading supports tensor parallelism, but pipeline parallelism is not supported yet."
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

if [[ -z "${DITTO_RECORD_TRANSFER_STATS}" ]]; then
    if [[ "${METHOD}" == offloading* || "${METHOD}" == *-offloading ]]; then
        DITTO_RECORD_TRANSFER_STATS="1"
    else
        DITTO_RECORD_TRANSFER_STATS="0"
    fi
fi
if [[ -z "${DITTO_RECORD_OVERLAP_STATS}" ]]; then
    if [[ "${DITTO_RECORD_TRANSFER_STATS}" == "1" ]]; then
        # Hit-rate stats are derived from overlap metrics.
        DITTO_RECORD_OVERLAP_STATS="1"
    else
        DITTO_RECORD_OVERLAP_STATS="0"
    fi
fi
if [[ -z "${DITTO_TRANSFER_STATS_FILE}" ]]; then
    DITTO_TRANSFER_STATS_FILE="${OUTPUT_DIR}/ditto_decode_transfer_stats.json"
fi

# =========================
# Build Commands
# =========================
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

if [[ "${DATASET_NAME}" == "longbench" ]] && is_longbench_e_task_list "${TASKS}"; then
    RUN_CMD+=(--e)
fi

if [[ "${DATASET_LIMIT}" != "0" ]]; then
    RUN_CMD+=(--dataset_limit "${DATASET_LIMIT}")
fi
if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
    RUN_CMD+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi
if [[ -n "${MEM_FRACTION_STATIC}" ]]; then
    RUN_CMD+=(--mem-fraction-static "${MEM_FRACTION_STATIC}")
fi
if [[ -n "${ENGINE_QUANTIZATION}" && "${ENGINE_QUANTIZATION}" != "auto" ]]; then
    RUN_CMD+=(--quantization "${ENGINE_QUANTIZATION}")
fi
if [[ "${RESUME}" == "1" ]]; then
    RUN_CMD+=(--resume)
fi

EVAL_CMD=(python3 "${EVAL_PY}" --model "${OUTPUT_DIR}")
if [[ "${DATASET_NAME}" == "longbench" ]] && is_longbench_e_task_list "${TASKS}"; then
    EVAL_CMD+=(--e)
fi

log_info "output_dir=${OUTPUT_DIR}"
log_info "model=$(basename "${MODEL_PATH}") method=${METHOD} topk=${TOPK} selective_start_len=${SELECTIVE_START_LEN}"
log_info "config_suffix=${CONFIG_SUFFIX} config_file=${CONFIG_FILE}"
log_info "max_seq_len=${MAX_SEQ_LEN} engine_context_length=${ENGINE_CONTEXT_LENGTH} max_total_tokens=${MAX_TOTAL_TOKENS}"
log_info "dataset=${DATASET_NAME} dataset_path=${DATASET_PATH} gpus=${CUDA_VISIBLE_DEVICES} mp=${MP_NUM} pp=${PP_NUM}"
if [[ "${MP_NUM}" == "1" ]]; then
    log_info "single-rank run will use first visible GPU=${CUDA_VISIBLE_DEVICES%%,*}"
fi
log_info "heartbeat_sec=${HEARTBEAT_SEC} decode_log_interval=${DECODE_LOG_INTERVAL}"
log_info "record_transfer_stats=${DITTO_RECORD_TRANSFER_STATS} record_overlap_stats=${DITTO_RECORD_OVERLAP_STATS}"
log_info "transfer_stats_file=${DITTO_TRANSFER_STATS_FILE}"
if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
    log_info "max_total_tokens=${MAX_TOTAL_TOKENS}"
fi
if [[ -n "${ENGINE_QUANTIZATION}" && "${ENGINE_QUANTIZATION}" != "auto" ]]; then
    log_info "engine_quantization=${ENGINE_QUANTIZATION}"
fi
log_info "resume=${RESUME}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] ${RUN_CMD[*]}"
    echo "[DRY_RUN] ${EVAL_CMD[*]}"
    exit 0
fi

# =========================
# Run Prediction + Eval
# =========================
DITTO_ROOT="${DITTO_ROOT}" \
PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
DITTO_DEBUG_STALL="${DITTO_DEBUG_STALL}" \
DITTO_DEBUG_STALL_MIN_MS="${DITTO_DEBUG_STALL_MIN_MS}" \
DITTO_DEBUG_CPUGATHER="${DITTO_DEBUG_CPUGATHER}" \
DITTO_RECORD_TRANSFER_STATS="${DITTO_RECORD_TRANSFER_STATS}" \
DITTO_RECORD_OVERLAP_STATS="${DITTO_RECORD_OVERLAP_STATS}" \
DITTO_TRANSFER_STATS_FILE="${DITTO_TRANSFER_STATS_FILE}" \
    "${RUN_CMD[@]}" | tee -a "${OUTPUT_DIR}/reuse_ratio.log"

"${EVAL_CMD[@]}" | tee -a "${OUTPUT_DIR}/eval.log"

RESULT_JSON="${OUTPUT_DIR}/result.json"
[[ -f "${RESULT_JSON}" ]] || die "missing result json: ${RESULT_JSON}"

# =========================
# Optional Baseline Compare
# =========================
python3 "${SUMMARIZE_PY}" \
    "${RESULT_JSON}" \
    "${BASELINE_RESULT}" \
    "${MAX_AVG_DROP}" \
    "${MAX_SINGLE_DROP}"
