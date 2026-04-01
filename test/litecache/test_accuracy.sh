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

# =========================
# Paths
# =========================
RUN_PRED_PY="${RUN_PRED_PY:-${SCRIPT_DIR}/run_pred.py}"
EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/eval_longbench_infinitebench.py}"

LITECACHE_ROOT="${LITECACHE_ROOT:-${SCRIPT_DIR}}"
MYTRANSFORMER_ROOT="${MYTRANSFORMER_ROOT:-${LITECACHE_ROOT}}"
MODEL_PATH="${MODEL_PATH:-/jhe/Llama-3-8B-Instruct-Gradient-1048k}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
DATASET_PATH="${DATASET_PATH:-/jhe/LongBench}"
if [[ ! -d "${DATASET_PATH}" && -d "/jhe/dataset/LongBench" ]]; then
    DATASET_PATH="/jhe/dataset/LongBench"
fi

# =========================
# Runtime Settings
# =========================
# offloading branch methods: offloading-hash / offloading-loki / offloading-quest / offloading-infinigen
METHOD="${METHOD:-offloading-hash}"
TOPK="${TOPK:-0.10}"
TASKS="${TASKS:-lcc_e,repobench-p_e,qasper_e,multifieldqa_en_e,hotpotqa_e,2wikimqa_e,trec_e,triviaqa_e,samsum_e,passage_count_e,passage_retrieval_en_e,gov_report_e,multi_news_e}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-131072}"
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
LITECACHE_DEBUG_STALL="${LITECACHE_DEBUG_STALL:-0}"
LITECACHE_DEBUG_STALL_MIN_MS="${LITECACHE_DEBUG_STALL_MIN_MS:-0}"
LITECACHE_DEBUG_CPUGATHER="${LITECACHE_DEBUG_CPUGATHER:-0}"

# Engine memory knobs
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-65536}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-}"

# Leave MAX_TOTAL_TOKENS empty by default so engine can profile token capacity from GPU memory.

CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/config/${MODEL_NAME}-64K-top${TOPK}.json}"

# Output + compare
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/preds}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}")}"
RUN_TAG="${RUN_TAG:-${MODEL_TAG}-longbench-top${TOPK}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD}/${RUN_TAG}"

DRY_RUN="${DRY_RUN:-0}"
BASELINE_RESULT="${BASELINE_RESULT:-}"
MAX_AVG_DROP="${MAX_AVG_DROP:-3.0}"
MAX_SINGLE_DROP="${MAX_SINGLE_DROP:-8.0}"

# =========================
# Validate Inputs
# =========================
[[ -f "${RUN_PRED_PY}" ]] || die "missing ${RUN_PRED_PY}"
[[ -f "${EVAL_PY}" ]] || die "missing ${EVAL_PY}"
[[ -e "${MODEL_PATH}" ]] || die "missing model: ${MODEL_PATH}"
[[ -d "${DATASET_PATH}" ]] || die "missing dataset dir: ${DATASET_PATH}"
[[ -f "${CONFIG_FILE}" ]] || die "missing config: ${CONFIG_FILE}"

if [[ "${METHOD}" == offloading* || "${METHOD}" == *-offloading ]]; then
    [[ "${MP_NUM}" == "1" ]] || die "LiteCache offloading currently requires MP_NUM=1."
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

# =========================
# Build Commands
# =========================
RUN_CMD=(
    python3 "${RUN_PRED_PY}"
    --model "${MODEL_PATH}"
    --dataset_name longbench --e
    --dataset_path "${DATASET_PATH}"
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
    --heartbeat-sec "${HEARTBEAT_SEC}"
    --decode-log-interval "${DECODE_LOG_INTERVAL}"
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

EVAL_CMD=(python3 "${EVAL_PY}" --model "${OUTPUT_DIR}")

log_info "output_dir=${OUTPUT_DIR}"
log_info "model=$(basename "${MODEL_PATH}") method=${METHOD} topk=${TOPK}"
log_info "config_file=${CONFIG_FILE}"
log_info "dataset_path=${DATASET_PATH} gpus=${CUDA_VISIBLE_DEVICES} mp=${MP_NUM} pp=${PP_NUM}"
if [[ "${MP_NUM}" == "1" ]]; then
    log_info "single-rank run will use first visible GPU=${CUDA_VISIBLE_DEVICES%%,*}"
fi
log_info "heartbeat_sec=${HEARTBEAT_SEC} decode_log_interval=${DECODE_LOG_INTERVAL}"
if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
    log_info "max_total_tokens=${MAX_TOTAL_TOKENS}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] ${RUN_CMD[*]}"
    echo "[DRY_RUN] ${EVAL_CMD[*]}"
    exit 0
fi

# =========================
# Run Prediction + Eval
# =========================
LITECACHE_ROOT="${LITECACHE_ROOT}" \
MYTRANSFORMER_ROOT="${MYTRANSFORMER_ROOT}" \
PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
LITECACHE_DEBUG_STALL="${LITECACHE_DEBUG_STALL}" \
LITECACHE_DEBUG_STALL_MIN_MS="${LITECACHE_DEBUG_STALL_MIN_MS}" \
LITECACHE_DEBUG_CPUGATHER="${LITECACHE_DEBUG_CPUGATHER}" \
    "${RUN_CMD[@]}" | tee -a "${OUTPUT_DIR}/reuse_ratio.log"

"${EVAL_CMD[@]}"

RESULT_JSON="${OUTPUT_DIR}/result.json"
[[ -f "${RESULT_JSON}" ]] || die "missing result json: ${RESULT_JSON}"

# =========================
# Optional Baseline Compare
# =========================
python3 - "${RESULT_JSON}" "${BASELINE_RESULT}" "${MAX_AVG_DROP}" "${MAX_SINGLE_DROP}" <<'PY'
import json
import os
import statistics
import sys

result_path, baseline_path, max_avg_drop, max_single_drop = sys.argv[1:]
max_avg_drop = float(max_avg_drop)
max_single_drop = float(max_single_drop)


def flatten_scores(raw):
    flat = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            vals = [float(x) for x in v.values()]
            flat[k] = sum(vals) / len(vals) if vals else 0.0
        else:
            flat[k] = float(v)
    return flat

with open(result_path, "r", encoding="utf-8") as f:
    result = flatten_scores(json.load(f))

vals = list(result.values())
avg = sum(vals) / len(vals) if vals else 0.0
med = statistics.median(vals) if vals else 0.0
print(f"[SUMMARY] task_count={len(vals)} avg_score={avg:.2f} median_score={med:.2f}")

if not baseline_path:
    print("[SUMMARY] no baseline comparison. set BASELINE_RESULT=/path/to/result.json to compare.")
    sys.exit(0)

if not os.path.exists(baseline_path):
    print(f"[ERROR] baseline not found: {baseline_path}")
    sys.exit(2)

with open(baseline_path, "r", encoding="utf-8") as f:
    baseline = flatten_scores(json.load(f))

common = sorted(set(result).intersection(baseline))
if not common:
    print("[ERROR] no overlapping tasks with baseline")
    sys.exit(2)

drops = {t: baseline[t] - result[t] for t in common}
avg_drop = sum(drops.values()) / len(drops)
worst_task = max(drops, key=drops.get)
worst_drop = drops[worst_task]

print(
    "[COMPARE] "
    f"tasks={len(common)} avg_drop={avg_drop:.2f} "
    f"worst_drop={worst_drop:.2f}({worst_task}) "
    f"thresholds(avg<={max_avg_drop}, worst<={max_single_drop})"
)

if avg_drop <= max_avg_drop and worst_drop <= max_single_drop:
    print("[PASS] accuracy drop is within tolerance.")
    sys.exit(0)

print("[FAIL] accuracy drop is larger than tolerance.")
sys.exit(3)
PY
