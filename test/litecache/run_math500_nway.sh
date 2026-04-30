#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BASE_RUNNER="${BASE_RUNNER:-${SCRIPT_DIR}/test_accuracy_math500.sh}"
EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/eval_longbench_infinitebench.py}"
SUMMARIZE_PY="${SUMMARIZE_PY:-${SCRIPT_DIR}/summarize_accuracy.py}"

[[ -x "${BASE_RUNNER}" ]] || { echo "[ERROR] missing executable runner: ${BASE_RUNNER}" >&2; exit 1; }
[[ -f "${EVAL_PY}" ]] || { echo "[ERROR] missing eval script: ${EVAL_PY}" >&2; exit 1; }
[[ -f "${SUMMARIZE_PY}" ]] || { echo "[ERROR] missing summarize script: ${SUMMARIZE_PY}" >&2; exit 1; }

DATASET_PATH="${DATASET_PATH:-/jhe/dataset/math500}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
N_WAY="${N_WAY:-}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_PATH="${MODEL_PATH:-/jhe/Qwen2.5-14B-Instruct-1M}"
MODEL_NAME="$(basename "${MODEL_PATH}")"
METHOD="${METHOD:-offloading}"
TOPK="${TOPK:-0.10}"
FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER:-}"
FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_LOWER:-}"
FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS:-${FIXED_NUM_SKIP_LAYER:-}}"
FIXED_DECAY_P="${FIXED_DECAY_P:-${DECAY_P:-3}}"
FIXED_MAX_REUSE_COUNT="${FIXED_MAX_REUSE_COUNT:-${MAX_REUSE_COUNT:-}}"

trim_spaces() {
    local s="${1:-}"
    s="${s//[[:space:]]/}"
    echo "${s}"
}

run_multi_reuse_sweep_if_needed() {
    local raw="${1:-}"
    shift || true
    local -a script_args=("$@")
    local cleaned
    cleaned="$(trim_spaces "${raw}")"
    [[ "${cleaned}" == *,* ]] || return 0

    IFS=',' read -r -a reuse_values <<< "${cleaned}"
    echo "[INFO] detected MAX_REUSE_COUNT list: ${cleaned}"
    for v in "${reuse_values[@]}"; do
        [[ -n "${v}" ]] || continue
        if ! [[ "${v}" =~ ^[0-9]+$ ]]; then
            echo "[ERROR] invalid MAX_REUSE_COUNT in list: ${v}" >&2
            exit 1
        fi
        echo "[INFO] launching reuse sweep item: ${v}"
        FIXED_MAX_REUSE_COUNT="${v}" MAX_REUSE_COUNT="${v}" \
            /usr/bin/env bash "$0" "${script_args[@]}"
    done
    exit 0
}

# One-command sweep support, e.g.:
#   MAX_REUSE_COUNT=4,6,12,16 ./run_math500_nway.sh
run_multi_reuse_sweep_if_needed "${FIXED_MAX_REUSE_COUNT}" "$@"

resolve_base_config_file() {
    local method="${METHOD:-offloading}"
    local cfg="${CONFIG_FILE:-}"
    if [[ -n "${cfg}" ]]; then
        echo "${cfg}"
        return 0
    fi
    case "${method}" in
        offloading)
            echo "${SCRIPT_DIR}/config/hata_offloading/${MODEL_NAME}-64K.yaml"
            ;;
        flash-attn|flashattn)
            echo "${SCRIPT_DIR}/config/full_attn/${MODEL_NAME}-64K.yaml"
            ;;
        *)
            echo "${SCRIPT_DIR}/config/${MODEL_NAME}-64K-top${TOPK}.json"
            ;;
    esac
}

read_thresholds_from_config() {
    local cfg_path="$1"
    python3 - "${cfg_path}" <<'PY'
import json
import os
import sys

import yaml

cfg_path = sys.argv[1]
if not os.path.isfile(cfg_path):
    print("|")
    sys.exit(0)

with open(cfg_path, "r", encoding="utf-8") as f:
    text = f.read()

ext = os.path.splitext(cfg_path)[1].lower()
if ext in {".yaml", ".yml"}:
    cfg = yaml.safe_load(text)
else:
    cfg = json.loads(text)

upper = None
lower = None

def visit(node):
    global upper, lower
    if isinstance(node, dict):
        if upper is None and "reuse_threshold_upper" in node:
            upper = node["reuse_threshold_upper"]
        if lower is None and "reuse_threshold_lower" in node:
            lower = node["reuse_threshold_lower"]
        for v in node.values():
            visit(v)
    elif isinstance(node, list):
        for v in node:
            visit(v)

visit(cfg)
u = "" if upper is None else str(upper)
l = "" if lower is None else str(lower)
print(f"{u}|{l}")
PY
}

if [[ -z "${FIXED_REUSE_THRESHOLD_UPPER}" || -z "${FIXED_REUSE_THRESHOLD_LOWER}" ]]; then
    BASE_CONFIG_FILE="$(resolve_base_config_file)"
    if [[ -f "${BASE_CONFIG_FILE}" ]]; then
        THRESH_LINE="$(read_thresholds_from_config "${BASE_CONFIG_FILE}")"
        CFG_UPPER="${THRESH_LINE%%|*}"
        CFG_LOWER="${THRESH_LINE#*|}"
        if [[ -z "${FIXED_REUSE_THRESHOLD_UPPER}" && -n "${CFG_UPPER}" ]]; then
            FIXED_REUSE_THRESHOLD_UPPER="${CFG_UPPER}"
        fi
        if [[ -z "${FIXED_REUSE_THRESHOLD_LOWER}" && -n "${CFG_LOWER}" ]]; then
            FIXED_REUSE_THRESHOLD_LOWER="${CFG_LOWER}"
        fi
    fi
fi

tag_value() {
    local value="${1:-}"
    if [[ -z "${value}" ]]; then
        echo "keep"
    else
        echo "${value//./p}"
    fi
}

UPPER_TAG="$(tag_value "${FIXED_REUSE_THRESHOLD_UPPER}")"
LOWER_TAG="$(tag_value "${FIXED_REUSE_THRESHOLD_LOWER}")"
SKIP_TAG="$(tag_value "${FIXED_NUM_SKIP_LAYERS}")"
DECAY_TAG="$(tag_value "${FIXED_DECAY_P}")"
MAX_REUSE_TAG="$(tag_value "${FIXED_MAX_REUSE_COUNT}")"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/preds_nway}"
RUN_TAG_BASE="${RUN_TAG_BASE:-${MODEL_NAME}-math500-top${TOPK}-nway-thup${UPPER_TAG}-thlo${LOWER_TAG}-skip${SKIP_TAG}-decay${DECAY_TAG}-reuse${MAX_REUSE_TAG}}"
WORKER_ROOT="${OUTPUT_ROOT}/workers"
FINAL_OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD}/${RUN_TAG_BASE}"
SHARD_ROOT="${SCRIPT_DIR}/.generated_datasets/math500_shards"

BASELINE_RESULT="${BASELINE_RESULT:-}"
MAX_AVG_DROP="${MAX_AVG_DROP:-3.0}"
MAX_SINGLE_DROP="${MAX_SINGLE_DROP:-8.0}"

resolve_source_csv() {
    local p="$1"
    if [[ -f "$p" ]]; then
        echo "$p"
        return 0
    fi
    if [[ -f "${p}/math500_test.csv" ]]; then
        echo "${p}/math500_test.csv"
        return 0
    fi
    echo "[ERROR] DATASET_PATH must be csv file or directory containing math500_test.csv: ${p}" >&2
    exit 1
}

SOURCE_CSV="$(resolve_source_csv "${DATASET_PATH}")"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
GPU_COUNT="${#GPUS[@]}"
if (( GPU_COUNT <= 0 )); then
    echo "[ERROR] no GPUs parsed from CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi

if [[ -z "${N_WAY}" ]]; then
    N_WAY="${GPU_COUNT}"
fi
if ! [[ "${N_WAY}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] N_WAY must be a positive integer, got: ${N_WAY}" >&2
    exit 1
fi
if (( N_WAY > GPU_COUNT )); then
    echo "[ERROR] N_WAY=${N_WAY} exceeds visible GPU count=${GPU_COUNT}" >&2
    exit 1
fi

rm -rf "${SHARD_ROOT}"
mkdir -p "${SHARD_ROOT}"
python3 - "${SOURCE_CSV}" "${SHARD_ROOT}" "${N_WAY}" <<'PY'
import csv
import os
import sys

src_csv, shard_root, n_way_raw = sys.argv[1:4]
n_way = int(n_way_raw)

with open(src_csv, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

counts = [0] * n_way
writers = []
files = []
for i in range(n_way):
    shard_dir = os.path.join(shard_root, f"shard_{i:02d}")
    os.makedirs(shard_dir, exist_ok=True)
    out_path = os.path.join(shard_dir, "math500_test.csv")
    f = open(out_path, "w", encoding="utf-8", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    files.append(f)
    writers.append(w)

for idx, row in enumerate(rows):
    sid = idx % n_way
    writers[sid].writerow(row)
    counts[sid] += 1

for f in files:
    f.close()

print("[INFO] shard counts:", ", ".join(f"{i}:{c}" for i, c in enumerate(counts)))
PY

echo "[INFO] Start ${N_WAY}-way shard-parallel run (sglang/test/litecache)"
echo "[INFO] effective knobs: th_up=${FIXED_REUSE_THRESHOLD_UPPER} th_lo=${FIXED_REUSE_THRESHOLD_LOWER} skip_layers=${FIXED_NUM_SKIP_LAYERS} decay_p=${FIXED_DECAY_P:-keep} max_reuse_count=${FIXED_MAX_REUSE_COUNT}"

declare -a PIDS=()
declare -a WORKER_PRED_FILES=()

to_p_tag() {
    echo "${1//./p}"
}
TOPK_TAG="$(to_p_tag "${TOPK}")"

for ((i=0; i<N_WAY; i++)); do
    gpu="${GPUS[$i]}"
    shard_dir="${SHARD_ROOT}/shard_$(printf "%02d" "${i}")"
    worker_tag="${RUN_TAG_BASE}-w$(printf "%02d" "${i}")"
    worker_log="${WORKER_ROOT}/${worker_tag}.log"
    worker_out_dir="${WORKER_ROOT}/${METHOD}/${worker_tag}"
    worker_pred_file="${worker_out_dir}/math500.jsonl"
    WORKER_PRED_FILES+=("${worker_pred_file}")

    mkdir -p "${WORKER_ROOT}"

    echo "[INFO] worker=${i} gpu=${gpu} dataset=${shard_dir} output=${worker_out_dir} max_reuse_count=${FIXED_MAX_REUSE_COUNT}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        AUTO_SELECT_GPUS=0 \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        DATASET_PATH="${shard_dir}" \
        OUTPUT_ROOT="${WORKER_ROOT}" \
        RUN_TAG="${worker_tag}" \
        FIXED_MAX_REUSE_COUNT="${FIXED_MAX_REUSE_COUNT}" \
        NUM_GPUS=1 MP_NUM=1 PP_NUM=1 DRY_RUN=1 \
            "${BASE_RUNNER}"
    else
        (
            AUTO_SELECT_GPUS=0 \
            CUDA_VISIBLE_DEVICES="${gpu}" \
            DATASET_PATH="${shard_dir}" \
            OUTPUT_ROOT="${WORKER_ROOT}" \
            RUN_TAG="${worker_tag}" \
            FIXED_MAX_REUSE_COUNT="${FIXED_MAX_REUSE_COUNT}" \
            NUM_GPUS=1 MP_NUM=1 PP_NUM=1 DRY_RUN=0 \
                "${BASE_RUNNER}" > "${worker_log}" 2>&1
        ) &
        PIDS+=("$!")
    fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] would merge into ${FINAL_OUTPUT_DIR}/math500.jsonl"
    echo "[DRY_RUN] would run: python3 ${EVAL_PY} --model ${FINAL_OUTPUT_DIR}"
    echo "[DRY_RUN] would run summarize against ${FINAL_OUTPUT_DIR}/result.json"
    exit 0
fi

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAIL=1
    fi
done
if (( FAIL != 0 )); then
    echo "[ERROR] one or more workers failed; check logs under ${WORKER_ROOT}/*.log" >&2
    exit 1
fi

mkdir -p "${FINAL_OUTPUT_DIR}"
MERGED_JSONL="${FINAL_OUTPUT_DIR}/math500.jsonl"
: > "${MERGED_JSONL}"
for f in "${WORKER_PRED_FILES[@]}"; do
    [[ -f "${f}" ]] || { echo "[ERROR] missing worker prediction: ${f}" >&2; exit 1; }
    cat "${f}" >> "${MERGED_JSONL}"
done

python3 "${EVAL_PY}" --model "${FINAL_OUTPUT_DIR}" | tee -a "${FINAL_OUTPUT_DIR}/eval.log"
RESULT_JSON="${FINAL_OUTPUT_DIR}/result.json"
[[ -f "${RESULT_JSON}" ]] || { echo "[ERROR] missing merged result json: ${RESULT_JSON}" >&2; exit 1; }

python3 "${SUMMARIZE_PY}" \
    "${RESULT_JSON}" \
    "${BASELINE_RESULT}" \
    "${MAX_AVG_DROP}" \
    "${MAX_SINGLE_DROP}"

echo "[DONE] ${N_WAY}-way shard run finished. result=${RESULT_JSON}"
