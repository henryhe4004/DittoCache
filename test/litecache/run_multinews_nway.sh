#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BASE_RUNNER="${BASE_RUNNER:-${SCRIPT_DIR}/test_accuracy_multinews.sh}"
EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/eval_longbench_infinitebench.py}"
SUMMARIZE_PY="${SUMMARIZE_PY:-${SCRIPT_DIR}/summarize_accuracy.py}"

[[ -x "${BASE_RUNNER}" ]] || { echo "[ERROR] missing executable runner: ${BASE_RUNNER}" >&2; exit 1; }
[[ -f "${EVAL_PY}" ]] || { echo "[ERROR] missing eval script: ${EVAL_PY}" >&2; exit 1; }
[[ -f "${SUMMARIZE_PY}" ]] || { echo "[ERROR] missing summarize script: ${SUMMARIZE_PY}" >&2; exit 1; }

DATASET_PATH="${DATASET_PATH:-/jhe/dataset/LongBench}"
if [[ ! -d "${DATASET_PATH}" && -d "/jhe/LongBench" ]]; then
    DATASET_PATH="/jhe/LongBench"
fi

resolve_multinews_jsonl() {
    local p="$1"
    if [[ -f "${p}" ]]; then
        echo "${p}"
        return 0
    fi
    local -a candidates=(
        "${p}/multi_news_e.jsonl"
        "${p}/data/multi_news_e.jsonl"
        "${p}/multinews_e.jsonl"
        "${p}/data/multinews_e.jsonl"
        "${p}/mulitinews_e.jsonl"
        "${p}/data/mulitinews_e.jsonl"
    )
    local c
    for c in "${candidates[@]}"; do
        if [[ -f "${c}" ]]; then
            echo "${c}"
            return 0
        fi
    done
    echo "[ERROR] cannot find multi_news_e.jsonl under DATASET_PATH=${p}" >&2
    exit 1
}

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

normalize_method_for_path() {
    local m="${1:-}"
    case "${m}" in
        flash-attn)
            echo "flashattn"
            ;;
        *)
            echo "${m}"
            ;;
    esac
}
METHOD_PATH_KEY="$(normalize_method_for_path "${METHOD}")"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/preds_nway}"
RUN_TAG_BASE="${RUN_TAG_BASE:-${MODEL_NAME}-multinews-top${TOPK}-nway-thup${UPPER_TAG}-thlo${LOWER_TAG}-skip${SKIP_TAG}-decay${DECAY_TAG}-reuse${MAX_REUSE_TAG}}"
WORKER_ROOT="${OUTPUT_ROOT}/workers"
FINAL_OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD_PATH_KEY}/${RUN_TAG_BASE}"
SHARD_ROOT="${SCRIPT_DIR}/.generated_datasets/multinews_shards"

BASELINE_RESULT="${BASELINE_RESULT:-}"
MAX_AVG_DROP="${MAX_AVG_DROP:-3.0}"
MAX_SINGLE_DROP="${MAX_SINGLE_DROP:-8.0}"

SOURCE_JSONL="$(resolve_multinews_jsonl "${DATASET_PATH}")"

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
python3 - "${SOURCE_JSONL}" "${SHARD_ROOT}" "${N_WAY}" <<'PY'
import os
import sys

src_jsonl, shard_root, n_way_raw = sys.argv[1:4]
n_way = int(n_way_raw)

writers = []
files = []
counts = [0] * n_way
for i in range(n_way):
    shard_dir = os.path.join(shard_root, f"shard_{i:02d}")
    os.makedirs(shard_dir, exist_ok=True)
    data_dir = os.path.join(shard_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "multi_news_e.jsonl")
    f = open(out_path, "w", encoding="utf-8")
    files.append(f)
    writers.append(f)

with open(src_jsonl, "r", encoding="utf-8") as fin:
    for idx, line in enumerate(fin):
        sid = idx % n_way
        writers[sid].write(line)
        counts[sid] += 1

for f in files:
    f.close()

print("[INFO] shard counts:", ", ".join(f"{i}:{c}" for i, c in enumerate(counts)))
PY

echo "[INFO] Start ${N_WAY}-way shard-parallel MultiNews run (sglang/test/litecache)"
echo "[INFO] source_jsonl=${SOURCE_JSONL}"
echo "[INFO] effective knobs: th_up=${FIXED_REUSE_THRESHOLD_UPPER} th_lo=${FIXED_REUSE_THRESHOLD_LOWER} skip_layers=${FIXED_NUM_SKIP_LAYERS} decay_p=${FIXED_DECAY_P:-keep} max_reuse_count=${FIXED_MAX_REUSE_COUNT}"

declare -a PIDS=()
declare -a WORKER_PRED_FILES=()

for ((i=0; i<N_WAY; i++)); do
    gpu="${GPUS[$i]}"
    shard_dir="${SHARD_ROOT}/shard_$(printf "%02d" "${i}")"
    worker_tag="${RUN_TAG_BASE}-w$(printf "%02d" "${i}")"
    worker_log="${WORKER_ROOT}/${METHOD_PATH_KEY}/${worker_tag}.log"
    worker_out_dir="${WORKER_ROOT}/${METHOD_PATH_KEY}/${worker_tag}"
    worker_pred_file="${worker_out_dir}/multi_news_e.jsonl"
    WORKER_PRED_FILES+=("${worker_pred_file}")

    mkdir -p "${WORKER_ROOT}/${METHOD_PATH_KEY}"

    echo "[INFO] worker=${i} gpu=${gpu} dataset=${shard_dir} output=${worker_out_dir} max_reuse_count=${FIXED_MAX_REUSE_COUNT}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        AUTO_SELECT_GPUS=0 \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        DATASET_PATH="${shard_dir}" \
        OUTPUT_ROOT="${WORKER_ROOT}" \
        RUN_TAG="${worker_tag}" \
        FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER}" \
        FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_LOWER}" \
        FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS}" \
        FIXED_DECAY_P="${FIXED_DECAY_P}" \
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
            FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER}" \
            FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_LOWER}" \
            FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS}" \
            FIXED_DECAY_P="${FIXED_DECAY_P}" \
            FIXED_MAX_REUSE_COUNT="${FIXED_MAX_REUSE_COUNT}" \
            NUM_GPUS=1 MP_NUM=1 PP_NUM=1 DRY_RUN=0 \
                "${BASE_RUNNER}" > "${worker_log}" 2>&1
        ) &
        PIDS+=("$!")
    fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] would merge into ${FINAL_OUTPUT_DIR}/multi_news_e.jsonl"
    echo "[DRY_RUN] would run: python3 ${EVAL_PY} --model ${FINAL_OUTPUT_DIR} --e"
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
    echo "[ERROR] one or more workers failed; check logs under ${WORKER_ROOT}/${METHOD_PATH_KEY}/*.log" >&2
    exit 1
fi

mkdir -p "${FINAL_OUTPUT_DIR}"
MERGED_JSONL="${FINAL_OUTPUT_DIR}/multi_news_e.jsonl"
: > "${MERGED_JSONL}"
for f in "${WORKER_PRED_FILES[@]}"; do
    [[ -f "${f}" ]] || { echo "[ERROR] missing worker prediction: ${f}" >&2; exit 1; }
    cat "${f}" >> "${MERGED_JSONL}"
done

python3 "${EVAL_PY}" --model "${FINAL_OUTPUT_DIR}" --e | tee -a "${FINAL_OUTPUT_DIR}/eval.log"
RESULT_JSON="${FINAL_OUTPUT_DIR}/result.json"
[[ -f "${RESULT_JSON}" ]] || { echo "[ERROR] missing merged result json: ${RESULT_JSON}" >&2; exit 1; }

python3 "${SUMMARIZE_PY}" \
    "${RESULT_JSON}" \
    "${BASELINE_RESULT}" \
    "${MAX_AVG_DROP}" \
    "${MAX_SINGLE_DROP}"

echo "[DONE] ${N_WAY}-way MultiNews shard run finished. result=${RESULT_JSON}"
