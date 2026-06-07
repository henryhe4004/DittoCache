#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BASE_RUNNER="${BASE_RUNNER:-${SCRIPT_DIR}/test_accuracy.sh}"
EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/eval_longbench_infinitebench.py}"
SUMMARIZE_PY="${SUMMARIZE_PY:-${SCRIPT_DIR}/summarize_accuracy.py}"

[[ -x "${BASE_RUNNER}" ]] || { echo "[ERROR] missing executable runner: ${BASE_RUNNER}" >&2; exit 1; }
[[ -f "${EVAL_PY}" ]] || { echo "[ERROR] missing eval script: ${EVAL_PY}" >&2; exit 1; }
[[ -f "${SUMMARIZE_PY}" ]] || { echo "[ERROR] missing summarize script: ${SUMMARIZE_PY}" >&2; exit 1; }

DATASET_PATH="${DATASET_PATH:-/datasets/LongBench-v2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_WAY="${N_WAY:-4}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
WORKER_NUM_GPUS="${WORKER_NUM_GPUS:-2}"
WORKER_MP_NUM="${WORKER_MP_NUM:-2}"
WORKER_PP_NUM="${WORKER_PP_NUM:-1}"

MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-14B-Instruct-1M}"
MODEL_NAME="$(basename "${MODEL_PATH}")"
METHOD="${METHOD:-offloading}"
TOPK="${TOPK:-0.10}"
CONFIG_SUFFIX="${CONFIG_SUFFIX:-128K}"
SELECTIVE_START_LEN="${SELECTIVE_START_LEN:-2048}"
TASKS="${TASKS:-longbench-v2}"
FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER:-1.0}"
FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_LOWER:-1.0}"
FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS:-${FIXED_NUM_SKIP_LAYER:-48}}"
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

normalize_method_for_path() {
    local m="${1:-}"
    case "${m}" in
        flash_attn|flash-attn)
            echo "flashattn"
            ;;
        *)
            echo "${m}"
            ;;
    esac
}
METHOD="$(normalize_method_for_path "${METHOD}")"
METHOD_PATH_KEY="${METHOD}"

trim_spaces() {
    local s="${1:-}"
    s="${s//[[:space:]]/}"
    echo "${s}"
}

tag_value() {
    local value="${1:-}"
    if [[ -z "${value}" ]]; then
        echo "keep"
    else
        echo "${value//./p}"
    fi
}

resolve_longbench_v2_dataset_root() {
    local base="${1:-}"
    if [[ -f "${base}/data.json" ]]; then
        echo "${base}"
    elif [[ -f "${base}/longbench-v2.json" ]]; then
        echo "${base}"
    elif [[ -f "${base}/longbench-v2.jsonl" ]]; then
        echo "${base}"
    else
        echo "${base}"
    fi
}
DATASET_ROOT="$(resolve_longbench_v2_dataset_root "${DATASET_PATH}")"
SHARD_ROOT="${SCRIPT_DIR}/.generated_datasets/longbench_v2_shards"

run_multi_method_sweep_if_needed() {
    local raw="${1:-}"
    shift || true
    local -a script_args=("$@")
    local cleaned
    cleaned="$(trim_spaces "${raw}")"
    [[ "${cleaned}" == *,* ]] || return 0

    IFS=',' read -r -a method_values <<< "${cleaned}"
    echo "[INFO] detected METHOD list: ${cleaned}"
    for v in "${method_values[@]}"; do
        [[ -n "${v}" ]] || continue
        case "${v}" in
            offloading|flash_attn|flash-attn|flashattn)
                ;;
            *)
                echo "[ERROR] invalid METHOD in list: ${v} (supported: offloading, flash_attn, flash-attn, flashattn)" >&2
                exit 1
                ;;
        esac
        echo "[INFO] launching method sweep item: ${v}"
        METHOD="${v}" /usr/bin/env bash "$0" "${script_args[@]}"
    done
    exit 0
}

run_multi_method_sweep_if_needed "${METHOD}" "$@"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/preds_nway}"
UPPER_TAG="$(tag_value "${FIXED_REUSE_THRESHOLD_UPPER}")"
LOWER_TAG="$(tag_value "${FIXED_REUSE_THRESHOLD_LOWER}")"
SKIP_TAG="$(tag_value "${FIXED_NUM_SKIP_LAYERS}")"
DECAY_TAG="$(tag_value "${FIXED_DECAY_P}")"
MAX_REUSE_TAG="$(tag_value "${FIXED_MAX_REUSE_COUNT}")"
RUN_TAG_BASE="${RUN_TAG_BASE:-${MODEL_NAME}-longbench-v2-top${TOPK}-nway-thup${UPPER_TAG}-thlo${LOWER_TAG}-skip${SKIP_TAG}-decay${DECAY_TAG}-reuse${MAX_REUSE_TAG}}"
WORKER_ROOT="${OUTPUT_ROOT}/workers"
FINAL_OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD_PATH_KEY}/${RUN_TAG_BASE}"

BASELINE_RESULT="${BASELINE_RESULT:-}"
MAX_AVG_DROP="${MAX_AVG_DROP:-3.0}"
MAX_SINGLE_DROP="${MAX_SINGLE_DROP:-8.0}"

[[ -d "${DATASET_ROOT}" ]] || { echo "[ERROR] missing dataset dir: ${DATASET_ROOT}" >&2; exit 1; }

SOURCE_JSON_PATH=""
if [[ -f "${DATASET_ROOT}/data.json" ]]; then
    SOURCE_JSON_PATH="${DATASET_ROOT}/data.json"
elif [[ -f "${DATASET_ROOT}/longbench-v2.json" ]]; then
    SOURCE_JSON_PATH="${DATASET_ROOT}/longbench-v2.json"
else
    echo "[ERROR] expected data.json or longbench-v2.json under ${DATASET_ROOT}" >&2
    exit 1
fi

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
GPU_COUNT="${#GPUS[@]}"
if (( GPU_COUNT <= 0 )); then
    echo "[ERROR] no GPUs parsed from CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi

if ! [[ "${WORKER_NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] WORKER_NUM_GPUS must be a positive integer, got: ${WORKER_NUM_GPUS}" >&2
    exit 1
fi
if (( WORKER_NUM_GPUS > GPU_COUNT )); then
    echo "[ERROR] WORKER_NUM_GPUS=${WORKER_NUM_GPUS} exceeds visible GPU count=${GPU_COUNT}" >&2
    exit 1
fi
if (( GPU_COUNT % WORKER_NUM_GPUS != 0 )); then
    echo "[ERROR] visible GPU count=${GPU_COUNT} is not divisible by WORKER_NUM_GPUS=${WORKER_NUM_GPUS}" >&2
    exit 1
fi

if [[ -z "${WORKER_MP_NUM}" && -z "${WORKER_PP_NUM}" ]]; then
    WORKER_MP_NUM=1
    WORKER_PP_NUM="${WORKER_NUM_GPUS}"
elif [[ -n "${WORKER_MP_NUM}" && -z "${WORKER_PP_NUM}" ]]; then
    if (( WORKER_NUM_GPUS % WORKER_MP_NUM != 0 )); then
        echo "[ERROR] WORKER_NUM_GPUS=${WORKER_NUM_GPUS} is not divisible by WORKER_MP_NUM=${WORKER_MP_NUM}" >&2
        exit 1
    fi
    WORKER_PP_NUM="$(( WORKER_NUM_GPUS / WORKER_MP_NUM ))"
elif [[ -z "${WORKER_MP_NUM}" && -n "${WORKER_PP_NUM}" ]]; then
    if (( WORKER_NUM_GPUS % WORKER_PP_NUM != 0 )); then
        echo "[ERROR] WORKER_NUM_GPUS=${WORKER_NUM_GPUS} is not divisible by WORKER_PP_NUM=${WORKER_PP_NUM}" >&2
        exit 1
    fi
    WORKER_MP_NUM="$(( WORKER_NUM_GPUS / WORKER_PP_NUM ))"
elif (( WORKER_MP_NUM * WORKER_PP_NUM != WORKER_NUM_GPUS )); then
    echo "[ERROR] WORKER_NUM_GPUS=${WORKER_NUM_GPUS} mismatches WORKER_MP_NUM*WORKER_PP_NUM=$(( WORKER_MP_NUM * WORKER_PP_NUM ))" >&2
    exit 1
fi

MAX_WORKERS_BY_GPU="$(( GPU_COUNT / WORKER_NUM_GPUS ))"

if [[ -z "${N_WAY}" ]]; then
    N_WAY="${MAX_WORKERS_BY_GPU}"
fi
if ! [[ "${N_WAY}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] N_WAY must be a positive integer, got: ${N_WAY}" >&2
    exit 1
fi
if (( N_WAY > MAX_WORKERS_BY_GPU )); then
    echo "[ERROR] N_WAY=${N_WAY} exceeds worker capacity=${MAX_WORKERS_BY_GPU} for WORKER_NUM_GPUS=${WORKER_NUM_GPUS} with visible GPU count=${GPU_COUNT}" >&2
    exit 1
fi

TASKS_CLEANED="$(trim_spaces "${TASKS}")"
if [[ "${TASKS_CLEANED}" != "longbench-v2" ]]; then
    echo "[ERROR] longbench-v2 nway runner only supports TASKS=longbench-v2, got: ${TASKS}" >&2
    exit 1
fi

rm -rf "${SHARD_ROOT}"
mkdir -p "${SHARD_ROOT}"
python3 - "${SOURCE_JSON_PATH}" "${SHARD_ROOT}" "${N_WAY}" <<'PY'
import json
import os
import sys

src_json, shard_root, n_way_raw = sys.argv[1:4]
n_way = int(n_way_raw)

with open(src_json, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise ValueError(f"Expected list in {src_json}")

shards = [[] for _ in range(n_way)]
for idx, row in enumerate(data):
    shards[idx % n_way].append(row)

for sid, rows in enumerate(shards):
    shard_dir = os.path.join(shard_root, f"shard_{sid:02d}")
    os.makedirs(shard_dir, exist_ok=True)
    with open(os.path.join(shard_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)

print("[INFO] shard counts:", ", ".join(f"{i}:{len(rows)}" for i, rows in enumerate(shards)))
PY

echo "[INFO] Start ${N_WAY}-way sample-parallel LongBench-v2 run"
echo "[INFO] method=${METHOD} config_suffix=${CONFIG_SUFFIX} topk=${TOPK} selective_start_len=${SELECTIVE_START_LEN}"
echo "[INFO] effective knobs: th_up=${FIXED_REUSE_THRESHOLD_UPPER} th_lo=${FIXED_REUSE_THRESHOLD_LOWER} skip_layers=${FIXED_NUM_SKIP_LAYERS} decay_p=${FIXED_DECAY_P:-keep} max_reuse_count=${FIXED_MAX_REUSE_COUNT}"
echo "[INFO] worker topology: worker_num_gpus=${WORKER_NUM_GPUS} mp=${WORKER_MP_NUM} pp=${WORKER_PP_NUM}"
echo "[INFO] dataset_root=${DATASET_ROOT} source_json=${SOURCE_JSON_PATH}"

declare -a PIDS=()
declare -a WORKER_OUT_DIRS=()

for ((i=0; i<N_WAY; i++)); do
    gpu_start=$(( i * WORKER_NUM_GPUS ))
    worker_gpu_list="$(IFS=,; echo "${GPUS[*]:gpu_start:WORKER_NUM_GPUS}")"
    shard_dir="${SHARD_ROOT}/shard_$(printf "%02d" "${i}")"
    worker_tag="${RUN_TAG_BASE}-w$(printf "%02d" "${i}")"
    worker_log="${WORKER_ROOT}/${METHOD_PATH_KEY}/${worker_tag}.log"
    worker_out_dir="${WORKER_ROOT}/${METHOD_PATH_KEY}/${worker_tag}"

    WORKER_OUT_DIRS+=("${worker_out_dir}")

    mkdir -p "${WORKER_ROOT}/${METHOD_PATH_KEY}"
    mkdir -p "${worker_out_dir}"

    echo "[INFO] worker=${i} gpus=${worker_gpu_list} shard=${shard_dir} output=${worker_out_dir}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        AUTO_SELECT_GPUS=0 \
        CUDA_VISIBLE_DEVICES="${worker_gpu_list}" \
        METHOD="${METHOD}" \
        FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER}" \
        FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_LOWER}" \
        FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS}" \
        FIXED_DECAY_P="${FIXED_DECAY_P}" \
        ENABLE_MAX_REUSE_COUNT="${ENABLE_MAX_REUSE_COUNT}" \
        FIXED_MAX_REUSE_COUNT="${FIXED_MAX_REUSE_COUNT}" \
        DATASET_NAME=longbench-v2 \
        DATASET_PATH="${shard_dir}" \
        TASKS="longbench-v2" \
        OUTPUT_ROOT="${WORKER_ROOT}" \
        RUN_TAG="${worker_tag}" \
        CONFIG_SUFFIX="${CONFIG_SUFFIX}" \
        SELECTIVE_START_LEN="${SELECTIVE_START_LEN}" \
        RESUME="${RESUME}" \
        MP_NUM="${WORKER_MP_NUM}" \
        PP_NUM="${WORKER_PP_NUM}" \
        DRY_RUN=1 \
            "${BASE_RUNNER}"
    else
        (
            AUTO_SELECT_GPUS=0 \
            CUDA_VISIBLE_DEVICES="${worker_gpu_list}" \
            METHOD="${METHOD}" \
            FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER}" \
            FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_LOWER}" \
            FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS}" \
            FIXED_DECAY_P="${FIXED_DECAY_P}" \
            ENABLE_MAX_REUSE_COUNT="${ENABLE_MAX_REUSE_COUNT}" \
            FIXED_MAX_REUSE_COUNT="${FIXED_MAX_REUSE_COUNT}" \
            DATASET_NAME=longbench-v2 \
            DATASET_PATH="${shard_dir}" \
            TASKS="longbench-v2" \
            OUTPUT_ROOT="${WORKER_ROOT}" \
            RUN_TAG="${worker_tag}" \
            CONFIG_SUFFIX="${CONFIG_SUFFIX}" \
            SELECTIVE_START_LEN="${SELECTIVE_START_LEN}" \
            RESUME="${RESUME}" \
            MP_NUM="${WORKER_MP_NUM}" \
            PP_NUM="${WORKER_PP_NUM}" \
            DRY_RUN=0 \
                "${BASE_RUNNER}" > "${worker_log}" 2>&1
        ) &
        PIDS+=("$!")
    fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] would merge worker jsonl files into ${FINAL_OUTPUT_DIR}/longbench-v2.jsonl"
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
    echo "[ERROR] one or more workers failed; check logs under ${WORKER_ROOT}/${METHOD_PATH_KEY}/*.log" >&2
    exit 1
fi

mkdir -p "${FINAL_OUTPUT_DIR}"
MERGED_JSONL="${FINAL_OUTPUT_DIR}/longbench-v2.jsonl"
TMP_MERGED="${MERGED_JSONL}.tmp"
rm -f "${MERGED_JSONL}" "${TMP_MERGED}"

python3 - "${TMP_MERGED}" "${WORKER_OUT_DIRS[@]}" <<'PY'
import json
import os
import sys

out_path = sys.argv[1]
worker_dirs = sys.argv[2:]
rows = []
for worker_dir in worker_dirs:
    src = os.path.join(worker_dir, "longbench-v2.jsonl")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"missing worker prediction: {src}")
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

rows.sort(key=lambda x: x.get("index", 0))
with open(out_path, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
mv "${TMP_MERGED}" "${MERGED_JSONL}"

python3 "${EVAL_PY}" --model "${FINAL_OUTPUT_DIR}" | tee -a "${FINAL_OUTPUT_DIR}/eval.log"
RESULT_JSON="${FINAL_OUTPUT_DIR}/result.json"
[[ -f "${RESULT_JSON}" ]] || { echo "[ERROR] missing merged result json: ${RESULT_JSON}" >&2; exit 1; }

python3 "${SUMMARIZE_PY}" \
    "${RESULT_JSON}" \
    "${BASELINE_RESULT}" \
    "${MAX_AVG_DROP}" \
    "${MAX_SINGLE_DROP}"

echo "[DONE] ${N_WAY}-way sample-parallel LongBench-v2 run finished. result=${RESULT_JSON}"
