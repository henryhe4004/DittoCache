#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BASE_RUNNER="${BASE_RUNNER:-${SCRIPT_DIR}/test_accuracy_infinitebench.sh}"
EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/eval_longbench_infinitebench.py}"
SUMMARIZE_PY="${SUMMARIZE_PY:-${SCRIPT_DIR}/summarize_accuracy.py}"

[[ -x "${BASE_RUNNER}" ]] || { echo "[ERROR] missing executable runner: ${BASE_RUNNER}" >&2; exit 1; }
[[ -f "${EVAL_PY}" ]] || { echo "[ERROR] missing eval script: ${EVAL_PY}" >&2; exit 1; }
[[ -f "${SUMMARIZE_PY}" ]] || { echo "[ERROR] missing summarize script: ${SUMMARIZE_PY}" >&2; exit 1; }

DATASET_PATH="${DATASET_PATH:-/datasets/InfiniteBench}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
N_WAY="${N_WAY:-}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
WORKER_NUM_GPUS="${WORKER_NUM_GPUS:-1}"
WORKER_MP_NUM="${WORKER_MP_NUM:-}"
WORKER_PP_NUM="${WORKER_PP_NUM:-}"

MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-14B-Instruct-1M}"
MODEL_NAME="$(basename "${MODEL_PATH}")"
METHOD="${METHOD:-offloading}"
TOPK="${TOPK:-0.10}"
FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER:-1.0}"
FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_LOWER:-1.0}"
FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS:-${FIXED_NUM_SKIP_LAYER:-48}}"
FIXED_DECAY_P="${FIXED_DECAY_P:-${DECAY_P:-}}"
ENABLE_MAX_REUSE_COUNT="${ENABLE_MAX_REUSE_COUNT:-}"
MAX_REUSE_COUNT_INPUT="${FIXED_MAX_REUSE_COUNT:-${MAX_REUSE_COUNT:-}}"
USE_MAX_REUSE_COUNT=0
if [[ "${ENABLE_MAX_REUSE_COUNT}" == "0" ]]; then
    FIXED_MAX_REUSE_COUNT=""
elif [[ "${ENABLE_MAX_REUSE_COUNT}" == "1" || -n "${MAX_REUSE_COUNT_INPUT}" ]]; then
    FIXED_MAX_REUSE_COUNT="${MAX_REUSE_COUNT_INPUT}"
    USE_MAX_REUSE_COUNT=1
else
    FIXED_MAX_REUSE_COUNT=""
fi
#longbook_qa_eng,longbook_qa_chn,
TASKS="${TASKS:-code_debug,code_run,passkey,number_string,kv_retrieval,math_find,math_calc,longdialogue_qa_eng}"

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
            offloading|flash-attn|flashattn)
                ;;
            *)
                echo "[ERROR] invalid METHOD in list: ${v} (supported: offloading, flash-attn, flashattn)" >&2
                exit 1
                ;;
        esac
        echo "[INFO] launching method sweep item: ${v}"
        METHOD="${v}" \
            /usr/bin/env bash "$0" "${script_args[@]}"
    done
    exit 0
}

# One-command method sweep support, e.g.:
#   METHOD=offloading,flash-attn ./run_infinitebench_nway.sh
run_multi_method_sweep_if_needed "${METHOD}" "$@"

# One-command sweep support, e.g.:
#   MAX_REUSE_COUNT=4,6,12,16 ./run_infinitebench_nway.sh
#   ENABLE_MAX_REUSE_COUNT=0 MAX_REUSE_COUNT=4,6,12,16 ./run_infinitebench_nway.sh  # disable explicitly
if [[ "${USE_MAX_REUSE_COUNT}" == "1" ]]; then
    run_multi_reuse_sweep_if_needed "${FIXED_MAX_REUSE_COUNT}" "$@"
fi

resolve_base_config_file() {
    local method="${METHOD:-offloading}"
    local cfg="${CONFIG_FILE:-}"
    if [[ -n "${cfg}" ]]; then
        echo "${cfg}"
        return 0
    fi
    case "${method}" in
        offloading)
            echo "${SCRIPT_DIR}/config/hata_offloading/${MODEL_NAME}-256K.yaml"
            ;;
        flash-attn|flashattn)
            echo "${SCRIPT_DIR}/config/full_attn/${MODEL_NAME}-256K.yaml"
            ;;
        *)
            echo "${SCRIPT_DIR}/config/${MODEL_NAME}-256K-top${TOPK}.json"
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
RUN_TAG_BASE="${RUN_TAG_BASE:-${MODEL_NAME}-infinitebench-top${TOPK}-nway-thup${UPPER_TAG}-thlo${LOWER_TAG}-skip${SKIP_TAG}-decay${DECAY_TAG}-reuse${MAX_REUSE_TAG}}"
WORKER_ROOT="${OUTPUT_ROOT}/workers"
FINAL_OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD_PATH_KEY}/${RUN_TAG_BASE}"

BASELINE_RESULT="${BASELINE_RESULT:-}"
MAX_AVG_DROP="${MAX_AVG_DROP:-3.0}"
MAX_SINGLE_DROP="${MAX_SINGLE_DROP:-8.0}"

[[ -d "${DATASET_PATH}" ]] || { echo "[ERROR] missing dataset dir: ${DATASET_PATH}" >&2; exit 1; }

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
if [[ -z "${TASKS_CLEANED}" ]]; then
    echo "[ERROR] TASKS is empty" >&2
    exit 1
fi

IFS=',' read -r -a TASK_ITEMS <<< "${TASKS_CLEANED}"
declare -A TASK_SEEN=()
declare -a TASK_LIST=()
for t in "${TASK_ITEMS[@]}"; do
    [[ -n "${t}" ]] || continue
    if [[ -z "${TASK_SEEN[${t}]+x}" ]]; then
        TASK_LIST+=("${t}")
        TASK_SEEN["${t}"]=1
    fi
    if [[ ! -f "${DATASET_PATH}/${t}.jsonl" ]]; then
        echo "[ERROR] missing task file: ${DATASET_PATH}/${t}.jsonl" >&2
        exit 1
    fi
done

TASK_COUNT="${#TASK_LIST[@]}"
if (( TASK_COUNT == 0 )); then
    echo "[ERROR] no valid tasks parsed from TASKS=${TASKS}" >&2
    exit 1
fi

if (( N_WAY > TASK_COUNT )); then
    echo "[INFO] N_WAY=${N_WAY} > task_count=${TASK_COUNT}; only ${TASK_COUNT} worker(s) will be launched"
fi

declare -a WORKER_TASKS=()
for ((i=0; i<N_WAY; i++)); do
    WORKER_TASKS[i]=""
done
for ((idx=0; idx<TASK_COUNT; idx++)); do
    wid=$(( idx % N_WAY ))
    if [[ -z "${WORKER_TASKS[$wid]}" ]]; then
        WORKER_TASKS[$wid]="${TASK_LIST[$idx]}"
    else
        WORKER_TASKS[$wid]="${WORKER_TASKS[$wid]},${TASK_LIST[$idx]}"
    fi
done

echo "[INFO] Start ${N_WAY}-way task-parallel run (sglang/test/ditto)"
echo "[INFO] effective knobs: th_up=${FIXED_REUSE_THRESHOLD_UPPER} th_lo=${FIXED_REUSE_THRESHOLD_LOWER} skip_layers=${FIXED_NUM_SKIP_LAYERS} decay_p=${FIXED_DECAY_P:-keep} max_reuse_count=${FIXED_MAX_REUSE_COUNT}"
echo "[INFO] worker topology: worker_num_gpus=${WORKER_NUM_GPUS} mp=${WORKER_MP_NUM} pp=${WORKER_PP_NUM}"
echo "[INFO] total tasks=${TASK_COUNT} tasks=${TASKS_CLEANED}"

declare -a PIDS=()
declare -a WORKER_OUT_DIRS=()
declare -a WORKER_TASK_ASSIGNMENTS=()
declare -A EXISTING_TASK_OUTPUTS=()

# Build a task->jsonl lookup from existing worker outputs so we can reuse
# prior partial results when N_WAY changes (e.g., 2-way -> 8-way).
while IFS= read -r existing_file; do
    task_name="$(basename "${existing_file}")"
    task_name="${task_name%.jsonl}"
    if [[ -z "${EXISTING_TASK_OUTPUTS[${task_name}]+x}" ]]; then
        EXISTING_TASK_OUTPUTS["${task_name}"]="${existing_file}"
    fi
done < <(compgen -G "${WORKER_ROOT}/${METHOD_PATH_KEY}/${RUN_TAG_BASE}-w*/*.jsonl" || true)

for ((i=0; i<N_WAY; i++)); do
    worker_tasks="${WORKER_TASKS[$i]}"
    [[ -n "${worker_tasks}" ]] || continue

    gpu_start=$(( i * WORKER_NUM_GPUS ))
    gpu_end=$(( gpu_start + WORKER_NUM_GPUS - 1 ))
    worker_gpu_list="$(IFS=,; echo "${GPUS[*]:gpu_start:WORKER_NUM_GPUS}")"
    worker_tag="${RUN_TAG_BASE}-w$(printf "%02d" "${i}")"
    worker_log="${WORKER_ROOT}/${METHOD_PATH_KEY}/${worker_tag}.log"
    worker_out_dir="${WORKER_ROOT}/${METHOD_PATH_KEY}/${worker_tag}"

    WORKER_OUT_DIRS+=("${worker_out_dir}")
    WORKER_TASK_ASSIGNMENTS+=("${worker_tasks}")

    mkdir -p "${WORKER_ROOT}/${METHOD_PATH_KEY}"
    mkdir -p "${worker_out_dir}"

    IFS=',' read -r -a one_worker_tasks <<< "${worker_tasks}"
    for task in "${one_worker_tasks[@]}"; do
        task_out="${worker_out_dir}/${task}.jsonl"
        if [[ ! -f "${task_out}" && -n "${EXISTING_TASK_OUTPUTS[${task}]+x}" ]]; then
            src_existing="${EXISTING_TASK_OUTPUTS[${task}]}"
            if [[ -f "${src_existing}" ]]; then
                cp "${src_existing}" "${task_out}"
                echo "[INFO] worker=${i} reused task=${task} from ${src_existing}"
            fi
        fi
    done

    echo "[INFO] worker=${i} gpus=${worker_gpu_list} tasks=${worker_tasks} output=${worker_out_dir} max_reuse_count=${FIXED_MAX_REUSE_COUNT}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        AUTO_SELECT_GPUS=0 \
        CUDA_VISIBLE_DEVICES="${worker_gpu_list}" \
        ENABLE_MAX_REUSE_COUNT="${ENABLE_MAX_REUSE_COUNT}" \
        METHOD="${METHOD}" \
        DATASET_PATH="${DATASET_PATH}" \
        TASKS="${worker_tasks}" \
        OUTPUT_ROOT="${WORKER_ROOT}" \
        RUN_TAG="${worker_tag}" \
        FIXED_MAX_REUSE_COUNT="${FIXED_MAX_REUSE_COUNT}" \
        RESUME="${RESUME}" \
        NUM_GPUS="${WORKER_NUM_GPUS}" MP_NUM="${WORKER_MP_NUM}" PP_NUM="${WORKER_PP_NUM}" DRY_RUN=1 \
            "${BASE_RUNNER}"
    else
        if [[ "${RESUME}" == "1" ]]; then
            echo "[INFO] worker=${i} RESUME=1: launch worker to verify task completeness and continue unfinished samples"
        fi
        (
            AUTO_SELECT_GPUS=0 \
            CUDA_VISIBLE_DEVICES="${worker_gpu_list}" \
            ENABLE_MAX_REUSE_COUNT="${ENABLE_MAX_REUSE_COUNT}" \
            METHOD="${METHOD}" \
            DATASET_PATH="${DATASET_PATH}" \
            TASKS="${worker_tasks}" \
            OUTPUT_ROOT="${WORKER_ROOT}" \
            RUN_TAG="${worker_tag}" \
            FIXED_MAX_REUSE_COUNT="${FIXED_MAX_REUSE_COUNT}" \
            RESUME="${RESUME}" \
            NUM_GPUS="${WORKER_NUM_GPUS}" MP_NUM="${WORKER_MP_NUM}" PP_NUM="${WORKER_PP_NUM}" DRY_RUN=0 \
                "${BASE_RUNNER}" > "${worker_log}" 2>&1
        ) &
        PIDS+=("$!")
    fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] would merge task jsonl files into ${FINAL_OUTPUT_DIR}"
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

for i in "${!WORKER_OUT_DIRS[@]}"; do
    worker_out_dir="${WORKER_OUT_DIRS[$i]}"
    worker_tasks="${WORKER_TASK_ASSIGNMENTS[$i]}"
    IFS=',' read -r -a one_worker_tasks <<< "${worker_tasks}"
    for task in "${one_worker_tasks[@]}"; do
        src="${worker_out_dir}/${task}.jsonl"
        dst="${FINAL_OUTPUT_DIR}/${task}.jsonl"
        [[ -f "${src}" ]] || { echo "[ERROR] missing worker prediction: ${src}" >&2; exit 1; }
        cp "${src}" "${dst}"
    done
done

python3 "${EVAL_PY}" --model "${FINAL_OUTPUT_DIR}" | tee -a "${FINAL_OUTPUT_DIR}/eval.log"
RESULT_JSON="${FINAL_OUTPUT_DIR}/result.json"
[[ -f "${RESULT_JSON}" ]] || { echo "[ERROR] missing merged result json: ${RESULT_JSON}" >&2; exit 1; }

python3 "${SUMMARIZE_PY}" \
    "${RESULT_JSON}" \
    "${BASELINE_RESULT}" \
    "${MAX_AVG_DROP}" \
    "${MAX_SINGLE_DROP}"

echo "[DONE] ${N_WAY}-way task-parallel run finished. result=${RESULT_JSON}"
