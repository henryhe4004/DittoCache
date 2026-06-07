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

DATASET_PATH="${DATASET_PATH:-/datasets/LongBench}"
if [[ ! -d "${DATASET_PATH}" && -d "/datasets/LongBench" ]]; then
    DATASET_PATH="/datasets/LongBench"
fi

resolve_longbench_task_root() {
    local base="${1:-}"
    if [[ -f "${base}/lcc_e.jsonl" ]]; then
        echo "${base}"
    elif [[ -f "${base}/data/lcc_e.jsonl" ]]; then
        echo "${base}/data"
    else
        echo "${base}"
    fi
}
DATASET_TASK_ROOT="$(resolve_longbench_task_root "${DATASET_PATH}")"

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
CONFIG_SUFFIX="${CONFIG_SUFFIX:-64K}"
SELECTIVE_START_LEN="${SELECTIVE_START_LEN:-2048}"
TASKS="${TASKS:-lcc_e,repobench-p_e,qasper_e,multifieldqa_en_e,hotpotqa_e,2wikimqa_e,trec_e,triviaqa_e,samsum_e,passage_count_e,passage_retrieval_en_e,gov_report_e,multi_news_e}"
FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS:-48}"
FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER:-1}"
FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_LOWER:-${FIXED_REUSE_THRESHOLD_UPPER}}"

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

normalize_longbench_task() {
    local t="${1:-}"
    case "${t}" in
        multinews) echo "multi_news" ;;
        mulitinews) echo "multi_news" ;;
        multinews_e) echo "multi_news_e" ;;
        mulitinews_e) echo "multi_news_e" ;;
        *) echo "${t}" ;;
    esac
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
RUN_TAG_BASE="${RUN_TAG_BASE:-${MODEL_NAME}-longbench-top${TOPK}-nway}"
WORKER_ROOT="${OUTPUT_ROOT}/workers"
FINAL_OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD_PATH_KEY}/${RUN_TAG_BASE}"

BASELINE_RESULT="${BASELINE_RESULT:-}"
MAX_AVG_DROP="${MAX_AVG_DROP:-3.0}"
MAX_SINGLE_DROP="${MAX_SINGLE_DROP:-8.0}"

[[ -d "${DATASET_PATH}" ]] || { echo "[ERROR] missing dataset dir: ${DATASET_PATH}" >&2; exit 1; }
[[ -d "${DATASET_TASK_ROOT}" ]] || { echo "[ERROR] missing task root dir: ${DATASET_TASK_ROOT}" >&2; exit 1; }

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
    t="$(normalize_longbench_task "${t}")"
    if [[ -z "${TASK_SEEN[${t}]+x}" ]]; then
        TASK_LIST+=("${t}")
        TASK_SEEN["${t}"]=1
    fi
    if [[ ! -f "${DATASET_TASK_ROOT}/${t}.jsonl" ]]; then
        echo "[ERROR] missing task file: ${DATASET_TASK_ROOT}/${t}.jsonl" >&2
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

echo "[INFO] Start ${N_WAY}-way task-parallel LongBench run"
echo "[INFO] method=${METHOD} config_suffix=${CONFIG_SUFFIX} topk=${TOPK} selective_start_len=${SELECTIVE_START_LEN}"
echo "[INFO] effective knobs: th_up=${FIXED_REUSE_THRESHOLD_UPPER} th_lo=${FIXED_REUSE_THRESHOLD_LOWER} skip_layers=${FIXED_NUM_SKIP_LAYERS}"
echo "[INFO] worker topology: worker_num_gpus=${WORKER_NUM_GPUS} mp=${WORKER_MP_NUM} pp=${WORKER_PP_NUM}"
echo "[INFO] total tasks=${TASK_COUNT} tasks=${TASKS_CLEANED}"

declare -a PIDS=()
declare -a WORKER_OUT_DIRS=()
declare -a WORKER_TASK_ASSIGNMENTS=()
declare -A EXISTING_TASK_OUTPUTS=()

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

    echo "[INFO] worker=${i} gpus=${worker_gpu_list} tasks=${worker_tasks} output=${worker_out_dir}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        AUTO_SELECT_GPUS=0 \
        CUDA_VISIBLE_DEVICES="${worker_gpu_list}" \
        METHOD="${METHOD}" \
        DATASET_NAME=longbench \
        DATASET_PATH="${DATASET_PATH}" \
        TASKS="${worker_tasks}" \
        OUTPUT_ROOT="${WORKER_ROOT}" \
        RUN_TAG="${worker_tag}" \
        CONFIG_SUFFIX="${CONFIG_SUFFIX}" \
        SELECTIVE_START_LEN="${SELECTIVE_START_LEN}" \
        FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS}" \
        FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER}" \
        FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_LOWER}" \
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
            DATASET_NAME=longbench \
            DATASET_PATH="${DATASET_PATH}" \
            TASKS="${worker_tasks}" \
            OUTPUT_ROOT="${WORKER_ROOT}" \
            RUN_TAG="${worker_tag}" \
            CONFIG_SUFFIX="${CONFIG_SUFFIX}" \
            SELECTIVE_START_LEN="${SELECTIVE_START_LEN}" \
            FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS}" \
            FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER}" \
            FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_LOWER}" \
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
    echo "[DRY_RUN] would merge task jsonl files into ${FINAL_OUTPUT_DIR}"
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

python3 "${EVAL_PY}" --model "${FINAL_OUTPUT_DIR}" --e | tee -a "${FINAL_OUTPUT_DIR}/eval.log"
RESULT_JSON="${FINAL_OUTPUT_DIR}/result.json"
[[ -f "${RESULT_JSON}" ]] || { echo "[ERROR] missing merged result json: ${RESULT_JSON}" >&2; exit 1; }

python3 "${SUMMARIZE_PY}" \
    "${RESULT_JSON}" \
    "${BASELINE_RESULT}" \
    "${MAX_AVG_DROP}" \
    "${MAX_SINGLE_DROP}"

echo "[DONE] ${N_WAY}-way task-parallel LongBench run finished. result=${RESULT_JSON}"
