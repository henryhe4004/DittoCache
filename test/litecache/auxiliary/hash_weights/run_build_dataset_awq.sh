#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/jhe/Qwen2.5-14B-Instruct-AWQ}"
SAVE_PATH="${SAVE_PATH:-${SCRIPT_DIR}/dataset/Qwen2.5-14B-Instruct-AWQ}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-65536}"
POS_SAMPLE_RATIO="${POS_SAMPLE_RATIO:-0.1}"
PP_NUM="${PP_NUM:-1}"
APPLY_TEMPLATE="${APPLY_TEMPLATE:-1}"
MAX_PROMPT_CHARS="${MAX_PROMPT_CHARS:-0}"
DRY_RUN="${DRY_RUN:-0}"

LONGBENCH_PATH="${LONGBENCH_PATH:-}"
if [[ -z "${LONGBENCH_PATH}" ]]; then
    if [[ -d "/jhe/dataset/LongBench/data" ]]; then
        LONGBENCH_PATH="/jhe/dataset/LongBench/data"
    elif [[ -d "/jhe/LongBench/data" ]]; then
        LONGBENCH_PATH="/jhe/LongBench/data"
    else
        LONGBENCH_PATH="/nfs/shared_LLM_dataset/LongBench/data"
    fi
fi

LONGBENCH_V2_PATH="${LONGBENCH_V2_PATH:-}"
if [[ -z "${LONGBENCH_V2_PATH}" ]]; then
    if [[ -d "/jhe/dataset/LongBench-v2" ]]; then
        LONGBENCH_V2_PATH="/jhe/dataset/LongBench-v2"
    else
        LONGBENCH_V2_PATH="/nfs/shared_LLM_dataset/LongBench-v2"
    fi
fi

mkdir -p "${SAVE_PATH}"

CMD=(
    python "${SCRIPT_DIR}/build_dataset.py"
    --model "${MODEL_PATH}"
    --save_path "${SAVE_PATH}"
    --longbench_path "${LONGBENCH_PATH}"
    --longbench_v2_path "${LONGBENCH_V2_PATH}"
    --max_context_length "${MAX_CONTEXT_LENGTH}"
    --pos_sample_ratio "${POS_SAMPLE_RATIO}"
    --max_prompt_chars "${MAX_PROMPT_CHARS}"
    --pp_num "${PP_NUM}"
)

if [[ "${APPLY_TEMPLATE}" == "1" ]]; then
    CMD+=(--apply_template)
fi

echo "[INFO] build AWQ hash dataset"
echo "[INFO] model=${MODEL_PATH}"
echo "[INFO] save_path=${SAVE_PATH}"
echo "[INFO] longbench=${LONGBENCH_PATH}"
echo "[INFO] longbench_v2=${LONGBENCH_V2_PATH}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] ${CMD[*]}"
    exit 0
fi

"${CMD[@]}"

