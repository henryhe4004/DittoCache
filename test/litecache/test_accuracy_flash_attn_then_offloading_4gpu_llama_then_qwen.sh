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

BASE_SCRIPT="${SCRIPT_DIR}/test_accuracy_flash_attn_then_offloading_4gpu.sh"
[[ -x "${BASE_SCRIPT}" ]] || die "missing executable base script: ${BASE_SCRIPT}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
FIRST_METHOD="${FIRST_METHOD:-flash-attn}"
SECOND_METHOD="${SECOND_METHOD:-offloading}"
TOPK="${TOPK:-0.10}"
SELECTIVE_START_LEN="${SELECTIVE_START_LEN:-512}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/preds}"
DRY_RUN="${DRY_RUN:-0}"

LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-/jhe/Llama-3-8B-Instruct-Gradient-1048k}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/jhe/Qwen2.5-14B-Instruct-1M}"

[[ -e "${LLAMA_MODEL_PATH}" ]] || die "missing llama model: ${LLAMA_MODEL_PATH}"
[[ -e "${QWEN_MODEL_PATH}" ]] || die "missing qwen model: ${QWEN_MODEL_PATH}"

run_model_suite() {
    local model_path="$1"
    local model_name
    model_name="$(basename "${model_path}")"

    log_info "starting model suite model=${model_name} gpus=${GPU_IDS} methods=${FIRST_METHOD}->${SECOND_METHOD}"

    (
        export MODEL_PATH="${model_path}"
        export MODEL_TAG="${model_name}"
        export GPU_IDS="${GPU_IDS}"
        export FIRST_METHOD="${FIRST_METHOD}"
        export SECOND_METHOD="${SECOND_METHOD}"
        export TOPK="${TOPK}"
        export SELECTIVE_START_LEN="${SELECTIVE_START_LEN}"
        export OUTPUT_ROOT="${OUTPUT_ROOT}"
        export DRY_RUN="${DRY_RUN}"
        exec "${BASE_SCRIPT}"
    )

    log_info "completed model suite model=${model_name}"
}

run_model_suite "${LLAMA_MODEL_PATH}"
run_model_suite "${QWEN_MODEL_PATH}"

log_info "all model suites completed"
