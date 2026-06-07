#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# You can override these envs when invoking the script.
export METHOD="${METHOD:-offloading}"
export MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-14B-Instruct-1M}"
export MODEL_NAME="${MODEL_NAME:-Qwen2.5-14B-Instruct-1M}"
export CONFIG_SUFFIX="${CONFIG_SUFFIX:-32K}"
export TOPK="${TOPK:-0.10}"
export SELECTIVE_START_LEN="${SELECTIVE_START_LEN:-512}"

# Optional: tune dataset/tasks if needed.
export DATASET_NAME="${DATASET_NAME:-longbench}"
export TASKS="${TASKS:-lcc_e,repobench-p_e,qasper_e,multifieldqa_en_e,hotpotqa_e,2wikimqa_e,trec_e,triviaqa_e,samsum_e,passage_count_e,passage_retrieval_en_e,gov_report_e,multi_news_e}"

# Runtime
export BATCH_SIZE="${BATCH_SIZE:-1}"
export MP_NUM="${MP_NUM:-1}"
export PP_NUM="${PP_NUM:-1}"
export AUTO_SELECT_GPUS="${AUTO_SELECT_GPUS:-1}"
export MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-20000}"
export ENGINE_QUANTIZATION="${ENGINE_QUANTIZATION:-auto}"

# Logging / observability
export DITTO_RECORD_TRANSFER_STATS="${DITTO_RECORD_TRANSFER_STATS:-1}"
export DITTO_RECORD_OVERLAP_STATS="${DITTO_RECORD_OVERLAP_STATS:-1}"
export RUN_TAG="${RUN_TAG:-Qwen2.5-14B-Instruct-1M-offloading-${CONFIG_SUFFIX}-top${TOPK}}"

bash "${SCRIPT_DIR}/test_accuracy.sh"
