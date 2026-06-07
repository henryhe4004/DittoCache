#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export AUTO_SELECT_GPUS=0

export METHOD="${METHOD:-offloading}"
export MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-14B-Instruct-1M}"

export NUM_GPUS="${NUM_GPUS:-4}"
export MP_NUM="${MP_NUM:-1}"
export PP_NUM="${PP_NUM:-4}"

export FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER:-1.0}"
export FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS:-48}"

export TOPK="${TOPK:-0.10}"
export SELECTIVE_START_LEN="${SELECTIVE_START_LEN:-2048}"
export DATASET_NAME="${DATASET_NAME:-infinitebench}"
export DATASET_PATH="${DATASET_PATH:-/datasets/InfiniteBench}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/preds}"
export RESUME="${RESUME:-0}"
export DRY_RUN="${DRY_RUN:-0}"

exec "${SCRIPT_DIR}/test_accuracy_infinitebench.sh" "$@"
