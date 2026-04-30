#!/usr/bin/env bash

set -euo pipefail
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,SHM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export AUTO_SELECT_GPUS=0

export METHOD="${METHOD:-offloading}"
export MODEL_PATH="${MODEL_PATH:-/jhe/Qwen2.5-14B-Instruct-1M}"
export DATASET_PATH="${DATASET_PATH:-/jhe/dataset/LongBench-v2}"

export N_WAY="${N_WAY:-4}"
export WORKER_NUM_GPUS="${WORKER_NUM_GPUS:-2}"
export WORKER_MP_NUM="${WORKER_MP_NUM:-1}"
export WORKER_PP_NUM="${WORKER_PP_NUM:-2}"

export FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER:-1.0}"
export FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS:-48}"

export TOPK="${TOPK:-0.10}"
export CONFIG_SUFFIX="${CONFIG_SUFFIX:-128K}"
export SELECTIVE_START_LEN="${SELECTIVE_START_LEN:-2048}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/preds_nway}"
export RESUME="${RESUME:-0}"
export DRY_RUN="${DRY_RUN:-0}"

exec "${SCRIPT_DIR}/run_longbench_v2_nway.sh" "$@"
