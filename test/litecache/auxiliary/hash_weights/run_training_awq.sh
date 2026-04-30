#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/dataset/Qwen2.5-14B-Instruct-AWQ}"
SAVE_PATH="${SAVE_PATH:-${SCRIPT_DIR}/Qwen2.5-14B-Instruct-AWQ-256}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
DRY_RUN="${DRY_RUN:-0}"

NUM_LAYERS="${NUM_LAYERS:-48}"
NUM_SKIP_LAYERS="${NUM_SKIP_LAYERS:-0}"
NUM_HEADS="${NUM_HEADS:-40}"
NUM_KV_HEADS="${NUM_KV_HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-128}"
RBIT="${RBIT:-256}"
CHUNK_NUM="${CHUNK_NUM:-3}"
MP_NUM="${MP_NUM:-8}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-15}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
REP_ITERS="${REP_ITERS:-10}"
LR="${LR:-0.1}"
EPSILON="${EPSILON:-0.01}"
LAMBDA="${LAMBDA:-1.0}"
ETA="${ETA:-2.0}"
SIGMA="${SIGMA:-0.1}"

mkdir -p "${SAVE_PATH}"

CMD=(
    python "${SCRIPT_DIR}/learn_hash_weights.py"
    --dataset_path "${DATASET_PATH}"
    --save_path "${SAVE_PATH}"
    --num_layers "${NUM_LAYERS}"
    --num_skip_layers "${NUM_SKIP_LAYERS}"
    --num_heads "${NUM_HEADS}"
    --num_kv_heads "${NUM_KV_HEADS}"
    --head_dim "${HEAD_DIM}"
    --rbit "${RBIT}"
    --chunk_num "${CHUNK_NUM}"
    --mp_num "${MP_NUM}"
    --train_epochs "${TRAIN_EPOCHS}"
    --train_iters "${TRAIN_ITERS}"
    --rep_iters "${REP_ITERS}"
    --lr "${LR}"
    --epsilon "${EPSILON}"
    --lambdda "${LAMBDA}"
    --eta "${ETA}"
    --sigma "${SIGMA}"
)

echo "[INFO] train AWQ hash weights"
echo "[INFO] dataset_path=${DATASET_PATH}"
echo "[INFO] save_path=${SAVE_PATH}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES}\" ${CMD[*]}"
    exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"

