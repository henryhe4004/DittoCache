#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="${SCRIPT_DIR}"
PROFILE_SCRIPT="${SCRIPT_DIR}/profile_heads_cosine.py"

MODEL_PATH="${MODEL_PATH:-/jhe/Qwen2.5-14B-Instruct-AWQ}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}")}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/${MODEL_TAG}}"
BASE_PATTERN_DIR="${BASE_PATTERN_DIR:-${SCRIPT_DIR}/Qwen2.5-14B-Instruct-1M}"

DATASET_PATH="${DATASET_PATH:-}"
if [[ -z "${DATASET_PATH}" ]]; then
    if [[ -d "/jhe/dataset/LongBench" ]]; then
        DATASET_PATH="/jhe/dataset/LongBench"
    elif [[ -d "/jhe/LongBench" ]]; then
        DATASET_PATH="/jhe/LongBench"
    else
        DATASET_PATH="/nfs/shared_LLM_dataset/LongBench"
    fi
fi

NUM_SAMPLES="${NUM_SAMPLES:-10}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-65536}"
PP_NUM="${PP_NUM:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUTPUT_DIR}"

Q_IMPORTANCE="${OUTPUT_DIR}/q_heads_importance.tsv"
K_IMPORTANCE="${OUTPUT_DIR}/k_heads_importance.tsv"

CMD=(
    python "${PROFILE_SCRIPT}"
    --model "${MODEL_PATH}"
    --dataset_path "${DATASET_PATH}"
    --num_samples "${NUM_SAMPLES}"
    --max_context_length "${MAX_CONTEXT_LENGTH}"
    --pp_num "${PP_NUM}"
)

echo "[INFO] profile AWQ attention pattern"
echo "[INFO] model=${MODEL_PATH}"
echo "[INFO] dataset_path=${DATASET_PATH}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] num_samples=${NUM_SAMPLES} max_context_length=${MAX_CONTEXT_LENGTH} pp_num=${PP_NUM}"

if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ ! -f "${Q_IMPORTANCE}" ]]; then
        echo "[DRY_RUN] cp ${BASE_PATTERN_DIR}/q_heads_importance.tsv ${Q_IMPORTANCE}"
    fi
    if [[ ! -f "${K_IMPORTANCE}" ]]; then
        echo "[DRY_RUN] cp ${BASE_PATTERN_DIR}/k_heads_importance.tsv ${K_IMPORTANCE}"
    fi
    echo "[DRY_RUN] (cd \"${PROFILE_DIR}\" && ${CMD[*]})"
    echo "[DRY_RUN] cp ${PROFILE_DIR}/${MODEL_TAG}/q_heads_cosine_similarity.csv ${OUTPUT_DIR}/heads_cosine_similarity.csv"
    exit 0
fi

if [[ ! -f "${Q_IMPORTANCE}" ]]; then
    cp "${BASE_PATTERN_DIR}/q_heads_importance.tsv" "${Q_IMPORTANCE}"
fi
if [[ ! -f "${K_IMPORTANCE}" ]]; then
    cp "${BASE_PATTERN_DIR}/k_heads_importance.tsv" "${K_IMPORTANCE}"
fi

(
    cd "${PROFILE_DIR}"
    "${CMD[@]}"
)

SRC_COSINE="${PROFILE_DIR}/${MODEL_TAG}/q_heads_cosine_similarity.csv"
[[ -f "${SRC_COSINE}" ]] || {
    echo "[ERROR] missing generated cosine file: ${SRC_COSINE}" >&2
    exit 1
}
cp "${SRC_COSINE}" "${OUTPUT_DIR}/heads_cosine_similarity.csv"

echo "[INFO] generated ${OUTPUT_DIR}/heads_cosine_similarity.csv"

