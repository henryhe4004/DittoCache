#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/jhe/Qwen2.5-14B-Instruct-1M}"
CONFIG_ROOT="${CONFIG_ROOT:-/jhe/myTransformer/config/full_attn}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs-cudagraph}"

METHODS="${METHODS:-flashattn}"
SEQ_LIST="${SEQ_LIST:-4000 8000 16000 32000 64000 128000}"
BSZ="${BSZ:-1}"
TOPK="${TOPK:-0.10}"
DECODE_STEPS="${DECODE_STEPS:-50}"
WARMUP="${WARMUP:-1}"
EPOCH="${EPOCH:-3}"
CPUSET="${CPUSET:-0-95}"
CUDA_DEVICE="${CUDA_DEVICE:-6}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"

mkdir -p "${LOG_DIR}"

for method in ${METHODS}; do
  for seq in ${SEQ_LIST}; do
    seq_in_k=$((seq / 1000))
    config_tokens=$((seq_in_k * BSZ))
    config_file="${CONFIG_ROOT}/Qwen2.5-14B-Instruct-1M-${config_tokens}K.yaml"
    data_file="${DATA_ROOT}/RULER-Qwen2.5-14B-Instruct-1M-${seq_in_k}K.jsonl"
    run_name="qwen2.5-14b-1m-${method}-bsz${BSZ}-seq${seq_in_k}K"
    log_file="${LOG_DIR}/${run_name}.log"
    result_json="${LOG_DIR}/${run_name}.json"

    if [[ ! -f "${data_file}" ]]; then
      echo "[WARN] missing data file: ${data_file}, skip"
      continue
    fi
    if [[ ! -f "${config_file}" ]]; then
      echo "[WARN] missing config file: ${config_file}, skip"
      continue
    fi

    cmd=(
      "${PYTHON_BIN}" "${SCRIPT_DIR}/n2n_offloading.py"
      --model "${MODEL_PATH}"
      --config_file "${config_file}"
      --data "${data_file}"
      --num_decode_steps "${DECODE_STEPS}"
      --warmup "${WARMUP}"
      --epoch "${EPOCH}"
      --method "${method}"
      --topk "${TOPK}"
      --batch_size "${BSZ}"
      --max_seq_len "${seq}"
      --max-running-requests "${BSZ}"
      --result-json "${result_json}"
    )
    if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
      cmd+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
    fi

    echo "[RUN] ${run_name}"
    if [[ -n "${CPUSET}" ]]; then
      CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" taskset -c "${CPUSET}" "${cmd[@]}" 2>&1 | tee "${log_file}"
    else
      CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${cmd[@]}" 2>&1 | tee "${log_file}"
    fi
  done
done
