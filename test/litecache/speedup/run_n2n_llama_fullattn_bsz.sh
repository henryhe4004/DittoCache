#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/jhe/Llama-3-8B-Instruct-Gradient-1048k}"
CONFIG_ROOT="${CONFIG_ROOT:-${SCRIPT_DIR}/../config/full_attn}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs-cudagraph}"

METHODS="${METHODS:-flashattn}"
SEQ_LIST="${SEQ_LIST:-8000}"
BSZ_LIST="${BSZ_LIST:-1 2 4 8 16 32}"
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
    for bsz in ${BSZ_LIST}; do
      config_tokens=$((seq_in_k * bsz))
      config_file="${CONFIG_ROOT}/Llama-3-8B-Instruct-Gradient-1048k-${config_tokens}K.yaml"
      data_file="${DATA_ROOT}/RULER-Llama-3-8B-Instruct-Gradient-1048k-${seq_in_k}K.jsonl"
      run_name="llama3-8b-1048k-${method}-bsz${bsz}-seq${seq_in_k}K"
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
        --batch_size "${bsz}"
        --max_seq_len "${seq}"
        --max-running-requests "${bsz}"
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
done
