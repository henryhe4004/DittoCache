#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-14B-Instruct-1M}"
CONFIG_ROOT="${CONFIG_ROOT:-${SCRIPT_DIR}/tmp_config_B2}"
DATA_ROOT="${DATA_ROOT:-/workspace/jhe/sglang-litecache/test/ditto/speedup/data}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs-batchless-B2}"
PROFILE="B2"

METHODS="${METHODS:-offloading}"
SEQ_LIST="${SEQ_LIST:-8000}"
BSZ_LIST="${BSZ_LIST:-64}"
TOPK="${TOPK:-0.10}"
DECODE_STEPS="${DECODE_STEPS:-50}"
WARMUP="${WARMUP:-1}"
EPOCH="${EPOCH:-3}"
CPUSET="${CPUSET:-0-31}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.92}"
SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-warning}"
DISABLE_SGLANG_BATCH_LOG="${DISABLE_SGLANG_BATCH_LOG:-1}"
# Keep SGLang global cuda graph off by default (can hang in offloading path),
# while keeping Ditto cuda graph on to match internal prototype configs.
SGLANG_CUDA_GRAPH="${SGLANG_CUDA_GRAPH:-0}"
DITTO_CUDA_GRAPH="${DITTO_CUDA_GRAPH:-0}"

mkdir -p "${LOG_DIR}"

for method in ${METHODS}; do
  for seq in ${SEQ_LIST}; do
    seq_in_k=$((seq / 1000))
    for bsz in ${BSZ_LIST}; do
      config_tokens=$((seq_in_k * bsz))
      config_file="${CONFIG_ROOT}/Qwen2.5-14B-Instruct-1M-${config_tokens}K.yaml"
      data_file="${DATA_ROOT}/RULER-Qwen2.5-14B-Instruct-1M-${seq_in_k}K.jsonl"
      run_name="${PROFILE}-qwen2.5-14b-1m-${method}-bsz${bsz}-seq${seq_in_k}K-topk${TOPK}"
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
        --mem-fraction-static "${MEM_FRACTION_STATIC}"
        --result-json "${result_json}"
      )
      if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
        cmd+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
      fi
      if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
        cmd+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
      fi
      if [[ -n "${PROFILE_RESERVE_RATIO:-}" ]]; then
        cmd+=(--profile-reserve-ratio "${PROFILE_RESERVE_RATIO}")
      fi
      if [[ "${ALLOW_AUTO_TRUNCATE:-0}" == "1" ]]; then
        cmd+=(--allow-auto-truncate)
      fi
      if [[ -n "${SGLANG_LOG_LEVEL}" ]]; then
        cmd+=(--sglang-log-level "${SGLANG_LOG_LEVEL}")
      fi
      if [[ "${DISABLE_SGLANG_BATCH_LOG}" == "1" ]]; then
        cmd+=(--disable-sglang-batch-log)
      fi
      if [[ "${SGLANG_CUDA_GRAPH}" == "1" ]]; then
        cmd+=(--enable-cuda-graph)
      else
        cmd+=(--disable-cuda-graph)
      fi
      if [[ "${DITTO_CUDA_GRAPH}" == "1" ]]; then
        cmd+=(--ditto-enable-cuda-graph)
      else
        cmd+=(--ditto-disable-cuda-graph)
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
