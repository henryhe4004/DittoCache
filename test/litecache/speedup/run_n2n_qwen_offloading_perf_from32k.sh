#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Fixed target from your request
MODEL_PATH="${MODEL_PATH:-/jhe/Qwen2.5-14B-Instruct-1M}"
METHOD="${METHOD:-offloading}"

CONFIG_ROOT="${CONFIG_ROOT:-${SCRIPT_DIR}/../config/hata_offloading}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs-perf-from32k-corr0.8}"
CSV_SUMMARY="${CSV_SUMMARY:-${LOG_DIR}/summary.csv}"
EXPORT_CSV="${EXPORT_CSV:-1}"

# Start from 32K
SEQ_LIST="${SEQ_LIST:-64000}"
BSZ="${BSZ:-1}"
TOPK="${TOPK:-0.10}"
DECODE_STEPS="${DECODE_STEPS:-50}"
WARMUP="${WARMUP:-1}"
EPOCH="${EPOCH:-3}"

CUDA_DEVICE="${CUDA_DEVICE:-7}"
CPUSET="${CPUSET:-}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.92}"
SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-warning}"
DISABLE_SGLANG_BATCH_LOG="${DISABLE_SGLANG_BATCH_LOG:-1}"
SGLANG_CUDA_GRAPH="${SGLANG_CUDA_GRAPH:-0}"
LITECACHE_CUDA_GRAPH="${LITECACHE_CUDA_GRAPH:-0}"

# Enable transfer + hit-rate related stats
LITECACHE_RECORD_OVERLAP_STATS="${LITECACHE_RECORD_OVERLAP_STATS:-1}"

mkdir -p "${LOG_DIR}"

for seq in ${SEQ_LIST}; do
  seq_in_k=$((seq / 1000))
  config_tokens=$((seq_in_k * BSZ))
  config_file="${CONFIG_ROOT}/Qwen2.5-14B-Instruct-1M-${config_tokens}K.yaml"
  data_file="${DATA_ROOT}/RULER-Qwen2.5-14B-Instruct-1M-${seq_in_k}K.jsonl"

  run_name="qwen2.5-14b-1m-${METHOD}-perf-bsz${BSZ}-seq${seq_in_k}K-topk${TOPK}"
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
    --method "${METHOD}"
    --topk "${TOPK}"
    --batch_size "${BSZ}"
    --max_seq_len "${seq}"
    --max-running-requests "${BSZ}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --record-transfer-stats
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
  if [[ "${LITECACHE_CUDA_GRAPH}" == "1" ]]; then
    cmd+=(--litecache-enable-cuda-graph)
  else
    cmd+=(--litecache-disable-cuda-graph)
  fi

  echo "[RUN] ${run_name}"
  if [[ -n "${CPUSET}" ]]; then
    LITECACHE_RECORD_OVERLAP_STATS="${LITECACHE_RECORD_OVERLAP_STATS}" \
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    taskset -c "${CPUSET}" "${cmd[@]}" 2>&1 | tee "${log_file}"
  else
    LITECACHE_RECORD_OVERLAP_STATS="${LITECACHE_RECORD_OVERLAP_STATS}" \
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    "${cmd[@]}" 2>&1 | tee "${log_file}"
  fi
done

if [[ "${EXPORT_CSV}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/export_speedup_csv.py" \
    --input-dir "${LOG_DIR}" \
    --glob "*.json" \
    --output-csv "${CSV_SUMMARY}"
  echo "[DONE] csv summary: ${CSV_SUMMARY}"
fi
