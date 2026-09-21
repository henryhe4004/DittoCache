#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_MODEL_PATH="/models/Qwen2.5-14B-Instruct-1M"
if [[ ! -d "${DEFAULT_MODEL_PATH}" && -d "/jhe/Qwen2.5-14B-Instruct-1M" ]]; then
  DEFAULT_MODEL_PATH="/jhe/Qwen2.5-14B-Instruct-1M"
fi
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
FULLATTN_CONFIG_ROOT="${FULLATTN_CONFIG_ROOT:-${SCRIPT_DIR}/../config/full_attn}"
OFFLOAD_CONFIG_ROOT="${OFFLOAD_CONFIG_ROOT:-${SCRIPT_DIR}/../config/hata_offloading}"
DEFAULT_DATA_ROOT="${SCRIPT_DIR}/data"
if [[ ! -d "${DEFAULT_DATA_ROOT}" && -d "/jhe/myTransformer/speedup/data" ]]; then
  DEFAULT_DATA_ROOT="/jhe/myTransformer/speedup/data"
fi
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
DEFAULT_ATTN_PATTERN_PATH="${SCRIPT_DIR}/../auxiliary/attn_pattern/Qwen2.5-14B-Instruct-1M"
if [[ ! -f "${DEFAULT_ATTN_PATTERN_PATH}/heads_cosine_similarity.csv" && \
      -f "/jhe/myTransformer/auxiliary/attn_pattern/Qwen2.5-14B-Instruct-1M/heads_cosine_similarity.csv" ]]; then
  DEFAULT_ATTN_PATTERN_PATH="/jhe/myTransformer/auxiliary/attn_pattern/Qwen2.5-14B-Instruct-1M"
fi
ATTN_PATTERN_PATH="${ATTN_PATTERN_PATH:-${DEFAULT_ATTN_PATTERN_PATH}}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs-ablation}"

STAGES="${STAGES:-fullattn b0_memcpy b1_gdr b2_prefetch b3_qsac_fixed b4_cudagraph b5_adaptive b6_resident}"
SEQ_LIST="${SEQ_LIST:-8000}"
BSZ_LIST="${BSZ_LIST:-1 2 4 8 16}"
TOPK="${TOPK:-0.10}"
DECODE_STEPS="${DECODE_STEPS:-50}"
WARMUP="${WARMUP:-1}"
EPOCH="${EPOCH:-3}"
CPUSET="${CPUSET:-144-191}"
CUDA_DEVICE="${CUDA_DEVICE:-5}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.92}"
GPU_MEMORY_BUDGET="${GPU_MEMORY_BUDGET:-16}"
SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-warning}"
FULLATTN_CUDA_GRAPH="${FULLATTN_CUDA_GRAPH:-0}"
RECORD_TRANSFER_STATS="${RECORD_TRANSFER_STATS:-1}"

# The stage config is authoritative. Legacy import-time switches would make
# separate stages silently collapse onto the same behavior.
unset USE_FIXED_THRESHOLDS
unset USE_INTRA_GQA_AGGREGATION
unset DISABLE_PERSISTENT_CACHING
unset DITTO_DISABLE_PREFETCH

mkdir -p "${LOG_DIR}"

for stage in ${STAGES}; do
  case "${stage}" in
    fullattn)
      method="flashattn"
      config_root="${FULLATTN_CONFIG_ROOT}"
      ;;
    b0_memcpy|b1_gdr|b2_prefetch|b3_qsac_fixed|b4_cudagraph|b5_adaptive|b6_resident)
      method="offloading"
      config_root="${OFFLOAD_CONFIG_ROOT}"
      ;;
    *)
      echo "[ERROR] unknown ablation stage: ${stage}" >&2
      exit 2
      ;;
  esac

  for seq in ${SEQ_LIST}; do
    seq_in_k=$((seq / 1000))
    for bsz in ${BSZ_LIST}; do
      config_tokens=$((seq_in_k * bsz))
      config_file="${config_root}/Qwen2.5-14B-Instruct-1M-${config_tokens}K.yaml"
      data_file="${DATA_ROOT}/RULER-Qwen2.5-14B-Instruct-1M-${seq_in_k}K.jsonl"
      run_name="qwen2.5-14b-1m-${stage}-bsz${bsz}-seq${seq_in_k}K-topk${TOPK}"
      log_file="${LOG_DIR}/${run_name}.log"
      result_json="${LOG_DIR}/${run_name}.json"

      if [[ ! -f "${data_file}" ]]; then
        echo "[ERROR] missing data file: ${data_file}" >&2
        exit 2
      fi
      if [[ ! -f "${config_file}" ]]; then
        echo "[ERROR] missing config file: ${config_file}" >&2
        exit 2
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
        --gpu-memory-budget "${GPU_MEMORY_BUDGET}"
        --sglang-log-level "${SGLANG_LOG_LEVEL}"
        --disable-sglang-batch-log
        --result-json "${result_json}"
      )

      if [[ "${stage}" == "fullattn" ]]; then
        if [[ "${FULLATTN_CUDA_GRAPH}" == "1" ]]; then
          cmd+=(--enable-cuda-graph)
        else
          cmd+=(--disable-cuda-graph)
        fi
      else
        cmd+=(
          --ablation-stage "${stage}"
          --attn-pattern-path "${ATTN_PATTERN_PATH}"
          --disable-cuda-graph
        )
        if [[ "${RECORD_TRANSFER_STATS}" == "1" ]]; then
          cmd+=(--record-transfer-stats)
        fi
      fi
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
