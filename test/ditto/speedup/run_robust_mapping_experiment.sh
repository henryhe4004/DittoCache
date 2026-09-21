#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv-py313/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv-py313/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
BENCH="${SCRIPT_DIR}/n2n_offloading.py"
OPTIMIZER="${SCRIPT_DIR}/build_tp_head_mapping_robust_refine.py"
SUMMARIZER="${SCRIPT_DIR}/summarize_robust_mapping_experiment.py"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

: "${MODEL_NAME:?Set MODEL_NAME to a stable experiment label}"
: "${MODEL_PATH:?Set MODEL_PATH to the model directory}"
: "${CONFIG_FILE:?Set CONFIG_FILE to the HATA YAML file}"
: "${DATA_FILE:?Set DATA_FILE to the benchmark JSONL file}"
: "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to a new or resumable output directory}"
: "${SEQ_LEN:?Set SEQ_LEN to the exact maximum sequence length}"

TOPK="${TOPK:-0.10}"
DECODE_STEPS="${DECODE_STEPS:-50}"
PROFILE_WARMUP="${PROFILE_WARMUP:-1}"
PROFILE_EPOCH="${PROFILE_EPOCH:-1}"
BENCH_WARMUP="${BENCH_WARMUP:-1}"
BENCH_EPOCH="${BENCH_EPOCH:-3}"
CPUSET="${CPUSET:-0-95}"
CUDA_DEVICE="${CUDA_DEVICE:-0,1}"
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-${SEQ_LEN}}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.92}"
MAX_CHANGES="${MAX_CHANGES:-5}"
MIN_SPLIT_IMPROVEMENT_PCT="${MIN_SPLIT_IMPROVEMENT_PCT:-4}"
SKIP_STEPS="${SKIP_STEPS:-10}"
SKIP_LAYERS="${SKIP_LAYERS:-1}"
RESIDENT_FILE="${RESIDENT_FILE:-}"
REFERENCE_MAPPING="${REFERENCE_MAPPING:-}"
PHASES="${PHASES:-profile generate benchmark summarize}"

PROFILE_DIR="${EXPERIMENT_DIR}/profile-linear"
TRANSFER_DIR="${PROFILE_DIR}/transfer"
TRANSFER_BASE="${TRANSFER_DIR}/profile-transfer.json"
PROFILE_RESULT="${PROFILE_DIR}/result.json"
MAPPING_FILE="${EXPERIMENT_DIR}/candidate.json"
PAIRED_DIR="${EXPERIMENT_DIR}/paired"
CANDIDATE_RESULT="${PAIRED_DIR}/candidate/result.json"
LINEAR_RESULT="${PAIRED_DIR}/linear/result.json"
COMPARISON_FILE="${EXPERIMENT_DIR}/comparison.json"

mkdir -p "${EXPERIMENT_DIR}"

wait_for_gpu_pair() {
  local deadline=$((SECONDS + 1800))
  local -a used
  while true; do
    mapfile -t used < <(
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${CUDA_DEVICE}"
    )
    if [[ "${#used[@]}" -eq 2 ]] && (( used[0] < 1024 && used[1] < 1024 )); then
      return
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for GPU pair ${CUDA_DEVICE}" >&2
      return 75
    fi
    sleep 2
  done
}

record_manifest() {
  local phase="$1"
  local output_dir="$2"
  mkdir -p "${output_dir}"
  {
    echo "phase=${phase}"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname)"
    echo "git_commit=$(git -c safe.directory="${REPO_ROOT}" -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "model_name=${MODEL_NAME}"
    echo "model_path=${MODEL_PATH}"
    echo "python=$(${PYTHON_BIN} --version 2>&1)"
    echo "torch=$(${PYTHON_BIN} -c 'import torch; print(torch.__version__)')"
    echo "active_mapping=${DITTO_TP_KV_HEAD_ORDER_FILE:-<linear>}"
    sha256sum "${REPO_ROOT}/sgl-kernel/python/sgl_kernel/sm100/common_ops.abi3.so"
    sha256sum "${MODEL_PATH}/config.json" "${CONFIG_FILE}" "${DATA_FILE}"
    if [[ -n "${RESIDENT_FILE}" ]]; then
      sha256sum "${RESIDENT_FILE}"
    fi
    if [[ -n "${REFERENCE_MAPPING}" ]]; then
      sha256sum "${REFERENCE_MAPPING}"
    fi
    if [[ -f "${MAPPING_FILE}" ]]; then
      sha256sum "${MAPPING_FILE}"
    fi
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total \
      --format=csv,noheader -i "${CUDA_DEVICE}"
    nvidia-smi topo -m
    echo "seq_len=${SEQ_LEN}"
    echo "topk=${TOPK}"
    echo "decode_steps=${DECODE_STEPS}"
    echo "ignore_eos=true"
    echo "tp_size=${TP_SIZE}"
    echo "pp_size=${PP_SIZE}"
    echo "cuda_device=${CUDA_DEVICE}"
    echo "cpuset=${CPUSET}"
    echo "max_total_tokens=${MAX_TOTAL_TOKENS}"
    echo "mem_fraction_static=${MEM_FRACTION_STATIC}"
    echo "max_changes=${MAX_CHANGES}"
    echo "min_split_improvement_pct=${MIN_SPLIT_IMPROVEMENT_PCT}"
    echo "skip_steps=${SKIP_STEPS}"
    echo "skip_layers=${SKIP_LAYERS}"
    echo "resident_file=${RESIDENT_FILE:-<none>}"
    echo "reference_mapping=${REFERENCE_MAPPING:-<linear>}"
  } > "${output_dir}/manifest.txt"
}

run_benchmark() {
  local phase="$1"
  local output_dir="$2"
  local result_file="$3"
  local warmup="$4"
  local epoch="$5"
  local record_transfer="$6"
  local mapping_file="$7"
  mkdir -p "${output_dir}"
  wait_for_gpu_pair

  if [[ -n "${mapping_file}" ]]; then
    export DITTO_TP_KV_HEAD_ORDER_FILE="${mapping_file}"
  else
    unset DITTO_TP_KV_HEAD_ORDER_FILE
  fi
  unset DITTO_TP_KV_HEAD_ORDER
  if [[ -n "${RESIDENT_FILE}" ]]; then
    export DITTO_RESIDENT_HEADS_FILE="${RESIDENT_FILE}"
  else
    unset DITTO_RESIDENT_HEADS_FILE
  fi
  export DITTO_TP_ENABLE_CUDA_GRAPH=1

  local -a command=(
    "${PYTHON_BIN}" "${BENCH}"
    --model "${MODEL_PATH}"
    --config_file "${CONFIG_FILE}"
    --data "${DATA_FILE}"
    --num_decode_steps "${DECODE_STEPS}"
    --ignore-eos
    --warmup "${warmup}"
    --epoch "${epoch}"
    --method offloading
    --topk "${TOPK}"
    --batch_size 1
    --max_seq_len "${SEQ_LEN}"
    --max-running-requests 1
    --max-total-tokens "${MAX_TOTAL_TOKENS}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --tp-size "${TP_SIZE}"
    --pp-size "${PP_SIZE}"
    --sglang-log-level warning
    --disable-sglang-batch-log
    --disable-cuda-graph
    --ditto-enable-cuda-graph
    --result-json "${result_file}"
  )
  if [[ "${record_transfer}" == "1" ]]; then
    mkdir -p "${TRANSFER_DIR}"
    export DITTO_TRANSFER_STATS_FILE="${TRANSFER_BASE}"
    export DITTO_RECORD_HEAD_MASKS=1
    command+=(--record-transfer-stats)
  else
    unset DITTO_TRANSFER_STATS_FILE
    unset DITTO_RECORD_HEAD_MASKS
  fi

  record_manifest "${phase}" "${output_dir}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START ${phase}"
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" taskset -c "${CPUSET}" \
    "${command[@]}" 2>&1 | tee "${output_dir}/run.log"
  local run_status=${PIPESTATUS[0]}
  set -e
  if [[ "${run_status}" -ne 0 ]]; then
    return "${run_status}"
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] END ${phase}"
}

generate_candidate() {
  local rank0="${TRANSFER_BASE%.json}.tp00.json"
  local rank1="${TRANSFER_BASE%.json}.tp01.json"
  local -a command=(
    "${PYTHON_BIN}" "${OPTIMIZER}"
    --transfer-json "${rank0}"
    --transfer-json "${rank1}"
    --output "${MAPPING_FILE}"
    --skip-steps "${SKIP_STEPS}"
    --skip-layers "${SKIP_LAYERS}"
    --max-changes "${MAX_CHANGES}"
    --min-split-improvement-pct "${MIN_SPLIT_IMPROVEMENT_PCT}"
  )
  if [[ -n "${REFERENCE_MAPPING}" ]]; then
    command+=(--reference-mapping "${REFERENCE_MAPPING}")
  fi
  if [[ -n "${RESIDENT_FILE}" ]]; then
    command+=(--resident-heads-file "${RESIDENT_FILE}")
  fi
  "${command[@]}" | tee "${EXPERIMENT_DIR}/generate.log"
  record_manifest generate "${EXPERIMENT_DIR}/generate"
}

summarize() {
  "${PYTHON_BIN}" "${SUMMARIZER}" \
    --candidate-result "${CANDIDATE_RESULT}" \
    --linear-result "${LINEAR_RESULT}" \
    --candidate-name robust-split-trust-region \
    --output "${COMPARISON_FILE}" \
    | tee "${EXPERIMENT_DIR}/summarize.log"
}

cd "${REPO_ROOT}"
for phase in ${PHASES}; do
  case "${phase}" in
    profile)
      run_benchmark profile "${PROFILE_DIR}" "${PROFILE_RESULT}" \
        "${PROFILE_WARMUP}" "${PROFILE_EPOCH}" 1 ""
      ;;
    generate)
      generate_candidate
      ;;
    benchmark)
      run_benchmark candidate "${PAIRED_DIR}/candidate" "${CANDIDATE_RESULT}" \
        "${BENCH_WARMUP}" "${BENCH_EPOCH}" 0 "${MAPPING_FILE}"
      run_benchmark linear "${PAIRED_DIR}/linear" "${LINEAR_RESULT}" \
        "${BENCH_WARMUP}" "${BENCH_EPOCH}" 0 ""
      ;;
    summarize)
      summarize
      ;;
    *)
      echo "Unknown phase: ${phase}" >&2
      exit 2
      ;;
  esac
done
