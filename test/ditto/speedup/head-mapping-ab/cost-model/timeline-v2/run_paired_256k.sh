#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
RUNNER="${REPO_ROOT}/test/ditto/speedup/run_n2n_qwen_offloading_seqlen.sh"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${SCRIPT_DIR}/perf-256K-paired-20260907}"
MAPPING_FILE="${MAPPING_FILE:-${SCRIPT_DIR}/per-profile/timeline-256K.json}"
CANDIDATE_NAME="${CANDIDATE_NAME:-timeline-256K}"
RESIDENT_FILE="${REPO_ROOT}/test/ditto/speedup/head-mapping-ab/resident-placement/l25-none.json"
VENV_BIN="${REPO_ROOT}/.venv-py313/bin"

mkdir -p "${EXPERIMENT_DIR}"
export PATH="${VENV_BIN}:${PATH}"

wait_for_gpu_pair() {
  local deadline=$((SECONDS + 1800))
  local -a used
  while true; do
    mapfile -t used < <(
      nvidia-smi \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits \
        -i 0,1
    )
    if (( used[0] < 1024 && used[1] < 1024 )); then
      return
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for GPU 0/1" >&2
      return 75
    fi
    sleep 2
  done
}

record_manifest() {
  local variant="$1"
  local output_dir="$2"
  {
    echo "variant=${variant}"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname)"
    echo "repo_root=${REPO_ROOT}"
    echo "git_commit=$(git -c safe.directory="${REPO_ROOT}" rev-parse HEAD)"
    echo "python=$(${VENV_BIN}/python --version 2>&1)"
    echo "torch=$(${VENV_BIN}/python -c 'import torch; print(torch.__version__)')"
    echo "mapping_file=${MAPPING_FILE}"
    echo "mapping_sha256=$(sha256sum "${MAPPING_FILE}" | cut -d' ' -f1)"
    echo "resident_file=${RESIDENT_FILE}"
    echo "resident_sha256=$(sha256sum "${RESIDENT_FILE}" | cut -d' ' -f1)"
    sha256sum \
      "${REPO_ROOT}/test/ditto/config/hata_offloading/Qwen2.5-14B-Instruct-1M-256K.yaml" \
      "${REPO_ROOT}/../myTransformer/speedup/data/RULER-Qwen2.5-14B-Instruct-1M-256K.jsonl"
    echo
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total \
      --format=csv,noheader -i 0,1
    echo
    nvidia-smi topo -m
    echo
    env | grep -E \
      '^(CUDA_DEVICE|TP_SIZE|PP_SIZE|BSZ|TOPK|DECODE_STEPS|WARMUP|EPOCH|MAX_TOTAL_TOKENS|MEM_FRACTION_STATIC|SGLANG_CUDA_GRAPH|DITTO_CUDA_GRAPH|DITTO_TP_ENABLE_CUDA_GRAPH|DITTO_TP_KV_HEAD_ORDER_FILE|DITTO_RESIDENT_HEADS_FILE)=' \
      | sort
  } > "${output_dir}/manifest.txt"
}

run_variant() {
  local variant="$1"
  local output_dir="${EXPERIMENT_DIR}/${variant}"
  mkdir -p "${output_dir}"

  if [[ "${variant}" != "linear" ]]; then
    export DITTO_TP_KV_HEAD_ORDER_FILE="${MAPPING_FILE}"
  else
    unset DITTO_TP_KV_HEAD_ORDER_FILE
  fi
  unset DITTO_TP_KV_HEAD_ORDER
  export DITTO_RESIDENT_HEADS_FILE="${RESIDENT_FILE}"

  export PYTHON_BIN="${VENV_BIN}/python"
  export MODEL_PATH="${REPO_ROOT}/../Qwen2.5-14B-Instruct-1M"
  export CONFIG_ROOT="${REPO_ROOT}/test/ditto/config/hata_offloading"
  export DATA_ROOT="${REPO_ROOT}/../myTransformer/speedup/data"
  export LOG_DIR="${output_dir}"
  export METHODS=offloading
  export SEQ_LIST=256000
  export BSZ=1
  export TOPK=0.10
  export DECODE_STEPS=50
  export WARMUP=1
  export EPOCH=3
  export CPUSET=0-95
  export CUDA_DEVICE=0,1
  export TP_SIZE=2
  export PP_SIZE=1
  export MAX_TOTAL_TOKENS=262144
  export MEM_FRACTION_STATIC=0.92
  export SGLANG_CUDA_GRAPH=0
  export DITTO_CUDA_GRAPH=1
  export DITTO_TP_ENABLE_CUDA_GRAPH=1
  export RECORD_TRANSFER_STATS=0
  export RECORD_HEAD_MASKS=0
  export RECORD_MAX_GPU_MEMORY=0

  wait_for_gpu_pair
  record_manifest "${variant}" "${output_dir}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START ${variant}"
  bash "${RUNNER}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] END ${variant}"
}

cd "${REPO_ROOT}"
run_variant "${CANDIDATE_NAME}"
run_variant linear
