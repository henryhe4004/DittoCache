#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python3}"
MODEL_PATH="${MODEL_PATH:-/jhe/Llama-3-8B-Instruct-Gradient-1048k}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PORT="${PORT:-30000}"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/../results/latency}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 24 32 48 64}"
REPEATS="${REPEATS:-3}"
INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
TARGET_SEQ_LEN="${TARGET_SEQ_LEN:-$((INPUT_LEN + OUTPUT_LEN))}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-1800}"
GPU_IDLE_TIMEOUT_SEC="${GPU_IDLE_TIMEOUT_SEC:-21600}"
GPU_IDLE_MAX_MIB="${GPU_IDLE_MAX_MIB:-512}"
GPU_IDLE_STABLE_SEC="${GPU_IDLE_STABLE_SEC:-120}"
METHOD_MAX_ATTEMPTS="${METHOD_MAX_ATTEMPTS:-20}"
RUN_METHODS="${RUN_METHODS:-full ditto}"

mkdir -p "${RESULT_ROOT}/server_logs"
server_pid=""

cleanup_server() {
  if [[ -z "${server_pid}" ]]; then
    return
  fi
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM -- "-${server_pid}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "${server_pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${server_pid}" 2>/dev/null; then
      kill -KILL -- "-${server_pid}" 2>/dev/null || true
    fi
  fi
  wait "${server_pid}" 2>/dev/null || true
  server_pid=""
}
trap cleanup_server EXIT INT TERM

wait_for_gpu_idle() {
  local deadline=$((SECONDS + GPU_IDLE_TIMEOUT_SEC))
  local idle_since=-1
  local idle_for=0
  local used_mib
  while (( SECONDS < deadline )); do
    used_mib="$(nvidia-smi --id="${CUDA_DEVICE}" \
      --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib <= GPU_IDLE_MAX_MIB )); then
      if (( idle_since < 0 )); then
        idle_since=${SECONDS}
      fi
      idle_for=$((SECONDS - idle_since))
      if (( idle_for >= GPU_IDLE_STABLE_SEC )); then
        echo "[IDLE] gpu=${CUDA_DEVICE} memory_used=${used_mib}MiB stable=${idle_for}s"
        return 0
      fi
      echo "[WAIT] gpu=${CUDA_DEVICE} idle=${idle_for}/${GPU_IDLE_STABLE_SEC}s"
    else
      idle_since=-1
      echo "[WAIT] gpu=${CUDA_DEVICE} memory_used=${used_mib:-unknown}MiB"
    fi
    sleep 5
  done
  echo "[ERROR] GPU ${CUDA_DEVICE} did not become idle within ${GPU_IDLE_TIMEOUT_SEC}s" >&2
  return 1
}

wait_until_ready() {
  local method="$1"
  local log_path="$2"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "[ERROR] ${method} server exited during startup" >&2
      tail -n 100 "${log_path}" >&2
      return 1
    fi
    if rg -q "Uvicorn running" "${log_path}"; then
      echo "[READY] ${method} server pid=${server_pid}"
      return 0
    fi
    sleep 2
  done
  echo "[ERROR] ${method} server did not become ready in ${STARTUP_TIMEOUT_SEC}s" >&2
  tail -n 100 "${log_path}" >&2
  return 1
}

run_method() {
  local method="$1"
  local server_script
  local attempt
  local log_path
  case "${method}" in
    full) server_script="${SCRIPT_DIR}/full_latency_server.sh" ;;
    ditto) server_script="${SCRIPT_DIR}/ditto_latency_server.sh" ;;
    *)
      echo "Unknown method: ${method}" >&2
      return 1
      ;;
  esac

  for attempt in $(seq 1 "${METHOD_MAX_ATTEMPTS}"); do
    wait_for_gpu_idle
    if curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[ERROR] port ${PORT} already has a running server" >&2
      return 1
    fi

    log_path="${RESULT_ROOT}/server_logs/${method}_attempt$(printf '%02d' "${attempt}").log"
    echo "[START] method=${method} attempt=${attempt}/${METHOD_MAX_ATTEMPTS} gpu=${CUDA_DEVICE} log=${log_path}"
    CUDA_DEVICE="${CUDA_DEVICE}" PORT="${PORT}" MODEL_PATH="${MODEL_PATH}" \
      DITTO_TARGET_SEQ_LEN="${TARGET_SEQ_LEN}" \
      PYTHON_BIN="${PYTHON_BIN}" setsid bash "${server_script}" >"${log_path}" 2>&1 &
    server_pid=$!
    if ! wait_until_ready "${method}" "${log_path}"; then
      cleanup_server
      echo "[RETRY] method=${method} failed during startup" >&2
      continue
    fi

    # Word splitting is intentional: CONCURRENCIES is a space-separated list.
    # shellcheck disable=SC2086
    if "${PYTHON_BIN}" "${SCRIPT_DIR}/run_latency_benchmark.py" \
      --method "${method}" \
      --base-url "http://127.0.0.1:${PORT}" \
      --model-path "${MODEL_PATH}" \
      --input-len "${INPUT_LEN}" \
      --output-len "${OUTPUT_LEN}" \
      --repeats "${REPEATS}" \
      --output-dir "${RESULT_ROOT}" \
      --concurrencies ${CONCURRENCIES}; then
      cleanup_server
      return 0
    fi

    echo "[RETRY] method=${method} benchmark interrupted; preserving validated results" >&2
    cleanup_server
  done

  echo "[ERROR] method=${method} failed after ${METHOD_MAX_ATTEMPTS} attempts" >&2
  return 1
}

for method in ${RUN_METHODS}; do
  run_method "${method}"
done

summary_files=()
for method in full ditto; do
  if [[ -f "${RESULT_ROOT}/${method}/summary.csv" ]]; then
    summary_files+=("${RESULT_ROOT}/${method}/summary.csv")
  fi
done
if (( ${#summary_files[@]} > 0 )); then
  awk 'FNR == 1 && NR != 1 { next } { print }' "${summary_files[@]}" \
    >"${RESULT_ROOT}/summary.csv"
fi

echo "[DONE] latency results: ${RESULT_ROOT}"
