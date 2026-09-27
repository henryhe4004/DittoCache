#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/jhe/sglang-litecache/.venv-py313/bin/python}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-14B-Instruct-1M}"
SERVER="${SERVER:-http://127.0.0.1:30000}"
MODE="${MODE:-ditto}"   # ditto or full
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/results_itl_tmp}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 24 32 48 64}"
CONCURRENCY_LIST="${CONCURRENCIES//,/ }"
DATA_FILE="${DATA_FILE:-${SCRIPT_DIR}/data/RULER-Qwen2.5-14B-Instruct-1M-8K.jsonl}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-64}"
TOTAL_REQUESTS_MULTIPLIER="${TOTAL_REQUESTS_MULTIPLIER:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-512}"
SEED="${SEED:-42}"
SLEEP_BETWEEN_CONCURRENCY="${SLEEP_BETWEEN_CONCURRENCY:-5}"

mode_dir="${OUT_ROOT}/${MODE}"
mkdir -p "${mode_dir}"

for c in ${CONCURRENCY_LIST}; do
  total_requests=$(( c * TOTAL_REQUESTS_MULTIPLIER ))
  if (( total_requests < c )); then total_requests=${c}; fi
  jsonl="${mode_dir}/concurrency_${c}.jsonl"
  log="${mode_dir}/concurrency_${c}.log"
  echo "[ITL] mode=${MODE} concurrency=${c} total_requests=${total_requests} jsonl=${jsonl}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/launch_client_itl_tmp.py" \
    --server "${SERVER}" \
    --log-file "${jsonl}" \
    --model-path "${MODEL_PATH}" \
    --data-file "${DATA_FILE}" \
    --dataset ruler \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --min-new-tokens "${MIN_NEW_TOKENS}" \
    --ignore-eos \
    --ordered \
    --max-samples "${MAX_SAMPLES}" \
    --concurrency "${c}" \
    --total-requests "${total_requests}" \
    --rid-prefix "${MODE}-itl-c${c}" \
    --seed "${SEED}" 2>&1 | tee "${log}"
  if [[ "${SLEEP_BETWEEN_CONCURRENCY}" != "0" ]]; then
    sleep "${SLEEP_BETWEEN_CONCURRENCY}"
  fi
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_itl_tmp.py" --root "${OUT_ROOT}" --concurrencies "${CONCURRENCY_LIST}" --mode "${MODE}"
