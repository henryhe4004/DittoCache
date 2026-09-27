#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILES=(B0 B1 B2 B3 B4 B5 B6)
SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/jhe/sglang-litecache/.venv-py313/bin/python}"

for profile in "${PROFILES[@]}"; do
  script="${SCRIPT_DIR}/run_n2n_qwen_offloading_bsz_${profile}.sh"
  if [[ ! -x "${script}" ]]; then
    echo "[ERROR] missing executable script: ${script}" >&2
    exit 1
  fi

  echo "[SEQUENCE] start ${profile}: ${script}"
  PYTHON_BIN="${PYTHON_BIN}" "${script}"
  echo "[SEQUENCE] done ${profile}"

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[SEQUENCE] gpu processes after ${profile}:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  fi

  if [[ "${profile}" != "B6" && "${SLEEP_BETWEEN}" != "0" ]]; then
    sleep "${SLEEP_BETWEEN}"
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[SEQUENCE] final gpu processes:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
fi
