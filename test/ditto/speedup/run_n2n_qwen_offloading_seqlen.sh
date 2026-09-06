#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-14B-Instruct-1M}"
CONFIG_ROOT="${CONFIG_ROOT:-${SCRIPT_DIR}/../config/hata_offloading}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs-offloading-corr0.31}"

METHODS="${METHODS:-offloading}"
SEQ_LIST="${SEQ_LIST:-64000}"
BSZ="${BSZ:-1}"
TOPK="${TOPK:-0.10}"
DECODE_STEPS="${DECODE_STEPS:-50}"
WARMUP="${WARMUP:-1}"
EPOCH="${EPOCH:-3}"
CPUSET="${CPUSET:-96-143}"
CUDA_DEVICE="${CUDA_DEVICE:-5}"
PP_SIZE="${PP_SIZE:-1}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.92}"
SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-warning}"
DISABLE_SGLANG_BATCH_LOG="${DISABLE_SGLANG_BATCH_LOG:-1}"
# Keep SGLang global cuda graph off by default (can hang in offloading path),
# while keeping Ditto cuda graph on to match internal prototype configs.
SGLANG_CUDA_GRAPH="${SGLANG_CUDA_GRAPH:-0}"
DITTO_CUDA_GRAPH="${DITTO_CUDA_GRAPH:-1}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-}"
RECORD_MAX_GPU_MEMORY="${RECORD_MAX_GPU_MEMORY:-1}"
GPU_MEM_MONITOR_INTERVAL_SEC="${GPU_MEM_MONITOR_INTERVAL_SEC:-0.2}"

start_gpu_mem_monitor() {
  local gpu_ids_csv="$1"
  local out_json="$2"
  local interval_sec="$3"
  local stop_flag="$4"
  "${PYTHON_BIN}" - "${gpu_ids_csv}" "${out_json}" "${interval_sec}" "${stop_flag}" <<'PY' &
import json
import os
import subprocess
import sys
import time
from pathlib import Path

gpu_csv = sys.argv[1]
out_path = Path(sys.argv[2])
interval_sec = float(sys.argv[3])
stop_flag = Path(sys.argv[4])

gpu_ids = [x.strip() for x in gpu_csv.split(",") if x.strip()]
peak_by_gpu = {gpu_id: 0 for gpu_id in gpu_ids}

def sample_once() -> None:
    if not gpu_ids:
        return
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return

    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        idx, used = parts
        if idx not in peak_by_gpu:
            continue
        try:
            used_mb = int(float(used))
        except Exception:
            continue
        if used_mb > peak_by_gpu[idx]:
            peak_by_gpu[idx] = used_mb

while not stop_flag.exists():
    sample_once()
    time.sleep(interval_sec)
sample_once()

peak_used_mb = max(peak_by_gpu.values()) if peak_by_gpu else 0
payload = {
    "gpu_ids": gpu_ids,
    "peak_used_mb": int(peak_used_mb),
    "peak_used_mb_by_gpu": {k: int(v) for k, v in peak_by_gpu.items()},
    "poll_interval_sec": float(interval_sec),
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  MONITOR_PID=$!
}

merge_peak_memory_into_result() {
  local mem_json="$1"
  local result_json="$2"
  "${PYTHON_BIN}" - "${mem_json}" "${result_json}" <<'PY'
import json
import sys
from pathlib import Path

mem_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])

if not mem_path.exists():
    print("[MEM] monitor output missing")
    raise SystemExit(0)

mem = json.loads(mem_path.read_text(encoding="utf-8"))
peak_mb = int(mem.get("peak_used_mb", 0))
peak_gb = float(peak_mb / 1024.0)
by_gpu = mem.get("peak_used_mb_by_gpu", {})

if result_path.exists():
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["peak_gpu_memory_used_mb"] = peak_mb
    result["peak_gpu_memory_used_gb"] = peak_gb
    result["peak_gpu_memory_by_gpu_mb"] = by_gpu
    runtime_meta = result.get("runtime_meta")
    if isinstance(runtime_meta, dict):
        runtime_meta["peak_gpu_memory_used_mb"] = peak_mb
        runtime_meta["peak_gpu_memory_used_gb"] = peak_gb
        runtime_meta["peak_gpu_memory_by_gpu_mb"] = by_gpu
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

print(
    "[MEM] peak_gpu_memory_used_mb="
    f"{peak_mb} peak_gpu_memory_used_gb={peak_gb:.2f} by_gpu_mb={by_gpu}"
)
PY
}

mkdir -p "${LOG_DIR}"

for method in ${METHODS}; do
  for seq in ${SEQ_LIST}; do
    seq_in_k=$((seq / 1000))
    config_tokens=$((seq_in_k * BSZ))
    config_file="${CONFIG_ROOT}/Qwen2.5-14B-Instruct-1M-${config_tokens}K.yaml"
    data_file="${DATA_ROOT}/RULER-Qwen2.5-14B-Instruct-1M-${seq_in_k}K.jsonl"
    run_name="qwen2.5-14b-1m-${method}-bsz${BSZ}-seq${seq_in_k}K-topk${TOPK}"
    log_file="${LOG_DIR}/${run_name}.log"
    result_json="${LOG_DIR}/${run_name}.json"
    mem_json="${LOG_DIR}/${run_name}.mem.json"

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
      --pp-size "${PP_SIZE}"
      --mem-fraction-static "${MEM_FRACTION_STATIC}"
      --result-json "${result_json}"
    )
    if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
      cmd+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
    fi
    if [[ -n "${CHUNKED_PREFILL_SIZE}" ]]; then
      cmd+=(--chunked-prefill-size "${CHUNKED_PREFILL_SIZE}")
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
    if [[ "${RECORD_MAX_GPU_MEMORY}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
      stop_flag="$(mktemp "${LOG_DIR}/.mem_stop_${run_name}_XXXXXX")"
      rm -f "${stop_flag}"
      start_gpu_mem_monitor "${CUDA_DEVICE}" "${mem_json}" "${GPU_MEM_MONITOR_INTERVAL_SEC}" "${stop_flag}"
      monitor_pid="${MONITOR_PID:-}"

      set +e
      if [[ -n "${CPUSET}" ]]; then
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" taskset -c "${CPUSET}" "${cmd[@]}" 2>&1 | tee "${log_file}"
      else
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${cmd[@]}" 2>&1 | tee "${log_file}"
      fi
      run_status=${PIPESTATUS[0]}
      set -e
      touch "${stop_flag}" || true
      if [[ -n "${monitor_pid}" ]]; then
        wait "${monitor_pid}" || true
      fi
      rm -f "${stop_flag}" || true

      if [[ "${run_status}" -ne 0 ]]; then
        exit "${run_status}"
      fi

      merge_peak_memory_into_result "${mem_json}" "${result_json}" | tee -a "${log_file}"
    else
      if [[ "${RECORD_MAX_GPU_MEMORY}" == "1" ]]; then
        echo "[WARN] nvidia-smi not found; skip peak GPU memory monitor" | tee -a "${log_file}"
      fi
      if [[ -n "${CPUSET}" ]]; then
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" taskset -c "${CPUSET}" "${cmd[@]}" 2>&1 | tee "${log_file}"
      else
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${cmd[@]}" 2>&1 | tee "${log_file}"
      fi
    fi
  done
done
