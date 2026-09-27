#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICE="${CUDA_DEVICE:-0}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
PORT="${PORT:-30000}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-14B-Instruct-1M}"

DITTO_ENABLE_CUDA_GRAPH="${DITTO_ENABLE_CUDA_GRAPH:-true}"
DITTO_MAX_TOKENS="${DITTO_MAX_TOKENS:-131072}"
DITTO_MAX_BATCH_SIZE="${DITTO_MAX_BATCH_SIZE:-16}"
DITTO_TARGET_SEQ_LEN="${DITTO_TARGET_SEQ_LEN:-}"
DITTO_GPU_MEMORY_BUDGET="${DITTO_GPU_MEMORY_BUDGET:-70.0}"
DITTO_CHUNK_PREFILL_SIZE="${DITTO_CHUNK_PREFILL_SIZE:-}"

SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-16}"
SGLANG_PREFILL_MAX_REQUESTS="${SGLANG_PREFILL_MAX_REQUESTS:-1}"
SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-8192}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-}"
SGLANG_DISABLE_OVERLAP_SCHEDULE="${SGLANG_DISABLE_OVERLAP_SCHEDULE:-0}"

if [[ -z "${DITTO_ARCHITECTURE:-}" ]]; then
  DITTO_ARCHITECTURE="$(${PYTHON_BIN:-/workspace/jhe/sglang-litecache/.venv-py313/bin/python} - "${MODEL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
config_path = model_path / "config.json" if model_path.is_dir() else model_path
with config_path.open() as f:
    config = json.load(f)

model_type = str(config.get("model_type", "")).lower()
architectures = [str(arch).lower() for arch in (config.get("architectures") or [])]

if "qwen2" in model_type or any("qwen2" in arch for arch in architectures):
    print("DittoQwen2ForCausalLM")
elif "llama" in model_type or any("llama" in arch for arch in architectures):
    print("DittoLlamaForCausalLM")
else:
    print("DittoLlamaForCausalLM")
PY
)"
fi

case "${DITTO_ARCHITECTURE}" in
  DittoLlamaForCausalLM|DittoQwen2ForCausalLM) ;;
  *)
    echo "Unsupported DITTO_ARCHITECTURE=${DITTO_ARCHITECTURE}" >&2
    exit 1
    ;;
esac

if [[ -z "${DITTO_CHUNK_PREFILL_SIZE}" ]]; then
  if (( DITTO_MAX_BATCH_SIZE >= 32 )); then
    DITTO_CHUNK_PREFILL_SIZE=8192
  else
    DITTO_CHUNK_PREFILL_SIZE=8192
  fi
fi

if [[ -n "${DITTO_TARGET_SEQ_LEN}" ]]; then
  DITTO_MAX_TOKENS="$(( DITTO_TARGET_SEQ_LEN * DITTO_MAX_BATCH_SIZE ))"
fi

if [[ -z "${SGLANG_PREFILL_MAX_REQUESTS}" && -n "${SGLANG_MAX_RUNNING_REQUESTS}" ]]; then
  SGLANG_PREFILL_MAX_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS}"
fi

if [[ -z "${SGLANG_CHUNKED_PREFILL_SIZE}" ]]; then
  SGLANG_CHUNKED_PREFILL_SIZE="${DITTO_CHUNK_PREFILL_SIZE}"
fi

JSON_MODEL_OVERRIDE="$(cat <<JSON
{
  "architectures": ["${DITTO_ARCHITECTURE}"],
  "ditto_variant": "flashattn",
  "custom_config": {
    "enable_cuda_graph": ${DITTO_ENABLE_CUDA_GRAPH},
    "chunk_prefill_size": ${DITTO_CHUNK_PREFILL_SIZE},
    "kvcache_manager_config": {
      "max_tokens": ${DITTO_MAX_TOKENS},
      "max_batch_size": ${DITTO_MAX_BATCH_SIZE},
      "gpu_memory_budget": ${DITTO_GPU_MEMORY_BUDGET}
    }
  }
}
JSON
)"

EXTRA_SERVER_ARGS=()
if [[ -n "${SGLANG_MAX_RUNNING_REQUESTS}" ]]; then
  EXTRA_SERVER_ARGS+=(--max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
fi
if [[ -n "${SGLANG_PREFILL_MAX_REQUESTS}" ]]; then
  EXTRA_SERVER_ARGS+=(--prefill-max-requests "${SGLANG_PREFILL_MAX_REQUESTS}")
fi
if [[ -n "${SGLANG_MAX_PREFILL_TOKENS}" ]]; then
  EXTRA_SERVER_ARGS+=(--max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}")
fi
if [[ -n "${SGLANG_CHUNKED_PREFILL_SIZE}" ]]; then
  EXTRA_SERVER_ARGS+=(--chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}")
fi
if [[ "${SGLANG_DISABLE_OVERLAP_SCHEDULE}" == "1" || "${SGLANG_DISABLE_OVERLAP_SCHEDULE}" == "true" ]]; then
  EXTRA_SERVER_ARGS+=(--disable-overlap-schedule)
fi

CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING}" CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" ${PYTHON_BIN:-/workspace/jhe/sglang-litecache/.venv-py313/bin/python} -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port "${PORT}" \
  --tp 1 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --decode-log-interval 1 \
  --enable-request-time-stats-logging \
  --enable-metrics \
  --disable-radix-cache \
  --skip-server-warmup \
  --json-model-override-args "${JSON_MODEL_OVERRIDE}" \
  "${EXTRA_SERVER_ARGS[@]}"
