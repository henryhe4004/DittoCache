#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICE="${CUDA_DEVICE:-0}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
PORT="${PORT:-30000}"
MODEL_PATH="${MODEL_PATH:-/jhe/Qwen2.5-14B-Instruct-1M}"

LITECACHE_ENABLE_CUDA_GRAPH="${LITECACHE_ENABLE_CUDA_GRAPH:-true}"
LITECACHE_MAX_TOKENS="${LITECACHE_MAX_TOKENS:-65536}"
LITECACHE_MAX_BATCH_SIZE="${LITECACHE_MAX_BATCH_SIZE:-1}"
LITECACHE_GPU_MEMORY_BUDGET="${LITECACHE_GPU_MEMORY_BUDGET:-40.0}"
LITECACHE_CHUNK_PREFILL_SIZE="${LITECACHE_CHUNK_PREFILL_SIZE:-4096}"

SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-}"
SGLANG_PREFILL_MAX_REQUESTS="${SGLANG_PREFILL_MAX_REQUESTS:-}"
SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-}"
SGLANG_DISABLE_OVERLAP_SCHEDULE="${SGLANG_DISABLE_OVERLAP_SCHEDULE:-0}"

JSON_MODEL_OVERRIDE="$(cat <<JSON
{
  "architectures": ["LiteCacheQwen2ForCausalLM"],
  "litecache_variant": "flashattn",
  "custom_config": {
    "enable_cuda_graph": ${LITECACHE_ENABLE_CUDA_GRAPH},
    "chunk_prefill_size": ${LITECACHE_CHUNK_PREFILL_SIZE},
    "kvcache_manager_config": {
      "max_tokens": ${LITECACHE_MAX_TOKENS},
      "max_batch_size": ${LITECACHE_MAX_BATCH_SIZE},
      "gpu_memory_budget": ${LITECACHE_GPU_MEMORY_BUDGET}
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

CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING}" CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port "${PORT}" \
  --tp 1 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --decode-log-interval 1 \
  --disable-radix-cache \
  --skip-server-warmup \
  --json-model-override-args "${JSON_MODEL_OVERRIDE}" \
  "${EXTRA_SERVER_ARGS[@]}"
