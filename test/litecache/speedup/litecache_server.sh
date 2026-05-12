#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICE="${CUDA_DEVICE:-0}"
PORT="${PORT:-30000}"
MODEL_PATH="${MODEL_PATH:-/jhe/Qwen2.5-14B-Instruct-1M}"

# Offline-parity defaults for accuracy debugging. Override these env vars when
# testing online multi-batch behavior.

LITECACHE_ENABLE_CUDA_GRAPH="${LITECACHE_ENABLE_CUDA_GRAPH:-true}"
LITECACHE_MAX_TOKENS="${LITECACHE_MAX_TOKENS:-262144}"
LITECACHE_MAX_BATCH_SIZE="${LITECACHE_MAX_BATCH_SIZE:-1}"
LITECACHE_TARGET_SEQ_LEN="${LITECACHE_TARGET_SEQ_LEN:-}"
LITECACHE_GPU_MEMORY_BUDGET="${LITECACHE_GPU_MEMORY_BUDGET:-40.0}"
LITECACHE_CHUNK_PREFILL_SIZE="${LITECACHE_CHUNK_PREFILL_SIZE:-}"
LITECACHE_TOKEN_BUDGET="${LITECACHE_TOKEN_BUDGET:-0.1}"
LITECACHE_SINK_BUDGET="${LITECACHE_SINK_BUDGET:-4}"
LITECACHE_RECENT_BUDGET="${LITECACHE_RECENT_BUDGET:-64}"
LITECACHE_REUSE_THRESHOLD_UPPER="${LITECACHE_REUSE_THRESHOLD_UPPER:-0.8}"
LITECACHE_REUSE_THRESHOLD_LOWER="${LITECACHE_REUSE_THRESHOLD_LOWER:--1.0}"
LITECACHE_NUM_SKIP_LAYERS="${LITECACHE_NUM_SKIP_LAYERS:-1}"
LITECACHE_NUM_OVERLAPPED_HEADS="${LITECACHE_NUM_OVERLAPPED_HEADS:-7}"
LITECACHE_NUM_OMP_THREADS="${LITECACHE_NUM_OMP_THREADS:-16}"

SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-}"
SGLANG_PREFILL_MAX_REQUESTS="${SGLANG_PREFILL_MAX_REQUESTS:-}"
SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-}"
SGLANG_DISABLE_OVERLAP_SCHEDULE="${SGLANG_DISABLE_OVERLAP_SCHEDULE:-0}"

if [[ -z "${LITECACHE_CHUNK_PREFILL_SIZE}" ]]; then
  if (( LITECACHE_MAX_BATCH_SIZE >= 32 )); then
    LITECACHE_CHUNK_PREFILL_SIZE=2048
  else
    LITECACHE_CHUNK_PREFILL_SIZE=8192
  fi
fi

if [[ -n "${LITECACHE_TARGET_SEQ_LEN}" ]]; then
  LITECACHE_MAX_TOKENS="$(( LITECACHE_TARGET_SEQ_LEN * LITECACHE_MAX_BATCH_SIZE ))"
fi

if [[ -z "${SGLANG_PREFILL_MAX_REQUESTS}" && -n "${SGLANG_MAX_RUNNING_REQUESTS}" ]]; then
  SGLANG_PREFILL_MAX_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS}"
fi

if [[ -z "${SGLANG_CHUNKED_PREFILL_SIZE}" ]]; then
  SGLANG_CHUNKED_PREFILL_SIZE="${LITECACHE_CHUNK_PREFILL_SIZE}"
fi

JSON_MODEL_OVERRIDE="$(cat <<JSON
{
  "architectures": ["LiteCacheQwen2ForCausalLM"],
  "litecache_variant": "offloading",
  "offloading_method": "hash",
  "custom_config": {
    "enable_cuda_graph": ${LITECACHE_ENABLE_CUDA_GRAPH},
    "new_config": true,
    "is_profiling": false,
    "offloading_method": "hash",
    "chunk_prefill_size": ${LITECACHE_CHUNK_PREFILL_SIZE},
    "kvcache_manager_config": {
      "max_tokens": ${LITECACHE_MAX_TOKENS},
      "max_batch_size": ${LITECACHE_MAX_BATCH_SIZE},
      "gpu_memory_budget": ${LITECACHE_GPU_MEMORY_BUDGET}
    },
    "sparse_attention_config": {
      "token_budget": ${LITECACHE_TOKEN_BUDGET},
      "sink_budget": ${LITECACHE_SINK_BUDGET},
      "recent_budget": ${LITECACHE_RECENT_BUDGET},
      "selective_start_len": 0
    },
    "offload_config": {
      "attn_pattern_path": "/jhe/sglang/test/litecache/auxiliary/attn_pattern/Qwen2.5-14B-Instruct-1M",
      "reuse_threshold_upper": ${LITECACHE_REUSE_THRESHOLD_UPPER},
      "reuse_threshold_lower": ${LITECACHE_REUSE_THRESHOLD_LOWER},
      "decay_p": 3.0,
      "cosine_padding": 0.05,
      "num_skip_layers": ${LITECACHE_NUM_SKIP_LAYERS},
      "num_overlapped_heads": ${LITECACHE_NUM_OVERLAPPED_HEADS},
      "num_omp_threads": ${LITECACHE_NUM_OMP_THREADS}
    },
    "aux_data_path": "/jhe/sglang/test/litecache/auxiliary/hash_weights/Qwen2.5-14B-Instruct-1M-256",
    "num_channels": 32,
    "rbits": 256,
    "block_size": 64
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

# CUDA_LAUNCH_BLOCKING=1 
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python -m sglang.launch_server \
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
