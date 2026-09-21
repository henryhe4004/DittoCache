#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python3}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
PORT="${PORT:-30000}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-14B-Instruct-1M}" #/models/Llama-3-8B-Instruct
# Offline-parity defaults for accuracy debugging. Override these env vars when
# testing online multi-batch behavior.

DITTO_ENABLE_CUDA_GRAPH="${DITTO_ENABLE_CUDA_GRAPH:-true}"
DITTO_MAX_TOKENS="${DITTO_MAX_TOKENS:-1048576}"
DITTO_MAX_BATCH_SIZE="${DITTO_MAX_BATCH_SIZE:-64}"
DITTO_TARGET_SEQ_LEN="${DITTO_TARGET_SEQ_LEN:-8192}"
DITTO_GPU_MEMORY_BUDGET="${DITTO_GPU_MEMORY_BUDGET:-40.0}"
DITTO_CHUNK_PREFILL_SIZE="${DITTO_CHUNK_PREFILL_SIZE:-}"
DITTO_TOKEN_BUDGET="${DITTO_TOKEN_BUDGET:-0.1}"
DITTO_SINK_BUDGET="${DITTO_SINK_BUDGET:-4}"
DITTO_RECENT_BUDGET="${DITTO_RECENT_BUDGET:-64}"
DITTO_REUSE_THRESHOLD_UPPER="${DITTO_REUSE_THRESHOLD_UPPER:-0.8}"
DITTO_REUSE_THRESHOLD_LOWER="${DITTO_REUSE_THRESHOLD_LOWER:--1.0}"
DITTO_NUM_SKIP_LAYERS="${DITTO_NUM_SKIP_LAYERS:-1}"
DITTO_NUM_OVERLAPPED_HEADS="${DITTO_NUM_OVERLAPPED_HEADS:-7}"
DITTO_NUM_OMP_THREADS="${DITTO_NUM_OMP_THREADS:-16}"
DITTO_NUM_CHANNELS="${DITTO_NUM_CHANNELS:-32}"
DITTO_RBITS="${DITTO_RBITS:-256}"
DITTO_BLOCK_SIZE="${DITTO_BLOCK_SIZE:-64}"

SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-64}"
SGLANG_PREFILL_MAX_REQUESTS="${SGLANG_PREFILL_MAX_REQUESTS:-}"
SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-}"
SGLANG_DISABLE_OVERLAP_SCHEDULE="${SGLANG_DISABLE_OVERLAP_SCHEDULE:-0}"

MODEL_NAME="$(basename "${MODEL_PATH%/}")"
DITTO_AUX_ROOT="${DITTO_AUX_ROOT:-/jhe/myTransformer/auxiliary}"
DITTO_ATTENTION_PATTERN_PATH="${DITTO_ATTENTION_PATTERN_PATH:-${DITTO_AUX_ROOT}/attn_pattern/${MODEL_NAME}}"
DITTO_AUX_DATA_PATH="${DITTO_AUX_DATA_PATH:-${DITTO_AUX_ROOT}/hash_weights/${MODEL_NAME}-${DITTO_RBITS}}"

for required_path in \
  "${MODEL_PATH}/config.json" \
  "${DITTO_ATTENTION_PATTERN_PATH}/heads_cosine_similarity.csv" \
  "${DITTO_ATTENTION_PATTERN_PATH}/k_heads_importance.tsv" \
  "${DITTO_ATTENTION_PATTERN_PATH}/q_heads_importance.tsv" \
  "${DITTO_AUX_DATA_PATH}/hash_weight_layer_00.pt"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing required file: ${required_path}" >&2
    exit 1
  fi
done

export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${DITTO_ARCHITECTURE:-}" ]]; then
  DITTO_ARCHITECTURE="$("${PYTHON_BIN}" - "${MODEL_PATH}" <<'PY'
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
    DITTO_CHUNK_PREFILL_SIZE=2048
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
  "ditto_variant": "offloading",
  "offloading_method": "hash",
  "custom_config": {
    "enable_cuda_graph": ${DITTO_ENABLE_CUDA_GRAPH},
    "new_config": true,
    "is_profiling": false,
    "offloading_method": "hash",
    "chunk_prefill_size": ${DITTO_CHUNK_PREFILL_SIZE},
    "kvcache_manager_config": {
      "max_tokens": ${DITTO_MAX_TOKENS},
      "max_batch_size": ${DITTO_MAX_BATCH_SIZE},
      "gpu_memory_budget": ${DITTO_GPU_MEMORY_BUDGET}
    },
    "sparse_attention_config": {
      "token_budget": ${DITTO_TOKEN_BUDGET},
      "sink_budget": ${DITTO_SINK_BUDGET},
      "recent_budget": ${DITTO_RECENT_BUDGET},
      "selective_start_len": 0
    },
    "offload_config": {
      "attn_pattern_path": "${DITTO_ATTENTION_PATTERN_PATH}",
      "reuse_threshold_upper": ${DITTO_REUSE_THRESHOLD_UPPER},
      "reuse_threshold_lower": ${DITTO_REUSE_THRESHOLD_LOWER},
      "decay_p": 3.0,
      "cosine_padding": 0.05,
      "num_skip_layers": ${DITTO_NUM_SKIP_LAYERS},
      "num_overlapped_heads": ${DITTO_NUM_OVERLAPPED_HEADS},
      "num_omp_threads": ${DITTO_NUM_OMP_THREADS}
    },
    "aux_data_path": "${DITTO_AUX_DATA_PATH}",
    "num_channels": ${DITTO_NUM_CHANNELS},
    "rbits": ${DITTO_RBITS},
    "block_size": ${DITTO_BLOCK_SIZE}
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
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -m sglang.launch_server \
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
