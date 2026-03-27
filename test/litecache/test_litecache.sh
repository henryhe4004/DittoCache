#!/usr/bin/env bash
set -euo pipefail

PROMPT="$(cat /jhe/longbench_qmsum_prompt_1k.txt)"

echo "prompt chars=${#PROMPT}"
PROMPT="$PROMPT" python3 - <<'PY'
import os
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('/jhe/Qwen2.5-14B-Instruct-1M', trust_remote_code=True)
print('prompt tokens=', len(tok(os.environ['PROMPT'], add_special_tokens=False)['input_ids']))
PY

# Toggle runtime behavior without editing this script.
# - CUDA_GRAPH_MODE controls CUDA graph branch:
#   - off       : disable both graphs
#   - litecache : enable LiteCache graph only (recommended for offloading)
#   - engine    : enable Engine graph only
# - LITECACHE_ENABLE_CUDA_GRAPH / ENGINE_ENABLE_CUDA_GRAPH remain supported for
#   backward compatibility when CUDA_GRAPH_MODE is unset.
# - LITECACHE_VARIANT selects LiteCache variant (default: offloading)
# - LITECACHE_OFFLOADING_METHOD selects backend when variant=offloading (default: hash)
# - LITECACHE_NUM_SKIP_LAYERS / LITECACHE_NUM_OVERLAPPED_HEADS align head placement
#   with hash-offloading defaults from run_pred config.
CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE:-}"
LITECACHE_ENABLE_CUDA_GRAPH="${LITECACHE_ENABLE_CUDA_GRAPH:-0}"
ENGINE_ENABLE_CUDA_GRAPH="${ENGINE_ENABLE_CUDA_GRAPH:-0}"
LITECACHE_VARIANT="${LITECACHE_VARIANT:-offloading}"
LITECACHE_OFFLOADING_METHOD="${LITECACHE_OFFLOADING_METHOD:-hash}"
LITECACHE_NUM_SKIP_LAYERS="${LITECACHE_NUM_SKIP_LAYERS:-1}"
LITECACHE_NUM_OVERLAPPED_HEADS="${LITECACHE_NUM_OVERLAPPED_HEADS:-7}"

if [[ -n "${CUDA_GRAPH_MODE}" ]]; then
  case "${CUDA_GRAPH_MODE}" in
    off)
      LITECACHE_ENABLE_CUDA_GRAPH="0"
      ENGINE_ENABLE_CUDA_GRAPH="0"
      ;;
    litecache)
      LITECACHE_ENABLE_CUDA_GRAPH="1"
      ENGINE_ENABLE_CUDA_GRAPH="0"
      ;;
    engine)
      LITECACHE_ENABLE_CUDA_GRAPH="0"
      ENGINE_ENABLE_CUDA_GRAPH="1"
      ;;
    both)
      echo "[litecache] CUDA_GRAPH_MODE=both is not supported. Pick 'litecache' or 'engine'." >&2
      exit 2
      ;;
    *)
      echo "[litecache] Invalid CUDA_GRAPH_MODE='${CUDA_GRAPH_MODE}'. Use off|litecache|engine." >&2
      exit 2
      ;;
  esac
fi

if [[ "${LITECACHE_ENABLE_CUDA_GRAPH}" == "1" && "${ENGINE_ENABLE_CUDA_GRAPH}" == "1" ]]; then
  echo "[litecache] LiteCache graph and Engine graph cannot be enabled simultaneously." >&2
  exit 2
fi

if [[ "${LITECACHE_ENABLE_CUDA_GRAPH}" == "1" ]]; then
  EFFECTIVE_CUDA_GRAPH_MODE="litecache"
elif [[ "${ENGINE_ENABLE_CUDA_GRAPH}" == "1" ]]; then
  EFFECTIVE_CUDA_GRAPH_MODE="engine"
else
  EFFECTIVE_CUDA_GRAPH_MODE="off"
fi

# Debug knobs.
# Note: CPUGather debug currently calls stream.query(), which is not capture-safe.
LITECACHE_DEBUG_STALL="${LITECACHE_DEBUG_STALL:-1}"
LITECACHE_DEBUG_STALL_MIN_MS="${LITECACHE_DEBUG_STALL_MIN_MS:-0}"
if [[ -z "${LITECACHE_DEBUG_CPUGATHER+x}" ]]; then
  if [[ "${ENGINE_ENABLE_CUDA_GRAPH}" == "1" || "${LITECACHE_ENABLE_CUDA_GRAPH}" == "1" ]]; then
    LITECACHE_DEBUG_CPUGATHER="0"
  else
    LITECACHE_DEBUG_CPUGATHER="1"
  fi
fi
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

EXTRA_ARGS=()
if [[ "${LITECACHE_ENABLE_CUDA_GRAPH}" == "1" ]]; then
  EXTRA_ARGS+=(--litecache-enable-cuda-graph)
fi
if [[ "${ENGINE_ENABLE_CUDA_GRAPH}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-cuda-graph)
fi
if [[ "${LITECACHE_VARIANT}" == "offloading" ]]; then
  EXTRA_ARGS+=(--offloading-method "${LITECACHE_OFFLOADING_METHOD}")
fi

echo "CUDA_GRAPH_MODE=${EFFECTIVE_CUDA_GRAPH_MODE} LITECACHE_ENABLE_CUDA_GRAPH=${LITECACHE_ENABLE_CUDA_GRAPH} ENGINE_ENABLE_CUDA_GRAPH=${ENGINE_ENABLE_CUDA_GRAPH} LITECACHE_VARIANT=${LITECACHE_VARIANT} LITECACHE_OFFLOADING_METHOD=${LITECACHE_OFFLOADING_METHOD} LITECACHE_NUM_SKIP_LAYERS=${LITECACHE_NUM_SKIP_LAYERS} LITECACHE_NUM_OVERLAPPED_HEADS=${LITECACHE_NUM_OVERLAPPED_HEADS} LITECACHE_DEBUG_STALL=${LITECACHE_DEBUG_STALL} LITECACHE_DEBUG_STALL_MIN_MS=${LITECACHE_DEBUG_STALL_MIN_MS} LITECACHE_DEBUG_CPUGATHER=${LITECACHE_DEBUG_CPUGATHER} CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}"

CMD=(
  python3 /jhe/sglang/python/sglang/srt/models/litecache/minimal_selftest.py
  --model-path /jhe/Qwen2.5-14B-Instruct-1M
  --prompt "$PROMPT"
  --max-new-tokens 16
  --max-tokens 4096
  --max-total-tokens 2048
  --max-batch-size 1
  --gpu-memory-budget 16
  --token-budget 0.2
  --sink-budget 4
  --recent-budget 128
  --num-skip-layers "${LITECACHE_NUM_SKIP_LAYERS}"
  --num-overlapped-heads "${LITECACHE_NUM_OVERLAPPED_HEADS}"
  --attn-pattern-path /jhe/myTransformer/auxiliary/attn_pattern/Qwen2.5-14B-Instruct-1M
  --variant "${LITECACHE_VARIANT}"
)

CMD+=("${EXTRA_ARGS[@]}")

CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING}" \
PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=1 \
LITECACHE_DEBUG_STALL="${LITECACHE_DEBUG_STALL}" \
LITECACHE_DEBUG_STALL_MIN_MS="${LITECACHE_DEBUG_STALL_MIN_MS}" \
LITECACHE_DEBUG_CPUGATHER="${LITECACHE_DEBUG_CPUGATHER}" \
PYTORCH_ALLOC_CONF=expandable_segments:True \
"${CMD[@]}"
