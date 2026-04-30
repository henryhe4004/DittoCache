#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "[litecache] $*" >&2
  exit 2
}

PROMPT_FILE="${PROMPT_FILE:-/jhe/longbench_qmsum_prompt_1k.txt}"
MODEL_PATH="${MODEL_PATH:-/jhe/Llama-3-8B-Instruct-Gradient-1048k}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
LITECACHE_ROOT="${LITECACHE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ATTN_PATTERN_PATH="${ATTN_PATTERN_PATH:-${LITECACHE_ROOT}/auxiliary/attn_pattern/${MODEL_NAME}}"
SELFTEST_PY="${SELFTEST_PY:-/jhe/sglang/python/sglang/srt/models/litecache/minimal_selftest.py}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-2048}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
GPU_MEMORY_BUDGET="${GPU_MEMORY_BUDGET:-16}"
LITECACHE_TOKEN_BUDGET="${LITECACHE_TOKEN_BUDGET:-0.10}"
LITECACHE_SINK_BUDGET="${LITECACHE_SINK_BUDGET:-4}"
LITECACHE_RECENT_BUDGET="${LITECACHE_RECENT_BUDGET:-64}"
LITECACHE_SELECTIVE_START_LEN="${LITECACHE_SELECTIVE_START_LEN:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

[[ -f "${PROMPT_FILE}" ]] || die "missing prompt file: ${PROMPT_FILE}"
[[ -e "${MODEL_PATH}" ]] || die "missing model path: ${MODEL_PATH}"
[[ -f "${SELFTEST_PY}" ]] || die "missing selftest entry: ${SELFTEST_PY}"
[[ -d "${ATTN_PATTERN_PATH}" ]] || die "missing attn pattern dir: ${ATTN_PATTERN_PATH}"

PROMPT="$(cat "${PROMPT_FILE}")"

echo "prompt chars=${#PROMPT}"
PROMPT="$PROMPT" MODEL_PATH="$MODEL_PATH" python3 - <<'PY'
import os
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"], trust_remote_code=True)
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
# - LITECACHE_MAX_REUSE_COUNT forces a refresh after N consecutive reuses per KV head.
CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE:-}"
LITECACHE_ENABLE_CUDA_GRAPH="${LITECACHE_ENABLE_CUDA_GRAPH:-0}"
ENGINE_ENABLE_CUDA_GRAPH="${ENGINE_ENABLE_CUDA_GRAPH:-0}"
LITECACHE_VARIANT="${LITECACHE_VARIANT:-offloading}"
LITECACHE_OFFLOADING_METHOD="${LITECACHE_OFFLOADING_METHOD:-hash}"
LITECACHE_NUM_SKIP_LAYERS="${LITECACHE_NUM_SKIP_LAYERS:-1}"
LITECACHE_MAX_REUSE_COUNT="${LITECACHE_MAX_REUSE_COUNT:-10}"
if [[ -z "${LITECACHE_NUM_OVERLAPPED_HEADS+x}" ]]; then
  case "${MODEL_NAME}" in
    Qwen2.5-14B-Instruct-1M)
      LITECACHE_NUM_OVERLAPPED_HEADS="7"
      ;;
    Llama-3-8B-Instruct-Gradient-1048k)
      LITECACHE_NUM_OVERLAPPED_HEADS="8"
      ;;
    *)
      LITECACHE_NUM_OVERLAPPED_HEADS="8"
      ;;
  esac
fi

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

echo "MODEL_NAME=${MODEL_NAME} MODEL_PATH=${MODEL_PATH} ATTN_PATTERN_PATH=${ATTN_PATTERN_PATH} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "CUDA_GRAPH_MODE=${EFFECTIVE_CUDA_GRAPH_MODE} LITECACHE_ENABLE_CUDA_GRAPH=${LITECACHE_ENABLE_CUDA_GRAPH} ENGINE_ENABLE_CUDA_GRAPH=${ENGINE_ENABLE_CUDA_GRAPH} LITECACHE_VARIANT=${LITECACHE_VARIANT} LITECACHE_OFFLOADING_METHOD=${LITECACHE_OFFLOADING_METHOD} LITECACHE_NUM_SKIP_LAYERS=${LITECACHE_NUM_SKIP_LAYERS} LITECACHE_NUM_OVERLAPPED_HEADS=${LITECACHE_NUM_OVERLAPPED_HEADS} LITECACHE_MAX_REUSE_COUNT=${LITECACHE_MAX_REUSE_COUNT} LITECACHE_TOKEN_BUDGET=${LITECACHE_TOKEN_BUDGET} LITECACHE_SINK_BUDGET=${LITECACHE_SINK_BUDGET} LITECACHE_RECENT_BUDGET=${LITECACHE_RECENT_BUDGET} LITECACHE_SELECTIVE_START_LEN=${LITECACHE_SELECTIVE_START_LEN} LITECACHE_DEBUG_STALL=${LITECACHE_DEBUG_STALL} LITECACHE_DEBUG_STALL_MIN_MS=${LITECACHE_DEBUG_STALL_MIN_MS} LITECACHE_DEBUG_CPUGATHER=${LITECACHE_DEBUG_CPUGATHER} CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}"

CMD=(
  python3 "${SELFTEST_PY}"
  --model-path "${MODEL_PATH}"
  --prompt "$PROMPT"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-tokens "${MAX_TOKENS}"
  --max-total-tokens "${MAX_TOTAL_TOKENS}"
  --max-batch-size "${MAX_BATCH_SIZE}"
  --gpu-memory-budget "${GPU_MEMORY_BUDGET}"
  --token-budget "${LITECACHE_TOKEN_BUDGET}"
  --sink-budget "${LITECACHE_SINK_BUDGET}"
  --recent-budget "${LITECACHE_RECENT_BUDGET}"
  --selective-start-len "${LITECACHE_SELECTIVE_START_LEN}"
  --num-skip-layers "${LITECACHE_NUM_SKIP_LAYERS}"
  --num-overlapped-heads "${LITECACHE_NUM_OVERLAPPED_HEADS}"
  --max-reuse-count "${LITECACHE_MAX_REUSE_COUNT}"
  --attn-pattern-path "${ATTN_PATTERN_PATH}"
  --variant "${LITECACHE_VARIANT}"
)

CMD+=("${EXTRA_ARGS[@]}")

CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING}" \
PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
LITECACHE_DEBUG_STALL="${LITECACHE_DEBUG_STALL}" \
LITECACHE_DEBUG_STALL_MIN_MS="${LITECACHE_DEBUG_STALL_MIN_MS}" \
LITECACHE_DEBUG_CPUGATHER="${LITECACHE_DEBUG_CPUGATHER}" \
PYTORCH_ALLOC_CONF=expandable_segments:True \
"${CMD[@]}"
