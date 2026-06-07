#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "[ditto] $*" >&2
  exit 2
}

PROMPT_FILE="${PROMPT_FILE:-/examples/longbench_qmsum_prompt_1k.txt}"
MODEL_PATH="${MODEL_PATH:-/models/Llama-3-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
DITTO_ROOT="${DITTO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ATTN_PATTERN_PATH="${ATTN_PATTERN_PATH:-${DITTO_ROOT}/auxiliary/attn_pattern/${MODEL_NAME}}"
SELFTEST_PY="${SELFTEST_PY:-/python/sglang/srt/models/ditto/minimal_selftest.py}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-2048}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
GPU_MEMORY_BUDGET="${GPU_MEMORY_BUDGET:-16}"
DITTO_TOKEN_BUDGET="${DITTO_TOKEN_BUDGET:-0.10}"
DITTO_SINK_BUDGET="${DITTO_SINK_BUDGET:-4}"
DITTO_RECENT_BUDGET="${DITTO_RECENT_BUDGET:-64}"
DITTO_SELECTIVE_START_LEN="${DITTO_SELECTIVE_START_LEN:-0}"
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
#   - ditto : enable Ditto graph only (recommended for offloading)
#   - engine    : enable Engine graph only
# - DITTO_ENABLE_CUDA_GRAPH / ENGINE_ENABLE_CUDA_GRAPH remain supported for
#   backward compatibility when CUDA_GRAPH_MODE is unset.
# - DITTO_VARIANT selects Ditto variant (default: offloading)
# - DITTO_OFFLOADING_METHOD selects backend when variant=offloading (default: hash)
# - DITTO_NUM_SKIP_LAYERS / DITTO_NUM_OVERLAPPED_HEADS align head placement
#   with hash-offloading defaults from run_pred config.
# - DITTO_MAX_REUSE_COUNT forces a refresh after N consecutive reuses per KV head.
CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE:-}"
DITTO_ENABLE_CUDA_GRAPH="${DITTO_ENABLE_CUDA_GRAPH:-0}"
ENGINE_ENABLE_CUDA_GRAPH="${ENGINE_ENABLE_CUDA_GRAPH:-0}"
DITTO_VARIANT="${DITTO_VARIANT:-offloading}"
DITTO_OFFLOADING_METHOD="${DITTO_OFFLOADING_METHOD:-hash}"
DITTO_NUM_SKIP_LAYERS="${DITTO_NUM_SKIP_LAYERS:-1}"
DITTO_MAX_REUSE_COUNT="${DITTO_MAX_REUSE_COUNT:-10}"
if [[ -z "${DITTO_NUM_OVERLAPPED_HEADS+x}" ]]; then
  case "${MODEL_NAME}" in
    Qwen2.5-14B-Instruct-1M)
      DITTO_NUM_OVERLAPPED_HEADS="7"
      ;;
    Llama-3-8B-Instruct-Gradient-1048k)
      DITTO_NUM_OVERLAPPED_HEADS="8"
      ;;
    *)
      DITTO_NUM_OVERLAPPED_HEADS="8"
      ;;
  esac
fi

if [[ -n "${CUDA_GRAPH_MODE}" ]]; then
  case "${CUDA_GRAPH_MODE}" in
    off)
      DITTO_ENABLE_CUDA_GRAPH="0"
      ENGINE_ENABLE_CUDA_GRAPH="0"
      ;;
    ditto)
      DITTO_ENABLE_CUDA_GRAPH="1"
      ENGINE_ENABLE_CUDA_GRAPH="0"
      ;;
    engine)
      DITTO_ENABLE_CUDA_GRAPH="0"
      ENGINE_ENABLE_CUDA_GRAPH="1"
      ;;
    both)
      echo "[ditto] CUDA_GRAPH_MODE=both is not supported. Pick 'ditto' or 'engine'." >&2
      exit 2
      ;;
    *)
      echo "[ditto] Invalid CUDA_GRAPH_MODE='${CUDA_GRAPH_MODE}'. Use off|ditto|engine." >&2
      exit 2
      ;;
  esac
fi

if [[ "${DITTO_ENABLE_CUDA_GRAPH}" == "1" && "${ENGINE_ENABLE_CUDA_GRAPH}" == "1" ]]; then
  echo "[ditto] Ditto graph and Engine graph cannot be enabled simultaneously." >&2
  exit 2
fi

if [[ "${DITTO_ENABLE_CUDA_GRAPH}" == "1" ]]; then
  EFFECTIVE_CUDA_GRAPH_MODE="ditto"
elif [[ "${ENGINE_ENABLE_CUDA_GRAPH}" == "1" ]]; then
  EFFECTIVE_CUDA_GRAPH_MODE="engine"
else
  EFFECTIVE_CUDA_GRAPH_MODE="off"
fi

# Debug knobs.
# Note: CPUGather debug currently calls stream.query(), which is not capture-safe.
DITTO_DEBUG_STALL="${DITTO_DEBUG_STALL:-1}"
DITTO_DEBUG_STALL_MIN_MS="${DITTO_DEBUG_STALL_MIN_MS:-0}"
if [[ -z "${DITTO_DEBUG_CPUGATHER+x}" ]]; then
  if [[ "${ENGINE_ENABLE_CUDA_GRAPH}" == "1" || "${DITTO_ENABLE_CUDA_GRAPH}" == "1" ]]; then
    DITTO_DEBUG_CPUGATHER="0"
  else
    DITTO_DEBUG_CPUGATHER="1"
  fi
fi
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

EXTRA_ARGS=()
if [[ "${DITTO_ENABLE_CUDA_GRAPH}" == "1" ]]; then
  EXTRA_ARGS+=(--ditto-enable-cuda-graph)
fi
if [[ "${ENGINE_ENABLE_CUDA_GRAPH}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-cuda-graph)
fi
if [[ "${DITTO_VARIANT}" == "offloading" ]]; then
  EXTRA_ARGS+=(--offloading-method "${DITTO_OFFLOADING_METHOD}")
fi

echo "MODEL_NAME=${MODEL_NAME} MODEL_PATH=${MODEL_PATH} ATTN_PATTERN_PATH=${ATTN_PATTERN_PATH} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "CUDA_GRAPH_MODE=${EFFECTIVE_CUDA_GRAPH_MODE} DITTO_ENABLE_CUDA_GRAPH=${DITTO_ENABLE_CUDA_GRAPH} ENGINE_ENABLE_CUDA_GRAPH=${ENGINE_ENABLE_CUDA_GRAPH} DITTO_VARIANT=${DITTO_VARIANT} DITTO_OFFLOADING_METHOD=${DITTO_OFFLOADING_METHOD} DITTO_NUM_SKIP_LAYERS=${DITTO_NUM_SKIP_LAYERS} DITTO_NUM_OVERLAPPED_HEADS=${DITTO_NUM_OVERLAPPED_HEADS} DITTO_MAX_REUSE_COUNT=${DITTO_MAX_REUSE_COUNT} DITTO_TOKEN_BUDGET=${DITTO_TOKEN_BUDGET} DITTO_SINK_BUDGET=${DITTO_SINK_BUDGET} DITTO_RECENT_BUDGET=${DITTO_RECENT_BUDGET} DITTO_SELECTIVE_START_LEN=${DITTO_SELECTIVE_START_LEN} DITTO_DEBUG_STALL=${DITTO_DEBUG_STALL} DITTO_DEBUG_STALL_MIN_MS=${DITTO_DEBUG_STALL_MIN_MS} DITTO_DEBUG_CPUGATHER=${DITTO_DEBUG_CPUGATHER} CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}"

CMD=(
  python3 "${SELFTEST_PY}"
  --model-path "${MODEL_PATH}"
  --prompt "$PROMPT"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-tokens "${MAX_TOKENS}"
  --max-total-tokens "${MAX_TOTAL_TOKENS}"
  --max-batch-size "${MAX_BATCH_SIZE}"
  --gpu-memory-budget "${GPU_MEMORY_BUDGET}"
  --token-budget "${DITTO_TOKEN_BUDGET}"
  --sink-budget "${DITTO_SINK_BUDGET}"
  --recent-budget "${DITTO_RECENT_BUDGET}"
  --selective-start-len "${DITTO_SELECTIVE_START_LEN}"
  --num-skip-layers "${DITTO_NUM_SKIP_LAYERS}"
  --num-overlapped-heads "${DITTO_NUM_OVERLAPPED_HEADS}"
  --max-reuse-count "${DITTO_MAX_REUSE_COUNT}"
  --attn-pattern-path "${ATTN_PATTERN_PATH}"
  --variant "${DITTO_VARIANT}"
)

CMD+=("${EXTRA_ARGS[@]}")

CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING}" \
PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
DITTO_DEBUG_STALL="${DITTO_DEBUG_STALL}" \
DITTO_DEBUG_STALL_MIN_MS="${DITTO_DEBUG_STALL_MIN_MS}" \
DITTO_DEBUG_CPUGATHER="${DITTO_DEBUG_CPUGATHER}" \
PYTORCH_ALLOC_CONF=expandable_segments:True \
"${CMD[@]}"
