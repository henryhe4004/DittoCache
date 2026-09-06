#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/jhe/Llama-3-8B-Instruct-Gradient-1048k}"
export DITTO_AUX_ROOT="${DITTO_AUX_ROOT:-/jhe/myTransformer/auxiliary}"
export DITTO_TARGET_SEQ_LEN="${DITTO_TARGET_SEQ_LEN:-8192}"
export DITTO_MAX_BATCH_SIZE="${DITTO_MAX_BATCH_SIZE:-64}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-64}"

exec bash "${SCRIPT_DIR}/ditto_server.sh"
