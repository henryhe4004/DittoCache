#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/jhe/Llama-3-8B-Instruct-Gradient-1048k}"
export DITTO_TARGET_SEQ_LEN="${DITTO_TARGET_SEQ_LEN:-8192}"
export DITTO_MAX_BATCH_SIZE="${DITTO_MAX_BATCH_SIZE:-16}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-16}"

exec bash "${SCRIPT_DIR}/flash_server.sh"
