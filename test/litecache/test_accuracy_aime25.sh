#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET_NAME="${DATASET_NAME:-aime25}"
export DATASET_PATH="${DATASET_PATH:-/jhe/dataset/aime25}"
export TASKS="${TASKS:-aime25}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-65536}"
export SELECTIVE_START_LEN="${SELECTIVE_START_LEN:-512}"

exec "${SCRIPT_DIR}/test_accuracy_benchmark.sh" "$@"
