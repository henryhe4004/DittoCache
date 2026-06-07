#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BENCHMARK_NAME="${BENCHMARK_NAME:-livecodebench}"
export SCENARIO="${SCENARIO:-codegeneration}"
export RELEASE_VERSION="${RELEASE_VERSION:-release_latest}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export AUTO_SELECT_GPUS="${AUTO_SELECT_GPUS:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export EVALUATE="${EVALUATE:-1}"
export SELECTIVE_START_LEN="${SELECTIVE_START_LEN:-512}"

exec "${SCRIPT_DIR}/test_accuracy_livecodebench_benchmark.sh" "$@"
