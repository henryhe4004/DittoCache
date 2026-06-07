#!/usr/bin/env bash
#
# AWQ Qwen：对比 flash-attn 基线 vs Ditto offloading（hash yaml）。
#
# 一次跑两种（默认）：
#   ./test_quantization_accuracy.sh
#
# 拆成两条单独跑（推荐各占一张卡）：
#   AWQ_METHODS=flashattn  ./test_quantization_accuracy.sh
#   AWQ_METHODS=offloading ./test_quantization_accuracy.sh
#
# 断点续跑（复用已生成的 jsonl）：
#   RESUME=1 AWQ_METHODS=offloading ./test_quantization_accuracy.sh
#
# 指定 GPU + LongBench（示例 GPU 4）：
#   AUTO_SELECT_GPUS=0 CUDA_VISIBLE_DEVICES=4 MODEL_PATH=/models/Qwen2.5-14B-Instruct-AWQ \
#     DATASET_NAME=longbench DATASET_PATH=/datasets/LongBench TOPK=0.10 CONFIG_SUFFIX=64K \
#     AWQ_METHODS=flashattn ./test_quantization_accuracy.sh
#
#   AUTO_SELECT_GPUS=0 CUDA_VISIBLE_DEVICES=5 MODEL_PATH=/models/Qwen2.5-14B-Instruct-AWQ \
#     DATASET_NAME=longbench DATASET_PATH=/datasets/LongBench TOPK=0.10 CONFIG_SUFFIX=64K \
#     AWQ_METHODS=offloading ./test_quantization_accuracy.sh
#
# CONFIG_SUFFIX（默认 64K）会同时决定 yaml 文件名，并在未手动设置时对齐：
#   MAX_SEQ_LEN、ENGINE_CONTEXT_LENGTH、MAX_TOTAL_TOKENS（8K→8192，32K→32768，64K→65536，128K→131072）。
# 覆盖示例：MAX_SEQ_LEN=131072 CONFIG_SUFFIX=64K …（仍用 64K yaml，但放宽序列上限）。
#
# 默认 ENGINE_QUANTIZATION=awq（普通 AWQ，不走 Marlin repack）。恢复 SGLang 自动选内核：ENGINE_QUANTIZATION=auto
#

set -euo pipefail

# Longer-than-HF-native context needs SGLang opt-in (CONFIG_SUFFIX≥64K 时常用).
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/test_accuracy.sh"

log_info() {
    echo "[INFO] $*"
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

[[ -x "${BASE_SCRIPT}" ]] || die "missing executable base script: ${BASE_SCRIPT}"

# AWQ quantized Qwen model path.
MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-14B-Instruct-AWQ}"

# Default to the model basename so AWQ picks AWQ config files directly.
MODEL_NAME_FOR_CONFIG="${MODEL_NAME_FOR_CONFIG:-$(basename "${MODEL_PATH}")}"

TOPK="${TOPK:-0.10}"
SELECTIVE_START_LEN="${SELECTIVE_START_LEN:-0}"
# Pick matching yaml under config/full_attn/ and config/hata_offloading/ (e.g. 64K, 32K, 8K).
CONFIG_SUFFIX="${CONFIG_SUFFIX:-64K}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/preds}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
# all | flashattn | offloading
AWQ_METHODS="${AWQ_METHODS:-all}"
# Plain AWQ avoids awq_marlin_repack shape errors on some HF AWQ checkpoints. Use ENGINE_QUANTIZATION=auto for SGLang default.
ENGINE_QUANTIZATION="${ENGINE_QUANTIZATION:-awq}"

[[ -e "${MODEL_PATH}" ]] || die "missing model: ${MODEL_PATH}"

run_method() {
    local method="$1"
    local run_tag_suffix="$2"

    log_info "start method=${method} model=$(basename "${MODEL_PATH}")"
    (
        export MODEL_PATH="${MODEL_PATH}"
        export MODEL_NAME="${MODEL_NAME_FOR_CONFIG}"
        export METHOD="${method}"
        export TOPK="${TOPK}"
        export SELECTIVE_START_LEN="${SELECTIVE_START_LEN}"
        export CONFIG_SUFFIX="${CONFIG_SUFFIX}"
        export OUTPUT_ROOT="${OUTPUT_ROOT}"
        export DRY_RUN="${DRY_RUN}"
        export RESUME="${RESUME}"
        export ENGINE_QUANTIZATION="${ENGINE_QUANTIZATION}"

        # Keep two runs isolated and easy to compare.
        export RUN_TAG="$(basename "${MODEL_PATH}")-longbench-top${TOPK}-${run_tag_suffix}"

        exec "${BASE_SCRIPT}"
    )
    log_info "done method=${method}"
}

case "${AWQ_METHODS}" in
    all)
        run_method "flashattn" "flashattn"
        run_method "offloading" "offloading"
        log_info "all done: flashattn + offloading on AWQ Qwen"
        ;;
    flashattn)
        run_method "flashattn" "flashattn"
        log_info "done: flashattn only (AWQ)"
        ;;
    offloading)
        run_method "offloading" "offloading"
        log_info "done: offloading only (AWQ)"
        ;;
    *)
        die "AWQ_METHODS must be all, flashattn, or offloading (got: ${AWQ_METHODS})"
        ;;
esac
