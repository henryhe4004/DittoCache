#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET_NAME="${DATASET_NAME:-longbench}"
export DATASET_PATH="${DATASET_PATH:-/jhe/dataset/LongBench}"
export TASKS="${TASKS:-multi_news_e}"
export DATASET_E_MODE="${DATASET_E_MODE:-1}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-65536}"
export SELECTIVE_START_LEN="${SELECTIVE_START_LEN:-2048}"
NUM_GPUS="${NUM_GPUS:-}"

if [[ -n "${NUM_GPUS}" ]]; then
    if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] NUM_GPUS must be a positive integer, got: ${NUM_GPUS}" >&2
        exit 1
    fi

    if [[ -z "${MP_NUM:-}" && -z "${PP_NUM:-}" ]]; then
        export MP_NUM=1
        export PP_NUM="${NUM_GPUS}"
    elif [[ -n "${MP_NUM:-}" && -z "${PP_NUM:-}" ]]; then
        if (( NUM_GPUS % MP_NUM != 0 )); then
            echo "[ERROR] NUM_GPUS=${NUM_GPUS} is not divisible by MP_NUM=${MP_NUM}" >&2
            exit 1
        fi
        export PP_NUM="$(( NUM_GPUS / MP_NUM ))"
    elif [[ -z "${MP_NUM:-}" && -n "${PP_NUM:-}" ]]; then
        if (( NUM_GPUS % PP_NUM != 0 )); then
            echo "[ERROR] NUM_GPUS=${NUM_GPUS} is not divisible by PP_NUM=${PP_NUM}" >&2
            exit 1
        fi
        export MP_NUM="$(( NUM_GPUS / PP_NUM ))"
    elif (( MP_NUM * PP_NUM != NUM_GPUS )); then
        echo "[ERROR] NUM_GPUS=${NUM_GPUS} mismatches MP_NUM*PP_NUM=$(( MP_NUM * PP_NUM ))" >&2
        exit 1
    fi
fi

FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER:-}"
FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_LOWER:-}"
FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS:-${FIXED_NUM_SKIP_LAYER:-}}"
FIXED_DECAY_P="${FIXED_DECAY_P:-${DECAY_P:-}}"
FIXED_MAX_REUSE_COUNT="${FIXED_MAX_REUSE_COUNT:-${MAX_REUSE_COUNT:-}}"

if [[ -n "${FIXED_REUSE_THRESHOLD_UPPER}" && -z "${FIXED_REUSE_THRESHOLD_LOWER}" ]]; then
    FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_UPPER}"
fi

METHOD="${METHOD:-offloading}"
TOPK="${TOPK:-0.10}"
MODEL_PATH="${MODEL_PATH:-/jhe/Qwen2.5-14B-Instruct-1M}"
export METHOD
export TOPK
export MODEL_PATH
DERIVED_MODEL_NAME="$(basename "${MODEL_PATH}")"
if [[ -n "${MODEL_NAME:-}" && "${MODEL_NAME}" != "${DERIVED_MODEL_NAME}" ]]; then
    echo "[WARN] MODEL_NAME=${MODEL_NAME} mismatches MODEL_PATH basename=${DERIVED_MODEL_NAME}; using MODEL_PATH basename." >&2
fi
MODEL_NAME="${DERIVED_MODEL_NAME}"
export MODEL_NAME

BASE_CONFIG_FILE="${CONFIG_FILE:-}"
if [[ -z "${BASE_CONFIG_FILE}" ]]; then
    case "${METHOD}" in
        offloading)
            BASE_CONFIG_FILE="${SCRIPT_DIR}/config/hata_offloading/${MODEL_NAME}-64K.yaml"
            ;;
        flash-attn|flashattn)
            BASE_CONFIG_FILE="${SCRIPT_DIR}/config/full_attn/${MODEL_NAME}-64K.yaml"
            ;;
        *)
            BASE_CONFIG_FILE="${SCRIPT_DIR}/config/${MODEL_NAME}-64K-top${TOPK}.json"
            ;;
    esac
fi

if [[ ! -f "${BASE_CONFIG_FILE}" ]]; then
    echo "[ERROR] missing config file: ${BASE_CONFIG_FILE}" >&2
    exit 1
fi

GENERATED_CONFIG_DIR="${SCRIPT_DIR}/config/.generated"
mkdir -p "${GENERATED_CONFIG_DIR}"
CONFIG_EXT="${BASE_CONFIG_FILE##*.}"
THRESHOLD_TAG="${FIXED_REUSE_THRESHOLD_UPPER//./p}"
SKIP_TAG="${FIXED_NUM_SKIP_LAYERS:-keep}"
DECAY_TAG="${FIXED_DECAY_P:-keep}"
DECAY_TAG="${DECAY_TAG//./p}"
MAX_REUSE_TAG="${FIXED_MAX_REUSE_COUNT:-keep}"
MAX_REUSE_TAG="${MAX_REUSE_TAG//./p}"
GENERATED_CONFIG_FILE="${GENERATED_CONFIG_DIR}/$(basename "${BASE_CONFIG_FILE%.*}")-multinews-th${THRESHOLD_TAG}-skip${SKIP_TAG}-decay${DECAY_TAG}-reuse${MAX_REUSE_TAG}.${CONFIG_EXT}"

python3 - "${BASE_CONFIG_FILE}" "${GENERATED_CONFIG_FILE}" "${FIXED_REUSE_THRESHOLD_UPPER}" "${FIXED_REUSE_THRESHOLD_LOWER}" "${FIXED_NUM_SKIP_LAYERS}" "${FIXED_DECAY_P}" "${FIXED_MAX_REUSE_COUNT}" <<'PY'
import json
import os
import sys

import yaml


src, dst, upper_raw, lower_raw, skip_raw, decay_raw, max_reuse_raw = sys.argv[1:8]
upper = None if upper_raw == "" else float(upper_raw)
lower = None if lower_raw == "" else float(lower_raw)
skip_layers = None if skip_raw == "" else int(skip_raw)
decay_p = None if decay_raw == "" else float(decay_raw)
max_reuse_count = None if max_reuse_raw == "" else int(max_reuse_raw)

with open(src, "r", encoding="utf-8") as f:
    text = f.read()

ext = os.path.splitext(src)[1].lower()
if ext in {".yaml", ".yml"}:
    cfg = yaml.safe_load(text)
else:
    cfg = json.loads(text)

patched = 0


def patch_threshold(node):
    global patched
    if isinstance(node, dict):
        if upper is not None and "reuse_threshold_upper" in node:
            node["reuse_threshold_upper"] = upper
            patched += 1
        if lower is not None and "reuse_threshold_lower" in node:
            node["reuse_threshold_lower"] = lower
            patched += 1
        if skip_layers is not None and "num_skip_layers" in node:
            node["num_skip_layers"] = skip_layers
            patched += 1
        if max_reuse_count is not None and "max_reuse_count" in node:
            node["max_reuse_count"] = max_reuse_count
            patched += 1
        if decay_p is not None:
            if "decay_p" in node:
                node["decay_p"] = decay_p
                patched += 1
            if "deacy_p" in node:
                node["deacy_p"] = decay_p
                patched += 1
        for value in node.values():
            patch_threshold(value)
    elif isinstance(node, list):
        for item in node:
            patch_threshold(item)


patch_threshold(cfg)

if patched == 0:
    if isinstance(cfg, dict) and isinstance(cfg.get("offload"), dict):
        if upper is not None:
            cfg["offload"]["reuse_threshold_upper"] = upper
        if lower is not None:
            cfg["offload"]["reuse_threshold_lower"] = lower
        if skip_layers is not None:
            cfg["offload"]["num_skip_layers"] = skip_layers
        if max_reuse_count is not None:
            cfg["offload"]["max_reuse_count"] = max_reuse_count
        if decay_p is not None:
            cfg["offload"]["decay_p"] = decay_p
    elif isinstance(cfg, dict):
        for value in cfg.values():
            if isinstance(value, dict):
                if upper is not None:
                    value["reuse_threshold_upper"] = upper
                if lower is not None:
                    value["reuse_threshold_lower"] = lower
                if skip_layers is not None:
                    value["num_skip_layers"] = skip_layers
                if max_reuse_count is not None:
                    value["max_reuse_count"] = max_reuse_count
                if decay_p is not None:
                    value["decay_p"] = decay_p

with open(dst, "w", encoding="utf-8") as f:
    if ext in {".yaml", ".yml"}:
        yaml.safe_dump(cfg, f, sort_keys=False)
    else:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
PY

export CONFIG_FILE="${GENERATED_CONFIG_FILE}"
echo "[INFO] base_config_file=${BASE_CONFIG_FILE}"
echo "[INFO] effective_config_file=${CONFIG_FILE}"
echo "[INFO] fixed_max_reuse_count=${FIXED_MAX_REUSE_COUNT:-keep}"
echo "[INFO] effective_config_begin"
cat "${CONFIG_FILE}"
echo "[INFO] effective_config_end"

exec "${SCRIPT_DIR}/test_accuracy_benchmark.sh" "$@"
