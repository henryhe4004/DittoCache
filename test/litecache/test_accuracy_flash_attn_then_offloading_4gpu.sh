#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_info() {
    echo "[INFO] $*"
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if (( ${#GPUS[@]} != 4 )); then
    die "GPU_IDS must contain exactly 4 GPU ids, got: ${GPU_IDS}"
fi

DATASETS=("aime25" "gpqa" "math500" "mmlu_pro")
SCRIPTS=(
    "${SCRIPT_DIR}/test_accuracy_aime25.sh"
    "${SCRIPT_DIR}/test_accuracy_gpqa.sh"
    "${SCRIPT_DIR}/test_accuracy_math500.sh"
    "${SCRIPT_DIR}/test_accuracy_mmlu_pro.sh"
)

FIRST_METHOD="${FIRST_METHOD:-flash-attn}"
SECOND_METHOD="${SECOND_METHOD:-offloading}"
MODEL_PATH="${MODEL_PATH:-/jhe/Llama-3-8B-Instruct-Gradient-1048k}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/preds}"
TOPK="${TOPK:-0.10}"
SELECTIVE_START_LEN="${SELECTIVE_START_LEN:-512}"
DRY_RUN="${DRY_RUN:-0}"

canonicalize_method() {
    local method="$1"
    case "${method}" in
        flash-attn)
            echo "flashattn"
            ;;
        *)
            echo "${method}"
            ;;
    esac
}

expected_output_dir() {
    local dataset="$1"
    local method="$2"
    local canonical_method
    canonical_method="$(canonicalize_method "${method}")"
    echo "${OUTPUT_ROOT}/${canonical_method}/${MODEL_TAG}-${dataset}-top${TOPK}"
}

verify_eval_artifacts() {
    local dataset="$1"
    local method="$2"
    local output_dir
    local result_json
    local eval_log
    local pred_jsonl

    output_dir="$(expected_output_dir "${dataset}" "${method}")"
    result_json="${output_dir}/result.json"
    eval_log="${output_dir}/eval.log"
    pred_jsonl="${output_dir}/${dataset}.jsonl"

    [[ -d "${output_dir}" ]] || die "missing output dir for ${dataset} method=${method}: ${output_dir}"
    [[ -s "${pred_jsonl}" ]] || die "missing or empty prediction file for ${dataset} method=${method}: ${pred_jsonl}"
    [[ -s "${eval_log}" ]] || die "missing or empty eval log for ${dataset} method=${method}: ${eval_log}"
    [[ -s "${result_json}" ]] || die "missing or empty result json for ${dataset} method=${method}: ${result_json}"

    python3 - "${result_json}" "${dataset}" <<'PY'
import json
import sys

result_path, dataset = sys.argv[1:]
with open(result_path, "r", encoding="utf-8") as f:
    data = json.load(f)

if dataset not in data:
    raise SystemExit(f"dataset '{dataset}' not found in {result_path}")

score = data[dataset]
if not isinstance(score, (int, float)):
    raise SystemExit(f"dataset '{dataset}' score is not numeric in {result_path}: {score!r}")
PY

    grep -q "^${dataset}:" "${eval_log}" || die "eval log does not contain dataset score line for ${dataset} method=${method}: ${eval_log}"
    log_info "verified artifacts dataset=${dataset} method=${method} output_dir=${output_dir}"
}

print_stage_summary() {
    local stage_method="$1"
    local dataset
    local output_dir
    local result_json

    log_info "stage summary method=${stage_method}"
    for dataset in "${DATASETS[@]}"; do
        output_dir="$(expected_output_dir "${dataset}" "${stage_method}")"
        result_json="${output_dir}/result.json"
        python3 - "${result_json}" "${dataset}" "${stage_method}" <<'PY'
import json
import sys

result_path, dataset, stage_method = sys.argv[1:]
with open(result_path, "r", encoding="utf-8") as f:
    data = json.load(f)

score = data[dataset]
print(f"[SUMMARY] method={stage_method} dataset={dataset} score={score}")
PY
    done
}

run_stage() {
    local stage_method="$1"
    local -a pids=()
    local -a labels=()
    local idx

    log_info "starting stage method=${stage_method}"

    for idx in "${!DATASETS[@]}"; do
        local dataset="${DATASETS[$idx]}"
        local script_path="${SCRIPTS[$idx]}"
        local gpu="${GPUS[$idx]}"

        [[ -x "${script_path}" ]] || die "missing executable dataset script: ${script_path}"

        log_info "launch dataset=${dataset} gpu=${gpu} method=${stage_method}"
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            export AUTO_SELECT_GPUS=0
            export METHOD="${stage_method}"
            export SELECTIVE_START_LEN="${SELECTIVE_START_LEN}"
            exec "${script_path}"
        ) &

        pids+=("$!")
        labels+=("${dataset}@gpu${gpu}")
    done

    local status=0
    for idx in "${!pids[@]}"; do
        local pid="${pids[$idx]}"
        local label="${labels[$idx]}"
        local dataset="${DATASETS[$idx]}"
        if wait "${pid}"; then
            log_info "finished ${label} method=${stage_method}"
            if [[ "${DRY_RUN}" != "1" ]]; then
                verify_eval_artifacts "${dataset}" "${stage_method}"
            else
                log_info "skip artifact verification for ${label} because DRY_RUN=1"
            fi
        else
            status=$?
            echo "[ERROR] failed ${label} method=${stage_method} exit=${status}" >&2
            local jdx
            for jdx in "${!pids[@]}"; do
                local other_pid="${pids[$jdx]}"
                if [[ "${other_pid}" != "${pid}" ]] && kill -0 "${other_pid}" 2>/dev/null; then
                    kill -TERM "${other_pid}" 2>/dev/null || true
                fi
            done
            wait || true
            return "${status}"
        fi
    done

    if [[ "${DRY_RUN}" != "1" ]]; then
        print_stage_summary "${stage_method}"
    fi
    log_info "stage completed method=${stage_method}"
}

run_stage "${FIRST_METHOD}"
run_stage "${SECOND_METHOD}"

log_info "all stages completed: ${FIRST_METHOD} -> ${SECOND_METHOD}"
