#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILES=(
    B0
    B1
    B2
    B3
    B4
    B5
    B6
)

if [[ -n "${ABLATION_PROFILES:-}" ]]; then
    read -r -a PROFILES <<< "${ABLATION_PROFILES}"
fi

BASE_RUN_TAG="${RUN_TAG:-}"
for profile in "${PROFILES[@]}"; do
    if [[ -n "${BASE_RUN_TAG}" ]]; then
        profile_run_tag="${BASE_RUN_TAG}-${profile}"
    else
        profile_run_tag=""
    fi

    ABLATION_PROFILE="${profile}" \
    RUN_TAG="${profile_run_tag}" \
        "${SCRIPT_DIR}/test_accuracy_benchmark.sh" "$@"
done
