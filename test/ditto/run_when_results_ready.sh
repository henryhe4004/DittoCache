#!/usr/bin/env bash

set -euo pipefail

log_info() {
    echo "[INFO] $*"
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

CHECK_ITEMS="${CHECK_ITEMS:-}"
TARGET_SCRIPT="${TARGET_SCRIPT:-}"
POLL_SEC="${POLL_SEC:-30}"
LOG_FILE="${LOG_FILE:-}"

[[ -n "${CHECK_ITEMS}" ]] || die "CHECK_ITEMS is required"
[[ -n "${TARGET_SCRIPT}" ]] || die "TARGET_SCRIPT is required"
[[ -x "${TARGET_SCRIPT}" ]] || die "TARGET_SCRIPT is not executable: ${TARGET_SCRIPT}"

all_items_ready() {
    python3 - "${CHECK_ITEMS}" <<'PY'
import json
import os
import sys

check_items = sys.argv[1]

ready = True
for item in check_items.split(";"):
    item = item.strip()
    if not item:
        continue
    try:
        dataset, result_path = item.split(":", 1)
    except ValueError:
        print(f"[ERROR] invalid CHECK_ITEMS entry: {item}", flush=True)
        sys.exit(2)

    if not os.path.isfile(result_path):
        print(f"[WAIT] dataset={dataset} missing_result={result_path}", flush=True)
        ready = False
        continue

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[WAIT] dataset={dataset} unreadable_result={result_path} err={exc}", flush=True)
        ready = False
        continue

    if dataset not in data:
        print(f"[WAIT] dataset={dataset} missing_key_in_result={result_path}", flush=True)
        ready = False
        continue

    score = data[dataset]
    if not isinstance(score, (int, float)):
        print(f"[WAIT] dataset={dataset} non_numeric_score={score!r} result={result_path}", flush=True)
        ready = False
        continue

    print(f"[READY] dataset={dataset} score={score} result={result_path}", flush=True)

sys.exit(0 if ready else 1)
PY
}

log_info "waiting for required result files"
while true; do
    if all_items_ready; then
        break
    fi
    sleep "${POLL_SEC}"
done

log_info "all required result files are ready"
log_info "launching target script: ${TARGET_SCRIPT}"

if [[ -n "${LOG_FILE}" ]]; then
    exec "${TARGET_SCRIPT}" >>"${LOG_FILE}" 2>&1
else
    exec "${TARGET_SCRIPT}"
fi
