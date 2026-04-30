#!/usr/bin/env bash

set -euo pipefail

log_info() {
    echo "[INFO] $*"
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

WAIT_MODEL_PATH="${WAIT_MODEL_PATH:-}"
TARGET_SCRIPT="${TARGET_SCRIPT:-}"
POLL_SEC="${POLL_SEC:-30}"
LOG_FILE="${LOG_FILE:-}"

[[ -n "${WAIT_MODEL_PATH}" ]] || die "WAIT_MODEL_PATH is required"
[[ -n "${TARGET_SCRIPT}" ]] || die "TARGET_SCRIPT is required"
[[ -x "${TARGET_SCRIPT}" ]] || die "TARGET_SCRIPT is not executable: ${TARGET_SCRIPT}"

find_matching_pids() {
    python3 - "${WAIT_MODEL_PATH}" "$$" <<'PY'
import os
import subprocess
import sys

model_path = sys.argv[1]
self_pid = int(sys.argv[2])
scanner_pid = os.getpid()

out = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
matches = []
for line in out.splitlines():
    line = line.rstrip()
    if not line:
        continue
    parts = line.strip().split(None, 1)
    if len(parts) != 2:
        continue
    pid = int(parts[0])
    cmd = parts[1]
    if pid in {self_pid, scanner_pid}:
        continue
    if model_path in cmd:
        matches.append((pid, cmd))

for pid, cmd in matches:
    print(f"{pid}\t{cmd}")
PY
}

log_info "waiting for model processes to finish: ${WAIT_MODEL_PATH}"
while true; do
    mapfile -t matches < <(find_matching_pids)
    if (( ${#matches[@]} == 0 )); then
        break
    fi

    log_info "still running ${#matches[@]} process(es) for ${WAIT_MODEL_PATH}"
    for line in "${matches[@]}"; do
        log_info "wait_match ${line}"
    done
    sleep "${POLL_SEC}"
done

log_info "no remaining process found for ${WAIT_MODEL_PATH}"
log_info "launching target script: ${TARGET_SCRIPT}"

if [[ -n "${LOG_FILE}" ]]; then
    exec "${TARGET_SCRIPT}" >>"${LOG_FILE}" 2>&1
else
    exec "${TARGET_SCRIPT}"
fi
