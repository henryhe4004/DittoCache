#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=ditto "${SCRIPT_DIR}/run_itl_client_sweep_tmp.sh" "$@"
