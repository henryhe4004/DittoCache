#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "[overlap] $*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/test_ditto.sh}"

[[ -f "${BASE_SCRIPT}" ]] || die "missing base script: ${BASE_SCRIPT}"

# Overlap profiling defaults.
# You can still override any of these from the environment.
CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE:-off}"
DITTO_VARIANT="${DITTO_VARIANT:-offloading}"
DITTO_OFFLOADING_METHOD="${DITTO_OFFLOADING_METHOD:-hash}"
DITTO_RECORD_OVERLAP_STATS="${DITTO_RECORD_OVERLAP_STATS:-1}"
DITTO_TRANSFER_STATS_FILE="${DITTO_TRANSFER_STATS_FILE:-/tmp/ditto_overlap_stats.json}"

if [[ "${DITTO_VARIANT}" != "offloading" ]]; then
  die "test_overlap.sh only supports DITTO_VARIANT=offloading"
fi

if [[ "${DITTO_OFFLOADING_METHOD}" != "hash" ]]; then
  die "test_overlap.sh currently only supports DITTO_OFFLOADING_METHOD=hash"
fi

mkdir -p "$(dirname "${DITTO_TRANSFER_STATS_FILE}")"

echo "[overlap] base_script=${BASE_SCRIPT}"
echo "[overlap] CUDA_GRAPH_MODE=${CUDA_GRAPH_MODE} DITTO_VARIANT=${DITTO_VARIANT} DITTO_OFFLOADING_METHOD=${DITTO_OFFLOADING_METHOD}"
echo "[overlap] DITTO_RECORD_OVERLAP_STATS=${DITTO_RECORD_OVERLAP_STATS} DITTO_TRANSFER_STATS_FILE=${DITTO_TRANSFER_STATS_FILE}"

CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE}" \
DITTO_VARIANT="${DITTO_VARIANT}" \
DITTO_OFFLOADING_METHOD="${DITTO_OFFLOADING_METHOD}" \
DITTO_RECORD_OVERLAP_STATS="${DITTO_RECORD_OVERLAP_STATS}" \
DITTO_TRANSFER_STATS_FILE="${DITTO_TRANSFER_STATS_FILE}" \
bash "${BASE_SCRIPT}"

[[ -f "${DITTO_TRANSFER_STATS_FILE}" ]] || die "missing overlap stats file: ${DITTO_TRANSFER_STATS_FILE}"

python3 - "${DITTO_TRANSFER_STATS_FILE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))

print(f"[overlap] stats_file={path}")
print(f"[overlap] decode_steps={data.get('decode_steps', 0)} overlap_steps={data.get('overlap_steps', 0)}")
print(
    "[overlap] "
    f"avg_mean_recall={data.get('avg_overlap_mean_recall', 0.0):.6f} "
    f"avg_mean_precision={data.get('avg_overlap_mean_precision', 0.0):.6f} "
    f"avg_mean_jaccard={data.get('avg_overlap_mean_jaccard', 0.0):.6f}"
)
print(
    "[overlap] "
    f"avg_union_recall={data.get('avg_overlap_union_recall', 0.0):.6f} "
    f"avg_union_precision={data.get('avg_overlap_union_precision', 0.0):.6f} "
    f"avg_union_jaccard={data.get('avg_overlap_union_jaccard', 0.0):.6f}"
)

steps = data.get("steps", [])
for step in reversed(steps):
    overlap = step.get("overlap")
    if isinstance(overlap, dict):
        print(
            "[overlap] "
            f"last_step={step.get('step')} seq_len={step.get('seq_len')} "
            f"matched_layers={overlap.get('matched_layers', 0)} "
            f"mean_recall={overlap.get('mean_recall', 0.0):.6f} "
            f"mean_precision={overlap.get('mean_precision', 0.0):.6f} "
            f"mean_jaccard={overlap.get('mean_jaccard', 0.0):.6f}"
        )
        break
PY
