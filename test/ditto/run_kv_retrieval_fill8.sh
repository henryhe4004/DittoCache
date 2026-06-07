#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-14B-Instruct-1M}"
FULL_DATASET="${FULL_DATASET:-/datasets/InfiniteBench/kv_retrieval.jsonl}"
EXISTING_FILE="${EXISTING_FILE:-/preds_nway/workers/offloading/Qwen2.5-14B-Instruct-1M-infinitebench-top0.10-nway-thup0p8-thlo-1p0-skipkeep-decaykeep-reusekeep-w02/kv_retrieval.jsonl}"

BASE_RUNNER="${BASE_RUNNER:-${SCRIPT_DIR}/test_accuracy_infinitebench.sh}"
EVAL_PY="${EVAL_PY:-${SCRIPT_DIR}/eval_longbench_infinitebench.py}"

METHOD="${METHOD:-offloading}"
TOPK="${TOPK:-0.10}"
FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER:-0.8}"
FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_LOWER:--1.0}"
FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS:-}"
FIXED_DECAY_P="${FIXED_DECAY_P:-}"
FIXED_MAX_REUSE_COUNT="${FIXED_MAX_REUSE_COUNT:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
GPU_COUNT="${#GPUS[@]}"
if (( GPU_COUNT == 0 )); then
  echo "[ERROR] CUDA_VISIBLE_DEVICES is empty" >&2
  exit 1
fi

N_WAY="${N_WAY:-${GPU_COUNT}}"
if ! [[ "${N_WAY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] N_WAY must be positive integer, got ${N_WAY}" >&2
  exit 1
fi
if (( N_WAY > GPU_COUNT )); then
  echo "[ERROR] N_WAY=${N_WAY} > gpu_count=${GPU_COUNT}" >&2
  exit 1
fi

[[ -f "${FULL_DATASET}" ]] || { echo "[ERROR] missing FULL_DATASET=${FULL_DATASET}" >&2; exit 1; }
[[ -f "${EXISTING_FILE}" ]] || { echo "[ERROR] missing EXISTING_FILE=${EXISTING_FILE}" >&2; exit 1; }
[[ -x "${BASE_RUNNER}" ]] || { echo "[ERROR] missing executable BASE_RUNNER=${BASE_RUNNER}" >&2; exit 1; }
[[ -f "${EVAL_PY}" ]] || { echo "[ERROR] missing EVAL_PY=${EVAL_PY}" >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/preds_nway}"
RUN_TAG_BASE="${RUN_TAG_BASE:-$(basename "${MODEL_PATH}")-kv_retrieval-fill${N_WAY}-${STAMP}}"
ROOT="${OUTPUT_ROOT}/${METHOD}/${RUN_TAG_BASE}"
SHARD_ROOT="${ROOT}/shards"
WORKER_ROOT="${ROOT}/workers"
FINAL_DIR="${ROOT}/final"
mkdir -p "${SHARD_ROOT}" "${WORKER_ROOT}" "${FINAL_DIR}"

python3 - "${FULL_DATASET}" "${EXISTING_FILE}" "${SHARD_ROOT}" "${N_WAY}" <<'PY'
import json, os, sys
full_path, existing_path, shard_root, n_way = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

# read done indices from existing predictions
completed = set()
with open(existing_path, 'r', encoding='utf-8') as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
            idx = o.get('index', None)
            if idx is not None:
                completed.add(int(idx))
        except Exception:
            pass

rows = []
with open(full_path, 'r', encoding='utf-8') as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        o = json.loads(ln)
        rid = int(o['id'])
        if rid not in completed:
            rows.append(o)

counts = [0] * n_way
for i in range(n_way):
    d = os.path.join(shard_root, f'shard_{i:02d}')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'kv_retrieval.jsonl'), 'w', encoding='utf-8'):
        pass

for i, row in enumerate(rows):
    sid = i % n_way
    out = os.path.join(shard_root, f'shard_{sid:02d}', 'kv_retrieval.jsonl')
    with open(out, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    counts[sid] += 1

print(f"[INFO] existing_done={len(completed)} remaining={len(rows)}")
print("[INFO] shard_counts=" + ",".join(f"{i}:{c}" for i, c in enumerate(counts)))
PY

declare -a PIDS=()
declare -a WORKER_FILES=()

for ((i=0; i<N_WAY; i++)); do
  gpu="${GPUS[$i]}"
  shard_dir="${SHARD_ROOT}/shard_$(printf '%02d' "$i")"
  shard_file="${shard_dir}/kv_retrieval.jsonl"
  if [[ ! -s "${shard_file}" ]]; then
    echo "[INFO] skip worker ${i}: empty shard"
    continue
  fi

  worker_tag="${RUN_TAG_BASE}-w$(printf '%02d' "$i")"
  worker_log="${WORKER_ROOT}/${worker_tag}.log"
  worker_out_root="${WORKER_ROOT}/preds"
  worker_out_dir="${worker_out_root}/${METHOD}/${worker_tag}"
  worker_jsonl="${worker_out_dir}/kv_retrieval.jsonl"
  WORKER_FILES+=("${worker_jsonl}")

  echo "[INFO] launch worker=${i} gpu=${gpu} shard=${shard_dir}"
  (
    AUTO_SELECT_GPUS=0 \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    MODEL_PATH="${MODEL_PATH}" \
    METHOD="${METHOD}" \
    TOPK="${TOPK}" \
    DATASET_PATH="${shard_dir}" \
    TASKS="kv_retrieval" \
    OUTPUT_ROOT="${worker_out_root}" \
    RUN_TAG="${worker_tag}" \
    RESUME=0 \
    NUM_GPUS=1 MP_NUM=1 PP_NUM=1 DRY_RUN=0 \
    FIXED_REUSE_THRESHOLD_UPPER="${FIXED_REUSE_THRESHOLD_UPPER}" \
    FIXED_REUSE_THRESHOLD_LOWER="${FIXED_REUSE_THRESHOLD_LOWER}" \
    FIXED_NUM_SKIP_LAYERS="${FIXED_NUM_SKIP_LAYERS}" \
    FIXED_DECAY_P="${FIXED_DECAY_P}" \
    FIXED_MAX_REUSE_COUNT="${FIXED_MAX_REUSE_COUNT}" \
      "${BASE_RUNNER}" > "${worker_log}" 2>&1
  ) &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "[INFO] all workers finished, merging..."
RAW_MERGED="${FINAL_DIR}/kv_retrieval.raw.jsonl"
MERGED="${FINAL_DIR}/kv_retrieval.jsonl"
: > "${RAW_MERGED}"
cat "${EXISTING_FILE}" >> "${RAW_MERGED}"
for f in "${WORKER_FILES[@]}"; do
  [[ -f "${f}" ]] || { echo "[ERROR] missing worker output: ${f}" >&2; exit 1; }
  cat "${f}" >> "${RAW_MERGED}"
done

python3 - "${RAW_MERGED}" "${MERGED}" "${FULL_DATASET}" <<'PY'
import json, sys
raw_path, merged_path, full_path = sys.argv[1], sys.argv[2], sys.argv[3]

keep = {}
with open(raw_path, 'r', encoding='utf-8') as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
            idx = o.get('index', None)
            if idx is None:
                continue
            idx = int(idx)
            if idx not in keep:
                keep[idx] = o
        except Exception:
            pass

with open(merged_path, 'w', encoding='utf-8') as f:
    for idx in sorted(keep):
        f.write(json.dumps(keep[idx], ensure_ascii=False) + '\n')

all_ids = set()
with open(full_path, 'r', encoding='utf-8') as f:
    for ln in f:
        ln = ln.strip()
        if ln:
            all_ids.add(int(json.loads(ln)['id']))

missing = sorted(all_ids - set(keep))
print(f"[INFO] merged={len(keep)} total={len(all_ids)} missing={len(missing)}")
if missing:
    print("[ERROR] first_missing=", missing[:20])
    sys.exit(2)
PY

python3 "${EVAL_PY}" --model "${FINAL_DIR}" | tee "${FINAL_DIR}/eval.log"

echo "[DONE] merged file: ${MERGED}"
echo "[DONE] eval dir   : ${FINAL_DIR}"
