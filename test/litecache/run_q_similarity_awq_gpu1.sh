#!/usr/bin/env bash
set -euo pipefail

SAMPLE_IDX="${SAMPLE_IDX:-0}"

CUDA_VISIBLE_DEVICES=1 python /jhe/sglang/test/litecache/auxiliary/attn_pattern/profile_heads_cosine.py \
  --model /jhe/Qwen2.5-14B-Instruct-AWQ \
  --dataset_path /jhe/dataset/LongBench \
  --num_samples 1 \
  --sample_idx "${SAMPLE_IDX}" \
  --max_context_length 65536 \
  --trace_task gov_report \
  --trace_layers 10 \
  --trace_kv_heads 2,12,26
