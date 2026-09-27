# Ditto cumulative ablation profiles

The ablation runner keeps HATA retrieval and CPU KV-cache offloading enabled in
all stages. Profiles are cumulative:

| Group | Transfer | Communication | Layer prefetch | QSAC / similarity | Ditto graph | Threshold switch | Resident cache |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `B0` | CUDA memcpy | off | off, always gather | off | `1.0/1.0` | off | off |
| `B1` | GDRCopy | off | off, always gather | off | `1.0/1.0` | off | off |
| `B2` | GDRCopy | on, cross-layer | on, cross-layer | off | `1.0/1.0` | off | off |
| `B3` | GDRCopy | on | on, cross-layer | on | `1.0/1.0` | off | off |
| `B4` | GDRCopy | on | on | on | fixed `0.8/0.8` | off | off |
| `B5` | GDRCopy | on | on | on | profile adaptive `-1.0/0.8` | off | off |
| `B6` | GDRCopy | on | on | on | profile adaptive `-1.0/0.8` | on | off |

`num_skip_layers` controls full-GPU layer placement in the current
implementation, so every profile sets it to `1`.

Cross-layer communication is implemented by Layer Prefetch (`enable_layer_prefetch`).

Intra-GQA aggregation is disabled explicitly for all B0-B6 profiles because it
is not one of the optimizations measured by this sequence.

Run one speedup group:

```bash
python3 test/ditto/speedup/n2n_offloading.py \
  --model /path/to/model \
  --config_file test/ditto/config/hata_offloading/Qwen2.5-14B-Instruct-1M-128K.yaml \
  --data /workspace/jhe/8K \
  --batch_size 16 \
  --max_seq_len 8192 \
  --method offloading-hash \
  --ablation-profile B0
```

Run B0-B6 for both `16x8K` and `64x8K`:

```bash
test/ditto/speedup/run_ablation_b0_b6.sh \
  --model /path/to/model
```

Results are written under `16x8K/` and `64x8K/` in
`test/ditto/speedup/results/ablation_b0_b6`. To run only a subset or
inspect commands without launching a model:

```bash
ABLATION_PROFILES="B0 B6" DRY_RUN=1 \
  test/ditto/speedup/run_ablation_b0_b6.sh --model /path/to/model
```
