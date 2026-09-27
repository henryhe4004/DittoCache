#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, statistics
from pathlib import Path
from typing import Any

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('/workspace/jhe/sglang-litecache/test/ditto/speedup/results_itl_tmp'))
parser.add_argument('--mode', choices=['full','ditto','both'], default='both')
parser.add_argument('--concurrencies', default='1 2 4 8 16 24 32 48 64')
args = parser.parse_args()
concs = [int(x) for x in args.concurrencies.replace(',', ' ').split()]


def first_obj(resp: Any) -> dict[str, Any]:
    if isinstance(resp, list) and resp:
        return resp[0] if isinstance(resp[0], dict) else {}
    if isinstance(resp, dict):
        return resp
    return {}


def meta_from_row(row: dict[str, Any]) -> dict[str, Any]:
    txt = row.get('response_text')
    if not txt:
        return {}
    try:
        resp = json.loads(txt)
    except Exception:
        return {}
    obj = first_obj(resp)
    meta = obj.get('meta_info')
    return meta if isinstance(meta, dict) else {}


def completion_tokens(row: dict[str, Any], meta: dict[str, Any]) -> int:
    for v in (meta.get('completion_tokens'), meta.get('output_tokens'), meta.get('num_output_tokens')):
        try:
            if v is not None:
                return int(v)
        except Exception:
            pass
    txt = row.get('response_text')
    if txt:
        try:
            obj = first_obj(json.loads(txt))
            if isinstance(obj.get('output_ids'), list):
                return len(obj['output_ids'])
        except Exception:
            pass
    return 0


def itl_ms(row: dict[str, Any]) -> float | None:
    meta = meta_from_row(row)
    # Preferred: internal server-side decode forward ms/step, with drop-first-20 if enough generated tokens.
    for key in (
        'decode_forward_latency_ms_per_step_drop_first_20',
        'decode_forward_latency_ms_per_step_drop_first_10',
        'decode_forward_latency_ms_per_step',
    ):
        v = meta.get(key)
        try:
            if v is not None and float(v) > 0:
                return float(v)
        except Exception:
            pass
    # Fallback: if TTFT exists in meta, estimate client-side ITL.
    ct = completion_tokens(row, meta)
    latency = row.get('latency_ms')
    ttft_s = meta.get('time_to_first_token') or meta.get('ttft') or meta.get('ttft_s')
    try:
        if ct > 1 and latency is not None and ttft_s is not None:
            ttft_ms = float(ttft_s) * 1000.0 if float(ttft_s) < 1000 else float(ttft_s)
            return max(0.0, (float(latency) - ttft_ms) / (ct - 1))
    except Exception:
        pass
    return None


def summarize_mode(mode: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for c in concs:
        path = args.root / mode / f'concurrency_{c}.jsonl'
        vals = []
        ok = 0
        total = 0
        if path.exists():
            with path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    total += 1
                    row = json.loads(line)
                    if row.get('ok'):
                        ok += 1
                        v = itl_ms(row)
                        if v is not None and math.isfinite(v):
                            vals.append(v)
        out[c] = {
            'path': str(path),
            'ok': ok,
            'total': total,
            'median_itl_ms': statistics.median(vals) if vals else None,
            'n_itl': len(vals),
        }
    return out

modes = ['full','ditto'] if args.mode == 'both' else [args.mode]
summary = {m: summarize_mode(m) for m in modes}
args.root.mkdir(parents=True, exist_ok=True)

# per-mode CSVs
for m, rows in summary.items():
    csv_path = args.root / f'{m}_itl_summary.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['concurrency','median_itl_ms','ok','total','n_itl','jsonl'])
        for c in concs:
            r = rows[c]
            val = '' if r['median_itl_ms'] is None else f"{r['median_itl_ms']:.3f}"
            w.writerow([c, val, r['ok'], r['total'], r['n_itl'], r['path']])
    print(f'[ITL] wrote {csv_path}')

# combined markdown
full = summary.get('full') or summarize_mode('full')
ditto = summary.get('ditto') or summarize_mode('ditto')
md_path = args.root / 'itl_table.md'
lines = [
    '| **并发** | **Full Median ITL** | **Ditto Median ITL** |',
    '| :----: | :-----------------: | :------------------: |',
]
for c in concs:
    def fmt(rows):
        v = rows[c]['median_itl_ms']
        return 'NA' if v is None else f'{v:.2f}'
    lines.append(f'| {c} | {fmt(full)} | {fmt(ditto)} |')
md_path.write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'[ITL] wrote {md_path}')
