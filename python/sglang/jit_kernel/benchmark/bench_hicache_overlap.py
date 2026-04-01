"""Benchmark overlap between HiCache direct layout conversion and GPU compute.

This script benchmarks the direct host->device HiCache layout conversion used by
the `page_first_direct -> layer_first` path, and compares:

1. Layout conversion alone
2. GPU compute alone
3. Layout conversion + compute serialized on one stream
4. Layout conversion + compute launched on two CUDA streams

The goal is to estimate how much of the layout-conversion cost can be hidden by
overlapping it with GPU work.
"""

from __future__ import annotations

import argparse
import sys
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PYTHON_ROOT = _REPO_ROOT / "python"
_SGL_KERNEL_PYTHON_ROOT = _REPO_ROOT / "sgl-kernel" / "python"
for _path in (_PYTHON_ROOT, _SGL_KERNEL_PYTHON_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from sgl_kernel.kvcacheio import transfer_kv_per_layer_direct_pf_lf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark overlap between HiCache direct PF->LF layout conversion "
            "and a GPU compute op."
        )
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--head-num", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=4096,
        help="Total token capacity of the synthetic KV cache.",
    )
    parser.add_argument(
        "--transfer-tokens",
        type=int,
        default=1024,
        help="Number of tokens transferred per iteration. Must be divisible by page size.",
    )
    parser.add_argument(
        "--compute-op",
        default="matmul",
        choices=["matmul", "sleep"],
        help="GPU compute op to overlap with the layout conversion.",
    )
    parser.add_argument(
        "--compute-tokens",
        type=int,
        default=1024,
        help="Token count used by the compute op when compute-op=matmul.",
    )
    parser.add_argument(
        "--compute-hidden",
        type=int,
        default=0,
        help="Hidden size used by the compute op. Defaults to head_num * head_dim.",
    )
    parser.add_argument(
        "--compute-out-multiplier",
        type=int,
        default=4,
        help="Output width multiplier for matmul compute.",
    )
    parser.add_argument(
        "--sleep-cycles",
        type=int,
        default=20_000_000,
        help="Cycle count for torch.cuda._sleep when compute-op=sleep.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mla",
        action="store_true",
        help="Benchmark the MLA variant with a single KV tensor instead of K/V tensors.",
    )
    return parser.parse_args()


def _to_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _expand_page_indices(page_ids: torch.Tensor, page_size: int) -> torch.Tensor:
    offsets = torch.arange(page_size, dtype=torch.int64)
    return (page_ids[:, None] * page_size + offsets[None, :]).reshape(-1)


def _summary(samples_ms: List[float]) -> Dict[str, float]:
    samples_ms = sorted(samples_ms)
    return {
        "median_ms": statistics.median(samples_ms),
        "min_ms": samples_ms[0],
        "max_ms": samples_ms[-1],
    }


def _summary_dict(samples: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {key: _summary([sample[key] for sample in samples]) for key in samples[0]}


@dataclass
class LayoutContext:
    src_ptrs: List[torch.Tensor]
    dst_ptrs: List[torch.Tensor]
    src_indices_cpu: torch.Tensor
    dst_indices_cpu: torch.Tensor
    layer_id: int
    page_size: int

    def run(self) -> None:
        transfer_kv_per_layer_direct_pf_lf(
            src_ptrs=self.src_ptrs,
            dst_ptrs=self.dst_ptrs,
            src_indices=self.src_indices_cpu,
            dst_indices=self.dst_indices_cpu,
            layer_id=self.layer_id,
            page_size=self.page_size,
        )


@dataclass
class ComputeContext:
    run: Callable[[], None]


def _make_layout_context(args: argparse.Namespace, dtype: torch.dtype) -> LayoutContext:
    if args.transfer_tokens % args.page_size != 0:
        raise ValueError("--transfer-tokens must be divisible by --page-size")
    if args.total_tokens % args.page_size != 0:
        raise ValueError("--total-tokens must be divisible by --page-size")
    if not (0 <= args.layer_id < args.num_layers):
        raise ValueError("--layer-id must be in [0, num_layers)")

    page_num = args.total_tokens // args.page_size
    transfer_pages = args.transfer_tokens // args.page_size
    if transfer_pages * 2 > page_num:
        raise ValueError("Need at least 2 * transfer_pages pages in the synthetic cache")

    torch.manual_seed(args.seed)
    perm = torch.randperm(page_num, dtype=torch.int64)
    src_pages = perm[:transfer_pages]
    dst_pages = perm[transfer_pages : 2 * transfer_pages]
    src_indices_cpu = _expand_page_indices(src_pages, args.page_size).cpu()
    dst_indices_cpu = _expand_page_indices(dst_pages, args.page_size).cpu()

    if args.mla:
        src_kv = torch.randn(
            page_num,
            args.num_layers,
            args.page_size,
            args.head_num,
            args.head_dim,
            dtype=dtype,
            pin_memory=True,
        )
        dst_kv = torch.empty(
            args.num_layers,
            args.total_tokens,
            args.head_num,
            args.head_dim,
            dtype=dtype,
            device="cuda",
        )
        src_ptrs = [src_kv]
        dst_ptrs = [dst_kv[args.layer_id]]
    else:
        src_k = torch.randn(
            page_num,
            args.num_layers,
            args.page_size,
            args.head_num,
            args.head_dim,
            dtype=dtype,
            pin_memory=True,
        )
        src_v = torch.randn(
            page_num,
            args.num_layers,
            args.page_size,
            args.head_num,
            args.head_dim,
            dtype=dtype,
            pin_memory=True,
        )
        dst_k = torch.empty(
            args.num_layers,
            args.total_tokens,
            args.head_num,
            args.head_dim,
            dtype=dtype,
            device="cuda",
        )
        dst_v = torch.empty_like(dst_k)
        src_ptrs = [src_k, src_v]
        dst_ptrs = [dst_k[args.layer_id], dst_v[args.layer_id]]

    return LayoutContext(
        src_ptrs=src_ptrs,
        dst_ptrs=dst_ptrs,
        src_indices_cpu=src_indices_cpu,
        dst_indices_cpu=dst_indices_cpu,
        layer_id=args.layer_id,
        page_size=args.page_size,
    )


def _make_compute_context(args: argparse.Namespace, dtype: torch.dtype) -> ComputeContext:
    if args.compute_op == "sleep":
        return ComputeContext(run=lambda: torch.cuda._sleep(args.sleep_cycles))

    hidden = args.compute_hidden or (args.head_num * args.head_dim)
    out_dim = hidden * args.compute_out_multiplier
    x = torch.randn(
        args.compute_tokens,
        hidden,
        dtype=dtype,
        device="cuda",
    )
    w = torch.randn(
        hidden,
        out_dim,
        dtype=dtype,
        device="cuda",
    )
    out = torch.empty(
        args.compute_tokens,
        out_dim,
        dtype=dtype,
        device="cuda",
    )

    def _run_matmul() -> None:
        torch.matmul(x, w, out=out)

    return ComputeContext(run=_run_matmul)


def _measure_single(op: Callable[[], None]) -> Dict[str, float]:
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record(stream)
    op()
    end.record(stream)
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return {
        "wall_ms": wall_ms,
        "gpu_ms": float(start.elapsed_time(end)),
    }


def _measure_serial(
    layout_ctx: LayoutContext, compute_ctx: ComputeContext
) -> Dict[str, float]:
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record(stream)
    layout_ctx.run()
    compute_ctx.run()
    end.record(stream)
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return {
        "wall_ms": wall_ms,
        "gpu_ms": float(start.elapsed_time(end)),
    }


def _measure_parallel(
    layout_ctx: LayoutContext,
    compute_ctx: ComputeContext,
    layout_stream: torch.cuda.Stream,
    compute_stream: torch.cuda.Stream,
) -> Dict[str, float]:
    main_stream = torch.cuda.current_stream()
    launch_event = torch.cuda.Event()
    stop_event = torch.cuda.Event(enable_timing=True)
    start_event = torch.cuda.Event(enable_timing=True)
    layout_start = torch.cuda.Event(enable_timing=True)
    layout_end = torch.cuda.Event(enable_timing=True)
    compute_start = torch.cuda.Event(enable_timing=True)
    compute_end = torch.cuda.Event(enable_timing=True)
    layout_done = torch.cuda.Event()
    compute_done = torch.cuda.Event()

    wall_start = time.perf_counter()
    start_event.record(main_stream)
    launch_event.record(main_stream)

    with torch.cuda.stream(layout_stream):
        layout_stream.wait_event(launch_event)
        layout_start.record(layout_stream)
        layout_ctx.run()
        layout_end.record(layout_stream)
        layout_done.record(layout_stream)

    with torch.cuda.stream(compute_stream):
        compute_stream.wait_event(launch_event)
        compute_start.record(compute_stream)
        compute_ctx.run()
        compute_end.record(compute_stream)
        compute_done.record(compute_stream)

    main_stream.wait_event(layout_done)
    main_stream.wait_event(compute_done)
    stop_event.record(main_stream)
    stop_event.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    return {
        "total_wall_ms": wall_ms,
        "total_gpu_ms": float(start_event.elapsed_time(stop_event)),
        "layout_gpu_ms": float(layout_start.elapsed_time(layout_end)),
        "compute_gpu_ms": float(compute_start.elapsed_time(compute_end)),
    }


def _benchmark(
    fn: Callable[[], Dict[str, float]],
    warmup: int,
    iters: int,
) -> Dict[str, Dict[str, float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = [fn() for _ in range(iters)]
    return _summary_dict(samples)


def _benchmark_parallel(
    fn: Callable[[], Dict[str, float]],
    warmup: int,
    iters: int,
) -> Dict[str, Dict[str, float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        samples.append(fn())
    return _summary_dict(samples)


def _print_config(args: argparse.Namespace, dtype: torch.dtype) -> None:
    hidden = args.compute_hidden or (args.head_num * args.head_dim)
    print("Configuration")
    print(f"  dtype               : {dtype}")
    print(f"  num_layers          : {args.num_layers}")
    print(f"  layer_id            : {args.layer_id}")
    print(f"  page_size           : {args.page_size}")
    print(f"  head_num            : {args.head_num}")
    print(f"  head_dim            : {args.head_dim}")
    print(f"  total_tokens        : {args.total_tokens}")
    print(f"  transfer_tokens     : {args.transfer_tokens}")
    print(f"  compute_op          : {args.compute_op}")
    print(f"  compute_tokens      : {args.compute_tokens}")
    print(f"  compute_hidden      : {hidden}")
    print(f"  mla                 : {args.mla}")
    print(f"  warmup              : {args.warmup}")
    print(f"  iters               : {args.iters}")
    print()


def _print_results(
    layout_stats: Dict[str, Dict[str, float]],
    compute_stats: Dict[str, Dict[str, float]],
    serial_stats: Dict[str, Dict[str, float]],
    parallel_stats: Dict[str, Dict[str, float]],
) -> None:
    layout_gpu_ms = layout_stats["gpu_ms"]["median_ms"]
    compute_gpu_ms = compute_stats["gpu_ms"]["median_ms"]
    serial_wall_ms = serial_stats["wall_ms"]["median_ms"]
    parallel_wall_ms = parallel_stats["total_wall_ms"]["median_ms"]
    serial_gpu_ms = serial_stats["gpu_ms"]["median_ms"]
    parallel_gpu_ms = parallel_stats["total_gpu_ms"]["median_ms"]
    overlap_gain_wall_ms = serial_wall_ms - parallel_wall_ms
    overlap_gain_gpu_ms = serial_gpu_ms - parallel_gpu_ms
    ideal_parallel_gpu_ms = max(layout_gpu_ms, compute_gpu_ms)
    hide_ratio = (
        0.0
        if layout_gpu_ms == 0
        else max(0.0, min(1.0, overlap_gain_gpu_ms / layout_gpu_ms))
    )
    speedup = float("inf") if parallel_wall_ms == 0 else serial_wall_ms / parallel_wall_ms

    print("Results (median / min / max, ms)")
    print(
        "  layout_only_wall    : "
        f"{layout_stats['wall_ms']['median_ms']:.3f} / "
        f"{layout_stats['wall_ms']['min_ms']:.3f} / "
        f"{layout_stats['wall_ms']['max_ms']:.3f}"
    )
    print(
        "  layout_only_gpu     : "
        f"{layout_stats['gpu_ms']['median_ms']:.3f} / "
        f"{layout_stats['gpu_ms']['min_ms']:.3f} / "
        f"{layout_stats['gpu_ms']['max_ms']:.3f}"
    )
    print(
        "  compute_only_wall   : "
        f"{compute_stats['wall_ms']['median_ms']:.3f} / "
        f"{compute_stats['wall_ms']['min_ms']:.3f} / "
        f"{compute_stats['wall_ms']['max_ms']:.3f}"
    )
    print(
        "  compute_only_gpu    : "
        f"{compute_stats['gpu_ms']['median_ms']:.3f} / "
        f"{compute_stats['gpu_ms']['min_ms']:.3f} / "
        f"{compute_stats['gpu_ms']['max_ms']:.3f}"
    )
    print(
        "  serial_total_wall   : "
        f"{serial_stats['wall_ms']['median_ms']:.3f} / "
        f"{serial_stats['wall_ms']['min_ms']:.3f} / "
        f"{serial_stats['wall_ms']['max_ms']:.3f}"
    )
    print(
        "  serial_total_gpu    : "
        f"{serial_stats['gpu_ms']['median_ms']:.3f} / "
        f"{serial_stats['gpu_ms']['min_ms']:.3f} / "
        f"{serial_stats['gpu_ms']['max_ms']:.3f}"
    )
    print(
        "  parallel_total_wall : "
        f"{parallel_stats['total_wall_ms']['median_ms']:.3f} / "
        f"{parallel_stats['total_wall_ms']['min_ms']:.3f} / "
        f"{parallel_stats['total_wall_ms']['max_ms']:.3f}"
    )
    print(
        "  parallel_total_gpu  : "
        f"{parallel_stats['total_gpu_ms']['median_ms']:.3f} / "
        f"{parallel_stats['total_gpu_ms']['min_ms']:.3f} / "
        f"{parallel_stats['total_gpu_ms']['max_ms']:.3f}"
    )
    print(
        "  parallel_layout_gpu : "
        f"{parallel_stats['layout_gpu_ms']['median_ms']:.3f} / "
        f"{parallel_stats['layout_gpu_ms']['min_ms']:.3f} / "
        f"{parallel_stats['layout_gpu_ms']['max_ms']:.3f}"
    )
    print(
        "  parallel_compute_gpu: "
        f"{parallel_stats['compute_gpu_ms']['median_ms']:.3f} / "
        f"{parallel_stats['compute_gpu_ms']['min_ms']:.3f} / "
        f"{parallel_stats['compute_gpu_ms']['max_ms']:.3f}"
    )
    print()
    print("Derived metrics")
    print(f"  overlap_gain_wall_ms: {overlap_gain_wall_ms:.3f}")
    print(f"  overlap_gain_gpu_ms : {overlap_gain_gpu_ms:.3f}")
    print(f"  ideal_parallel_gpu  : {ideal_parallel_gpu_ms:.3f}")
    print(f"  serial_vs_parallel  : {speedup:.3f}x")
    print(f"  hidden_layout_ratio : {hide_ratio:.3f}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    args = _parse_args()
    dtype = _to_dtype(args.dtype)

    layout_ctx = _make_layout_context(args, dtype)
    compute_ctx = _make_compute_context(args, dtype)
    layout_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    _print_config(args, dtype)

    layout_stats = _benchmark(lambda: _measure_single(layout_ctx.run), args.warmup, args.iters)
    compute_stats = _benchmark(lambda: _measure_single(compute_ctx.run), args.warmup, args.iters)
    serial_stats = _benchmark(
        lambda: _measure_serial(layout_ctx, compute_ctx),
        args.warmup,
        args.iters,
    )
    parallel_stats = _benchmark_parallel(
        lambda: _measure_parallel(layout_ctx, compute_ctx, layout_stream, compute_stream),
        args.warmup,
        args.iters,
    )

    _print_results(layout_stats, compute_stats, serial_stats, parallel_stats)


if __name__ == "__main__":
    main()
