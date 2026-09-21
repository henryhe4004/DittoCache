#!/usr/bin/env python3
"""Rebuild only Ditto's CPU gather + CUDA prefetch extension (no downloads).

python scripts/build_ditto_gather.py --cuda-arch 8.6 --install
Requires the installed PyTorch headers, CUDA toolkit, GDRCopy and a C++ compiler.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-arch", default="8.6", help="8.6 for A40; 9.0 for H100")
    parser.add_argument("--build-dir", type=Path, default=root / "sgl-kernel/build/ditto-gather")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    os.environ["MAX_JOBS"] = str(args.jobs)
    os.environ["TORCH_CUDA_ARCH_LIST"] = args.cuda_arch
    from torch.utils.cpp_extension import load

    kernel = root / "sgl-kernel"
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    module = load(
        name="kvlib_cpu_gather",
        sources=[str(kernel / "csrc/kvlib" / name) for name in
                 ("py_cpu_gather_engine_v3.cc", "cpu_gather_engine_v3.cc", "prefetch.cu")],
        extra_cflags=["-O3", "-fopenmp", "-mavx512f", "-mf16c",
                      "-DKVLIB_GDR_AVAILABLE", "-DKVLIB_RAFT_AVAILABLE"],
        extra_cuda_cflags=["-O3"],
        extra_include_paths=[str(kernel / "csrc"), str(kernel / "include"), "/usr/local/include"],
        extra_ldflags=["-fopenmp", "-L/usr/local/lib", "-lgdrapi", "-lcuda"],
        build_directory=str(build), verbose=True,
    )
    print(module.CPUGatherEngineV3.__init__.__doc__)
    if args.install:
        target = kernel / "python/sgl_kernel/kvlib_cpu_gather.so"
        if target.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = build / f"kvlib_cpu_gather.previous.{stamp}.so"
            shutil.copy2(target, backup)
            print(f"Previous binary: {backup}")
        shutil.copy2(module.__file__, target)
        print(f"Installed: {target}")


if __name__ == "__main__":
    main()
