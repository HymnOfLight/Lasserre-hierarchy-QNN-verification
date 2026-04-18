#!/usr/bin/env python3
"""
Verify VNN-COMP benchmark instances with the Lasserre hierarchy framework.

Supports ACAS Xu and all VNN-COMP 2024 complex benchmarks:
  acasxu, cGAN, NN4Sys, LinearizeNN, ml4acopf, ViT,
  Collins Aerospace, LSNC-ReLU, CCTSDB

Models are downloaded to ./benchmarks_data/ in the project directory.

Usage:
    # List all benchmarks
    python verify_benchmark.py --list

    # Download ACAS Xu
    python verify_benchmark.py --download acasxu

    # Verify ACAS Xu instance 0
    python verify_benchmark.py --benchmark acasxu --instance 0

    # Verify instances 0-9
    python verify_benchmark.py --benchmark acasxu --instances 0-9

    # Verify all instances of a benchmark
    python verify_benchmark.py --benchmark linearizenn --all
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from qnn_verifier.benchmarks import (
    BENCHMARKS,
    download_benchmark,
    download_all,
    load_benchmark_instance,
    verify_instance,
)
from qnn_verifier.benchmarks.loader import list_instances

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def print_benchmarks():
    print("\n" + "=" * 76)
    print(f"  {'Key':<22} {'Cat':<8} {'Description':<40} {'DL'}")
    print("=" * 76)
    for key, info in BENCHMARKS.items():
        large = "LARGE" if info.needs_large_download else ""
        print(f"  {key:<22} {info.category:<8} {info.description[:40]:<40} {large}")
    print("=" * 76)
    print(f"  Total: {len(BENCHMARKS)} benchmarks  "
          f"(Classic: {sum(1 for b in BENCHMARKS.values() if b.category == 'classic')}, "
          f"Complex: {sum(1 for b in BENCHMARKS.values() if b.category == 'complex')})")


def run_verification(benchmark_name: str, indices: list, timeout: float = None):
    """Download, load, and verify benchmark instances."""
    try:
        bench_dir = download_benchmark(benchmark_name, skip_large=False)
    except Exception:
        logger.info("Retrying with skip_large=True...")
        bench_dir = download_benchmark(benchmark_name, skip_large=True)

    info = BENCHMARKS[benchmark_name]
    all_inst = list_instances(benchmark_name)

    if not indices:
        indices = list(range(len(all_inst)))

    print(f"\n{'='*76}")
    print(f"  Verifying: {info.name}  ({len(indices)} instances)")
    print(f"{'='*76}")

    results_summary = {"verified": 0, "violated": 0, "unknown": 0, "error": 0, "timeout": 0}
    total_time = 0.0

    for idx in indices:
        if idx >= len(all_inst):
            print(f"  [SKIP] Instance {idx} out of range (max {len(all_inst)-1})")
            continue

        try:
            inst = load_benchmark_instance(benchmark_name, idx)
            res = verify_instance(inst, timeout=timeout)
            results_summary[res.result] = results_summary.get(res.result, 0) + 1
            total_time += res.time_seconds
            print(f"  [{idx:3d}] {res}")
        except Exception as e:
            results_summary["error"] += 1
            print(f"  [{idx:3d}] [ERROR     ] {e}")

    print(f"\n{'='*76}")
    print(f"  Summary: {info.name}")
    print(f"    Verified : {results_summary['verified']}")
    print(f"    Violated : {results_summary['violated']}")
    print(f"    Unknown  : {results_summary['unknown']}")
    print(f"    Error    : {results_summary['error']}")
    print(f"    Total    : {sum(results_summary.values())} instances, {total_time:.2f}s")
    print(f"{'='*76}")


def main():
    parser = argparse.ArgumentParser(description="VNN-COMP benchmark verification")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--download", type=str, default=None)
    parser.add_argument("--download-all", action="store_true")
    parser.add_argument("--skip-large", action="store_true")
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--instance", type=int, default=None)
    parser.add_argument("--instances", type=str, default=None,
                        help="Range e.g. '0-9'")
    parser.add_argument("--all", action="store_true",
                        help="Verify all instances")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--list-instances", type=str, default=None)
    args = parser.parse_args()

    if args.list:
        print_benchmarks()
        return

    if args.download:
        path = download_benchmark(args.download, skip_large=args.skip_large)
        print(f"Downloaded to: {path}")
        return

    if args.download_all:
        paths = download_all(skip_large=args.skip_large)
        for name, path in paths.items():
            print(f"  {name:<22} {'OK' if path else 'FAILED'}")
        return

    if args.list_instances:
        download_benchmark(args.list_instances, skip_large=True)
        instances = list_instances(args.list_instances)
        print(f"\n{BENCHMARKS[args.list_instances].name}: {len(instances)} instances")
        for i, (m, p, t) in enumerate(instances[:20]):
            print(f"  [{i:3d}] {Path(m).name:<45} {Path(p).name}")
        if len(instances) > 20:
            print(f"  ... and {len(instances)-20} more")
        return

    if args.benchmark:
        if args.all:
            indices = []  # empty = all
        elif args.instances:
            parts = args.instances.split("-")
            indices = list(range(int(parts[0]), int(parts[1]) + 1))
        elif args.instance is not None:
            indices = [args.instance]
        else:
            indices = [0]
        run_verification(args.benchmark, indices, args.timeout)
        return

    print_benchmarks()
    print("\nExamples:")
    print("  python verify_benchmark.py --download acasxu")
    print("  python verify_benchmark.py --benchmark acasxu --instances 0-9")
    print("  python verify_benchmark.py --benchmark linearizenn --all")


if __name__ == "__main__":
    main()
