#!/usr/bin/env python3
"""
Comprehensive VNN-COMP benchmark verification.

Runs full-coverage testing on:
  - VGGNet16 (2023): 18 ImageNet robustness instances
  - YOLO (2023): 72 TinyYOLO detection instances
  - cGAN (2023): 21 conditional GAN generation instances
  - CIFAR100 (2024): 200 ResNet classification instances
  - And any other registered benchmark

Usage:
    # Run all four benchmarks
    python verify_vnncomp.py

    # Run a specific benchmark
    python verify_vnncomp.py --benchmark vggnet16
    python verify_vnncomp.py --benchmark yolo
    python verify_vnncomp.py --benchmark cgan
    python verify_vnncomp.py --benchmark cifar100

    # With specific solver
    python verify_vnncomp.py --benchmark yolo --solver smt --cores 32 --timeout 60

    # Compare solvers on CIFAR100
    python verify_vnncomp.py --benchmark cifar100 --compare --cores 32

    # Run all benchmarks with a limit on instances per benchmark
    python verify_vnncomp.py --max-instances 10

    # Skip large downloads (only run already-downloaded benchmarks)
    python verify_vnncomp.py --skip-large
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qnn_verifier.benchmarks import (
    BENCHMARKS, download_benchmark, load_benchmark_instance, verify_instance,
)
from qnn_verifier.benchmarks.loader import list_instances

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_BENCHMARKS = ["vggnet16", "yolo", "cgan", "cifar100"]


def run_benchmark(name, solver, n_cores, timeout, max_instances=None, skip_large=False):
    """Download, load, and verify all instances of a benchmark."""
    info = BENCHMARKS[name]

    print(f"\n{'='*76}")
    print(f"  {info.name} | {info.description[:60]}")
    print(f"  Solver: {solver} | Cores: {n_cores} | Timeout: {timeout}s")
    if info.needs_large_download:
        print(f"  NOTE: requires large model download")
    print(f"{'='*76}")

    try:
        bench_dir = download_benchmark(name, skip_large=skip_large)
    except Exception as e:
        print(f"  Download failed: {e}")
        if skip_large and info.needs_large_download:
            print(f"  (Use --no-skip-large to download large models)")
        return []

    instances = list_instances(name)
    if not instances:
        print(f"  No instances found (models may need large download)")
        return []

    # Check if ONNX models actually exist
    onnx_dir = bench_dir / "onnx"
    n_onnx = len(list(onnx_dir.glob("*.onnx"))) if onnx_dir.exists() else 0
    if n_onnx == 0:
        print(f"  No ONNX models available (need --no-skip-large to download)")
        return []

    n = len(instances)
    if max_instances and max_instances < n:
        n = max_instances
        print(f"  Running {n}/{len(instances)} instances (--max-instances {max_instances})")
    else:
        print(f"  Running all {n} instances")

    # Analyze instance structure
    models = set()
    props = set()
    for model, prop, _ in instances[:n]:
        models.add(Path(model).stem)
        props.add(Path(prop).stem)
    print(f"  Models: {len(models)} | Properties: {len(set(Path(p).stem for _, p, _ in instances[:n]))}")

    results = []
    t_total = time.time()

    for idx in range(n):
        model_name = Path(instances[idx][0]).stem
        prop_name = Path(instances[idx][1]).stem
        try:
            inst = load_benchmark_instance(name, idx)
            res = verify_instance(inst, timeout=timeout, method=solver,
                                  n_workers=n_cores, threads_per_worker=n_cores)
            tag = {"verified": "V", "violated": "X", "unknown": "?"}.get(res.result, "E")
            results.append({"idx": idx, "model": model_name[:30], "prop": prop_name[:30],
                            "result": res.result, "time": res.time_seconds, "tag": tag})
            print(f"  [{idx:3d}/{n}] [{tag}] {model_name[:25]:<25} {prop_name[:25]:<25} {res.time_seconds:.2f}s")
        except Exception as e:
            results.append({"idx": idx, "model": model_name[:30], "prop": prop_name[:30],
                            "result": "error", "time": 0, "tag": "E"})
            print(f"  [{idx:3d}/{n}] [E] {model_name[:25]:<25} ERROR: {str(e)[:40]}")

    elapsed = time.time() - t_total

    # Summary
    n_v = sum(1 for r in results if r["result"] == "verified")
    n_x = sum(1 for r in results if r["result"] == "violated")
    n_u = sum(1 for r in results if r["result"] == "unknown")
    n_e = sum(1 for r in results if r["result"] == "error")

    print(f"\n  {info.name} Summary:")
    print(f"    Verified: {n_v}  Violated: {n_x}  Unknown: {n_u}  Error: {n_e}")
    print(f"    Total: {len(results)} instances in {elapsed:.2f}s")

    # Per-model breakdown if multiple models
    if len(models) > 1:
        by_model = defaultdict(list)
        for r in results:
            by_model[r["model"]].append(r)
        print(f"\n    Per-model breakdown:")
        for model in sorted(by_model.keys()):
            mrs = by_model[model]
            mv = sum(1 for r in mrs if r["result"] == "verified")
            mx = sum(1 for r in mrs if r["result"] == "violated")
            mu = sum(1 for r in mrs if r["result"] == "unknown")
            me = sum(1 for r in mrs if r["result"] == "error")
            print(f"      {model[:35]:<35} V={mv:3d} X={mx:3d} ?={mu:3d} E={me:3d}")

    print(f"{'='*76}")
    return results


def main():
    parser = argparse.ArgumentParser(description="VNN-COMP comprehensive verification")
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=list(BENCHMARKS.keys()),
                        help="Run specific benchmark (default: vggnet16+yolo+cgan+cifar100)")
    parser.add_argument("--solver", type=str, default="jacobian",
                        choices=["jacobian", "smt", "gurobi"])
    parser.add_argument("--cores", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-instances", type=int, default=None,
                        help="Limit instances per benchmark")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip benchmarks needing large downloads")
    parser.add_argument("--no-skip-large", action="store_true",
                        help="Force download of large models")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all solvers")
    parser.add_argument("--all", action="store_true",
                        help="Run ALL registered benchmarks")
    args = parser.parse_args()

    n_cores = args.cores or os.cpu_count() or 4
    skip_large = args.skip_large and not args.no_skip_large

    if args.benchmark:
        benchmarks = [args.benchmark]
    elif args.all:
        benchmarks = list(BENCHMARKS.keys())
    else:
        benchmarks = TARGET_BENCHMARKS

    print(f"\n{'#'*76}")
    print(f"  VNN-COMP Benchmark Suite")
    print(f"  Benchmarks: {', '.join(benchmarks)}")
    print(f"  Solver: {args.solver} | Cores: {n_cores} | Timeout: {args.timeout}s")
    print(f"{'#'*76}")

    all_results = {}

    if args.compare:
        for bench in benchmarks:
            bench_results = {}
            for solver in ["jacobian", "smt", "gurobi"]:
                print(f"\n  >>> {BENCHMARKS[bench].name} with {solver.upper()} <<<")
                try:
                    results = run_benchmark(bench, solver, n_cores, args.timeout,
                                            args.max_instances, skip_large)
                except Exception as e:
                    print(f"  {solver} failed: {e}")
                    results = []
                bench_results[solver] = results
            all_results[bench] = bench_results

        # Comparison table
        print(f"\n{'#'*76}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'#'*76}")
        print(f"  {'Benchmark':<15} {'Solver':<10} {'V':>4} {'X':>4} {'?':>4} {'E':>4} {'Time':>8}")
        print(f"  {'-'*55}")
        for bench in benchmarks:
            for solver in ["jacobian", "smt", "gurobi"]:
                results = all_results.get(bench, {}).get(solver, [])
                nv = sum(1 for r in results if r["result"] == "verified")
                nx = sum(1 for r in results if r["result"] == "violated")
                nu = sum(1 for r in results if r["result"] == "unknown")
                ne = sum(1 for r in results if r["result"] == "error")
                t = sum(r["time"] for r in results)
                print(f"  {BENCHMARKS[bench].name:<15} {solver:<10} {nv:4d} {nx:4d} {nu:4d} {ne:4d} {t:7.1f}s")
            print()
    else:
        for bench in benchmarks:
            try:
                results = run_benchmark(bench, args.solver, n_cores, args.timeout,
                                        args.max_instances, skip_large)
            except Exception as e:
                print(f"  Benchmark {bench} failed: {e}")
                results = []
            all_results[bench] = results

        # Grand summary
        print(f"\n{'#'*76}")
        print(f"  GRAND SUMMARY")
        print(f"{'#'*76}")
        print(f"  {'Benchmark':<15} {'Instances':>10} {'V':>5} {'X':>5} {'?':>5} {'E':>5} {'Time':>8}")
        print(f"  {'-'*60}")
        grand_v = grand_x = grand_u = grand_e = 0
        grand_t = 0.0
        for bench in benchmarks:
            results = all_results.get(bench, [])
            nv = sum(1 for r in results if r["result"] == "verified")
            nx = sum(1 for r in results if r["result"] == "violated")
            nu = sum(1 for r in results if r["result"] == "unknown")
            ne = sum(1 for r in results if r["result"] == "error")
            t = sum(r["time"] for r in results)
            n = len(results)
            bname = BENCHMARKS[bench].name if bench in BENCHMARKS else bench
            print(f"  {bname:<15} {n:>10} {nv:>5} {nx:>5} {nu:>5} {ne:>5} {t:>7.1f}s")
            grand_v += nv; grand_x += nx; grand_u += nu; grand_e += ne; grand_t += t
        print(f"  {'-'*60}")
        grand_n = grand_v + grand_x + grand_u + grand_e
        print(f"  {'TOTAL':<15} {grand_n:>10} {grand_v:>5} {grand_x:>5} {grand_u:>5} {grand_e:>5} {grand_t:>7.1f}s")
        print(f"{'#'*76}")


if __name__ == "__main__":
    main()
