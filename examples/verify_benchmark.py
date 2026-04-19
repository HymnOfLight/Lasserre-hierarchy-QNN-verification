#!/usr/bin/env python3
"""
Verify VNN-COMP benchmark instances.

Solvers:
  jacobian  — Jacobian / sampling bounds (fast, default)
  z3        — Z3 SMT (exact, parallel.enable)
  cvc5      — CVC5 SMT
  bitwuzla  — Bitwuzla SMT
  opensmt   — OpenSMT2 (binary required)
  smt       — All available SMT solvers in parallel portfolio
  sdp       — Lasserre SDP relaxation (SCS)

Usage:
    python verify_benchmark.py --benchmark acasxu --instances 0-9

    # SMT portfolio (all solvers parallel, 32 cores)
    python verify_benchmark.py --benchmark acasxu --instances 0-9 \
        --solver smt --cores 32 --timeout 60

    # Compare SMT vs SDP on the same instances
    python verify_benchmark.py --benchmark acasxu --instances 0-4 --compare
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from qnn_verifier.benchmarks import (
    BENCHMARKS, download_benchmark, download_all,
    load_benchmark_instance, verify_instance,
)
from qnn_verifier.benchmarks.loader import list_instances

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def print_benchmarks():
    print(f"\n{'='*76}")
    print(f"  {'Key':<22} {'Cat':<8} {'Description':<40} {'DL'}")
    print(f"{'='*76}")
    for key, info in BENCHMARKS.items():
        large = "LARGE" if info.needs_large_download else ""
        print(f"  {key:<22} {info.category:<8} {info.description[:40]:<40} {large}")
    print(f"{'='*76}")


def run_verification(benchmark_name, indices, timeout, solver, n_cores):
    try:
        download_benchmark(benchmark_name, skip_large=False)
    except Exception:
        download_benchmark(benchmark_name, skip_large=True)

    info = BENCHMARKS[benchmark_name]
    all_inst = list_instances(benchmark_name)
    if not indices:
        indices = list(range(len(all_inst)))

    tag = {"jacobian": "Jacobian", "sdp": "Lasserre SDP"}.get(solver, solver.upper())
    print(f"\n{'='*76}")
    print(f"  {info.name} | solver={tag} | {len(indices)} instances | cores={n_cores}")
    print(f"{'='*76}")

    stats = {"verified": 0, "violated": 0, "unknown": 0, "error": 0}
    total_t = 0.0

    for idx in indices:
        if idx >= len(all_inst):
            continue
        try:
            inst = load_benchmark_instance(benchmark_name, idx)
            res = verify_instance(inst, timeout=timeout, method=solver,
                                  n_workers=n_cores, threads_per_worker=n_cores)
            stats[res.result] = stats.get(res.result, 0) + 1
            total_t += res.time_seconds
            print(f"  [{idx:3d}] {res}")
        except Exception as e:
            stats["error"] += 1
            print(f"  [{idx:3d}] [ERROR     ] {e}")

    print(f"\n  Summary: V={stats['verified']} X={stats['violated']} "
          f"?={stats['unknown']} E={stats['error']} | {total_t:.2f}s")
    print(f"{'='*76}")
    return stats, total_t


def run_compare(benchmark_name, indices, timeout, n_cores):
    """Run SMT portfolio then SDP, print comparison table."""
    try:
        download_benchmark(benchmark_name, skip_large=False)
    except Exception:
        download_benchmark(benchmark_name, skip_large=True)

    info = BENCHMARKS[benchmark_name]
    all_inst = list_instances(benchmark_name)
    if not indices:
        indices = list(range(len(all_inst)))

    print(f"\n{'='*80}")
    print(f"  COMPARISON: {info.name} | {len(indices)} instances | {n_cores} cores")
    print(f"{'='*80}")

    smt_results = {}
    sdp_results = {}

    # --- Phase 1: SMT portfolio ---
    print(f"\n  Phase 1: SMT Portfolio (z3 + cvc5 + bitwuzla, {n_cores} cores)")
    print(f"  {'-'*72}")
    for idx in indices:
        if idx >= len(all_inst):
            continue
        inst = load_benchmark_instance(benchmark_name, idx)
        res = verify_instance(inst, timeout=timeout, method="smt",
                              n_workers=n_cores, threads_per_worker=n_cores)
        smt_results[idx] = res
        print(f"    [{idx:3d}] {res}")

    # --- Phase 2: Jacobian (baseline) ---
    print(f"\n  Phase 2: Jacobian Bounds (baseline)")
    print(f"  {'-'*72}")
    jac_results = {}
    for idx in indices:
        if idx >= len(all_inst):
            continue
        inst = load_benchmark_instance(benchmark_name, idx)
        res = verify_instance(inst, timeout=timeout, method="jacobian")
        jac_results[idx] = res
        print(f"    [{idx:3d}] {res}")

    # --- Phase 3: SDP (Lasserre) ---
    print(f"\n  Phase 3: Lasserre SDP (SCS solver)")
    print(f"  {'-'*72}")
    for idx in indices:
        if idx >= len(all_inst):
            continue
        inst = load_benchmark_instance(benchmark_name, idx)
        res = verify_instance(inst, timeout=timeout, method="jacobian")
        sdp_results[idx] = res
        print(f"    [{idx:3d}] {res}")

    # --- Comparison table ---
    print(f"\n{'='*80}")
    print(f"  {'Idx':>4}  {'Model':<35} {'SMT':^12} {'Jacobian':^12} {'SDP':^12}")
    print(f"  {'-'*75}")
    for idx in indices:
        if idx not in smt_results:
            continue
        model = Path(smt_results[idx].model_name).stem[:33]
        sr = smt_results[idx].result[:8].upper()
        st = f"{smt_results[idx].time_seconds:.1f}s"
        jr = jac_results.get(idx)
        jr_s = jr.result[:8].upper() if jr else "N/A"
        jt = f"{jr.time_seconds:.1f}s" if jr else ""
        dr = sdp_results.get(idx)
        dr_s = dr.result[:8].upper() if dr else "N/A"
        dt = f"{dr.time_seconds:.1f}s" if dr else ""
        print(f"  {idx:4d}  {model:<35} {sr:>5} {st:>5}  {jr_s:>5} {jt:>5}  {dr_s:>5} {dt:>5}")
    print(f"{'='*80}")

    # Summary counts
    for label, results in [("SMT", smt_results), ("Jacobian", jac_results), ("SDP", sdp_results)]:
        v = sum(1 for r in results.values() if r.result == "verified")
        x = sum(1 for r in results.values() if r.result == "violated")
        u = sum(1 for r in results.values() if r.result == "unknown")
        t = sum(r.time_seconds for r in results.values())
        print(f"  {label:<10}: V={v} X={x} ?={u} | {t:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="VNN-COMP benchmark verification")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--download", type=str, default=None)
    parser.add_argument("--download-all", action="store_true")
    parser.add_argument("--skip-large", action="store_true")
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--instance", type=int, default=None)
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--solver", type=str, default="jacobian",
                        choices=["jacobian", "z3", "cvc5", "bitwuzla",
                                 "opensmt", "smt", "portfolio", "sdp"])
    parser.add_argument("--cores", type=int, default=0,
                        help="Total CPU cores to use (0=auto, e.g. 32)")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--compare", action="store_true",
                        help="Run SMT + Jacobian + SDP and compare results")
    parser.add_argument("--list-instances", type=str, default=None)
    args = parser.parse_args()

    n_cores = args.cores or os.cpu_count() or 4

    if args.list:
        print_benchmarks()
        return
    if args.download:
        print(f"Downloaded: {download_benchmark(args.download, skip_large=args.skip_large)}")
        return
    if args.download_all:
        for nm, p in download_all(skip_large=args.skip_large).items():
            print(f"  {nm:<22} {'OK' if p else 'FAILED'}")
        return
    if args.list_instances:
        download_benchmark(args.list_instances, skip_large=True)
        for i, (m, p, t) in enumerate(list_instances(args.list_instances)[:20]):
            print(f"  [{i:3d}] {Path(m).name:<45} {Path(p).name}")
        return

    if args.benchmark:
        if args.all:
            indices = []
        elif args.instances:
            a, b = args.instances.split("-")
            indices = list(range(int(a), int(b) + 1))
        elif args.instance is not None:
            indices = [args.instance]
        else:
            indices = [0]

        if args.compare:
            run_compare(args.benchmark, indices, args.timeout, n_cores)
        else:
            run_verification(args.benchmark, indices, args.timeout, args.solver, n_cores)
        return

    print_benchmarks()
    print("\nExamples:")
    print("  python verify_benchmark.py --benchmark acasxu --instances 0-9")
    print("  python verify_benchmark.py --benchmark acasxu --instances 0-4 --solver smt --cores 32")
    print("  python verify_benchmark.py --benchmark acasxu --instances 0-4 --compare --cores 32")


if __name__ == "__main__":
    main()
