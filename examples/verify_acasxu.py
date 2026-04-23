#!/usr/bin/env python3
"""
Comprehensive ACAS Xu verification: all 10 properties on all 45 networks.

The ACAS Xu benchmark consists of 45 neural networks arranged in a 5×9
grid (a_prev × tau indices), each with 5 inputs, 5 outputs, 6 hidden
layers of 50 ReLU neurons (300 ReLU total).

Properties (from Reluplex paper):
  prop_1:  COC output < 1500 (Y_0 < 3.991)        — 45 networks
  prop_2:  COC is not maximal (¬(Y_0 ≥ all))       — 45 networks
  prop_3:  COC is not minimal (¬(Y_0 ≤ all))       — 45 networks
  prop_4:  COC is not minimal (different input)     — 45 networks
  prop_5:  Strong right is minimal                  — 1 network (1_1)
  prop_6:  COC is minimal                           — 1 network (1_1)
  prop_7:  Strong left/right not minimal (OR)       — 1 network (1_9)
  prop_8:  Either output 0 or 1 is minimal (OR)     — 1 network (2_9)
  prop_9:  Strong left is minimal                   — 1 network (3_3)
  prop_10: COC is not maximal                       — 1 network (4_5)

Usage:
    # Verify all 186 instances with Jacobian (fast)
    python verify_acasxu.py

    # Verify all with SMT (exact, needs time)
    python verify_acasxu.py --solver smt --cores 32 --timeout 120

    # Verify all with Gurobi MILP
    python verify_acasxu.py --solver gurobi --cores 32 --timeout 120

    # Compare all solvers
    python verify_acasxu.py --compare --cores 32 --timeout 60

    # Only specific property
    python verify_acasxu.py --prop 1

    # Only specific network
    python verify_acasxu.py --network 1_1

    # Specific property on specific network
    python verify_acasxu.py --prop 2 --network 3_5
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
    download_benchmark, load_benchmark_instance, verify_instance,
)
from qnn_verifier.benchmarks.loader import list_instances

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROP_DESCRIPTIONS = {
    "prop_1": "COC < 1500 (output bound)",
    "prop_2": "COC not maximal (all nets)",
    "prop_3": "COC not minimal (all nets)",
    "prop_4": "COC not minimal (diff input)",
    "prop_5": "Strong right minimal (1_1)",
    "prop_6": "COC minimal (1_1)",
    "prop_7": "Left/right not minimal (1_9)",
    "prop_8": "Out 0 or 1 minimal (2_9)",
    "prop_9": "Strong left minimal (3_3)",
    "prop_10": "COC not maximal (4_5)",
}


def parse_instance_info(instances):
    """Parse instance list into structured form."""
    parsed = []
    for idx, (model, prop, timeout) in enumerate(instances):
        m = Path(model).stem.replace("ACASXU_run2a_", "").replace("_batch_2000", "")
        p = Path(prop).stem
        parts = m.split("_")
        net_i, net_j = (int(parts[0]), int(parts[1])) if len(parts) == 2 else (0, 0)
        parsed.append({
            "idx": idx, "model": m, "prop": p,
            "net_i": net_i, "net_j": net_j, "timeout": timeout,
        })
    return parsed


def run_verification(indices, instances, solver, n_cores, timeout):
    """Run verification on selected instances."""
    results = []
    for info in indices:
        idx = info["idx"]
        try:
            inst = load_benchmark_instance("acasxu", idx)
            res = verify_instance(inst, timeout=timeout, method=solver,
                                  n_workers=n_cores, threads_per_worker=n_cores)
            tag = {"verified": "V", "violated": "X", "unknown": "?"}.get(res.result, "E")
            results.append({"info": info, "result": res.result, "time": res.time_seconds,
                            "tag": tag, "details": res.details})
            print(f"  [{tag}] net={info['model']:>5} {info['prop']:<8} "
                  f"bound={res.lower_bound:+.4f} {res.time_seconds:.2f}s")
        except Exception as e:
            results.append({"info": info, "result": "error", "time": 0, "tag": "E", "details": str(e)})
            print(f"  [E] net={info['model']:>5} {info['prop']:<8} ERROR: {e}")
    return results


def print_summary_table(results, prop_filter=None):
    """Print a summary table organized by property and network."""
    by_prop = defaultdict(list)
    for r in results:
        by_prop[r["info"]["prop"]].append(r)

    props = sorted(by_prop.keys(), key=lambda p: int(p.replace("prop_", "")))
    if prop_filter:
        props = [p for p in props if p == f"prop_{prop_filter}"]

    for prop in props:
        prop_results = by_prop[prop]
        n_v = sum(1 for r in prop_results if r["result"] == "verified")
        n_x = sum(1 for r in prop_results if r["result"] == "violated")
        n_u = sum(1 for r in prop_results if r["result"] == "unknown")
        n_e = sum(1 for r in prop_results if r["result"] == "error")
        total_t = sum(r["time"] for r in prop_results)

        desc = PROP_DESCRIPTIONS.get(prop, "")
        print(f"\n  {prop} — {desc}")
        print(f"    V={n_v} X={n_x} ?={n_u} E={n_e} | {total_t:.2f}s")

        # Grid display for props with 45 instances
        if len(prop_results) > 1:
            grid = {}
            for r in prop_results:
                key = (r["info"]["net_i"], r["info"]["net_j"])
                grid[key] = r["tag"]

            if grid:
                js = sorted(set(j for _, j in grid.keys()))
                is_ = sorted(set(i for i, _ in grid.keys()))
                header = "        " + "  ".join(f"{j:>2}" for j in js)
                print(header)
                for i in is_:
                    row = f"    {i:>2}  " + "  ".join(
                        f" {grid.get((i,j), '-')}" for j in js
                    )
                    print(row)


def main():
    parser = argparse.ArgumentParser(description="ACAS Xu comprehensive verification")
    parser.add_argument("--solver", type=str, default="jacobian",
                        choices=["jacobian", "smt", "gurobi"])
    parser.add_argument("--cores", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--prop", type=int, default=None,
                        help="Only verify this property (1-10)")
    parser.add_argument("--network", type=str, default=None,
                        help="Only verify this network (e.g. 1_1, 3_5)")
    parser.add_argument("--compare", action="store_true",
                        help="Run all solvers and compare")
    args = parser.parse_args()

    n_cores = args.cores or os.cpu_count() or 4

    # Download
    download_benchmark("acasxu", skip_large=True)

    # Parse instances
    all_instances = list_instances("acasxu")
    parsed = parse_instance_info(all_instances)

    # Filter
    selected = parsed
    if args.prop:
        selected = [p for p in selected if p["prop"] == f"prop_{args.prop}"]
    if args.network:
        ni, nj = args.network.split("_")
        selected = [p for p in selected if p["net_i"] == int(ni) and p["net_j"] == int(nj)]

    if not selected:
        print("No matching instances found.")
        return

    print(f"\n{'='*72}")
    print(f"  ACAS Xu Verification: {len(selected)} instances")
    print(f"  Solver: {args.solver} | Cores: {n_cores} | Timeout: {args.timeout}s")
    print(f"{'='*72}")

    if args.compare:
        all_results = {}
        for solver in ["jacobian", "smt", "gurobi"]:
            print(f"\n--- {solver.upper()} ---")
            try:
                results = run_verification(selected, all_instances, solver, n_cores, args.timeout)
            except Exception as e:
                print(f"  Solver {solver} failed: {e}")
                results = []
            all_results[solver] = results

        # Comparison summary
        print(f"\n{'='*72}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*72}")
        for solver, results in all_results.items():
            n_v = sum(1 for r in results if r["result"] == "verified")
            n_x = sum(1 for r in results if r["result"] == "violated")
            n_u = sum(1 for r in results if r["result"] == "unknown")
            t = sum(r["time"] for r in results)
            print(f"  {solver:<10}: V={n_v:3d}  X={n_x:3d}  ?={n_u:3d}  | {t:.2f}s")
    else:
        results = run_verification(selected, all_instances, args.solver, n_cores, args.timeout)

        # Summary
        print(f"\n{'='*72}")
        print(f"  SUMMARY")
        print(f"{'='*72}")
        print_summary_table(results, args.prop)

        n_v = sum(1 for r in results if r["result"] == "verified")
        n_x = sum(1 for r in results if r["result"] == "violated")
        n_u = sum(1 for r in results if r["result"] == "unknown")
        n_e = sum(1 for r in results if r["result"] == "error")
        total_t = sum(r["time"] for r in results)
        print(f"\n  TOTAL: V={n_v} X={n_x} ?={n_u} E={n_e} | {total_t:.2f}s")
        print(f"{'='*72}")


if __name__ == "__main__":
    main()
