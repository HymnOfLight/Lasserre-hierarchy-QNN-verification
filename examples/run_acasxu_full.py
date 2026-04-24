#!/usr/bin/env python3
"""
Run ALL 186 ACAS Xu benchmark instances with full output:
  - Every instance generates a C program (saved to ./acasxu_results/<net>/<prop>.c)
  - VERIFIED: output bounds as safety witness
  - VIOLATED: concrete counterexample input/output
  - UNKNOWN: nominal input/output + margin
  - JSON report saved to ./acasxu_results/report.json
  - Human-readable summary to ./acasxu_results/summary.txt

Usage:
    python run_acasxu_full.py
    python run_acasxu_full.py --timeout 10 --cores 32
    python run_acasxu_full.py --prop 2             # only property 2
    python run_acasxu_full.py --network 1_1         # only network 1_1
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from qnn_verifier.benchmarks import (
    download_benchmark, load_benchmark_instance, verify_instance,
)
from qnn_verifier.benchmarks.loader import list_instances
from qnn_verifier.benchmarks.framac_solver import generate_c_program, save_c_program
from qnn_verifier.benchmarks.smt_solver import (
    _extract_weights_from_onnx, _ibp_stable_neurons, generate_smtlib2, save_smtlib2,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("acasxu_results")

PROP_DESC = {
    "prop_1": "If COC >= 1500, property violated (output bound on Y_0)",
    "prop_2": "COC should not be maximal output (Y_i <= Y_0 for all i)",
    "prop_3": "COC should not be minimal output (Y_0 <= Y_i for all i)",
    "prop_4": "COC should not be minimal (different input region)",
    "prop_5": "Strong right should be minimal (network 1_1)",
    "prop_6": "COC should be minimal (network 1_1)",
    "prop_7": "Strong left or strong right not minimal (network 1_9, OR)",
    "prop_8": "Output 0 or 1 is minimal (network 2_9, OR)",
    "prop_9": "Strong left should be minimal (network 3_3)",
    "prop_10": "COC should not be maximal (network 4_5)",
}


def run_all(args):
    t_global = time.time()
    n_cores = args.cores or os.cpu_count() or 4

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download
    download_benchmark("acasxu", skip_large=True)
    all_instances = list_instances("acasxu")

    # Parse structure
    parsed = []
    for idx, (model, prop, timeout) in enumerate(all_instances):
        m = Path(model).stem.replace("ACASXU_run2a_", "").replace("_batch_2000", "")
        p = Path(prop).stem
        parts = m.split("_")
        net_i, net_j = (int(parts[0]), int(parts[1])) if len(parts) == 2 else (0, 0)
        parsed.append({"idx": idx, "model": m, "prop": p,
                        "net_i": net_i, "net_j": net_j, "timeout": timeout,
                        "model_path": model, "prop_path": prop})

    # Filter
    selected = parsed
    if args.prop:
        selected = [p for p in selected if p["prop"] == f"prop_{args.prop}"]
    if args.network:
        ni, nj = args.network.split("_")
        selected = [p for p in selected if p["net_i"] == int(ni) and p["net_j"] == int(nj)]

    print(f"{'='*80}")
    print(f"  ACAS Xu Full Verification: {len(selected)} instances")
    print(f"  Output directory: {OUTPUT_DIR.resolve()}")
    print(f"  Timeout: {args.timeout}s | Cores: {n_cores}")
    print(f"{'='*80}")

    records = []
    summary_lines = []

    for info in selected:
        idx = info["idx"]
        net = info["model"]
        prop = info["prop"]
        net_dir = OUTPUT_DIR / net
        net_dir.mkdir(parents=True, exist_ok=True)

        # Load instance
        inst = load_benchmark_instance("acasxu", idx)

        # ---- Generate and save C program ----
        c_path = ""
        try:
            onnx_path = inst.model_path
            layers = _extract_weights_from_onnx(onnx_path)
            prop_obj = inst.property
            lb = np.where(np.isfinite(prop_obj.input_lower), prop_obj.input_lower, -1e6)
            ub = np.where(np.isfinite(prop_obj.input_upper), prop_obj.input_upper, 1e6)
            stable = _ibp_stable_neurons(layers, lb, ub)

            c_code = generate_c_program(
                layers, prop_obj.n_inputs, prop_obj.n_outputs,
                lb, ub, prop_obj.output_constraints, stable,
            )
            c_path = str(net_dir / f"{prop}.c")
            Path(c_path).write_text(c_code)
        except Exception as e:
            c_path = f"ERROR: {e}"

        # ---- Generate and save SMT-LIB2 ----
        smt_path = ""
        try:
            smt_code = generate_smtlib2(
                layers, prop_obj.n_inputs, prop_obj.n_outputs,
                lb, ub, prop_obj.output_constraints, stable,
            )
            smt_path = str(net_dir / f"{prop}.smt2")
            Path(smt_path).write_text(smt_code)
        except Exception:
            pass

        # ---- Run verification ----
        res = verify_instance(inst, timeout=args.timeout, method="jacobian",
                              n_workers=n_cores)

        # ---- Build record ----
        record = {
            "network": net,
            "property": prop,
            "property_description": PROP_DESC.get(prop, ""),
            "result": res.result,
            "margin": float(res.lower_bound),
            "time_seconds": res.time_seconds,
            "method": res.method,
            "c_file": c_path,
            "smt_file": smt_path,
            "witness": {},
        }

        if res.witness_input:
            record["witness"]["input"] = [round(v, 10) for v in res.witness_input]
        if res.witness_output:
            record["witness"]["output"] = [round(v, 10) for v in res.witness_output]
        if res.output_bounds_lower:
            record["witness"]["output_lower_bound"] = [round(v, 6) for v in res.output_bounds_lower]
        if res.output_bounds_upper:
            record["witness"]["output_upper_bound"] = [round(v, 6) for v in res.output_bounds_upper]

        records.append(record)

        # ---- Print ----
        tag = {"verified": "V", "violated": "X", "unknown": "?"}.get(res.result, "E")
        line = f"[{tag}] net={net:>5} {prop:<8} margin={res.lower_bound:+.4f} {res.time_seconds:.2f}s"
        print(f"  {line}")

        if res.result == "violated" and res.witness_input:
            print(f"       CEX input:  {[f'{v:.6f}' for v in res.witness_input]}")
            print(f"       CEX output: {[f'{v:.6f}' for v in (res.witness_output or [])]}")
        elif res.result == "verified" and res.output_bounds_upper:
            print(f"       Output UB:  {[f'{v:.4f}' for v in res.output_bounds_upper[:5]]}")

        # Save per-instance witness file
        witness_path = net_dir / f"{prop}_witness.json"
        with open(witness_path, "w") as f:
            json.dump(record, f, indent=2)

        summary_lines.append(line)

    elapsed = time.time() - t_global

    # ---- Save JSON report ----
    report = {
        "benchmark": "ACAS Xu",
        "total_instances": len(records),
        "verified": sum(1 for r in records if r["result"] == "verified"),
        "violated": sum(1 for r in records if r["result"] == "violated"),
        "unknown": sum(1 for r in records if r["result"] == "unknown"),
        "error": sum(1 for r in records if r["result"] == "error"),
        "total_time_seconds": elapsed,
        "instances": records,
    }
    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ---- Save summary text ----
    summary_path = OUTPUT_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"ACAS Xu Verification Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total: {len(records)} instances\n")
        f.write(f"Verified: {report['verified']}\n")
        f.write(f"Violated: {report['violated']}\n")
        f.write(f"Unknown:  {report['unknown']}\n")
        f.write(f"Time:     {elapsed:.2f}s\n\n")
        for line in summary_lines:
            f.write(line + "\n")

    # ---- Print summary ----
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")

    by_prop = defaultdict(list)
    for r in records:
        by_prop[r["property"]].append(r)

    for prop in sorted(by_prop.keys(), key=lambda p: int(p.replace("prop_", ""))):
        prs = by_prop[prop]
        nv = sum(1 for r in prs if r["result"] == "verified")
        nx = sum(1 for r in prs if r["result"] == "violated")
        nu = sum(1 for r in prs if r["result"] == "unknown")
        desc = PROP_DESC.get(prop, "")
        print(f"  {prop}: V={nv:2d} X={nx:2d} ?={nu:2d}  | {desc}")

    print(f"\n  TOTAL: V={report['verified']} X={report['violated']} "
          f"?={report['unknown']} | {elapsed:.2f}s")
    print(f"\n  Files saved:")
    print(f"    {report_path}")
    print(f"    {summary_path}")
    n_c = sum(1 for r in records if r["c_file"] and not r["c_file"].startswith("ERROR"))
    n_smt = sum(1 for r in records if r["smt_file"])
    n_witness = sum(1 for r in records if r["witness"])
    print(f"    {n_c} C programs in {OUTPUT_DIR}/*/prop_*.c")
    print(f"    {n_smt} SMT-LIB2 files in {OUTPUT_DIR}/*/prop_*.smt2")
    print(f"    {n_witness} witness files in {OUTPUT_DIR}/*/prop_*_witness.json")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="ACAS Xu full verification with witnesses")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--cores", type=int, default=0)
    parser.add_argument("--prop", type=int, default=None)
    parser.add_argument("--network", type=str, default=None)
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
