#!/usr/bin/env python3
"""
Cross-verify all ACAS Xu witness files.

For each witness JSON in acasxu_results/:
  1. Load the ONNX model for that network.
  2. Feed the witness input into the model.
  3. Compare the actual output with the claimed witness output.
  4. Check whether the output actually satisfies/violates the VNNLIB property.
  5. Report: PASS (witness confirmed) or FAIL (witness incorrect).

This is an independent verification — it does NOT use the verification
framework at all, only onnxruntime for inference and the VNNLIB parser
for property checking.

Usage:
    # Verify all witnesses (run run_acasxu_full.py first)
    python verify_witnesses.py

    # Verify specific network
    python verify_witnesses.py --network 1_1

    # Verify specific property
    python verify_witnesses.py --prop 2
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from qnn_verifier.benchmarks.vnnlib_parser import parse_vnnlib
from qnn_verifier.benchmarks.registry import BENCHMARKS, DEFAULT_DATA_DIR

logging.basicConfig(level=logging.WARNING, format="%(message)s")

RESULTS_DIR = Path("acasxu_results")
BENCH_DIR = DEFAULT_DATA_DIR / "_vnncomp_repo" / "benchmarks" / "acasxu_2023"


def load_onnx_session(model_name: str):
    """Load an ONNX model via onnxruntime (independent of verification framework)."""
    import onnxruntime as ort
    onnx_path = BENCH_DIR / "onnx" / f"ACASXU_run2a_{model_name}_batch_2000.onnx"
    if not onnx_path.exists():
        return None
    return ort.InferenceSession(str(onnx_path))


def run_inference(session, input_vec: list) -> np.ndarray:
    """Run inference on the ONNX model with the given input."""
    x = np.array(input_vec, dtype=np.float32).reshape(1, 1, 1, 5)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})
    return outputs[0].flatten()


def check_property_violated(prop_name: str, output: np.ndarray) -> bool:
    """
    Check if the output satisfies the UNSAFE constraints (VNNLIB semantics).
    If ALL constraints are satisfied → property is VIOLATED (unsafe region reached).
    """
    vnnlib_path = BENCH_DIR / "vnnlib" / f"{prop_name}.vnnlib"
    if not vnnlib_path.exists():
        return None

    prop = parse_vnnlib(str(vnnlib_path))
    if not prop.output_constraints:
        return None

    for c in prop.output_constraints:
        sat = _check_constraint_sat(c, output)
        if not sat:
            return False  # At least one constraint not satisfied → not violated
    return True  # All constraints satisfied → violated


def _check_constraint_sat(c, output):
    """Check if a single constraint is satisfied by the output."""
    t = c["type"]
    if t == "output_bound":
        v = c["var"]
        if v >= len(output):
            return False
        if c["op"] == ">=":
            return output[v] >= c["bound"]
        else:
            return output[v] <= c["bound"]
    elif t == "comparison":
        l, r = c["left"], c["right"]
        if l >= len(output) or r >= len(output):
            return False
        if c["op"] == "<=":
            return output[l] <= output[r]
        else:
            return output[l] >= output[r]
    elif t == "disjunction":
        # OR: at least one clause must be satisfied
        for clause in c["clauses"]:
            clause_sat = True
            for atom in clause:
                if "right" in atom:
                    sub = {"type": "comparison", **atom}
                else:
                    sub = {"type": "output_bound", **atom}
                if not _check_constraint_sat(sub, output):
                    clause_sat = False
                    break
            if clause_sat:
                return True
        return False
    return False


def verify_one_witness(witness_path: Path) -> dict:
    """Verify a single witness JSON file against the actual ONNX model."""
    with open(witness_path) as f:
        record = json.load(f)

    net = record["network"]
    prop = record["property"]
    claimed_result = record["result"]
    witness = record.get("witness", {})

    result = {
        "file": str(witness_path),
        "network": net,
        "property": prop,
        "claimed_result": claimed_result,
        "check": "skip",
        "details": "",
    }

    if claimed_result not in ("verified", "violated"):
        result["check"] = "skip"
        result["details"] = f"Claimed result is '{claimed_result}', no witness to verify"
        return result

    witness_input = witness.get("input")
    witness_output = witness.get("output")

    if not witness_input:
        result["check"] = "skip"
        result["details"] = "No witness input in JSON"
        return result

    # Load ONNX model
    session = load_onnx_session(net)
    if session is None:
        result["check"] = "skip"
        result["details"] = f"ONNX model not found for {net}"
        return result

    # Run actual inference
    actual_output = run_inference(session, witness_input)
    result["actual_output"] = actual_output.tolist()

    # Check output match (if witness output was recorded)
    if witness_output:
        max_diff = np.max(np.abs(actual_output[:len(witness_output)] -
                                  np.array(witness_output)))
        result["output_max_diff"] = float(max_diff)
        if max_diff > 0.01:
            result["check"] = "FAIL"
            result["details"] = f"Output mismatch: max_diff={max_diff:.6f}"
            return result

    # Check input bounds
    vnnlib_path = BENCH_DIR / "vnnlib" / f"{prop}.vnnlib"
    if vnnlib_path.exists():
        vnn = parse_vnnlib(str(vnnlib_path))
        for i, v in enumerate(witness_input):
            if i < len(vnn.input_lower) and np.isfinite(vnn.input_lower[i]):
                if v < vnn.input_lower[i] - 1e-6:
                    result["check"] = "FAIL"
                    result["details"] = f"Input x[{i}]={v:.6f} < lower bound {vnn.input_lower[i]:.6f}"
                    return result
            if i < len(vnn.input_upper) and np.isfinite(vnn.input_upper[i]):
                if v > vnn.input_upper[i] + 1e-6:
                    result["check"] = "FAIL"
                    result["details"] = f"Input x[{i}]={v:.6f} > upper bound {vnn.input_upper[i]:.6f}"
                    return result

    # Check property satisfaction
    actually_violated = check_property_violated(prop, actual_output)

    if claimed_result == "violated":
        if actually_violated:
            result["check"] = "PASS"
            result["details"] = "Counterexample confirmed: actual output violates property"
        else:
            result["check"] = "FAIL"
            result["details"] = "SPURIOUS counterexample: actual output does NOT violate property"

    elif claimed_result == "verified":
        # For verified: the witness input is just the nominal point.
        # We can't fully verify safety by checking one point, but we check
        # that the nominal point does NOT violate the property.
        if actually_violated:
            result["check"] = "FAIL"
            result["details"] = "Claimed verified but nominal input violates property!"
        else:
            result["check"] = "PASS"
            result["details"] = "Nominal input does not violate property (consistent with verified)"

    return result


def main():
    parser = argparse.ArgumentParser(description="Cross-verify ACAS Xu witnesses")
    parser.add_argument("--network", type=str, default=None)
    parser.add_argument("--prop", type=int, default=None)
    parser.add_argument("--dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run examples/run_acasxu_full.py first.")
        return

    # Find all witness JSON files
    witness_files = sorted(results_dir.rglob("*_witness.json"))
    if not witness_files:
        print("No witness files found.")
        return

    # Filter
    if args.network:
        witness_files = [f for f in witness_files if f.parent.name == args.network]
    if args.prop:
        witness_files = [f for f in witness_files if f"prop_{args.prop}_" in f.name]

    print(f"{'='*80}")
    print(f"  ACAS Xu Witness Cross-Verification")
    print(f"  Checking {len(witness_files)} witness files against ONNX models")
    print(f"{'='*80}")

    stats = {"PASS": 0, "FAIL": 0, "skip": 0}
    failures = []

    for wf in witness_files:
        result = verify_one_witness(wf)
        stats[result["check"]] = stats.get(result["check"], 0) + 1

        tag = {"PASS": "PASS", "FAIL": "FAIL", "skip": "SKIP"}.get(result["check"], "????")
        net = result["network"]
        prop = result["property"]
        claimed = result["claimed_result"]

        if result["check"] == "FAIL":
            failures.append(result)
            print(f"  [{tag}] {net:>5} {prop:<8} claimed={claimed:<8} | {result['details']}")
        elif result["check"] == "PASS":
            diff_str = ""
            if "output_max_diff" in result:
                diff_str = f" | output_diff={result['output_max_diff']:.8f}"
            print(f"  [{tag}] {net:>5} {prop:<8} claimed={claimed:<8} | {result['details']}{diff_str}")
        else:
            print(f"  [SKIP] {net:>5} {prop:<8} claimed={claimed:<8} | {result['details']}")

    # Summary
    print(f"\n{'='*80}")
    print(f"  CROSS-VERIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"  PASS: {stats['PASS']}")
    print(f"  FAIL: {stats['FAIL']}")
    print(f"  SKIP: {stats['skip']}")
    total = stats["PASS"] + stats["FAIL"]
    if total > 0:
        print(f"  Accuracy: {stats['PASS']}/{total} ({100*stats['PASS']/total:.1f}%)")

    if failures:
        print(f"\n  FAILURES:")
        for f in failures:
            print(f"    {f['network']} {f['property']}: {f['details']}")
            if "actual_output" in f:
                print(f"      Actual output: {[f'{v:.6f}' for v in f['actual_output'][:5]]}")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
