#!/usr/bin/env python3
"""
SMT Solver Performance Benchmark for Neural Network Verification.

Compares multiple SMT/SAT solvers on the same neural network verification
instances using the SMT-LIB2 file format as the universal interface.

Supported solvers (auto-detected):
  Python API:     z3, cvc5, bitwuzla (QF_LRA/QF_NRA)
  Binary (PATH):  z3, cvc5, bitwuzla, boolector, yices-smt2,
                  mathsat, opensmt, any SMT-LIB2 binary

Usage:
    # Benchmark all available solvers on ACAS Xu prop_1
    python benchmark_solvers.py --benchmark acasxu --instances 0-4

    # Specific solvers
    python benchmark_solvers.py --benchmark acasxu --instance 0 \
        --solvers z3,cvc5,bitwuzla

    # Custom timeout
    python benchmark_solvers.py --benchmark acasxu --instances 0-9 --timeout 60

    # Use pre-generated SMT-LIB2 files
    python benchmark_solvers.py --smt2-dir smt_formulas/acasxu/

    # Add custom solver binary
    python benchmark_solvers.py --add-solver /path/to/mysolver --benchmark acasxu
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Solver Registry
# ------------------------------------------------------------------

class SolverBackend:
    """A single SMT solver backend."""
    def __init__(self, name: str, mode: str = "api", binary: str = "", theory: str = "QF_NRA"):
        self.name = name
        self.mode = mode  # "api" or "binary"
        self.binary = binary
        self.theory = theory
        self.available = False

    def check_available(self) -> bool:
        if self.mode == "api":
            try:
                if self.name == "z3":
                    import z3; self.available = True
                elif self.name == "cvc5":
                    import cvc5; self.available = True
                elif self.name == "bitwuzla":
                    import bitwuzla
                    # Bitwuzla only supports QF_BV/QF_FP, not real arithmetic
                    self.available = self.theory in ("QF_BV", "QF_FP", "QF_ABV")
                else:
                    self.available = False
            except ImportError:
                self.available = False
        elif self.mode == "binary":
            self.available = shutil.which(self.binary) is not None or os.path.isfile(self.binary)
        return self.available

    def solve(self, smt2_path: str, timeout: float) -> Tuple[str, float]:
        """Solve an SMT-LIB2 file. Returns (result, elapsed)."""
        t0 = time.time()
        try:
            if self.mode == "api":
                result = self._solve_api(smt2_path, timeout)
            else:
                result = self._solve_binary(smt2_path, timeout)
            return result, time.time() - t0
        except Exception as e:
            return "error", time.time() - t0

    def _solve_api(self, smt2_path: str, timeout: float) -> str:
        if self.name == "z3":
            return self._solve_z3(smt2_path, timeout)
        elif self.name == "cvc5":
            return self._solve_cvc5(smt2_path, timeout)
        return "unknown"

    def _solve_z3(self, smt2_path: str, timeout: float) -> str:
        import z3
        s = z3.Solver()
        s.set("timeout", int(timeout * 1000))
        s.from_file(smt2_path)
        r = s.check()
        if r == z3.sat: return "sat"
        if r == z3.unsat: return "unsat"
        return "unknown"

    def _solve_cvc5(self, smt2_path: str, timeout: float) -> str:
        cmd = [
            "python3", "-c",
            f"""import cvc5
tm = cvc5.TermManager()
s = cvc5.Solver(tm)
s.setOption("tlimit", "{int(timeout*1000)}")
parser = cvc5.InputParser(s)
parser.setFileInput(cvc5.InputLanguage.SMT_LIB_2_6, "{smt2_path}")
sm = parser.getSymbolManager()
while True:
    cmd = parser.nextCommand()
    if cmd.isNull(): break
    cmd.invoke(s, sm)
"""
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+10)
            out = (proc.stdout + proc.stderr).lower()
            if "unsat" in out: return "unsat"
            if "sat" in out and "unsat" not in out: return "sat"
        except Exception:
            pass
        return "unknown"

    def _solve_binary(self, smt2_path: str, timeout: float) -> str:
        try:
            proc = subprocess.run(
                [self.binary, smt2_path],
                capture_output=True, text=True, timeout=timeout+5
            )
            out = proc.stdout.lower()
            if "unsat" in out: return "unsat"
            if "sat" in out and "unsat" not in out: return "sat"
        except subprocess.TimeoutExpired:
            return "timeout"
        except FileNotFoundError:
            return "error"
        except Exception:
            pass
        return "unknown"


def detect_all_solvers(extra_binaries: Optional[List[str]] = None) -> List[SolverBackend]:
    """Detect all available SMT solvers."""
    solvers = [
        SolverBackend("z3", "api", theory="QF_NRA"),
        SolverBackend("cvc5", "api", theory="QF_NRA"),
        # Binary-based solvers
        SolverBackend("z3-bin", "binary", binary="z3", theory="QF_NRA"),
        SolverBackend("cvc5-bin", "binary", binary="cvc5", theory="QF_NRA"),
        SolverBackend("bitwuzla-bin", "binary", binary="bitwuzla", theory="QF_BV"),
        SolverBackend("boolector", "binary", binary="boolector", theory="QF_BV"),
        SolverBackend("yices2", "binary", binary="yices-smt2", theory="QF_NRA"),
        SolverBackend("mathsat", "binary", binary="mathsat", theory="QF_NRA"),
        SolverBackend("opensmt", "binary", binary="opensmt", theory="QF_NRA"),
    ]

    if extra_binaries:
        for b in extra_binaries:
            name = Path(b).stem
            solvers.append(SolverBackend(name, "binary", binary=b, theory="QF_NRA"))

    available = []
    for s in solvers:
        if s.check_available():
            # Skip binary duplicates if API version exists
            if s.mode == "binary" and any(a.name == s.name.replace("-bin","") and a.mode == "api" for a in available):
                continue
            available.append(s)

    return available


# ------------------------------------------------------------------
# Benchmark Runner
# ------------------------------------------------------------------

def run_solver_benchmark(
    smt2_files: List[str],
    solvers: List[SolverBackend],
    timeout: float = 60.0,
) -> Dict:
    """Run all solvers on all SMT-LIB2 files and collect results."""
    results = {s.name: [] for s in solvers}

    n_files = len(smt2_files)
    n_solvers = len(solvers)

    print(f"\n{'='*90}")
    print(f"  SMT Solver Benchmark: {n_files} instances × {n_solvers} solvers")
    print(f"  Solvers: {[s.name for s in solvers]}")
    print(f"  Timeout: {timeout}s per instance per solver")
    print(f"{'='*90}")

    # Header
    header = f"  {'Instance':<35}"
    for s in solvers:
        header += f" {s.name:^12}"
    print(header)
    print(f"  {'-'*85}")

    for fi, smt2_path in enumerate(smt2_files):
        fname = Path(smt2_path).stem[:33]
        row = f"  {fname:<35}"

        for solver in solvers:
            result, elapsed = solver.solve(smt2_path, timeout)
            results[solver.name].append({
                "file": smt2_path,
                "result": result,
                "time": elapsed,
            })
            # Format cell
            tag = {"sat": "SAT", "unsat": "UNS", "unknown": " ? ", "timeout": "T/O", "error": "ERR"}.get(result, "???")
            cell = f"{tag} {elapsed:.1f}s"
            row += f" {cell:^12}"

        print(row)

    return results


def print_summary(results: Dict, solvers: List[SolverBackend]):
    """Print summary statistics."""
    print(f"\n{'='*90}")
    print(f"  SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Solver':<15} {'SAT':>5} {'UNSAT':>6} {'UNK':>5} {'T/O':>5} {'ERR':>5} {'Time':>8} {'Avg':>7}")
    print(f"  {'-'*60}")

    for solver in solvers:
        data = results[solver.name]
        n_sat = sum(1 for d in data if d["result"] == "sat")
        n_unsat = sum(1 for d in data if d["result"] == "unsat")
        n_unk = sum(1 for d in data if d["result"] == "unknown")
        n_to = sum(1 for d in data if d["result"] == "timeout")
        n_err = sum(1 for d in data if d["result"] == "error")
        total_t = sum(d["time"] for d in data)
        avg_t = total_t / max(len(data), 1)
        print(f"  {solver.name:<15} {n_sat:>5} {n_unsat:>6} {n_unk:>5} {n_to:>5} {n_err:>5} {total_t:>7.1f}s {avg_t:>6.2f}s")

    # Comparison: which solver solved the most?
    print(f"\n  {'Solver':<15} {'Solved':>7} {'Rate':>6}")
    print(f"  {'-'*30}")
    for solver in solvers:
        data = results[solver.name]
        solved = sum(1 for d in data if d["result"] in ("sat", "unsat"))
        rate = solved / max(len(data), 1) * 100
        print(f"  {solver.name:<15} {solved:>7} {rate:>5.1f}%")

    print(f"{'='*90}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SMT Solver Performance Benchmark")
    parser.add_argument("--benchmark", type=str, default="acasxu",
                        help="Benchmark name (generates SMT-LIB2 if needed)")
    parser.add_argument("--instance", type=int, default=None)
    parser.add_argument("--instances", type=str, default="0-4",
                        help="Instance range (e.g. 0-9)")
    parser.add_argument("--smt2-dir", type=str, default=None,
                        help="Use pre-generated SMT-LIB2 files from this directory")
    parser.add_argument("--timeout", type=float, default=30,
                        help="Timeout per solver per instance")
    parser.add_argument("--solvers", type=str, default=None,
                        help="Comma-separated solver names (default: all available)")
    parser.add_argument("--add-solver", type=str, default=None,
                        help="Add custom solver binary path")
    parser.add_argument("--no-padic", action="store_true")
    args = parser.parse_args()

    if args.no_padic:
        from qnn_verifier.benchmarks.symbolic_rewrite import set_padic_enabled
        set_padic_enabled(False)

    # Detect solvers
    extra = [args.add_solver] if args.add_solver else None
    all_solvers = detect_all_solvers(extra)

    if args.solvers:
        names = [s.strip() for s in args.solvers.split(",")]
        all_solvers = [s for s in all_solvers if s.name in names]

    if not all_solvers:
        print("No SMT solvers available!")
        return

    print(f"  Detected solvers: {[s.name for s in all_solvers]}")

    # Get or generate SMT-LIB2 files
    smt2_files = []

    if args.smt2_dir:
        smt2_dir = Path(args.smt2_dir)
        smt2_files = sorted(str(f) for f in smt2_dir.glob("*.smt2"))
    else:
        # Generate from benchmark
        from qnn_verifier.benchmarks import download_benchmark, load_benchmark_instance
        from qnn_verifier.benchmarks.loader import list_instances
        from qnn_verifier.benchmarks.smt_solver import (
            _extract_weights_from_onnx, _ibp_stable_neurons,
            generate_smtlib2, save_smtlib2,
        )
        from qnn_verifier.benchmarks.symbolic_rewrite import symbolic_rewrite_preprocess

        download_benchmark(args.benchmark, skip_large=True)
        instances = list_instances(args.benchmark)

        if args.instance is not None:
            indices = [args.instance]
        else:
            a, b = args.instances.split("-")
            indices = list(range(int(a), int(b) + 1))

        for idx in indices:
            if idx >= len(instances):
                continue
            inst = load_benchmark_instance(args.benchmark, idx)
            if not inst.property or not Path(inst.model_path).exists():
                continue

            prop = inst.property
            layers = _extract_weights_from_onnx(inst.model_path)
            lb = np.where(np.isfinite(prop.input_lower), prop.input_lower, -1e6)
            ub = np.where(np.isfinite(prop.input_upper), prop.input_upper, 1e6)

            layers, lb, ub, stable, out_c, _ = symbolic_rewrite_preprocess(
                layers, lb, ub, prop.output_constraints, prop.n_outputs)

            smt2 = generate_smtlib2(layers, prop.n_inputs, prop.n_outputs,
                                    lb, ub, out_c, stable)
            smt2_path = save_smtlib2(smt2, args.benchmark,
                                     Path(inst.property_path).stem)
            smt2_files.append(smt2_path)

    if not smt2_files:
        print("No SMT-LIB2 files to benchmark!")
        return

    print(f"  SMT-LIB2 files: {len(smt2_files)}")

    # Run benchmark
    results = run_solver_benchmark(smt2_files, all_solvers, args.timeout)
    print_summary(results, all_solvers)


if __name__ == "__main__":
    main()
