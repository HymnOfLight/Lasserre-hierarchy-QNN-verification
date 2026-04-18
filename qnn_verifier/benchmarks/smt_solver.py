"""
Multi-solver SMT-based neural network verification.

Supports Z3, CVC5, Bitwuzla, OpenSMT2, and any SMT-LIB2-compatible
solver.  All solvers can run in parallel (portfolio mode) — the first
solver to return a definitive answer wins.

Solver backends:
  - z3:       Python API with parallel.enable (multi-threaded)
  - cvc5:     Python API
  - bitwuzla: Python API (bit-vector / floating-point theories)
  - opensmt:  SMT-LIB2 file + subprocess (must be on PATH or specified)
  - smtlib:   Generic SMT-LIB2 file + any solver binary

Multi-core strategy:
  1. IBP pre-processing eliminates stable ReLU neurons.
  2. Portfolio: launch each solver in a separate process, first answer wins.
  3. Within each solver: use solver-native parallelism (Z3 parallel.enable,
     CVC5 --tlimit-per, etc.)
"""

import logging
import multiprocessing
import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SMTSolverConfig:
    """Configuration for a single SMT solver backend."""
    name: str
    enabled: bool = True
    timeout: float = 300.0
    threads: int = 0           # 0 = use all available cores
    binary_path: str = ""      # for subprocess-based solvers


# ------------------------------------------------------------------
# ONNX weight extraction + IBP (shared with z3_solver.py)
# ------------------------------------------------------------------

def _extract_weights_from_onnx(onnx_path: str) -> List[Dict]:
    import onnx
    from onnx import numpy_helper
    model = onnx.load(onnx_path)
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    raw = []
    for node in model.graph.node:
        op = node.op_type
        if op in ("Gemm", "MatMul"):
            W, b, trans = None, None, False
            if op == "Gemm":
                for a in node.attribute:
                    if a.name == "transB":
                        trans = bool(a.i)
            for nm in node.input:
                if nm in initializers:
                    arr = initializers[nm]
                    if arr.ndim == 2: W = arr
                    elif arr.ndim == 1: b = arr
            if W is not None:
                We = W.astype(np.float64) if trans else W.T.astype(np.float64)
                raw.append({"type": "linear", "W": We,
                            "b": b.astype(np.float64) if b is not None else np.zeros(We.shape[0])})
        elif op == "Relu": raw.append({"type": "relu"})
        elif op in ("Flatten", "Reshape"): raw.append({"type": "flatten"})
        elif op == "Add":
            b = None
            for nm in node.input:
                if nm in initializers and initializers[nm].ndim == 1:
                    b = initializers[nm]
            if b is not None: raw.append({"type": "add_bias", "b": b.astype(np.float64)})
    layers = []
    for l in raw:
        if l["type"] == "add_bias" and layers and layers[-1]["type"] == "linear":
            n = min(len(l["b"]), len(layers[-1]["b"]))
            layers[-1]["b"][:n] += l["b"][:n]
        else:
            layers.append(l)
    return layers


def _ibp_stable_neurons(layers, lb, ub):
    cur_lb, cur_ub = lb.copy(), ub.copy()
    stable = {}
    rl = 0
    for layer in layers:
        if layer["type"] == "linear":
            W, b = layer["W"], layer["b"]
            _, ni = W.shape
            li = cur_lb[:ni] if len(cur_lb) >= ni else np.pad(cur_lb, (0, ni - len(cur_lb)))
            ui = cur_ub[:ni] if len(cur_ub) >= ni else np.pad(cur_ub, (0, ni - len(cur_ub)))
            Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
            cur_lb, cur_ub = Wp @ li + Wn @ ui + b, Wp @ ui + Wn @ li + b
        elif layer["type"] == "relu":
            for j in range(len(cur_lb)):
                if cur_lb[j] >= 0: stable[(rl, j)] = "active"
                elif cur_ub[j] <= 0: stable[(rl, j)] = "inactive"
                else: stable[(rl, j)] = "unstable"
            cur_lb, cur_ub = np.maximum(cur_lb, 0), np.maximum(cur_ub, 0)
            rl += 1
    return stable


# ------------------------------------------------------------------
# SMT-LIB2 file generation
# ------------------------------------------------------------------

def generate_smtlib2(
    layers: List[Dict],
    n_inputs: int,
    n_outputs: int,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    output_constraints: List[Dict],
    stable_neurons: Dict,
) -> str:
    """Generate a complete SMT-LIB2 file encoding the NN + property."""
    lines = ["(set-logic QF_NRA)", "(set-option :produce-models true)", ""]

    # Declare input variables
    for i in range(n_inputs):
        lines.append(f"(declare-const X_{i} Real)")
    lines.append("")

    # Input bounds
    for i in range(n_inputs):
        if np.isfinite(input_lb[i]):
            lines.append(f"(assert (>= X_{i} {input_lb[i]:.15e}))")
        if np.isfinite(input_ub[i]):
            lines.append(f"(assert (<= X_{i} {input_ub[i]:.15e}))")
    lines.append("")

    # Encode layers
    current = [f"X_{i}" for i in range(n_inputs)]
    relu_layer = 0
    layer_idx = 0

    for layer in layers:
        if layer["type"] == "linear":
            W, b = layer["W"], layer["b"]
            no, ni = W.shape
            inv = current[:ni]
            while len(inv) < ni:
                inv.append("0.0")
            new_vars = []
            for j in range(no):
                vn = f"h_{layer_idx}_{j}"
                lines.append(f"(declare-const {vn} Real)")
                nz = np.nonzero(np.abs(W[j, :]) > 1e-15)[0]
                if len(nz) == 0:
                    lines.append(f"(assert (= {vn} {b[j]:.15e}))")
                else:
                    terms = [f"{b[j]:.15e}"]
                    for k in nz:
                        terms.append(f"(* {W[j, k]:.15e} {inv[k]})")
                    lines.append(f"(assert (= {vn} (+ {' '.join(terms)})))")
                new_vars.append(vn)
            current = new_vars
            layer_idx += 1

        elif layer["type"] == "relu":
            new_vars = []
            for j, v in enumerate(current):
                key = (relu_layer, j)
                st = stable_neurons.get(key, "unstable")
                rn = f"r_{relu_layer}_{j}"
                if st == "active":
                    new_vars.append(v)
                    continue
                elif st == "inactive":
                    lines.append(f"(declare-const {rn} Real)")
                    lines.append(f"(assert (= {rn} 0.0))")
                else:
                    lines.append(f"(declare-const {rn} Real)")
                    lines.append(f"(assert (>= {rn} 0.0))")
                    lines.append(f"(assert (>= {rn} {v}))")
                    lines.append(f"(assert (or (= {rn} 0.0) (= {rn} {v})))")
                new_vars.append(rn)
            current = new_vars
            relu_layer += 1

        elif layer["type"] == "flatten":
            pass

    lines.append("")

    # Output variables
    for i in range(min(n_outputs, len(current))):
        lines.append(f"(declare-const Y_{i} Real)")
        lines.append(f"(assert (= Y_{i} {current[i]}))")
    lines.append("")

    # Output constraints
    for c in output_constraints:
        lines.append(_constraint_to_smtlib2(c))
    lines.append("")

    lines.append("(check-sat)")
    lines.append("(exit)")
    return "\n".join(lines)


def _constraint_to_smtlib2(c: Dict) -> str:
    t = c["type"]
    if t == "output_bound":
        return f"(assert ({c['op']} Y_{c['var']} {c['bound']:.15e}))"
    elif t == "comparison":
        return f"(assert ({c['op']} Y_{c['left']} Y_{c['right']}))"
    elif t == "disjunction":
        clauses = []
        for cl in c["clauses"]:
            atoms = []
            for a in cl:
                if "right" in a:
                    atoms.append(f"({a['op']} Y_{a['left']} Y_{a['right']})")
                elif "bound" in a:
                    atoms.append(f"({a['op']} Y_{a['var']} {a['bound']:.15e})")
            if len(atoms) == 1:
                clauses.append(atoms[0])
            elif atoms:
                clauses.append(f"(and {' '.join(atoms)})")
        if len(clauses) == 1:
            return f"(assert {clauses[0]})"
        return f"(assert (or {' '.join(clauses)}))"
    return ""


# ------------------------------------------------------------------
# Individual solver backends
# ------------------------------------------------------------------

def _solve_z3(smtlib2_text: str, timeout_ms: int, threads: int) -> Tuple[str, float]:
    """Solve with Z3 Python API using parallel mode."""
    import z3
    t0 = time.time()
    z3.set_param("parallel.enable", threads > 1)
    if threads > 1:
        z3.set_param("parallel.threads.max", threads)
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.from_string(smtlib2_text.replace("(check-sat)", "").replace("(exit)", ""))
    result = solver.check()
    elapsed = time.time() - t0
    if result == z3.sat:
        return "sat", elapsed
    elif result == z3.unsat:
        return "unsat", elapsed
    return "unknown", elapsed


def _solve_cvc5(smtlib2_text: str, timeout_ms: int, threads: int) -> Tuple[str, float]:
    """Solve with CVC5 Python API."""
    import cvc5
    t0 = time.time()
    tm = cvc5.TermManager()
    solver = cvc5.Solver(tm)
    solver.setOption("produce-models", "true")
    solver.setOption("tlimit-per", str(timeout_ms))
    if threads > 1:
        solver.setOption("parallel", "true")
    solver.setLogic("QF_NRA")

    # Parse SMT-LIB2 via file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".smt2", delete=False) as f:
        f.write(smtlib2_text)
        tmp_path = f.name
    try:
        # Use subprocess for CVC5 since the Python API parser is complex
        proc = subprocess.run(
            ["python3", "-c", f"""
import cvc5, sys
tm = cvc5.TermManager()
s = cvc5.Solver(tm)
s.setOption("produce-models", "true")
s.setOption("tlimit-per", "{timeout_ms}")
s.setLogic("QF_NRA")
parser = cvc5.InputParser(s)
parser.setFileInput(cvc5.InputLanguage.SMT_LIB_2_6, "{tmp_path}")
sm = parser.getSymbolManager()
while True:
    cmd = parser.nextCommand()
    if cmd.isNull(): break
    cmd.invoke(s, sm)
"""],
            capture_output=True, text=True,
            timeout=timeout_ms / 1000 + 5,
        )
        output = proc.stdout.strip().lower()
        elapsed = time.time() - t0
        if "unsat" in output: return "unsat", elapsed
        elif "sat" in output: return "sat", elapsed
        return "unknown", elapsed
    except Exception:
        return "unknown", time.time() - t0
    finally:
        os.unlink(tmp_path)


def _solve_bitwuzla(smtlib2_text: str, timeout_ms: int, threads: int) -> Tuple[str, float]:
    """Solve with Bitwuzla via SMT-LIB2 file subprocess."""
    t0 = time.time()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".smt2", delete=False) as f:
        # Bitwuzla uses QF_BV / QF_FP; adapt logic for real arithmetic
        # Use it only when appropriate, fallback to unknown otherwise
        f.write(smtlib2_text)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            ["python3", "-c", f"""
import bitwuzla
options = bitwuzla.Options()
options.set(bitwuzla.Option.TIME_LIMIT, {timeout_ms})
tm = bitwuzla.TermManager()
parser = bitwuzla.Parser(tm, options)
try:
    parser.parse("{tmp_path}")
    bz = parser.bitwuzla()
    result = bz.check_sat()
    if result == bitwuzla.Result.SAT: print("sat")
    elif result == bitwuzla.Result.UNSAT: print("unsat")
    else: print("unknown")
except Exception as e:
    print("unknown")
"""],
            capture_output=True, text=True,
            timeout=timeout_ms / 1000 + 5,
        )
        output = proc.stdout.strip().lower()
        elapsed = time.time() - t0
        if "unsat" in output: return "unsat", elapsed
        elif "sat" in output: return "sat", elapsed
        return "unknown", elapsed
    except Exception:
        return "unknown", time.time() - t0
    finally:
        os.unlink(tmp_path)


def _solve_subprocess(binary: str, smtlib2_text: str, timeout_ms: int) -> Tuple[str, float]:
    """Solve with any SMT-LIB2-compatible solver binary (OpenSMT2, etc.)."""
    t0 = time.time()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".smt2", delete=False) as f:
        f.write(smtlib2_text)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            [binary, tmp_path],
            capture_output=True, text=True,
            timeout=timeout_ms / 1000 + 5,
        )
        output = proc.stdout.strip().lower()
        elapsed = time.time() - t0
        if "unsat" in output: return "unsat", elapsed
        elif "sat" in output: return "sat", elapsed
        return "unknown", elapsed
    except FileNotFoundError:
        return "error", time.time() - t0
    except subprocess.TimeoutExpired:
        return "unknown", time.time() - t0
    except Exception:
        return "unknown", time.time() - t0
    finally:
        try: os.unlink(tmp_path)
        except: pass


# ------------------------------------------------------------------
# Worker function for portfolio parallel execution
# ------------------------------------------------------------------

def _portfolio_worker(args) -> Tuple[str, str, float]:
    """Run a single solver backend. Returns (solver_name, result, elapsed)."""
    solver_name, smtlib2_text, timeout_ms, threads, binary_path = args
    try:
        if solver_name == "z3":
            result, elapsed = _solve_z3(smtlib2_text, timeout_ms, threads)
        elif solver_name == "cvc5":
            result, elapsed = _solve_cvc5(smtlib2_text, timeout_ms, threads)
        elif solver_name == "bitwuzla":
            result, elapsed = _solve_bitwuzla(smtlib2_text, timeout_ms, threads)
        elif solver_name == "opensmt" or binary_path:
            bin_path = binary_path or "opensmt"
            result, elapsed = _solve_subprocess(bin_path, smtlib2_text, timeout_ms)
        else:
            result, elapsed = "error", 0.0
        return solver_name, result, elapsed
    except Exception as e:
        return solver_name, "error", 0.0


# ------------------------------------------------------------------
# Main multi-solver entry point
# ------------------------------------------------------------------

def verify_with_smt(
    onnx_path: str,
    property,
    timeout: float = 300.0,
    solvers: Optional[List[str]] = None,
    n_threads: int = 0,
    opensmt_path: str = "",
) -> Dict:
    """
    Verify a neural network property using multiple SMT solvers in parallel.

    Each solver runs in its own process.  The first solver to return
    SAT or UNSAT wins; the rest are cancelled.

    Args:
        onnx_path: Path to ONNX model.
        property: Parsed VNNLIBProperty.
        timeout: Overall timeout in seconds.
        solvers: List of solver names to use. Default: all available.
            Options: "z3", "cvc5", "bitwuzla", "opensmt", or a path to
            any SMT-LIB2-compatible binary.
        n_threads: Threads per solver (0=auto).
        opensmt_path: Path to the OpenSMT2 binary (if not on PATH).

    Returns:
        Dict with 'result', 'solver', 'time_seconds', 'details'.
    """
    t0 = time.time()

    total_cores = os.cpu_count() or 4

    # Detect available solvers
    if solvers is None:
        solvers = _detect_available_solvers(opensmt_path)

    if not solvers:
        return {"result": "error", "time_seconds": 0.0,
                "details": "No SMT solvers available", "solver": "none"}

    # Extract network and compute stable neurons
    layers = _extract_weights_from_onnx(onnx_path)
    if not any(l["type"] == "linear" for l in layers):
        return {"result": "error", "time_seconds": 0.0,
                "details": "No linear layers in ONNX", "solver": "none"}

    lb = np.where(np.isfinite(property.input_lower), property.input_lower, -1e6)
    ub = np.where(np.isfinite(property.input_upper), property.input_upper, 1e6)
    stable = _ibp_stable_neurons(layers, lb, ub)

    n_stable = sum(1 for s in stable.values() if s != "unstable")
    n_unstable = sum(1 for s in stable.values() if s == "unstable")

    logger.info(
        f"SMT portfolio: solvers={solvers} | "
        f"ReLU: {len(stable)} total, {n_stable} stable, {n_unstable} unstable"
    )

    # Generate SMT-LIB2 formula (shared across all solvers)
    smtlib2 = generate_smtlib2(
        layers, property.n_inputs, property.n_outputs,
        lb, ub, property.output_constraints, stable,
    )

    timeout_ms = int(timeout * 1000)
    if n_threads <= 0:
        n_threads = max(1, total_cores // len(solvers))

    # Build tasks for portfolio
    tasks = []
    for s in solvers:
        bp = ""
        if s == "opensmt":
            bp = opensmt_path or "opensmt"
        elif s not in ("z3", "cvc5", "bitwuzla"):
            bp = s  # treat as binary path
        tasks.append((s, smtlib2, timeout_ms, n_threads, bp))

    # Single solver: no need for multiprocessing
    if len(tasks) == 1:
        solver_name, result, elapsed = _portfolio_worker(tasks[0])
        return _format_result(result, solver_name, elapsed, n_unstable, solvers)

    # Multi-solver portfolio
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=ctx) as pool:
        futures = {pool.submit(_portfolio_worker, t): t[0] for t in tasks}

        remaining = timeout - (time.time() - t0) + 2
        try:
            for future in as_completed(futures, timeout=max(remaining, 1)):
                solver_name = futures[future]
                try:
                    name, result, elapsed = future.result(timeout=5)
                except Exception:
                    continue

                if result in ("sat", "unsat"):
                    # Definitive answer — cancel the rest
                    for f in futures:
                        f.cancel()
                    return _format_result(result, name, elapsed, n_unstable, solvers)
        except Exception:
            pass

    # No solver returned a definitive answer
    return _format_result("unknown", "portfolio", time.time() - t0, n_unstable, solvers)


def _detect_available_solvers(opensmt_path: str = "") -> List[str]:
    """Detect which SMT solver backends are available."""
    available = []
    try:
        import z3
        available.append("z3")
    except ImportError:
        pass
    try:
        import cvc5
        available.append("cvc5")
    except ImportError:
        pass
    try:
        import bitwuzla
        available.append("bitwuzla")
    except ImportError:
        pass
    # Check OpenSMT2 binary
    osmt = opensmt_path or "opensmt"
    try:
        subprocess.run([osmt, "--version"], capture_output=True, timeout=5)
        available.append("opensmt")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return available


def _format_result(result, solver_name, elapsed, n_unstable, all_solvers):
    if result == "unsat":
        return {
            "result": "verified",
            "solver": solver_name,
            "time_seconds": elapsed,
            "details": f"UNSAT by {solver_name} — exact proof "
                       f"({n_unstable} unstable ReLUs, portfolio: {all_solvers})",
        }
    elif result == "sat":
        return {
            "result": "violated",
            "solver": solver_name,
            "time_seconds": elapsed,
            "details": f"SAT by {solver_name} — counterexample found "
                       f"(portfolio: {all_solvers})",
        }
    else:
        return {
            "result": "unknown",
            "solver": solver_name,
            "time_seconds": elapsed,
            "details": f"No definitive answer ({n_unstable} unstable ReLUs, "
                       f"portfolio: {all_solvers})",
        }
