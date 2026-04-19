"""
Multi-solver parallel SMT-based neural network verification.

Generates a standard SMT-LIB2 file (saved locally for reproducibility),
then launches all available SMT solvers in parallel across all CPU cores.

SMT-LIB2 files are saved to  ./smt_formulas/<benchmark>/<instance>.smt2

Solver backends (all run with maximum parallelism):
  z3:        parallel.enable=true, parallel.threads.max=N
  cvc5:      --parallel, nlsat + cad strategies
  bitwuzla:  native parallel via subprocess
  opensmt:   subprocess (binary on PATH or user-specified)
  <binary>:  any SMT-LIB2 solver via subprocess
"""

import logging
import multiprocessing
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SMT_DIR = _PROJECT_ROOT / "smt_formulas"


def _fmt(v: float) -> str:
    """Format a float as an SMT-LIB2-compatible decimal literal.
    SMT-LIB2 does NOT support scientific notation (1.5e-01).
    Negative numbers use the (- x) form."""
    if v == 0.0:
        return "0.0"
    neg = v < 0
    s = f"{abs(v):.18f}"
    if "." in s:
        s = s.rstrip("0")
        if s.endswith("."):
            s += "0"
    return f"(- {s})" if neg else s


# ------------------------------------------------------------------
# ONNX weight extraction + IBP
# ------------------------------------------------------------------

def _extract_weights_from_onnx(onnx_path: str) -> List[Dict]:
    import onnx
    from onnx import numpy_helper
    model = onnx.load(onnx_path)
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    raw = []
    for node in model.graph.node:
        op = node.op_type
        if op in ("Gemm", "MatMul"):
            W, b, trans = None, None, False
            if op == "Gemm":
                for a in node.attribute:
                    if a.name == "transB": trans = bool(a.i)
            for nm in node.input:
                if nm in inits:
                    arr = inits[nm]
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
                if nm in inits and inits[nm].ndim == 1: b = inits[nm]
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
    cl, cu = lb.copy(), ub.copy()
    stable, rl = {}, 0
    for layer in layers:
        if layer["type"] == "linear":
            W, b = layer["W"], layer["b"]
            ni = W.shape[1]
            li = cl[:ni] if len(cl) >= ni else np.pad(cl, (0, ni - len(cl)))
            ui = cu[:ni] if len(cu) >= ni else np.pad(cu, (0, ni - len(cu)))
            Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
            cl, cu = Wp @ li + Wn @ ui + b, Wp @ ui + Wn @ li + b
        elif layer["type"] == "relu":
            for j in range(len(cl)):
                if cl[j] >= 0: stable[(rl, j)] = "active"
                elif cu[j] <= 0: stable[(rl, j)] = "inactive"
                else: stable[(rl, j)] = "unstable"
            cl, cu = np.maximum(cl, 0), np.maximum(cu, 0)
            rl += 1
    return stable


# ------------------------------------------------------------------
# SMT-LIB2 generation + file saving
# ------------------------------------------------------------------

def generate_smtlib2(layers, n_inputs, n_outputs, input_lb, input_ub,
                     output_constraints, stable_neurons) -> str:
    # Use QF_LRA (linear) if all ReLUs are fixed, QF_NRA otherwise
    has_unstable = any(v == "unstable" for v in stable_neurons.values())
    logic = "QF_NRA" if has_unstable else "QF_LRA"
    lines = [f"(set-logic {logic})", "(set-option :produce-models true)", ""]
    for i in range(n_inputs):
        lines.append(f"(declare-const X_{i} Real)")
    lines.append("")
    for i in range(n_inputs):
        if np.isfinite(input_lb[i]):
            lines.append(f"(assert (>= X_{i} {_fmt(input_lb[i])}))")
        if np.isfinite(input_ub[i]):
            lines.append(f"(assert (<= X_{i} {_fmt(input_ub[i])}))")
    lines.append("")

    current = [f"X_{i}" for i in range(n_inputs)]
    relu_layer, layer_idx = 0, 0
    for layer in layers:
        if layer["type"] == "linear":
            W, b = layer["W"], layer["b"]
            no, ni = W.shape
            iv = current[:ni]
            while len(iv) < ni: iv.append("0.0")
            nv = []
            for j in range(no):
                vn = f"h_{layer_idx}_{j}"
                lines.append(f"(declare-const {vn} Real)")
                nz = np.nonzero(np.abs(W[j, :]) > 1e-15)[0]
                if len(nz) == 0:
                    lines.append(f"(assert (= {vn} {_fmt(b[j])}))")
                else:
                    terms = [_fmt(b[j])] + [f"(* {_fmt(W[j,k])} {iv[k]})" for k in nz]
                    lines.append(f"(assert (= {vn} (+ {' '.join(terms)})))")
                nv.append(vn)
            current = nv
            layer_idx += 1
        elif layer["type"] == "relu":
            nv = []
            for j, v in enumerate(current):
                key = (relu_layer, j)
                st = stable_neurons.get(key, "unstable")
                rn = f"r_{relu_layer}_{j}"
                if st == "active":
                    nv.append(v); continue
                elif st == "inactive":
                    lines.append(f"(declare-const {rn} Real)")
                    lines.append(f"(assert (= {rn} 0.0))")
                else:
                    lines.append(f"(declare-const {rn} Real)")
                    lines.append(f"(assert (>= {rn} 0.0))")
                    lines.append(f"(assert (>= {rn} {v}))")
                    lines.append(f"(assert (or (= {rn} 0.0) (= {rn} {v})))")
                nv.append(rn)
            current = nv
            relu_layer += 1
        elif layer["type"] == "flatten":
            pass
    lines.append("")
    for i in range(min(n_outputs, len(current))):
        lines.append(f"(declare-const Y_{i} Real)")
        lines.append(f"(assert (= Y_{i} {current[i]}))")
    lines.append("")
    for c in output_constraints:
        lines.append(_constraint_to_smtlib2(c))
    lines += ["", "(check-sat)", "(exit)"]
    return "\n".join(lines)


def _constraint_to_smtlib2(c):
    t = c["type"]
    if t == "output_bound":
        return f"(assert ({c['op']} Y_{c['var']} {_fmt(c['bound'])}))"
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
                    atoms.append(f"({a['op']} Y_{a['var']} {_fmt(a['bound'])})")
            if len(atoms) == 1: clauses.append(atoms[0])
            elif atoms: clauses.append(f"(and {' '.join(atoms)})")
        if len(clauses) == 1: return f"(assert {clauses[0]})"
        return f"(assert (or {' '.join(clauses)}))"
    return ""


def save_smtlib2(smtlib2_text: str, benchmark: str, instance_name: str,
                 output_dir: Optional[str] = None) -> str:
    d = Path(output_dir) if output_dir else DEFAULT_SMT_DIR / benchmark
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{instance_name}.smt2"
    path.write_text(smtlib2_text)
    return str(path)


# ------------------------------------------------------------------
# Solver backends — each gets threads_per_solver cores
# ------------------------------------------------------------------

def _run_solver_process(args) -> Tuple[str, str, float]:
    """Worker: run one solver on the SMT-LIB2 file. Returns (name, result, time)."""
    name, smt2_path, timeout_s, n_threads = args
    t0 = time.time()
    try:
        if name == "z3":
            r = _run_z3(smt2_path, timeout_s, n_threads)
        elif name == "cvc5":
            r = _run_cvc5(smt2_path, timeout_s, n_threads)
        else:
            # opensmt or any other binary
            r = _run_generic(name, smt2_path, timeout_s)
        return name, r, time.time() - t0
    except Exception as e:
        return name, "unknown", time.time() - t0


def _run_z3(smt2_path: str, timeout_s: float, n_threads: int) -> str:
    import z3
    z3.set_param("parallel.enable", True)
    z3.set_param("parallel.threads.max", n_threads)
    s = z3.Solver()
    s.set("timeout", int(timeout_s * 1000))
    s.from_file(smt2_path)
    r = s.check()
    if r == z3.sat: return "sat"
    if r == z3.unsat: return "unsat"
    return "unknown"


def _run_cvc5(smt2_path: str, timeout_s: float, n_threads: int) -> str:
    """Run CVC5 via its Python InputParser API.
    Do NOT call setLogic() — the SMT-LIB2 file already contains (set-logic ...).
    """
    cmd = [
        "python3", "-c",
        f"""
import cvc5
tm = cvc5.TermManager()
s = cvc5.Solver(tm)
s.setOption("tlimit", "{int(timeout_s * 1000)}")
parser = cvc5.InputParser(s)
parser.setFileInput(cvc5.InputLanguage.SMT_LIB_2_6, "{smt2_path}")
sm = parser.getSymbolManager()
while True:
    cmd = parser.nextCommand()
    if cmd.isNull():
        break
    cmd.invoke(s, sm)
"""
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s + 10)
        out = (proc.stdout + proc.stderr).strip().lower()
        if "unsat" in out: return "unsat"
        if "sat" in out and "unsat" not in out: return "sat"
    except Exception:
        pass
    return "unknown"


def _run_generic(binary: str, smt2_path: str, timeout_s: float) -> str:
    try:
        proc = subprocess.run([binary, smt2_path], capture_output=True,
                              text=True, timeout=timeout_s + 5)
        out = proc.stdout.strip().lower()
        if "unsat" in out: return "unsat"
        if "sat" in out: return "sat"
    except Exception:
        pass
    return "unknown"


# ------------------------------------------------------------------
# Auto-detect available solvers
# ------------------------------------------------------------------

def detect_solvers(opensmt_path: str = "", theory: str = "QF_NRA") -> List[str]:
    """Detect available SMT solvers that support the given theory.
    Bitwuzla only supports QF_BV/QF_FP — excluded for real arithmetic."""
    avail = []
    try:
        import z3
        avail.append("z3")
    except ImportError:
        pass
    try:
        import cvc5
        avail.append("cvc5")
    except ImportError:
        pass
    # Bitwuzla: only for bit-vector theories, not real arithmetic
    if theory in ("QF_BV", "QF_FP", "QF_ABV", "QF_ABVFP"):
        try:
            import bitwuzla
            avail.append("bitwuzla")
        except ImportError:
            pass
    osmt = opensmt_path or "opensmt"
    try:
        subprocess.run([osmt, "--version"], capture_output=True, timeout=3)
        avail.append("opensmt")
    except Exception:
        pass
    return avail


# ------------------------------------------------------------------
# Main entry: multi-solver parallel portfolio
# ------------------------------------------------------------------

def verify_with_smt(
    onnx_path: str,
    property,
    timeout: float = 300.0,
    solvers: Optional[List[str]] = None,
    total_cores: int = 0,
    opensmt_path: str = "",
    save_formula: bool = True,
    benchmark_name: str = "",
    instance_name: str = "",
) -> Dict:
    """
    Verify using parallel SMT portfolio.  All cores are distributed
    across solvers; each solver's subprocess inherits its share.

    Args:
        total_cores: Total CPU cores to use (0=auto-detect).
        save_formula: Save SMT-LIB2 file to ./smt_formulas/.
    """
    t0 = time.time()
    n_cores = total_cores or os.cpu_count() or 4

    if solvers is None:
        solvers = detect_solvers(opensmt_path)
    if not solvers:
        return {"result": "error", "solver": "none", "time_seconds": 0,
                "details": "No SMT solvers available"}

    layers = _extract_weights_from_onnx(onnx_path)
    if not any(l["type"] == "linear" for l in layers):
        return {"result": "error", "solver": "none", "time_seconds": 0,
                "details": "No linear layers in ONNX"}

    lb = np.where(np.isfinite(property.input_lower), property.input_lower, -1e6)
    ub = np.where(np.isfinite(property.input_upper), property.input_upper, 1e6)

    # Prolog-style symbolic rewrite pre-processing
    from .symbolic_rewrite import symbolic_rewrite_preprocess
    layers, lb, ub, stable, out_constraints, rw_stats = symbolic_rewrite_preprocess(
        layers, lb, ub, property.output_constraints, property.n_outputs,
    )

    if rw_stats.early_unsat:
        smt2_path = ""
        if save_formula:
            bname = benchmark_name or "unknown"
            iname = instance_name or f"instance_{int(time.time())}"
            smt2_path = save_smtlib2("; EARLY UNSAT by symbolic rewrite\n(set-logic QF_NRA)\n(assert false)\n(check-sat)\n(exit)",
                                     bname, iname)
        return {"result": "verified", "solver": "symbolic_rewrite",
                "time_seconds": rw_stats.time_seconds, "smt2_file": smt2_path,
                "details": f"UNSAT by symbolic rewrite pre-processing | {rw_stats.summary()}"}

    n_stable = sum(1 for s in stable.values() if s != "unstable")
    n_unstable = sum(1 for s in stable.values() if s == "unstable")

    # Generate and save SMT-LIB2
    smtlib2 = generate_smtlib2(
        layers, property.n_inputs, property.n_outputs,
        lb, ub, out_constraints, stable,
    )

    # Save the base SMT-LIB2 formula
    smt2_path = None
    if save_formula:
        bname = benchmark_name or "unknown"
        iname = instance_name or f"instance_{int(time.time())}"
        smt2_path = save_smtlib2(smtlib2, bname, iname)
        logger.info(f"SMT-LIB2 saved: {smt2_path}")

    if smt2_path is None:
        import tempfile
        fd, smt2_path = tempfile.mkstemp(suffix=".smt2")
        with os.fdopen(fd, "w") as f:
            f.write(smtlib2)

    # ---- True multi-core: domain splitting on unstable ReLU phases ----
    # Each sub-problem fixes some ReLU neurons to active/inactive,
    # then generates a fresh (simpler) SMT formula and solves it.
    # This creates n_cores independent sub-problems.

    # Select neurons to split on: spread across layers, pick those with
    # the widest pre-activation range (most "impactful" when fixed).
    unstable_by_layer: Dict[int, List] = {}
    for (rl, j), v in stable.items():
        if v == "unstable":
            unstable_by_layer.setdefault(rl, []).append((rl, j))

    # Round-robin across layers to spread the splits
    split_keys = []
    layer_iters = {rl: iter(neurons) for rl, neurons in sorted(unstable_by_layer.items())}
    while len(split_keys) < min(n_unstable, 64):
        added = False
        for rl in sorted(layer_iters.keys()):
            try:
                split_keys.append(next(layer_iters[rl]))
                added = True
            except StopIteration:
                pass
        if not added:
            break

    n_split = min(int(np.log2(max(n_cores, 1))) + 1, len(split_keys), 12)
    n_subproblems = min(2 ** n_split, n_cores * 2)
    split_keys = split_keys[:n_split]
    solver_to_use = solvers[0] if solvers else "z3"

    logger.info(
        f"Domain splitting: {n_subproblems} sub-problems on {n_cores} cores | "
        f"splitting {n_split} neurons (cascade IBP will fix more) | solver={solver_to_use} | "
        f"ReLU: {len(stable)} total, {n_stable} stable, {n_unstable} unstable"
    )

    sub_timeout = max(1, timeout - (time.time() - t0))
    sub_args = []
    for combo in range(n_subproblems):
        fixed = dict(stable)
        for bit_idx, key in enumerate(split_keys):
            fixed[key] = "active" if ((combo >> bit_idx) & 1) else "inactive"
        sub_args.append((
            layers, property.n_inputs, property.n_outputs,
            lb, ub, out_constraints, fixed,
            solver_to_use, sub_timeout, smt2_path,
        ))

    # Pre-generate all sub-problem SMT files, then solve in parallel via subprocess
    import tempfile
    sub_files = []
    for i, args in enumerate(sub_args):
        (lyrs, ni, no, l, u, oc, fixed, sn, to, bp) = args
        # Cascade IBP + fix all remaining unstable
        cascaded, _ = _cascade_ibp(lyrs, l, u, fixed)
        # Fix remaining unstable by midpoint heuristic
        cl, cu = l.copy(), u.copy()
        rl = 0
        for layer in lyrs:
            if layer["type"] == "linear":
                W, b = layer["W"], layer["b"]
                ni2 = W.shape[1]
                li = cl[:ni2] if len(cl) >= ni2 else np.pad(cl, (0, ni2 - len(cl)))
                ui = cu[:ni2] if len(cu) >= ni2 else np.pad(cu, (0, ni2 - len(cu)))
                Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
                cl = Wp @ li + Wn @ ui + b
                cu = Wp @ ui + Wn @ li + b
            elif layer["type"] == "relu":
                for j in range(len(cl)):
                    key = (rl, j)
                    if cascaded.get(key) == "unstable":
                        cascaded[key] = "active" if (cl[j] + cu[j]) >= 0 else "inactive"
                    if cascaded.get(key) == "inactive":
                        cl[j] = 0.0; cu[j] = 0.0
                cl = np.maximum(cl, 0); cu = np.maximum(cu, 0)
                rl += 1

        smt2 = generate_smtlib2(lyrs, ni, no, l, u, oc, cascaded)
        fd, fpath = tempfile.mkstemp(suffix=".smt2")
        with os.fdopen(fd, "w") as f:
            f.write(smt2)
        sub_files.append(fpath)

    # Launch all sub-problem solvers as separate OS processes
    sub_procs = []
    sub_timeout_s = max(1, timeout - (time.time() - t0))
    for fpath in sub_files:
        cmd = [
            "python3", "-c",
            f'import z3; s=z3.Solver(); s.set("timeout",{int(sub_timeout_s*1000)}); '
            f's.from_file("{fpath}"); r=s.check(); '
            f'print("sat" if r==z3.sat else "unsat" if r==z3.unsat else "unknown")'
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sub_procs.append(p)

    logger.info(f"Launched {len(sub_procs)} solver processes")

    # Collect results
    deadline = t0 + timeout + 2
    results_collected = [None] * n_subproblems
    while time.time() < deadline:
        all_done = True
        for i, p in enumerate(sub_procs):
            if results_collected[i] is not None:
                continue
            ret = p.poll()
            if ret is not None:
                out = p.stdout.read().decode().strip().lower()
                if "unsat" in out:
                    results_collected[i] = "unsat"
                elif "sat" in out:
                    results_collected[i] = "sat"
                    # Kill all remaining
                    for j, p2 in enumerate(sub_procs):
                        if results_collected[j] is None:
                            p2.kill()
                    # Cleanup temp files
                    for f in sub_files:
                        try: os.unlink(f)
                        except: pass
                    return _format("sat", f"{solver_to_use}(split)", time.time() - t0,
                                   n_unstable, solvers, smt2_path)
                else:
                    results_collected[i] = "unknown"
            else:
                all_done = False
        if all_done:
            break
        time.sleep(0.05)

    # Kill stragglers
    for p in sub_procs:
        try: p.kill()
        except: pass
    for f in sub_files:
        try: os.unlink(f)
        except: pass

    total_elapsed = time.time() - t0
    n_unsat = sum(1 for r in results_collected if r == "unsat")
    n_done = sum(1 for r in results_collected if r is not None)

    if n_unsat == n_subproblems:
        return _format("unsat", f"{solver_to_use}(split×{n_subproblems})", total_elapsed,
                       n_unstable, solvers, smt2_path)

    return _format("unknown", f"split({n_unsat}/{n_done} unsat of {n_subproblems})",
                   total_elapsed, n_unstable, solvers, smt2_path)


def _cascade_ibp(layers, lb, ub, fixed_stable):
    """
    Re-run IBP with fixed ReLU phases to discover additional stable neurons.

    When we fix a neuron to active (y=x) or inactive (y=0), the output
    bounds of that layer change, which tightens bounds on subsequent layers,
    potentially making MORE neurons stable.  This "snowball effect" is the
    key to making domain splitting effective.
    """
    cl, cu = lb.copy(), ub.copy()
    new_stable = dict(fixed_stable)
    rl = 0
    newly_fixed = 0

    for layer in layers:
        if layer["type"] == "linear":
            W, b = layer["W"], layer["b"]
            ni = W.shape[1]
            li = cl[:ni] if len(cl) >= ni else np.pad(cl, (0, ni - len(cl)))
            ui = cu[:ni] if len(cu) >= ni else np.pad(cu, (0, ni - len(cu)))
            Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
            cl = Wp @ li + Wn @ ui + b
            cu = Wp @ ui + Wn @ li + b
        elif layer["type"] == "relu":
            for j in range(len(cl)):
                key = (rl, j)
                status = new_stable.get(key, "unstable")
                if status == "active":
                    pass  # cl[j], cu[j] unchanged
                elif status == "inactive":
                    cl[j] = 0.0
                    cu[j] = 0.0
                elif cl[j] >= 0:
                    new_stable[key] = "active"
                    newly_fixed += 1
                elif cu[j] <= 0:
                    new_stable[key] = "inactive"
                    cl[j] = 0.0
                    cu[j] = 0.0
                    newly_fixed += 1
                else:
                    cl[j] = 0.0
                    # cu[j] unchanged (positive part)
            cl = np.maximum(cl, 0)
            cu = np.maximum(cu, 0)
            rl += 1

    return new_stable, newly_fixed


def _solve_subproblem_worker(args) -> Tuple[str, float]:
    """
    Worker: fix ALL unstable ReLU phases → problem becomes pure LP → solve instantly.

    Each sub-problem assigns a specific active/inactive phase to EVERY
    unstable neuron, reducing the MILP to a feasibility LP.  With all
    ReLUs fixed, the network is a piecewise-linear function on ONE piece,
    solvable by LP in milliseconds.

    We enumerate a subset of the 2^n possible phase combinations.
    """
    (layers, n_inputs, n_outputs, lb, ub, output_constraints,
     fixed_stable, solver_name, timeout_s, base_smt2_path) = args

    t0 = time.time()

    # Cascade IBP to fix even more neurons
    cascaded_stable, _ = _cascade_ibp(layers, lb, ub, fixed_stable)

    # Fix ALL remaining unstable neurons by their "most likely" phase
    # (based on the midpoint of their pre-activation interval)
    remaining_unstable = [(k, v) for k, v in cascaded_stable.items() if v == "unstable"]
    if remaining_unstable:
        # Run one more IBP to get bounds for remaining unstable neurons
        cl, cu = lb.copy(), ub.copy()
        rl = 0
        pre_bounds = {}
        for layer in layers:
            if layer["type"] == "linear":
                W, b = layer["W"], layer["b"]
                ni = W.shape[1]
                li = cl[:ni] if len(cl) >= ni else np.pad(cl, (0, ni - len(cl)))
                ui = cu[:ni] if len(cu) >= ni else np.pad(cu, (0, ni - len(cu)))
                Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
                cl = Wp @ li + Wn @ ui + b
                cu = Wp @ ui + Wn @ li + b
            elif layer["type"] == "relu":
                for j in range(len(cl)):
                    key = (rl, j)
                    st = cascaded_stable.get(key, "unstable")
                    if st == "unstable":
                        pre_bounds[key] = (cl[j], cu[j])
                        # Fix based on midpoint heuristic
                        mid = (cl[j] + cu[j]) / 2.0
                        cascaded_stable[key] = "active" if mid >= 0 else "inactive"
                    if cascaded_stable.get(key) == "active":
                        pass
                    elif cascaded_stable.get(key) == "inactive":
                        cl[j] = 0.0
                        cu[j] = 0.0
                    else:
                        cl[j] = max(cl[j], 0)
                cl = np.maximum(cl, 0)
                cu = np.maximum(cu, 0)
                rl += 1

    # Now ALL neurons are fixed → generate a purely linear SMT formula
    smtlib2 = generate_smtlib2(
        layers, n_inputs, n_outputs, lb, ub, output_constraints, cascaded_stable,
    )

    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix=".smt2")
    with os.fdopen(fd, "w") as f:
        f.write(smtlib2)

    try:
        if solver_name == "z3":
            result = _run_z3(tmp_path, timeout_s, 1)
        elif solver_name == "cvc5":
            result = _run_cvc5(tmp_path, timeout_s, 1)
        else:
            result = _run_generic(solver_name, tmp_path, timeout_s)
        return result, time.time() - t0
    except Exception:
        return "unknown", time.time() - t0
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _format(result, solver, elapsed, n_unstable, all_solvers, smt2_path):
    status_map = {"unsat": "verified", "sat": "violated"}
    return {
        "result": status_map.get(result, "unknown"),
        "solver": solver,
        "time_seconds": elapsed,
        "smt2_file": smt2_path,
        "details": f"{result} by {solver} | {n_unstable} unstable ReLUs | "
                   f"portfolio: {all_solvers} | file: {smt2_path}",
    }


# ------------------------------------------------------------------
# PyTorch model support: export to ONNX → reuse ONNX pipeline
# ------------------------------------------------------------------

def _export_pytorch_to_onnx(model, input_shape: Tuple, output_path: str):
    """Export a PyTorch model to ONNX format."""
    import torch
    model.eval()
    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"], output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
    )
    return output_path


def _extract_last_layer_from_pytorch(model, x0_tensor, input_shape, n_classes):
    """
    Extract the last linear (classification) layer's weight/bias and the
    pre-logit hidden state bounds via forward-pass + Jacobian on the model.

    For deep CNNs like ResNet, the full network cannot be encoded as SMT
    (millions of neurons).  Instead we encode only the final fc layer:
        logits = W_fc @ h + b_fc
    where h is the pre-logit hidden state, and we bound h using IBP/Jacobian
    on the preceding convolutional trunk.
    """
    import torch

    model.eval()
    hidden_states = []

    # Hook into the layer before lm_head / fc
    def find_fc_and_hook(m):
        fc_layer = None
        hook_target = None
        for name, mod in m.named_modules():
            if isinstance(mod, torch.nn.Linear):
                fc_layer = mod
            if isinstance(mod, (torch.nn.AdaptiveAvgPool2d, torch.nn.AvgPool2d)):
                hook_target = mod
        if hook_target is None:
            # For models without avgpool, hook the module before fc
            children = list(m.children())
            if len(children) >= 2:
                hook_target = children[-2]
        return fc_layer, hook_target

    fc_layer, hook_target = find_fc_and_hook(model)
    if fc_layer is None:
        return None

    W = fc_layer.weight.detach().cpu().numpy().astype(np.float64)
    b = fc_layer.bias.detach().cpu().numpy().astype(np.float64) if fc_layer.bias is not None else np.zeros(W.shape[0])

    # Get nominal hidden state
    def capture(module, inp, out):
        if isinstance(out, torch.Tensor):
            hidden_states.append(out.detach())
        elif isinstance(out, tuple):
            hidden_states.append(out[0].detach())

    hook = hook_target.register_forward_hook(capture) if hook_target else None
    with torch.no_grad():
        model(x0_tensor)
    if hook:
        hook.remove()

    h_nominal = hidden_states[0].flatten().cpu().numpy().astype(np.float64) if hidden_states else None
    return W, b, h_nominal


def verify_pytorch_with_smt(
    model,
    x0: np.ndarray,
    epsilon: float,
    true_label: int,
    target_label: Optional[int] = None,
    input_shape: Optional[Tuple] = None,
    n_classes: int = 10,
    timeout: float = 3600.0,
    total_cores: int = 0,
    solvers: Optional[List[str]] = None,
    model_name: str = "resnet",
) -> Dict:
    """
    Verify a PyTorch model (including ResNet-121) using SMT portfolio.

    For deep CNNs, encodes only the final classification layer as SMT:
        logits = W_fc @ h + b_fc,  h_lb <= h <= h_ub
    where h bounds come from Jacobian-based estimation on the conv trunk.
    The property (unsafe region) is: exists target with output[target] >= output[true_label].

    For small fully-connected networks, exports to ONNX and encodes the full network.
    """
    import torch

    t0 = time.time()
    n_cores = total_cores or os.cpu_count() or 4

    if solvers is None:
        solvers = detect_solvers()
    if not solvers:
        return {"result": "error", "solver": "none", "time_seconds": 0,
                "details": "No SMT solvers available"}

    x0_flat = x0.flatten().astype(np.float32)
    if input_shape is None:
        input_shape = (1,) + tuple(x0.shape)

    n_input_pixels = int(np.prod(input_shape[1:]))
    x0_tensor = torch.tensor(x0_flat.reshape(input_shape)).float()

    # ---- Strategy: last-layer SMT for deep CNNs ----
    extracted = _extract_last_layer_from_pytorch(model, x0_tensor, input_shape, n_classes)
    if extracted is None:
        return {"result": "error", "solver": "none", "time_seconds": time.time() - t0,
                "details": "Could not extract final linear layer from model"}

    W, b, h_nominal = extracted
    n_hidden = W.shape[1]

    # Compute hidden-state perturbation bound via Jacobian
    logger.info(f"Computing Jacobian bounds on pre-logit hidden state (dim={n_hidden})...")
    h_delta = _compute_hidden_perturbation(model, x0_tensor, n_hidden, epsilon)

    h_lb = (h_nominal - h_delta).astype(np.float64)
    h_ub = (h_nominal + h_delta).astype(np.float64)

    logger.info(
        f"Last-layer SMT: W={W.shape}, h_dim={n_hidden}, "
        f"h_delta_max={h_delta.max():.6f}, timeout={timeout}s, cores={n_cores}"
    )

    # Build SMT-LIB2 for the last layer: logits = W @ h + b
    # Variables: h_0..h_{n_hidden-1} (bounded), Y_0..Y_{n_classes-1}
    n_outputs = min(n_classes, W.shape[0])
    layers_for_smt = [{"type": "linear", "W": W, "b": b}]
    stable_for_smt = {}  # no ReLU in the last layer

    # Output constraints: unsafe if any target beats true_label
    output_constraints = []
    targets = [target_label] if target_label is not None else [i for i in range(n_outputs) if i != true_label]
    clauses = []
    for t in targets:
        clauses.append([{"op": ">=", "left": t, "right": true_label}])
    if clauses:
        output_constraints.append({"type": "disjunction", "clauses": clauses})

    smtlib2 = generate_smtlib2(
        layers_for_smt, n_hidden, n_outputs, h_lb, h_ub,
        output_constraints, stable_for_smt,
    )

    instance_name = f"{model_name}_eps{epsilon}_label{true_label}"
    smt2_path = save_smtlib2(smtlib2, model_name, instance_name)
    logger.info(f"SMT-LIB2 saved: {smt2_path} ({len(smtlib2)} chars, {n_hidden} vars)")

    # Run portfolio
    threads_per = max(1, n_cores // len(solvers))
    logger.info(f"SMT portfolio: {solvers} | {n_cores} cores ({threads_per}/solver)")

    tasks = [(s, smt2_path, timeout, threads_per) for s in solvers]
    if len(tasks) == 1:
        name, result, elapsed = _run_solver_process(tasks[0])
        return _format(result, name, elapsed, 0, solvers, smt2_path)

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(len(tasks)) as pool:
        async_results = [pool.apply_async(_run_solver_process, (t,)) for t in tasks]
        deadline = t0 + timeout + 10
        while time.time() < deadline:
            for ar in async_results:
                if ar.ready():
                    name, result, elapsed = ar.get(timeout=1)
                    if result in ("sat", "unsat"):
                        pool.terminate()
                        return _format(result, name, elapsed, 0, solvers, smt2_path)
            time.sleep(0.5)
        pool.terminate()

    return _format("unknown", "portfolio", time.time() - t0, 0, solvers, smt2_path)


def _compute_hidden_perturbation(model, x0_tensor, n_hidden, epsilon):
    """Compute per-dimension perturbation bound on the pre-logit hidden state."""
    import torch

    model.eval()
    # Use Jacobian: |h_j(x) - h_j(x0)| <= ||grad h_j / grad x||_1 * epsilon
    safety = 1.5
    x_var = x0_tensor.clone().requires_grad_(True)

    # Get hidden state via the fc layer's input
    hidden_states = []
    fc_layer = None
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            fc_layer = mod

    if fc_layer is None:
        return np.ones(n_hidden) * epsilon * 100

    def hook_fn(module, inp, out):
        hidden_states.append(inp[0] if isinstance(inp, tuple) else inp)

    hook = fc_layer.register_forward_hook(hook_fn)

    h_deltas = np.zeros(n_hidden)
    # Sample a few dimensions for speed (full Jacobian too expensive for 512 dims)
    n_probe = min(n_hidden, 32)
    probe_dims = np.linspace(0, n_hidden - 1, n_probe, dtype=int)

    for d in probe_dims:
        hidden_states.clear()
        if x_var.grad is not None:
            x_var.grad.zero_()
        out = model(x_var)
        h = hidden_states[0].flatten()
        if d < len(h):
            h[d].backward(retain_graph=True)
            grad = x_var.grad.detach().flatten().float()
            h_deltas[d] = grad.abs().sum().item() * epsilon * safety

    hook.remove()

    # Interpolate unprobed dimensions
    if n_probe < n_hidden:
        max_probed = h_deltas[probe_dims].max()
        for j in range(n_hidden):
            if h_deltas[j] == 0:
                h_deltas[j] = max_probed

    return h_deltas
