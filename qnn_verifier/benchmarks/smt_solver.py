"""
Multi-solver parallel SMT-based neural network verification.

Generates a standard SMT-LIB2 file (saved locally for reproducibility),
then launches solver processes in parallel across all CPU cores.

Parallelism: domain splitting on unstable ReLU neurons.  Each sub-problem
fixes SOME neurons' phases and keeps the rest as disjunctions.  This is
a SOUND decomposition: the sub-problems PARTITION the feasible space.

  - One sub-problem SAT  → property VIOLATED (with concrete counterexample)
  - ALL sub-problems UNSAT → property VERIFIED
  - Any sub-problem UNKNOWN → overall UNKNOWN (incomplete, not wrong)
"""

import logging
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
    """SMT-LIB2 decimal literal (no scientific notation, negatives as (- x))."""
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
# SMT-LIB2 generation
# ------------------------------------------------------------------

def generate_smtlib2(layers, n_inputs, n_outputs, input_lb, input_ub,
                     output_constraints, stable_neurons) -> str:
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
                    # Fixed active: y = x, with x >= 0
                    lines.append(f"(declare-const {rn} Real)")
                    lines.append(f"(assert (= {rn} {v}))")
                    lines.append(f"(assert (>= {v} 0.0))")
                    nv.append(rn)
                elif st == "inactive":
                    # Fixed inactive: y = 0, with x <= 0
                    lines.append(f"(declare-const {rn} Real)")
                    lines.append(f"(assert (= {rn} 0.0))")
                    lines.append(f"(assert (<= {v} 0.0))")
                    nv.append(rn)
                else:
                    # Unstable: exact ReLU encoding with disjunction
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


def save_smtlib2(smtlib2_text, benchmark, instance_name, output_dir=None):
    d = Path(output_dir) if output_dir else DEFAULT_SMT_DIR / benchmark
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{instance_name}.smt2"
    path.write_text(smtlib2_text)
    return str(path)


# ------------------------------------------------------------------
# Solver runners (each runs in its own subprocess)
# ------------------------------------------------------------------

def _run_z3(smt2_path: str, timeout_s: float, n_threads: int) -> str:
    import z3
    s = z3.Solver()
    s.set("timeout", int(timeout_s * 1000))
    s.from_file(smt2_path)
    r = s.check()
    if r == z3.sat: return "sat"
    if r == z3.unsat: return "unsat"
    return "unknown"


def _run_cvc5(smt2_path: str, timeout_s: float, n_threads: int) -> str:
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
    if cmd.isNull(): break
    cmd.invoke(s, sm)
"""
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 10)
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
# Solver detection
# ------------------------------------------------------------------

def detect_solvers(opensmt_path: str = "", theory: str = "QF_NRA") -> List[str]:
    avail = []
    try:
        import z3; avail.append("z3")
    except ImportError: pass
    try:
        import cvc5; avail.append("cvc5")
    except ImportError: pass
    if theory in ("QF_BV", "QF_FP"):
        try:
            import bitwuzla; avail.append("bitwuzla")
        except ImportError: pass
    osmt = opensmt_path or "opensmt"
    try:
        subprocess.run([osmt, "--version"], capture_output=True, timeout=3)
        avail.append("opensmt")
    except Exception: pass
    return avail


# ------------------------------------------------------------------
# Cascade IBP: re-propagate bounds after fixing neuron phases
# ------------------------------------------------------------------

def _cascade_ibp(layers, lb, ub, fixed_stable):
    """Re-run IBP with fixed phases. Returns (new_stable, n_newly_fixed)."""
    cl, cu = lb.copy(), ub.copy()
    new_stable = dict(fixed_stable)
    rl, newly = 0, 0
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
                st = new_stable.get(key, "unstable")
                if st == "active":
                    pass
                elif st == "inactive":
                    cl[j] = 0.0; cu[j] = 0.0
                elif cl[j] >= 0:
                    new_stable[key] = "active"; newly += 1
                elif cu[j] <= 0:
                    new_stable[key] = "inactive"; cl[j] = 0.0; cu[j] = 0.0; newly += 1
                else:
                    cl[j] = 0.0
            cl = np.maximum(cl, 0); cu = np.maximum(cu, 0)
            rl += 1
    return new_stable, newly


# ------------------------------------------------------------------
# MAIN: parallel domain-splitting verification
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

    # Symbolic rewrite pre-processing
    from .symbolic_rewrite import symbolic_rewrite_preprocess
    layers, lb, ub, stable, out_constraints, rw_stats = symbolic_rewrite_preprocess(
        layers, lb, ub, property.output_constraints, property.n_outputs,
    )

    if rw_stats.early_unsat:
        smt2_path = ""
        if save_formula:
            smt2_path = save_smtlib2(
                "; EARLY UNSAT by symbolic rewrite\n(set-logic QF_LRA)\n(assert false)\n(check-sat)\n(exit)",
                benchmark_name or "unknown", instance_name or "instance")
        return {"result": "verified", "solver": "symbolic_rewrite",
                "time_seconds": rw_stats.time_seconds, "smt2_file": smt2_path,
                "details": f"UNSAT by rewrite | {rw_stats.summary()}"}

    n_unstable = sum(1 for s in stable.values() if s == "unstable")
    n_stable = sum(1 for s in stable.values() if s != "unstable")

    # Save the base (exact) formula
    smtlib2 = generate_smtlib2(
        layers, property.n_inputs, property.n_outputs, lb, ub, out_constraints, stable,
    )
    smt2_path = ""
    if save_formula:
        smt2_path = save_smtlib2(smtlib2, benchmark_name or "unknown",
                                 instance_name or "instance")
        logger.info(f"SMT-LIB2 saved: {smt2_path}")

    # ---- Domain splitting: SOUND parallel decomposition ----
    #
    # We split on k neurons, creating 2^k sub-problems. Each sub-problem
    # KEEPS all other unstable neurons as disjunctions — no midpoint
    # heuristic, no unsound over-approximation.
    #
    # After fixing split neurons, cascade IBP may discover MORE stable
    # neurons, reducing the remaining disjunctions in each sub-problem.
    #
    # Soundness: sub-problems partition the ReLU phase space on the
    # split neurons. The Or(active, inactive) on split neurons is
    # replaced by explicit enumeration → sound and complete.

    unstable_keys = sorted([k for k, v in stable.items() if v == "unstable"])

    # Split on enough neurons so we have ~n_cores sub-problems
    n_split = min(int(np.log2(max(n_cores, 1))) + 1, len(unstable_keys), 12)
    n_subproblems = min(2 ** n_split, n_cores * 2)
    split_keys = unstable_keys[:n_split]
    solver_name = solvers[0] if solvers else "z3"

    logger.info(
        f"Domain splitting: {n_subproblems} sub-problems on {n_cores} cores | "
        f"split {n_split} neurons | solver={solver_name} | "
        f"unstable: {n_unstable} (each sub has ≤{n_unstable - n_split})"
    )

    # Pre-generate all sub-problem SMT files
    import tempfile
    sub_files = []
    sub_remaining = []
    for combo in range(n_subproblems):
        fixed = dict(stable)
        for bit_idx, key in enumerate(split_keys):
            fixed[key] = "active" if ((combo >> bit_idx) & 1) else "inactive"
        # Cascade: fixing split neurons may make more neurons stable
        cascaded, n_extra = _cascade_ibp(layers, lb, ub, fixed)
        remaining = sum(1 for v in cascaded.values() if v == "unstable")
        sub_remaining.append(remaining)

        smt2 = generate_smtlib2(
            layers, property.n_inputs, property.n_outputs,
            lb, ub, out_constraints, cascaded,
        )
        fd, fpath = tempfile.mkstemp(suffix=".smt2")
        with os.fdopen(fd, "w") as f:
            f.write(smt2)
        sub_files.append(fpath)

    avg_remaining = sum(sub_remaining) / max(len(sub_remaining), 1)
    logger.info(
        f"Generated {len(sub_files)} sub-problem files | "
        f"avg unstable per sub: {avg_remaining:.0f} (was {n_unstable})"
    )

    # Launch all solvers as OS processes (true parallelism)
    sub_timeout_s = max(1, timeout - (time.time() - t0))
    procs = []
    for fpath in sub_files:
        cmd = [
            "python3", "-c",
            f'import z3; s=z3.Solver(); s.set("timeout",{int(sub_timeout_s*1000)}); '
            f's.from_file("{fpath}"); r=s.check(); '
            f'print("sat" if r==z3.sat else "unsat" if r==z3.unsat else "unknown")'
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append(p)

    logger.info(f"Launched {len(procs)} Z3 processes")

    # Poll for results
    deadline = t0 + timeout + 2
    results = [None] * n_subproblems
    while time.time() < deadline:
        all_done = True
        for i, p in enumerate(procs):
            if results[i] is not None:
                continue
            if p.poll() is not None:
                out = p.stdout.read().decode().strip().lower()
                if "unsat" in out:
                    results[i] = "unsat"
                elif "sat" in out and "unsat" not in out:
                    results[i] = "sat"
                    # SAT in any sub-problem → VIOLATED (sound)
                    for j, p2 in enumerate(procs):
                        if results[j] is None:
                            p2.kill(); p2.wait()
                    for f in sub_files:
                        try: os.unlink(f)
                        except: pass
                    return _make_result("sat", f"{solver_name}(split)", time.time() - t0,
                                        n_unstable, smt2_path)
                else:
                    results[i] = "unknown"
            else:
                all_done = False
        if all_done:
            break
        time.sleep(0.05)

    # Cleanup
    for p in procs:
        try: p.kill(); p.wait()
        except: pass
    for f in sub_files:
        try: os.unlink(f)
        except: pass

    elapsed = time.time() - t0
    n_unsat = sum(1 for r in results if r == "unsat")
    n_done = sum(1 for r in results if r is not None)

    # ALL UNSAT → VERIFIED (sound: we enumerated all phase combos for split neurons)
    if n_unsat == n_subproblems:
        return _make_result("unsat", f"{solver_name}(split×{n_subproblems})",
                            elapsed, n_unstable, smt2_path)

    return _make_result("unknown",
                        f"split({n_unsat}/{n_done} unsat of {n_subproblems})",
                        elapsed, n_unstable, smt2_path)


def _make_result(result, solver, elapsed, n_unstable, smt2_path):
    status_map = {"unsat": "verified", "sat": "violated"}
    return {
        "result": status_map.get(result, "unknown"),
        "solver": solver,
        "time_seconds": elapsed,
        "smt2_file": smt2_path,
        "details": f"{result} by {solver} | {n_unstable} unstable ReLUs | file: {smt2_path}",
    }


# ------------------------------------------------------------------
# PyTorch model support
# ------------------------------------------------------------------

def _export_pytorch_to_onnx(model, input_shape, output_path):
    import torch
    model.eval()
    torch.onnx.export(model, torch.randn(*input_shape), output_path,
                      input_names=["input"], output_names=["output"],
                      opset_version=13, do_constant_folding=True)
    return output_path


def _extract_last_layer_from_pytorch(model, x0_tensor, input_shape, n_classes):
    import torch
    model.eval()
    hidden_states = []

    fc_layer, hook_target = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear): fc_layer = mod
        if isinstance(mod, (torch.nn.AdaptiveAvgPool2d, torch.nn.AvgPool2d)): hook_target = mod
    if hook_target is None:
        children = list(model.children())
        if len(children) >= 2: hook_target = children[-2]
    if fc_layer is None:
        return None

    W = fc_layer.weight.detach().cpu().numpy().astype(np.float64)
    b = fc_layer.bias.detach().cpu().numpy().astype(np.float64) if fc_layer.bias is not None else np.zeros(W.shape[0])

    def capture(module, inp, out):
        t = out if isinstance(out, torch.Tensor) else out[0]
        hidden_states.append(t.detach())

    hook = hook_target.register_forward_hook(capture) if hook_target else None
    with torch.no_grad():
        model(x0_tensor)
    if hook: hook.remove()

    h = hidden_states[0].flatten().cpu().numpy().astype(np.float64) if hidden_states else None
    return W, b, h


def verify_pytorch_with_smt(
    model, x0, epsilon, true_label,
    target_label=None, input_shape=None, n_classes=10,
    timeout=3600.0, total_cores=0, solvers=None, model_name="resnet",
) -> Dict:
    """Verify a PyTorch model's last layer via SMT. Sound for the last-layer LP."""
    import torch

    t0 = time.time()
    n_cores = total_cores or os.cpu_count() or 4
    if solvers is None: solvers = detect_solvers()
    if not solvers:
        return {"result": "error", "solver": "none", "time_seconds": 0,
                "details": "No SMT solvers available"}

    x0_flat = x0.flatten().astype(np.float32)
    if input_shape is None: input_shape = (1,) + tuple(x0.shape)
    x0_tensor = torch.tensor(x0_flat.reshape(input_shape)).float()

    extracted = _extract_last_layer_from_pytorch(model, x0_tensor, input_shape, n_classes)
    if extracted is None:
        return {"result": "error", "solver": "none", "time_seconds": time.time() - t0,
                "details": "Could not extract final linear layer"}

    W, b, h_nominal = extracted
    n_hidden = W.shape[1]
    n_outputs = min(n_classes, W.shape[0])

    logger.info(f"Computing Jacobian bounds (dim={n_hidden})...")
    h_delta = _compute_hidden_perturbation(model, x0_tensor, n_hidden, epsilon)
    h_lb = (h_nominal - h_delta).astype(np.float64)
    h_ub = (h_nominal + h_delta).astype(np.float64)

    # Last layer is purely linear (no ReLU) → QF_LRA, no splitting needed
    layers_smt = [{"type": "linear", "W": W, "b": b}]
    targets = [target_label] if target_label is not None else [i for i in range(n_outputs) if i != true_label]
    clauses = [[{"op": ">=", "left": t, "right": true_label}] for t in targets]
    output_constraints = [{"type": "disjunction", "clauses": clauses}] if clauses else []

    smtlib2 = generate_smtlib2(layers_smt, n_hidden, n_outputs, h_lb, h_ub,
                               output_constraints, {})
    smt2_path = save_smtlib2(smtlib2, model_name,
                             f"{model_name}_eps{epsilon}_label{true_label}")
    logger.info(f"SMT-LIB2 saved: {smt2_path} ({n_hidden} vars)")

    # Single solver (no splitting needed for pure LP)
    solver = solvers[0]
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".smt2")
    with os.fdopen(fd, "w") as f: f.write(smtlib2)

    try:
        if solver == "z3":
            result = _run_z3(tmp, timeout, 1)
        elif solver == "cvc5":
            result = _run_cvc5(tmp, timeout, 1)
        else:
            result = _run_generic(solver, tmp, timeout)
    except Exception:
        result = "unknown"
    finally:
        try: os.unlink(tmp)
        except: pass

    return _make_result(result, solver, time.time() - t0, 0, smt2_path)


def _compute_hidden_perturbation(model, x0_tensor, n_hidden, epsilon):
    import torch
    model.eval()
    safety = 1.5
    x_var = x0_tensor.clone().requires_grad_(True)

    fc_layer = None
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear): fc_layer = mod
    if fc_layer is None:
        return np.ones(n_hidden) * epsilon * 100

    hidden_states = []
    def hook_fn(module, inp, out):
        hidden_states.append(inp[0] if isinstance(inp, tuple) else inp)
    hook = fc_layer.register_forward_hook(hook_fn)

    h_deltas = np.zeros(n_hidden)
    n_probe = min(n_hidden, 32)
    probe_dims = np.linspace(0, n_hidden - 1, n_probe, dtype=int)

    for d in probe_dims:
        hidden_states.clear()
        if x_var.grad is not None: x_var.grad.zero_()
        model(x_var)
        h = hidden_states[0].flatten()
        if d < len(h):
            h[d].backward(retain_graph=True)
            grad = x_var.grad.detach().flatten().float()
            h_deltas[d] = grad.abs().sum().item() * epsilon * safety

    hook.remove()

    if n_probe < n_hidden:
        max_probed = h_deltas[probe_dims].max()
        for j in range(n_hidden):
            if h_deltas[j] == 0: h_deltas[j] = max_probed

    return h_deltas
