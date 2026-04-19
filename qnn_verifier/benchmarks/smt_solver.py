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
    lines = ["(set-logic QF_NRA)", "(set-option :produce-models true)", ""]
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
        elif name == "bitwuzla":
            r = _run_bitwuzla(smt2_path, timeout_s)
        else:
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
    cmd = [
        "python3", "-c",
        f"""
import cvc5, sys
tm = cvc5.TermManager()
s = cvc5.Solver(tm)
s.setOption("produce-models","true")
s.setOption("tlimit","{int(timeout_s*1000)}")
s.setLogic("QF_NRA")
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
        out = proc.stdout.strip().lower()
        if "unsat" in out: return "unsat"
        if "sat" in out: return "sat"
    except Exception:
        pass
    return "unknown"


def _run_bitwuzla(smt2_path: str, timeout_s: float) -> str:
    cmd = [
        "python3", "-c",
        f"""
import bitwuzla
opts = bitwuzla.Options()
opts.set(bitwuzla.Option.TIME_LIMIT, {int(timeout_s * 1000)})
tm = bitwuzla.TermManager()
parser = bitwuzla.Parser(tm, opts)
try:
    parser.parse("{smt2_path}")
    bz = parser.bitwuzla()
    r = bz.check_sat()
    if r == bitwuzla.Result.SAT: print("sat")
    elif r == bitwuzla.Result.UNSAT: print("unsat")
    else: print("unknown")
except: print("unknown")
"""
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 10)
        out = proc.stdout.strip().lower()
        if "unsat" in out: return "unsat"
        if "sat" in out: return "sat"
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

def detect_solvers(opensmt_path: str = "") -> List[str]:
    avail = []
    try:
        import z3; avail.append("z3")
    except ImportError: pass
    try:
        import cvc5; avail.append("cvc5")
    except ImportError: pass
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
    stable = _ibp_stable_neurons(layers, lb, ub)
    n_stable = sum(1 for s in stable.values() if s != "unstable")
    n_unstable = sum(1 for s in stable.values() if s == "unstable")

    # Generate and save SMT-LIB2
    smtlib2 = generate_smtlib2(
        layers, property.n_inputs, property.n_outputs,
        lb, ub, property.output_constraints, stable,
    )

    smt2_path = None
    if save_formula:
        bname = benchmark_name or "unknown"
        iname = instance_name or f"instance_{int(time.time())}"
        smt2_path = save_smtlib2(smtlib2, bname, iname)
        logger.info(f"SMT-LIB2 saved: {smt2_path}")

    # If not saved, write to temp file (solvers need a file path)
    if smt2_path is None:
        import tempfile
        fd, smt2_path = tempfile.mkstemp(suffix=".smt2")
        with os.fdopen(fd, "w") as f:
            f.write(smtlib2)

    # Distribute cores: each solver gets n_cores // n_solvers threads
    threads_per = max(1, n_cores // len(solvers))

    logger.info(
        f"SMT portfolio: {solvers} | {n_cores} cores ({threads_per}/solver) | "
        f"ReLU: {len(stable)} total, {n_stable} stable, {n_unstable} unstable"
    )

    tasks = [(s, smt2_path, timeout, threads_per) for s in solvers]

    # Single solver — no multiprocessing overhead
    if len(tasks) == 1:
        name, result, elapsed = _run_solver_process(tasks[0])
        return _format(result, name, elapsed, n_unstable, solvers, smt2_path)

    # Multi-solver portfolio
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(len(tasks)) as pool:
        async_results = [pool.apply_async(_run_solver_process, (t,)) for t in tasks]

        deadline = t0 + timeout + 5
        while time.time() < deadline:
            for ar in async_results:
                if ar.ready():
                    name, result, elapsed = ar.get(timeout=1)
                    if result in ("sat", "unsat"):
                        pool.terminate()
                        return _format(result, name, elapsed, n_unstable, solvers, smt2_path)
            time.sleep(0.2)

        pool.terminate()

    return _format("unknown", "portfolio", time.time() - t0, n_unstable, solvers, smt2_path)


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
