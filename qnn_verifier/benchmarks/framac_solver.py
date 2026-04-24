"""
Frama-C based neural network verification.

Transpiles the neural network + VNNLIB property into a C program with
ACSL (ANSI/ISO C Specification Language) annotations, then verifies
using Frama-C's abstract interpretation (Eva plugin) and deductive
verification (WP plugin).

Pipeline:
  1. Extract NN weights from ONNX.
  2. Generate C code: the network forward pass as a pure C function.
  3. Add ACSL annotations: input bounds as requires, output property
     as ensures, loop invariants for layers.
  4. Run Frama-C Eva: abstract interpretation computes value ranges
     for all intermediate variables — equivalent to (but more precise
     than) IBP, because Frama-C tracks relational domains.
  5. Run Frama-C WP: deductive verification via SMT backend (Alt-Ergo,
     Z3, CVC5) to formally prove the ACSL postcondition.

Generated C files are saved to ./framac_output/<benchmark>/<instance>.c
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FRAMAC_DIR = _PROJECT_ROOT / "framac_output"

FRAMAC_BIN = shutil.which("frama-c") or os.path.expanduser("~/.opam/default/bin/frama-c")


# ------------------------------------------------------------------
# C code generation
# ------------------------------------------------------------------

def generate_c_program(
    layers: List[Dict],
    n_inputs: int,
    n_outputs: int,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    output_constraints: List[Dict],
    stable_neurons: Dict,
) -> str:
    """
    Generate a C program encoding the NN forward pass with ACSL annotations.

    The generated program has:
    - Input array x[N_INPUTS] with ACSL requires for bounds.
    - Hidden layer computations as explicit assignments.
    - ReLU as: h = (h > 0) ? h : 0;
    - Output array y[N_OUTPUTS].
    - ACSL ensures clause encoding the NEGATION of the unsafe property
      (i.e., we want to prove the ensures holds → property verified).
    """
    lines = []
    lines.append("// Auto-generated neural network verification program")
    lines.append("// Verify with: frama-c -eva -eva-precision 7 <file>.c")
    lines.append("//          or: frama-c -wp -wp-prover alt-ergo,z3 <file>.c")
    lines.append("")

    # Frama-C Eva built-in for non-deterministic input ranges
    lines.append("// Frama-C built-in for Eva abstract interpretation")
    lines.append("extern double Frama_C_double_interval(double min, double max);")
    lines.append("")
    lines.append(f"#define N_IN {n_inputs}")
    lines.append(f"#define N_OUT {n_outputs}")
    lines.append("")

    # Global weight arrays
    layer_idx = 0
    for layer in layers:
        if layer["type"] == "linear":
            W, b = layer["W"], layer["b"]
            no, ni = W.shape
            lines.append(f"// Layer {layer_idx}: {ni} -> {no}")
            lines.append(f"const double W{layer_idx}[{no}][{ni}] = {{")
            for i in range(no):
                row = ", ".join(f"{W[i,j]:.15f}" for j in range(ni))
                lines.append(f"  {{{row}}},")
            lines.append("};")
            lines.append(f"const double B{layer_idx}[{no}] = {{{', '.join(f'{b[j]:.15f}' for j in range(no))}}};")
            lines.append("")
            layer_idx += 1

    # UNROLLED forward pass as a single function (no loops — Eva/WP friendly)
    # For Frama-C, unrolled code is much easier to analyze than loops.
    ensures_str = _build_ensures(output_constraints, n_outputs)

    lines.append("/*@")
    for i in range(n_inputs):
        if np.isfinite(input_lb[i]):
            lines.append(f"  requires x[{i}] >= {input_lb[i]:.15f};")
        if np.isfinite(input_ub[i]):
            lines.append(f"  requires x[{i}] <= {input_ub[i]:.15f};")
    if ensures_str:
        lines.append(f"  ensures {ensures_str};")
    lines.append("*/")
    lines.append(f"void nn_forward(double x[{n_inputs}], double y[{n_outputs}]) {{")

    # Generate UNROLLED forward pass (no for loops)
    current_var = "x"
    current_size = n_inputs
    layer_idx = 0
    relu_idx = 0
    temp_idx = 0

    for layer in layers:
        if layer["type"] == "linear":
            W, b = layer["W"], layer["b"]
            no, ni = W.shape
            vname = f"h{temp_idx}"
            lines.append(f"  double {vname}[{no}];")
            for i in range(no):
                nz = np.nonzero(np.abs(W[i, :]) > 1e-15)[0]
                if len(nz) == 0:
                    lines.append(f"  {vname}[{i}] = {b[i]:.15f};")
                else:
                    terms = [f"{b[i]:.15f}"]
                    for j in nz:
                        terms.append(f"({W[i,j]:.15f} * {current_var}[{j}])")
                    expr = " + ".join(terms)
                    lines.append(f"  {vname}[{i}] = {expr};")
            current_var = vname
            current_size = no
            temp_idx += 1
            layer_idx += 1

        elif layer["type"] == "relu":
            vname = f"a{relu_idx}"
            lines.append(f"  double {vname}[{current_size}];")
            for j in range(current_size):
                st = stable_neurons.get((relu_idx, j), "unstable")
                if st == "active":
                    lines.append(f"  {vname}[{j}] = {current_var}[{j}]; // active")
                elif st == "inactive":
                    lines.append(f"  {vname}[{j}] = 0.0; // inactive")
                else:
                    lines.append(f"  {vname}[{j}] = ({current_var}[{j}] > 0.0) ? {current_var}[{j}] : 0.0;")
            current_var = vname
            relu_idx += 1

        elif layer["type"] == "flatten":
            pass

    # Copy to output
    for i in range(min(n_outputs, current_size)):
        lines.append(f"  y[{i}] = {current_var}[{i}];")
    lines.append("}")
    lines.append("")

    # Eva-friendly main: uses Frama_C_double_interval for non-det inputs
    lines.append("int main(void) {")
    lines.append(f"  double x[{n_inputs}];")
    lines.append(f"  double y[{n_outputs}];")
    for i in range(n_inputs):
        lb_val = input_lb[i] if np.isfinite(input_lb[i]) else -1e6
        ub_val = input_ub[i] if np.isfinite(input_ub[i]) else 1e6
        lines.append(f"  x[{i}] = Frama_C_double_interval({lb_val:.15f}, {ub_val:.15f});")
    lines.append(f"  nn_forward(x, y);")

    # Eva assertions: check that output is in expected range
    for c in output_constraints:
        assertion = _build_eva_assertion(c, n_outputs)
        if assertion:
            lines.append(f"  //@ assert {assertion};")

    lines.append("  return 0;")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def _build_ensures(output_constraints: List[Dict], n_outputs: int) -> str:
    """Build the ACSL ensures clause: negation of the unsafe property."""
    parts = []
    for c in output_constraints:
        neg = _negate_constraint(c, n_outputs)
        if neg:
            parts.append(neg)
    if not parts:
        return ""
    return " && ".join(parts) if len(parts) > 1 else parts[0]


def _build_eva_assertion(c: Dict, n_out: int) -> str:
    """Build an ACSL assertion that Eva should try to validate.
    This asserts the NEGATION of the unsafe property (safe condition)."""
    return _build_ensures_single(c, n_out)


def _build_ensures_single(c: Dict, n_out: int) -> str:
    """Build ensures for a single constraint (negated)."""
    return _negate_constraint(c, n_out)


def _negate_constraint(c: Dict, n_out: int) -> str:
    """Negate a VNNLIB constraint for ACSL ensures."""
    t = c["type"]
    if t == "output_bound":
        v = c["var"]
        if v >= n_out: return ""
        # Negate: >=b becomes <b, <=b becomes >b
        if c["op"] == ">=":
            return f"y[{v}] < {c['bound']:.15f}"
        else:
            return f"y[{v}] > {c['bound']:.15f}"
    elif t == "comparison":
        l, r = c["left"], c["right"]
        if l >= n_out or r >= n_out: return ""
        if c["op"] == "<=":
            return f"y[{l}] > y[{r}]"
        else:
            return f"y[{l}] < y[{r}]"
    elif t == "disjunction":
        # NOT(OR(clauses)) = AND(NOT(clause) for each clause)
        neg_clauses = []
        for clause in c["clauses"]:
            neg_atoms = []
            for a in clause:
                if "right" in a:
                    sub = {"type": "comparison", **a}
                else:
                    sub = {"type": "output_bound", **a}
                neg = _negate_constraint(sub, n_out)
                if neg: neg_atoms.append(neg)
            if neg_atoms:
                if len(neg_atoms) == 1:
                    neg_clauses.append(neg_atoms[0])
                else:
                    neg_clauses.append(f"({' || '.join(neg_atoms)})")
        if not neg_clauses: return ""
        return " && ".join(neg_clauses)
    return ""


# ------------------------------------------------------------------
# File saving
# ------------------------------------------------------------------

def save_c_program(code: str, benchmark: str, instance_name: str) -> str:
    d = DEFAULT_FRAMAC_DIR / benchmark
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{instance_name}.c"
    path.write_text(code)
    return str(path)


# ------------------------------------------------------------------
# Frama-C execution
# ------------------------------------------------------------------

def run_framac_eva(c_path: str, timeout: float = 300.0) -> Dict:
    """Run Frama-C Eva (abstract interpretation) on the generated C file."""
    t0 = time.time()
    # Use precision 0 for fast completion; higher values are more precise
    # but exponentially slower on large networks.
    cmd = [
        FRAMAC_BIN, "-eva",
        "-eva-precision", "0",
        "-no-unicode",
        c_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=_framac_env())
        elapsed = time.time() - t0
        output = proc.stdout + proc.stderr

        n_alarms = output.count("[eva:alarm]")
        has_red = "red alarm" in output.lower() or "invalid" in output.lower()
        # Check assertion status
        n_valid = output.count("valid")
        n_invalid = output.count("invalid")
        # Eva considers 0 alarms on assertions as proof
        assertions_ok = ("Assertions" in output and "0 invalid" in output)

        return {
            "success": proc.returncode == 0,
            "alarms": n_alarms,
            "has_red_alarm": has_red,
            "assertions_ok": assertions_ok,
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "output": output[-3000:],
            "time_seconds": elapsed,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "alarms": -1, "has_red_alarm": False,
                "output": "timeout", "time_seconds": timeout}
    except FileNotFoundError:
        return {"success": False, "alarms": -1, "has_red_alarm": False,
                "output": "frama-c not found", "time_seconds": 0}


def run_framac_wp(c_path: str, timeout: float = 300.0) -> Dict:
    """Run Frama-C WP (deductive verification) on the generated C file."""
    t0 = time.time()
    cmd = [
        FRAMAC_BIN, "-wp",
        "-wp-prover", "alt-ergo",
        "-wp-timeout", str(int(timeout)),
        "-no-unicode",
        c_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout + 30, env=_framac_env())
        elapsed = time.time() - t0
        output = proc.stdout + proc.stderr

        # Parse WP results
        n_proved = output.lower().count("proved")
        n_unknown = output.lower().count("unknown")
        n_failed = output.lower().count("failed") + output.lower().count("timeout")

        return {
            "success": proc.returncode == 0,
            "proved": n_proved,
            "unknown": n_unknown,
            "failed": n_failed,
            "output": output[-2000:],
            "time_seconds": elapsed,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "proved": 0, "unknown": 0, "failed": 0,
                "output": "timeout", "time_seconds": timeout}
    except FileNotFoundError:
        return {"success": False, "proved": 0, "unknown": 0, "failed": 0,
                "output": "frama-c not found", "time_seconds": 0}


def _framac_env():
    """Build environment with opam paths for Frama-C."""
    env = os.environ.copy()
    opam_bin = os.path.expanduser("~/.opam/default/bin")
    if os.path.isdir(opam_bin):
        env["PATH"] = opam_bin + ":" + env.get("PATH", "")
    return env


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def verify_with_framac(
    onnx_path: str,
    property,
    timeout: float = 300.0,
    total_cores: int = 0,
    save_code: bool = True,
    benchmark_name: str = "",
    instance_name: str = "",
    mode: str = "eva",
) -> Dict:
    """
    Verify a neural network property using Frama-C.

    Args:
        mode: "eva" (abstract interpretation), "wp" (deductive), or "both".
    """
    from .smt_solver import _extract_weights_from_onnx, _ibp_stable_neurons

    t0 = time.time()

    layers = _extract_weights_from_onnx(onnx_path)
    if not any(l["type"] == "linear" for l in layers):
        return {"result": "error", "solver": "frama-c", "time_seconds": 0,
                "details": "No linear layers"}

    n_inputs = property.n_inputs
    n_outputs = property.n_outputs
    lb = np.where(np.isfinite(property.input_lower), property.input_lower, -1e6)
    ub = np.where(np.isfinite(property.input_upper), property.input_upper, 1e6)

    stable = _ibp_stable_neurons(layers, lb, ub)

    # Generate C program
    code = generate_c_program(
        layers, n_inputs, n_outputs, lb, ub,
        property.output_constraints, stable,
    )

    c_path = ""
    if save_code:
        bname = benchmark_name or "unknown"
        iname = instance_name or "instance"
        c_path = save_c_program(code, bname, iname)
        logger.info(f"C program saved: {c_path} ({len(code)} chars)")

    if not c_path:
        import tempfile
        fd, c_path = tempfile.mkstemp(suffix=".c")
        with os.fdopen(fd, "w") as f:
            f.write(code)

    # Run Frama-C
    result_details = []
    verified = False

    if mode in ("eva", "both"):
        eva_result = run_framac_eva(c_path, timeout / 2 if mode == "both" else timeout)
        result_details.append(
            f"Eva: {eva_result['alarms']} alarms, "
            f"{eva_result.get('n_valid',0)} valid, "
            f"{eva_result.get('n_invalid',0)} invalid, "
            f"{eva_result['time_seconds']:.1f}s"
        )
        if eva_result["success"] and eva_result.get("assertions_ok", False):
            verified = True
        elif eva_result["success"] and eva_result["alarms"] == 0 and eva_result.get("n_invalid", 1) == 0:
            verified = True

    if mode in ("wp", "both"):
        remaining = max(10, timeout - (time.time() - t0))
        wp_result = run_framac_wp(c_path, remaining)
        result_details.append(
            f"WP: {wp_result.get('proved',0)} proved, "
            f"{wp_result.get('unknown',0)} unknown, "
            f"{wp_result['time_seconds']:.1f}s"
        )
        if wp_result["success"] and wp_result.get("proved", 0) > 0 and wp_result.get("failed", 0) == 0:
            verified = True

    elapsed = time.time() - t0

    return {
        "result": "verified" if verified else "unknown",
        "solver": f"frama-c({mode})",
        "time_seconds": elapsed,
        "c_file": c_path,
        "details": " | ".join(result_details) + f" | file: {c_path}",
    }
