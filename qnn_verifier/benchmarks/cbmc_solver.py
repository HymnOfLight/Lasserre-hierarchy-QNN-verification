"""
CBMC (Bounded Model Checking) based neural network verification.

Encodes the neural network + VNNLIB property as a C program using
CBMC's built-in primitives:
  - __CPROVER_assume() for input bounds
  - __CPROVER_assert() for safety property (negated unsafe region)
  - nondet_double() for non-deterministic inputs
  - ReLU encoded as if-else (CBMC's SAT solver handles the branching)

CBMC unrolls all paths through the ReLU network and checks the
assertion using a SAT/SMT backend — this is COMPLETE for the
bounded verification problem.

Generated C files saved to ./cbmc_output/<benchmark>/<instance>.c

Pipeline:
  1. Generate C with CBMC primitives (no ACSL, no Frama-C dependency)
  2. Run: cbmc --unwind 1 --unwinding-assertions <file>.c
  3. Parse result: VERIFICATION SUCCESSFUL / VERIFICATION FAILED + trace
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
DEFAULT_CBMC_DIR = _PROJECT_ROOT / "cbmc_output"
CBMC_BIN = shutil.which("cbmc") or "/usr/local/bin/cbmc"


def generate_cbmc_c(
    layers: List[Dict],
    n_inputs: int,
    n_outputs: int,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    output_constraints: List[Dict],
    stable_neurons: Dict,
) -> str:
    """
    Generate a C program for CBMC verification.

    Uses CBMC primitives:
    - nondet_double(): non-deterministic double value
    - __CPROVER_assume(cond): constrain non-deterministic values
    - __CPROVER_assert(cond, msg): property to verify

    The program is UNROLLED (no loops) for CBMC's bounded model checker.
    """
    lines = []
    lines.append("// CBMC neural network verification")
    lines.append("// Run: cbmc --unwind 1 <file>.c")
    lines.append("")
    lines.append("double nondet_double(void);")
    lines.append("")

    # Weight arrays
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

    # Main function with CBMC primitives
    lines.append("int main() {")
    lines.append(f"  double x[{n_inputs}];")
    lines.append("")

    # Non-deterministic inputs with assumed bounds
    lines.append("  // Non-deterministic inputs")
    for i in range(n_inputs):
        lines.append(f"  x[{i}] = nondet_double();")
    lines.append("")

    # Input bounds as assumptions
    lines.append("  // Input bounds (from VNNLIB)")
    for i in range(n_inputs):
        if np.isfinite(input_lb[i]):
            lines.append(f"  __CPROVER_assume(x[{i}] >= {input_lb[i]:.15f});")
        if np.isfinite(input_ub[i]):
            lines.append(f"  __CPROVER_assume(x[{i}] <= {input_ub[i]:.15f});")
    lines.append("")

    # Unrolled forward pass
    lines.append("  // Forward pass (unrolled)")
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
                        terms.append(f"(W{layer_idx}[{i}][{j}] * {current_var}[{j}])")
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
                    lines.append(f"  {vname}[{j}] = {current_var}[{j}];")
                elif st == "inactive":
                    lines.append(f"  {vname}[{j}] = 0.0;")
                else:
                    lines.append(f"  {vname}[{j}] = ({current_var}[{j}] > 0.0) ? {current_var}[{j}] : 0.0;")
            current_var = vname
            relu_idx += 1

        elif layer["type"] == "flatten":
            pass

    # Output
    lines.append("")
    lines.append(f"  double y[{n_outputs}];")
    for i in range(min(n_outputs, current_size)):
        lines.append(f"  y[{i}] = {current_var}[{i}];")
    lines.append("")

    # Property assertion
    # VNN-COMP: VNNLIB describes UNSAFE region.
    # We assert the NEGATION of the unsafe property.
    # If CBMC finds a counterexample to the assertion → property VIOLATED.
    # If CBMC proves the assertion → property VERIFIED.
    assertion = _build_cbmc_assertion(output_constraints, n_outputs)
    if assertion:
        lines.append(f"  // Safety property (negation of VNNLIB unsafe region)")
        lines.append(f'  __CPROVER_assert({assertion}, "safety property");')
    else:
        lines.append(f'  __CPROVER_assert(1, "no output constraints");')

    lines.append("  return 0;")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def _build_cbmc_assertion(constraints: List[Dict], n_out: int) -> str:
    """Build C assertion: negation of the VNNLIB unsafe property."""
    parts = []
    for c in constraints:
        neg = _negate_for_c(c, n_out)
        if neg:
            parts.append(neg)
    if not parts:
        return ""
    return " && ".join(parts)


def _negate_for_c(c: Dict, n_out: int) -> str:
    t = c["type"]
    if t == "output_bound":
        v = c["var"]
        if v >= n_out: return ""
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
        # NOT(OR(clauses)) = AND(NOT(clause))
        # NOT(clause) = OR(NOT(atom))
        neg_clauses = []
        for clause in c["clauses"]:
            neg_atoms = []
            for a in clause:
                if "right" in a:
                    sub = {"type": "comparison", **a}
                else:
                    sub = {"type": "output_bound", **a}
                neg = _negate_for_c(sub, n_out)
                if neg: neg_atoms.append(neg)
            if neg_atoms:
                if len(neg_atoms) == 1:
                    neg_clauses.append(neg_atoms[0])
                else:
                    neg_clauses.append(f"({' || '.join(neg_atoms)})")
        if not neg_clauses: return ""
        return " && ".join(neg_clauses)
    return ""


def save_cbmc_c(code: str, benchmark: str, instance_name: str) -> str:
    d = DEFAULT_CBMC_DIR / benchmark
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{instance_name}.c"
    path.write_text(code)
    return str(path)


def run_cbmc(c_path: str, timeout: float = 300.0) -> Dict:
    """
    Run CBMC on the generated C file.

    CBMC performs bounded model checking:
    - Encodes the C program + assertions as a SAT/SMT problem
    - Uses MiniSat/CaDiCaL as the backend solver
    - Reports VERIFICATION SUCCESSFUL or VERIFICATION FAILED + trace
    """
    t0 = time.time()
    cmd = [
        CBMC_BIN,
        "--unwind", "1",              # no loops to unwind (already unrolled)
        "--no-unwinding-assertions",
        "--verbosity", "4",
        c_path,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - t0
        output = proc.stdout + proc.stderr

        if "VERIFICATION SUCCESSFUL" in output:
            return {
                "result": "verified",
                "time_seconds": elapsed,
                "details": "CBMC: VERIFICATION SUCCESSFUL (all assertions hold)",
                "output": output[-2000:],
            }
        elif "VERIFICATION FAILED" in output:
            # Extract counterexample from trace
            cex = _parse_cbmc_trace(output)
            return {
                "result": "violated",
                "time_seconds": elapsed,
                "details": "CBMC: VERIFICATION FAILED (assertion violated)",
                "counterexample": cex,
                "output": output[-2000:],
            }
        else:
            return {
                "result": "unknown",
                "time_seconds": elapsed,
                "details": f"CBMC: inconclusive (returncode={proc.returncode})",
                "output": output[-2000:],
            }
    except subprocess.TimeoutExpired:
        return {"result": "unknown", "time_seconds": timeout,
                "details": f"CBMC: timeout ({timeout}s)", "output": ""}
    except FileNotFoundError:
        return {"result": "error", "time_seconds": 0,
                "details": "cbmc not found (install from github.com/diffblue/cbmc)",
                "output": ""}


def _parse_cbmc_trace(output: str) -> Dict:
    """Extract counterexample values from CBMC trace output."""
    cex = {"inputs": {}, "outputs": {}}
    for line in output.split("\n"):
        # Look for: x[0]=0.639929
        if "x[" in line and "=" in line:
            try:
                parts = line.split("=")
                idx = int(parts[0].split("[")[1].split("]")[0])
                val = float(parts[-1].strip())
                cex["inputs"][f"x[{idx}]"] = val
            except (ValueError, IndexError):
                pass
        if "y[" in line and "=" in line:
            try:
                parts = line.split("=")
                idx = int(parts[0].split("[")[1].split("]")[0])
                val = float(parts[-1].strip())
                cex["outputs"][f"y[{idx}]"] = val
            except (ValueError, IndexError):
                pass
    return cex


def verify_with_cbmc(
    onnx_path: str,
    property,
    timeout: float = 300.0,
    total_cores: int = 0,
    save_code: bool = True,
    benchmark_name: str = "",
    instance_name: str = "",
) -> Dict:
    """Verify a neural network property using CBMC."""
    from .smt_solver import _extract_weights_from_onnx, _ibp_stable_neurons

    t0 = time.time()

    layers = _extract_weights_from_onnx(onnx_path)
    if not any(l["type"] == "linear" for l in layers):
        return {"result": "error", "solver": "cbmc", "time_seconds": 0,
                "details": "No linear layers"}

    lb = np.where(np.isfinite(property.input_lower), property.input_lower, -1e6)
    ub = np.where(np.isfinite(property.input_upper), property.input_upper, 1e6)
    stable = _ibp_stable_neurons(layers, lb, ub)

    code = generate_cbmc_c(
        layers, property.n_inputs, property.n_outputs,
        lb, ub, property.output_constraints, stable,
    )

    c_path = ""
    if save_code:
        bname = benchmark_name or "unknown"
        iname = instance_name or "instance"
        c_path = save_cbmc_c(code, bname, iname)
        logger.info(f"CBMC C saved: {c_path} ({len(code)} chars)")

    if not c_path:
        import tempfile
        fd, c_path = tempfile.mkstemp(suffix=".c")
        with os.fdopen(fd, "w") as f:
            f.write(code)

    cbmc_result = run_cbmc(c_path, timeout)
    elapsed = time.time() - t0

    return {
        "result": cbmc_result["result"],
        "solver": "cbmc",
        "time_seconds": elapsed,
        "c_file": c_path,
        "details": cbmc_result["details"],
        "counterexample": cbmc_result.get("counterexample"),
    }
