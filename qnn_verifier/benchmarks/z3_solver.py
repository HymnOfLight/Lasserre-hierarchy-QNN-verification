"""
Z3-based exact neural network verification solver.

Encodes the neural network as an SMT formula over real arithmetic,
adds the VNNLIB input/output constraints, and checks satisfiability.

  SAT   → unsafe region reachable → property VIOLATED
  UNSAT → unsafe region unreachable → property VERIFIED (exact proof)

Supports ReLU networks loaded from ONNX via weight matrix extraction.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _extract_weights_from_onnx(onnx_path: str) -> List[Dict]:
    """
    Extract weight matrices and biases from an ONNX model.

    Returns a list of layers with types "linear" (with W, b) and "relu".
    The weight matrix W has shape (n_out, n_in), so y = W @ x + b.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    graph = model.graph

    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)

    # Infer output name → shape from graph value_info and outputs
    raw_layers = []
    for node in graph.node:
        op = node.op_type

        if op in ("Gemm", "MatMul"):
            W = None
            b = None
            transpose_B = False

            if op == "Gemm":
                for attr in node.attribute:
                    if attr.name == "transB":
                        transpose_B = bool(attr.i)

            for inp_name in node.input:
                if inp_name in initializers:
                    arr = initializers[inp_name]
                    if arr.ndim == 2:
                        W = arr
                    elif arr.ndim == 1:
                        b = arr

            if W is not None:
                # For Gemm: y = x @ W^T + b  (when transB=1) → W_eff = W
                # For Gemm: y = x @ W + b    (when transB=0) → W_eff = W^T
                # We want W_eff such that y = W_eff @ x + b
                if transpose_B:
                    W_eff = W.astype(np.float64)  # W is already (n_out, n_in)
                else:
                    W_eff = W.T.astype(np.float64)

                raw_layers.append({
                    "type": "linear",
                    "W": W_eff,
                    "b": b.astype(np.float64) if b is not None else np.zeros(W_eff.shape[0], dtype=np.float64),
                })

        elif op == "Relu":
            raw_layers.append({"type": "relu"})

        elif op in ("Flatten", "Reshape"):
            raw_layers.append({"type": "flatten"})

        elif op == "Add":
            b = None
            for inp_name in node.input:
                if inp_name in initializers:
                    arr = initializers[inp_name]
                    if arr.ndim == 1:
                        b = arr
            if b is not None:
                raw_layers.append({"type": "add_bias", "b": b.astype(np.float64)})

    # Merge add_bias into preceding linear layer
    layers = []
    for l in raw_layers:
        if l["type"] == "add_bias" and layers and layers[-1]["type"] == "linear":
            b = l["b"]
            n = min(len(b), len(layers[-1]["b"]))
            layers[-1]["b"][:n] += b[:n]
        else:
            layers.append(l)

    return layers


def verify_with_z3(
    onnx_path: str,
    property,  # VNNLIBProperty
    timeout: float = 300.0,
    input_shape: Optional[Tuple] = None,
) -> Dict:
    """
    Verify a neural network property using Z3.

    Args:
        onnx_path: Path to the ONNX model file.
        property: Parsed VNNLIBProperty.
        timeout: Z3 solver timeout in seconds.
        input_shape: Optional input tensor shape.

    Returns:
        Dict with 'result' ("verified"/"violated"/"unknown"),
        'time_seconds', 'details'.
    """
    import z3

    t0 = time.time()

    layers = _extract_weights_from_onnx(onnx_path)
    linear_layers = [l for l in layers if l["type"] == "linear"]
    if not linear_layers:
        return {"result": "error", "time_seconds": 0.0,
                "details": "No linear layers extracted from ONNX"}

    n_inputs = property.n_inputs
    n_outputs = property.n_outputs

    logger.info(f"Z3 encoding: {n_inputs} inputs, {n_outputs} outputs, "
                f"{len(linear_layers)} linear layers")

    # Create Z3 real variables for inputs
    X = [z3.Real(f"X_{i}") for i in range(n_inputs)]
    Y_output = [z3.Real(f"Y_{i}") for i in range(n_outputs)]

    solver = z3.Solver()
    solver.set("timeout", int(timeout * 1000))

    # Input bounds
    lb = property.input_lower
    ub = property.input_upper
    for i in range(n_inputs):
        if np.isfinite(lb[i]):
            solver.add(X[i] >= float(lb[i]))
        if np.isfinite(ub[i]):
            solver.add(X[i] <= float(ub[i]))

    # Encode network layer by layer
    current_vars = X
    relu_count = 0
    layer_idx = 0

    for layer in layers:
        if layer["type"] == "linear":
            W = layer["W"]
            b = layer["b"]
            n_out, n_in = W.shape

            # Truncate / pad input dimension
            in_vars = current_vars[:n_in]
            if len(in_vars) < n_in:
                in_vars = in_vars + [z3.RealVal(0)] * (n_in - len(in_vars))

            out_vars = []
            for j in range(n_out):
                name = f"h_{layer_idx}_{j}"
                h = z3.Real(name)
                # h = W[j,:] @ in_vars + b[j]
                expr = z3.RealVal(float(b[j]))
                for k in range(n_in):
                    w_val = float(W[j, k])
                    if abs(w_val) > 1e-15:
                        expr = expr + z3.RealVal(w_val) * in_vars[k]
                solver.add(h == expr)
                out_vars.append(h)

            current_vars = out_vars
            layer_idx += 1

        elif layer["type"] == "relu":
            relu_vars = []
            for j, v in enumerate(current_vars):
                name = f"r_{relu_count}_{j}"
                r = z3.Real(name)
                solver.add(r == z3.If(v >= 0, v, z3.RealVal(0)))
                relu_vars.append(r)
                relu_count += 1
            current_vars = relu_vars

        elif layer["type"] == "flatten":
            pass

        elif layer["type"] == "add_bias":
            b = layer["b"]
            for j in range(min(len(current_vars), len(b))):
                name = f"ab_{layer_idx}_{j}"
                h = z3.Real(name)
                solver.add(h == current_vars[j] + z3.RealVal(float(b[j])))
                current_vars[j] = h
            layer_idx += 1

    # Map network outputs to Y variables
    for i in range(min(n_outputs, len(current_vars))):
        solver.add(Y_output[i] == current_vars[i])

    # Add output constraints (VNNLIB: describes unsafe region)
    for c in property.output_constraints:
        _add_z3_constraint(solver, c, Y_output)

    # Solve
    logger.info(f"Z3 solving ({solver.num_scopes()} scopes, "
                f"{len(linear_layers)} linear layers, {relu_count} ReLU neurons)...")

    check_result = solver.check()
    elapsed = time.time() - t0

    if check_result == z3.sat:
        # Unsafe region is reachable → property VIOLATED
        model = solver.model()
        cex_input = []
        for i in range(n_inputs):
            val = model.evaluate(X[i], model_completion=True)
            cex_input.append(_z3_to_float(val))
        cex_output = []
        for i in range(n_outputs):
            val = model.evaluate(Y_output[i], model_completion=True)
            cex_output.append(_z3_to_float(val))

        return {
            "result": "violated",
            "time_seconds": elapsed,
            "details": "Z3 found counterexample (unsafe region reachable)",
            "counterexample_input": cex_input,
            "counterexample_output": cex_output,
        }

    elif check_result == z3.unsat:
        return {
            "result": "verified",
            "time_seconds": elapsed,
            "details": "Z3 proved UNSAT (unsafe region unreachable — exact proof)",
        }

    else:
        return {
            "result": "unknown",
            "time_seconds": elapsed,
            "details": f"Z3 returned {check_result} (likely timeout)",
        }


def _add_z3_constraint(solver, constraint: Dict, Y: list):
    """Add a VNNLIB output constraint to the Z3 solver."""
    import z3

    ctype = constraint["type"]

    if ctype == "output_bound":
        var = constraint["var"]
        if var >= len(Y):
            return
        if constraint["op"] == ">=":
            solver.add(Y[var] >= z3.RealVal(float(constraint["bound"])))
        else:
            solver.add(Y[var] <= z3.RealVal(float(constraint["bound"])))

    elif ctype == "comparison":
        l, r = constraint["left"], constraint["right"]
        if l >= len(Y) or r >= len(Y):
            return
        if constraint["op"] == "<=":
            solver.add(Y[l] <= Y[r])
        else:
            solver.add(Y[l] >= Y[r])

    elif ctype == "disjunction":
        or_clauses = []
        for clause in constraint["clauses"]:
            and_atoms = []
            for atom in clause:
                if "right" in atom:
                    l, r = atom["left"], atom["right"]
                    if l < len(Y) and r < len(Y):
                        if atom["op"] == "<=":
                            and_atoms.append(Y[l] <= Y[r])
                        else:
                            and_atoms.append(Y[l] >= Y[r])
                elif "bound" in atom:
                    v = atom["var"]
                    if v < len(Y):
                        if atom["op"] == ">=":
                            and_atoms.append(Y[v] >= z3.RealVal(float(atom["bound"])))
                        else:
                            and_atoms.append(Y[v] <= z3.RealVal(float(atom["bound"])))
            if and_atoms:
                or_clauses.append(z3.And(*and_atoms) if len(and_atoms) > 1 else and_atoms[0])
        if or_clauses:
            solver.add(z3.Or(*or_clauses) if len(or_clauses) > 1 else or_clauses[0])


def _z3_to_float(val) -> float:
    """Convert a Z3 value to Python float."""
    import z3
    try:
        if z3.is_rational_value(val):
            return float(val.numerator_as_long()) / float(val.denominator_as_long())
        if z3.is_algebraic_value(val):
            return float(val.approx(20))
        return float(str(val))
    except Exception:
        return 0.0
