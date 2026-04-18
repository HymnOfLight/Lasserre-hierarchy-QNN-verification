"""
Z3-based exact neural network verification solver with multi-core support.

Encodes the neural network as an SMT formula over real arithmetic,
adds the VNNLIB input/output constraints, and checks satisfiability.

  SAT   → unsafe region reachable → property VIOLATED
  UNSAT → unsafe region unreachable → property VERIFIED (exact proof)

Parallelism strategy:
  1. IBP pre-processing: determine stable neurons (always active / always
     inactive) to eliminate trivial ReLU branches.
  2. Z3 parallel mode: enable built-in parallel solving with configurable
     thread count.
  3. Multi-process portfolio: split the problem by fixing unstable neuron
     phases and solve sub-problems in parallel worker processes.
"""

import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# ONNX weight extraction
# ------------------------------------------------------------------

def _extract_weights_from_onnx(onnx_path: str) -> List[Dict]:
    """
    Extract weight matrices and biases from an ONNX model.
    Returns a list of layers with types "linear" (W, b) and "relu".
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    graph = model.graph

    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)

    raw_layers = []
    for node in graph.node:
        op = node.op_type

        if op in ("Gemm", "MatMul"):
            W, b, transpose_B = None, None, False
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
                W_eff = W.astype(np.float64) if transpose_B else W.T.astype(np.float64)
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

    layers = []
    for l in raw_layers:
        if l["type"] == "add_bias" and layers and layers[-1]["type"] == "linear":
            b = l["b"]
            n = min(len(b), len(layers[-1]["b"]))
            layers[-1]["b"][:n] += b[:n]
        else:
            layers.append(l)

    return layers


# ------------------------------------------------------------------
# IBP pre-processing: identify stable ReLU neurons
# ------------------------------------------------------------------

def _ibp_stable_neurons(
    layers: List[Dict],
    input_lb: np.ndarray,
    input_ub: np.ndarray,
) -> Dict[Tuple[int, int], str]:
    """
    Run interval bound propagation to classify each ReLU neuron as:
      "active"   — always non-negative → ReLU is identity
      "inactive" — always negative → ReLU is zero
      "unstable" — crosses zero → needs branching

    Returns dict mapping (relu_layer_idx, neuron_idx) → status.
    """
    current_lb = input_lb.copy()
    current_ub = input_ub.copy()

    stable = {}
    relu_layer = 0

    for layer in layers:
        if layer["type"] == "linear":
            W = layer["W"]
            b = layer["b"]
            n_out, n_in = W.shape
            lb_in = current_lb[:n_in] if len(current_lb) >= n_in else np.pad(current_lb, (0, n_in - len(current_lb)))
            ub_in = current_ub[:n_in] if len(current_ub) >= n_in else np.pad(current_ub, (0, n_in - len(current_ub)))
            W_pos = np.maximum(W, 0)
            W_neg = np.minimum(W, 0)
            current_lb = W_pos @ lb_in + W_neg @ ub_in + b
            current_ub = W_pos @ ub_in + W_neg @ lb_in + b

        elif layer["type"] == "relu":
            n = len(current_lb)
            for j in range(n):
                if current_lb[j] >= 0:
                    stable[(relu_layer, j)] = "active"
                elif current_ub[j] <= 0:
                    stable[(relu_layer, j)] = "inactive"
                else:
                    stable[(relu_layer, j)] = "unstable"
            current_lb = np.maximum(current_lb, 0)
            current_ub = np.maximum(current_ub, 0)
            relu_layer += 1

        elif layer["type"] == "flatten":
            pass

    return stable


# ------------------------------------------------------------------
# Z3 formula builder
# ------------------------------------------------------------------

def _build_z3_formula(
    layers: List[Dict],
    n_inputs: int,
    n_outputs: int,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    output_constraints: List[Dict],
    stable_neurons: Dict[Tuple[int, int], str],
    fixed_phases: Optional[Dict[Tuple[int, int], bool]] = None,
    parallel_threads: int = 1,
    timeout_ms: int = 60000,
):
    """
    Build and solve a Z3 formula encoding the neural network.

    Args:
        fixed_phases: Optional dict mapping (relu_layer, neuron) → True (active)
                      / False (inactive) for portfolio sub-problem splitting.
        parallel_threads: Z3 internal parallel thread count.
    """
    import z3

    if parallel_threads > 1:
        z3.set_param("parallel.enable", True)
        z3.set_param("parallel.threads.max", parallel_threads)
    else:
        z3.set_param("parallel.enable", False)

    X = [z3.Real(f"X_{i}") for i in range(n_inputs)]
    Y_out = [z3.Real(f"Y_{i}") for i in range(n_outputs)]

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    # Input bounds
    for i in range(n_inputs):
        if np.isfinite(input_lb[i]):
            solver.add(X[i] >= z3.RealVal(float(input_lb[i])))
        if np.isfinite(input_ub[i]):
            solver.add(X[i] <= z3.RealVal(float(input_ub[i])))

    current_vars = X
    relu_layer = 0
    layer_idx = 0

    # Pre-compute Z3 RealVal constants for all weights (batch for speed)
    _rv_cache = {}
    def _rv(f: float):
        if f not in _rv_cache:
            _rv_cache[f] = z3.RealVal(f)
        return _rv_cache[f]

    for layer in layers:
        if layer["type"] == "linear":
            W = layer["W"]
            b = layer["b"]
            n_out, n_in = W.shape
            in_vars = current_vars[:n_in]
            if len(in_vars) < n_in:
                in_vars = in_vars + [_rv(0.0)] * (n_in - len(in_vars))

            # Sparse encoding: only add non-zero weight terms
            out_vars = []
            for j in range(n_out):
                h = z3.Real(f"h_{layer_idx}_{j}")
                nz = np.nonzero(np.abs(W[j, :]) > 1e-15)[0]
                if len(nz) == 0:
                    solver.add(h == _rv(float(b[j])))
                else:
                    terms = [_rv(float(b[j]))]
                    for k in nz:
                        terms.append(_rv(float(W[j, k])) * in_vars[k])
                    solver.add(h == z3.Sum(terms))
                out_vars.append(h)
            current_vars = out_vars
            layer_idx += 1

        elif layer["type"] == "relu":
            relu_vars = []
            for j, v in enumerate(current_vars):
                key = (relu_layer, j)
                status = stable_neurons.get(key, "unstable")

                if fixed_phases and key in fixed_phases:
                    phase = fixed_phases[key]
                    r = z3.Real(f"r_{relu_layer}_{j}")
                    if phase:
                        solver.add(v >= 0)
                        solver.add(r == v)
                    else:
                        solver.add(v <= 0)
                        solver.add(r == 0)
                    relu_vars.append(r)

                elif status == "active":
                    relu_vars.append(v)

                elif status == "inactive":
                    r = z3.Real(f"r_{relu_layer}_{j}")
                    solver.add(r == 0)
                    relu_vars.append(r)

                else:
                    r = z3.Real(f"r_{relu_layer}_{j}")
                    solver.add(r >= 0)
                    solver.add(r >= v)
                    solver.add(z3.Or(r == 0, r == v))
                    relu_vars.append(r)

            current_vars = relu_vars
            relu_layer += 1

        elif layer["type"] == "flatten":
            pass

    # Map outputs
    for i in range(min(n_outputs, len(current_vars))):
        solver.add(Y_out[i] == current_vars[i])

    # Output constraints (VNNLIB unsafe region)
    for c in output_constraints:
        _add_z3_constraint(solver, c, Y_out)

    # Solve
    result = solver.check()

    if result == z3.sat:
        m = solver.model()
        cex_in = [_z3_to_float(m.evaluate(X[i], True)) for i in range(n_inputs)]
        cex_out = [_z3_to_float(m.evaluate(Y_out[i], True)) for i in range(n_outputs)]
        return "violated", cex_in, cex_out
    elif result == z3.unsat:
        return "verified", None, None
    else:
        return "unknown", None, None


# ------------------------------------------------------------------
# Portfolio parallel sub-problem worker (for ProcessPoolExecutor)
# ------------------------------------------------------------------

def _solve_subproblem(args) -> Tuple[str, Optional[List], Optional[List]]:
    """Worker function for multi-process portfolio solving."""
    (onnx_path, n_inputs, n_outputs, input_lb, input_ub,
     output_constraints, stable_neurons, fixed_phases,
     threads_per_worker, timeout_ms) = args

    layers = _extract_weights_from_onnx(onnx_path)
    return _build_z3_formula(
        layers, n_inputs, n_outputs, input_lb, input_ub,
        output_constraints, stable_neurons, fixed_phases,
        threads_per_worker, timeout_ms,
    )


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def verify_with_z3(
    onnx_path: str,
    property,  # VNNLIBProperty
    timeout: float = 300.0,
    input_shape: Optional[Tuple] = None,
    n_workers: int = 0,
    threads_per_worker: int = 0,
) -> Dict:
    """
    Verify a neural network property using Z3 with multi-core support.

    Parallelism:
      - n_workers=0 (default): auto-detect CPU cores.
      - n_workers=1: single-process, Z3 internal parallelism only.
      - n_workers>1: multi-process portfolio with domain splitting.

    Args:
        onnx_path: Path to the ONNX model file.
        property: Parsed VNNLIBProperty.
        timeout: Overall timeout in seconds.
        input_shape: Optional input tensor shape.
        n_workers: Number of parallel worker processes (0=auto).
        threads_per_worker: Z3 internal threads per worker (0=auto).
    """
    t0 = time.time()

    total_cores = os.cpu_count() or 4

    # Default: single process, Z3 internal parallel using all cores.
    # Multi-process portfolio only when explicitly requested (n_workers > 1).
    if n_workers <= 0:
        n_workers = 1  # single process, Z3 handles parallelism internally
    if threads_per_worker <= 0:
        threads_per_worker = total_cores

    layers = _extract_weights_from_onnx(onnx_path)
    linear_layers = [l for l in layers if l["type"] == "linear"]
    if not linear_layers:
        return {"result": "error", "time_seconds": 0.0,
                "details": "No linear layers extracted from ONNX"}

    n_inputs = property.n_inputs
    n_outputs = property.n_outputs

    # IBP: identify stable neurons
    lb = property.input_lower.copy()
    ub = property.input_upper.copy()
    lb = np.where(np.isfinite(lb), lb, -1e6)
    ub = np.where(np.isfinite(ub), ub, 1e6)

    stable = _ibp_stable_neurons(layers, lb, ub)
    n_stable = sum(1 for s in stable.values() if s != "unstable")
    n_unstable = sum(1 for s in stable.values() if s == "unstable")
    n_total = len(stable)

    logger.info(
        f"Z3 parallel: {n_workers} workers × {threads_per_worker} threads | "
        f"ReLU: {n_total} total, {n_stable} stable, {n_unstable} unstable"
    )

    timeout_ms = int(timeout * 1000)
    output_constraints = property.output_constraints

    # --- Single process with Z3 internal parallelism ---
    # This is the default and most efficient path: Z3's built-in parallel
    # mode uses all available cores via its internal thread pool.
    if n_workers == 1 or n_unstable <= 10:
        status, cex_in, cex_out = _build_z3_formula(
            layers, n_inputs, n_outputs, lb, ub,
            output_constraints, stable, None,
            total_cores, timeout_ms,
        )
        return _make_result(status, cex_in, cex_out, time.time() - t0, n_unstable, total_cores)

    # --- Multi-process portfolio: split by fixing unstable neuron phases ---
    unstable_keys = [k for k, v in stable.items() if v == "unstable"]

    # Pick the first few unstable neurons to split on
    n_split = min(int(np.log2(n_workers)) + 1, len(unstable_keys), 8)
    split_keys = unstable_keys[:n_split]
    n_subproblems = 2 ** n_split

    logger.info(f"Portfolio: splitting on {n_split} neurons → {n_subproblems} sub-problems")

    sub_timeout_ms = int((timeout - (time.time() - t0)) * 1000)
    if sub_timeout_ms <= 0:
        return {"result": "unknown", "time_seconds": time.time() - t0,
                "details": "Timeout before solving"}

    tasks = []
    for combo in range(n_subproblems):
        fixed = {}
        for bit_idx, key in enumerate(split_keys):
            fixed[key] = bool((combo >> bit_idx) & 1)
        tasks.append((
            onnx_path, n_inputs, n_outputs, lb, ub,
            output_constraints, stable, fixed,
            threads_per_worker, sub_timeout_ms,
        ))

    # Execute in parallel
    any_violated = False
    all_verified = True
    cex_in_result = None
    cex_out_result = None
    n_verified = 0
    n_unknown = 0

    ctx = multiprocessing.get_context("spawn")
    pool_workers = min(n_workers, n_subproblems)

    with ProcessPoolExecutor(max_workers=pool_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_solve_subproblem, t): i for i, t in enumerate(tasks)}

        remaining_timeout = timeout - (time.time() - t0) + 2
        for future in as_completed(futures, timeout=max(remaining_timeout, 1)):
            try:
                status, cex_in, cex_out = future.result(timeout=5)
            except Exception as e:
                logger.warning(f"Sub-problem failed: {e}")
                all_verified = False
                n_unknown += 1
                continue

            if status == "violated":
                any_violated = True
                cex_in_result = cex_in
                cex_out_result = cex_out
                for f in futures:
                    f.cancel()
                break
            elif status == "verified":
                n_verified += 1
            else:
                all_verified = False
                n_unknown += 1

    elapsed = time.time() - t0

    if any_violated:
        return _make_result("violated", cex_in_result, cex_out_result, elapsed, n_unstable, n_workers)

    if all_verified and n_verified == n_subproblems:
        return _make_result("verified", None, None, elapsed, n_unstable, n_workers)

    return _make_result("unknown", None, None, elapsed, n_unstable, n_workers,
                        extra=f"{n_verified}/{n_subproblems} sub-problems verified, {n_unknown} unknown")


def _make_result(status, cex_in, cex_out, elapsed, n_unstable, n_workers, extra=""):
    details_map = {
        "verified": f"Z3 proved UNSAT — exact proof ({n_unstable} unstable ReLUs, {n_workers} workers)",
        "violated": f"Z3 found counterexample ({n_workers} workers)",
        "unknown": f"Z3 timeout/unknown ({n_unstable} unstable ReLUs, {n_workers} workers)",
    }
    details = details_map.get(status, status)
    if extra:
        details += f" | {extra}"

    result = {"result": status, "time_seconds": elapsed, "details": details}
    if cex_in is not None:
        result["counterexample_input"] = cex_in
    if cex_out is not None:
        result["counterexample_output"] = cex_out
    return result


# ------------------------------------------------------------------
# Z3 constraint helpers
# ------------------------------------------------------------------

def _add_z3_constraint(solver, constraint: Dict, Y: list):
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
    import z3
    try:
        if z3.is_rational_value(val):
            return float(val.numerator_as_long()) / float(val.denominator_as_long())
        if z3.is_algebraic_value(val):
            return float(val.approx(20))
        return float(str(val))
    except Exception:
        return 0.0
