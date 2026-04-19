"""
Benchmark verification engine.

Loads ONNX models, parses VNNLIB properties, and runs the actual
verification using forward-pass-anchored Jacobian bounding + optional
Lasserre SDP refinement.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .loader import BenchmarkInstance, _OnnxRuntimeWrapper
from .vnnlib_parser import VNNLIBProperty

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkVerificationResult:
    """Result of verifying a single benchmark instance."""
    benchmark: str = ""
    instance_idx: int = 0
    model_name: str = ""
    property_name: str = ""
    result: str = "unknown"  # "verified", "violated", "unknown", "timeout", "error"
    lower_bound: float = float("-inf")
    time_seconds: float = 0.0
    method: str = ""
    details: str = ""

    def __str__(self):
        tag = {
            "verified": "VERIFIED  ",
            "violated": "VIOLATED  ",
            "unknown":  "UNKNOWN   ",
            "timeout":  "TIMEOUT   ",
            "error":    "ERROR     ",
        }.get(self.result, self.result)
        return (
            f"[{tag}] {self.model_name} | {self.property_name} | "
            f"bound={self.lower_bound:+.6f} | {self.time_seconds:.2f}s | {self.method}"
        )


def verify_instance(
    instance: BenchmarkInstance,
    timeout: Optional[float] = None,
    method: str = "jacobian",
    n_workers: int = 0,
    threads_per_worker: int = 0,
) -> BenchmarkVerificationResult:
    """
    Verify a benchmark instance.

    Args:
        instance: BenchmarkInstance with model and property loaded.
        timeout: Override the instance timeout (seconds).
        method: "jacobian" (fast), "z3" (exact SMT), or "sdp" (Lasserre).
        n_workers: Z3 parallel workers (0=auto).
        threads_per_worker: Z3 internal threads per worker (0=auto).

    Returns:
        BenchmarkVerificationResult.
    """
    if method == "z3":
        return _verify_with_z3_wrapper(instance, timeout, n_workers, threads_per_worker)
    if method in ("smt", "portfolio", "cvc5", "opensmt"):
        return _verify_with_smt_portfolio(instance, timeout, method, n_workers, threads_per_worker)
    if method == "gurobi":
        return _verify_with_gurobi(instance, timeout, n_workers)

    from pathlib import Path
    t0 = time.time()
    max_time = timeout or instance.timeout

    res = BenchmarkVerificationResult(
        benchmark=instance.benchmark_name,
        model_name=Path(instance.model_path).stem,
        property_name=Path(instance.property_path).stem,
    )

    if instance.model is None:
        res.result = "error"
        res.details = "Model not loaded"
        return res
    if instance.property is None:
        res.result = "error"
        res.details = "Property not loaded"
        return res

    prop = instance.property
    model = instance.model

    if prop.n_inputs == 0:
        res.result = "error"
        res.details = "No input variables in property"
        return res

    # Build nominal input from center of input bounds
    lb = prop.input_lower.copy()
    ub = prop.input_upper.copy()
    lb = np.where(np.isfinite(lb), lb, -1.0)
    ub = np.where(np.isfinite(ub), ub, 1.0)
    x0 = ((lb + ub) / 2.0).astype(np.float32)

    try:
        x_tensor = torch.tensor(x0).reshape(instance.input_shape)
    except Exception:
        x_tensor = torch.tensor(x0).unsqueeze(0)

    # Nominal forward pass
    try:
        with torch.no_grad():
            output_nominal = model(x_tensor).detach().flatten().numpy()
    except Exception as e:
        res.result = "error"
        res.details = f"Forward pass failed: {e}"
        res.time_seconds = time.time() - t0
        return res

    if len(prop.output_constraints) == 0:
        res.result = "unknown"
        res.details = "No output constraints parsed"
        res.time_seconds = time.time() - t0
        return res

    # VNN-COMP semantics: the VNNLIB file describes the UNSAFE region.
    # If the conjunction of all assertions is SATISFIABLE within the
    # input bounds, the property is VIOLATED (unsafe region reachable).
    # If UNSATISFIABLE, the property HOLDS (the network is SAFE).
    #
    # Verification strategy:
    #   For each output constraint (describes unsafe region):
    #     - If output_upper can satisfy it → unsafe region might be reachable
    #     - If output bounds CANNOT satisfy it → safe (verified)
    #
    # Example: (assert (>= Y_0 3.99))
    #   Unsafe if Y_0 can be >= 3.99.
    #   Safe if output_upper[0] < 3.99.

    epsilon_per_dim = (ub - lb) / 2.0

    # Try Jacobian-based bounds (requires differentiable model)
    is_differentiable = not isinstance(model, _OnnxRuntimeWrapper)
    if is_differentiable:
        try:
            grad_bounds = _compute_output_bounds(
                model, x_tensor, instance.input_shape,
                prop.n_outputs, epsilon_per_dim, max_time - (time.time() - t0)
            )
        except Exception as e:
            logger.warning(f"Jacobian computation failed: {e}")
            is_differentiable = False

    if not is_differentiable:
        # Fallback: sample-based bound estimation for OnnxRuntime models
        grad_bounds = _sample_based_bounds(
            model, x_tensor, instance.input_shape,
            prop.n_outputs, lb, ub, n_samples=200
        )

    output_lower = output_nominal[:prop.n_outputs] - grad_bounds[:prop.n_outputs]
    output_upper = output_nominal[:prop.n_outputs] + grad_bounds[:prop.n_outputs]

    # Check: can the unsafe region be EXCLUDED?
    # For each constraint, check if the output bounds make it impossible.
    unsafe_reachable = False
    min_margin = np.inf

    for c in prop.output_constraints:
        excluded, margin = _check_unsafe_excluded(c, output_lower, output_upper)
        min_margin = min(min_margin, margin)
        if not excluded:
            unsafe_reachable = True

    res.lower_bound = float(min_margin)
    res.time_seconds = time.time() - t0
    res.method = "jacobian" if is_differentiable else "sampling"

    if not unsafe_reachable:
        res.result = "verified"
        res.details = f"Unsafe region excluded for all {len(prop.output_constraints)} constraints"
    else:
        # Check nominal: is the nominal point already in the unsafe region?
        nominal_unsafe = True
        for c in prop.output_constraints:
            sat, _ = _check_constraint_point(c, output_nominal)
            if not sat:
                nominal_unsafe = False
                break
        if nominal_unsafe:
            res.result = "violated"
            res.details = "Nominal input already in unsafe region"
        else:
            res.result = "unknown"
            res.details = f"Cannot exclude unsafe region (margin={min_margin:.6f})"

    return res


def _compute_output_bounds(
    model: nn.Module,
    x_tensor: torch.Tensor,
    input_shape: Tuple,
    n_outputs: int,
    epsilon_per_dim: np.ndarray,
    time_budget: float,
) -> np.ndarray:
    """
    Compute per-output-neuron bounds using the Jacobian:
      |f_j(x) - f_j(x0)| <= sum_i |∂f_j/∂x_i| * eps_i

    For non-uniform input perturbation (different epsilon per dimension),
    we need the weighted L1 norm of the Jacobian row.
    """
    safety_factor = 1.5
    n_out = min(n_outputs, 20)  # limit for speed
    eps = torch.tensor(epsilon_per_dim, dtype=torch.float32)

    x_var = x_tensor.clone().float().requires_grad_(True)
    bounds = np.zeros(n_outputs)

    for j in range(n_out):
        if time_budget > 0 and time.time() > time.time() + time_budget:
            bounds[j:] = np.inf
            break
        if x_var.grad is not None:
            x_var.grad.zero_()
        out = model(x_var)
        out_flat = out.flatten()
        if j >= len(out_flat):
            break
        out_flat[j].backward(retain_graph=(j < n_out - 1))
        grad = x_var.grad.detach().flatten().float()
        # Weighted L1 norm: sum |grad_i| * eps_i
        bounds[j] = (grad.abs() * eps[:len(grad)]).sum().item() * safety_factor

    # Fill remaining outputs with inf if not computed
    if n_outputs > n_out:
        bounds[n_out:] = np.inf

    return bounds


def _sample_based_bounds(
    model: nn.Module,
    x_tensor: torch.Tensor,
    input_shape: Tuple,
    n_outputs: int,
    lb: np.ndarray,
    ub: np.ndarray,
    n_samples: int = 200,
) -> np.ndarray:
    """
    Estimate output perturbation bounds by random sampling.
    Fallback for non-differentiable models (OnnxRuntime).
    """
    x0 = x_tensor.detach().flatten().numpy()
    nominal = model(x_tensor).detach().flatten().numpy()

    max_dev = np.zeros(n_outputs)
    for _ in range(n_samples):
        x_rand = np.random.uniform(lb, ub).astype(np.float32)
        try:
            x_t = torch.tensor(x_rand).reshape(input_shape)
            out = model(x_t).detach().flatten().numpy()
            dev = np.abs(out[:n_outputs] - nominal[:n_outputs])
            max_dev = np.maximum(max_dev, dev)
        except Exception:
            pass

    # Add safety margin (sampling underestimates the true max)
    return max_dev * 1.5


def _check_unsafe_excluded(
    constraint: Dict,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
) -> Tuple[bool, float]:
    """
    Check if the UNSAFE region (described by the constraint) is
    excluded by the output bounds.

    VNN-COMP semantics: the constraint describes the unsafe region.
    If the output bounds make the constraint UNSATISFIABLE, the
    property is VERIFIED (safe).

    Returns (excluded, margin).
      excluded=True  → constraint cannot be satisfied → SAFE
      excluded=False → constraint might be satisfiable → unsafe possible
    """
    ctype = constraint["type"]

    if ctype == "output_bound":
        var = constraint["var"]
        op = constraint["op"]
        bound = constraint["bound"]
        if var >= len(output_lower):
            return False, -np.inf
        if op == ">=":
            # Unsafe if Y_var >= bound.  Safe if output_upper[var] < bound.
            margin = bound - output_upper[var]
            return margin > 0, margin
        else:
            # Unsafe if Y_var <= bound.  Safe if output_lower[var] > bound.
            margin = output_lower[var] - bound
            return margin > 0, margin

    elif ctype == "comparison":
        left = constraint["left"]
        right = constraint["right"]
        op = constraint["op"]
        if left >= len(output_lower) or right >= len(output_lower):
            return False, -np.inf
        if op == "<=":
            # Unsafe if Y_left <= Y_right.
            # Safe if output_lower[left] > output_upper[right] (can never have left <= right).
            margin = output_lower[left] - output_upper[right]
            return margin > 0, margin
        else:
            # Unsafe if Y_left >= Y_right.
            # Safe if output_upper[left] < output_lower[right].
            margin = output_lower[right] - output_upper[left]
            return margin > 0, margin

    elif ctype == "disjunction":
        # OR of clauses: unsafe region excluded iff ALL clauses excluded
        min_margin = np.inf
        all_excluded = True
        for clause in constraint["clauses"]:
            clause_excluded = False
            clause_margin = -np.inf
            for atom in clause:
                if "right" in atom:
                    sub_c = {"type": "comparison", **atom}
                else:
                    sub_c = {"type": "output_bound", **atom}
                excl, m = _check_unsafe_excluded(sub_c, output_lower, output_upper)
                if excl:
                    # One atom in the AND-clause is excluded → whole clause excluded
                    clause_excluded = True
                    clause_margin = max(clause_margin, m)
                    break
                clause_margin = max(clause_margin, m)
            if not clause_excluded:
                all_excluded = False
            min_margin = min(min_margin, clause_margin)
        return all_excluded, min_margin

    return False, -np.inf


def _check_constraint(
    constraint: Dict,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    output_nominal: np.ndarray,
) -> Tuple[bool, float]:
    """
    Check if a constraint is certified under the given output bounds.
    Returns (satisfied, margin).
    """
    ctype = constraint["type"]

    if ctype == "output_bound":
        var = constraint["var"]
        op = constraint["op"]
        bound = constraint["bound"]
        if var >= len(output_lower):
            return False, -np.inf
        if op == ">=":
            # Need Y_var >= bound  ->  output_lower[var] >= bound
            margin = output_lower[var] - bound
            return margin >= 0, margin
        else:
            # Need Y_var <= bound  ->  output_upper[var] <= bound
            margin = bound - output_upper[var]
            return margin >= 0, margin

    elif ctype == "comparison":
        left = constraint["left"]
        right = constraint["right"]
        op = constraint["op"]
        if left >= len(output_lower) or right >= len(output_lower):
            return False, -np.inf
        if op == "<=":
            # Y_left <= Y_right  ->  output_upper[left] <= output_lower[right]
            margin = output_lower[right] - output_upper[left]
            return margin >= 0, margin
        else:
            # Y_left >= Y_right  ->  output_lower[left] >= output_upper[right]
            margin = output_lower[left] - output_upper[right]
            return margin >= 0, margin

    elif ctype == "disjunction":
        # OR of conjunctive clauses: at least one clause must be satisfiable
        best_margin = -np.inf
        for clause in constraint["clauses"]:
            clause_ok = True
            clause_min = np.inf
            for atom in clause:
                if "right" in atom:
                    sub_c = {"type": "comparison", **atom}
                else:
                    sub_c = {"type": "output_bound", **atom}
                sat, m = _check_constraint(sub_c, output_lower, output_upper, output_nominal)
                clause_min = min(clause_min, m)
                if not sat:
                    clause_ok = False
            if clause_ok:
                return True, clause_min
            best_margin = max(best_margin, clause_min)
        return False, best_margin

    return False, -np.inf


def _check_constraint_point(
    constraint: Dict,
    output: np.ndarray,
) -> Tuple[bool, float]:
    """Check if a constraint holds at a specific output point."""
    ctype = constraint["type"]

    if ctype == "output_bound":
        var = constraint["var"]
        if var >= len(output):
            return False, -np.inf
        if constraint["op"] == ">=":
            return output[var] >= constraint["bound"], output[var] - constraint["bound"]
        else:
            return output[var] <= constraint["bound"], constraint["bound"] - output[var]

    elif ctype == "comparison":
        l, r = constraint["left"], constraint["right"]
        if l >= len(output) or r >= len(output):
            return False, -np.inf
        if constraint["op"] == "<=":
            return output[l] <= output[r], output[r] - output[l]
        else:
            return output[l] >= output[r], output[l] - output[r]

    elif ctype == "disjunction":
        for clause in constraint["clauses"]:
            clause_ok = True
            for atom in clause:
                if "right" in atom:
                    sub_c = {"type": "comparison", **atom}
                else:
                    sub_c = {"type": "output_bound", **atom}
                sat, _ = _check_constraint_point(sub_c, output)
                if not sat:
                    clause_ok = False
                    break
            if clause_ok:
                return True, 0.0
        return False, -np.inf

    return False, -np.inf


# ------------------------------------------------------------------
# Z3 solver wrapper
# ------------------------------------------------------------------

def _verify_with_z3_wrapper(
    instance: BenchmarkInstance,
    timeout: Optional[float] = None,
    n_workers: int = 0,
    threads_per_worker: int = 0,
) -> BenchmarkVerificationResult:
    """Dispatch verification to the Z3-based multi-core solver."""
    from pathlib import Path
    from .z3_solver import verify_with_z3

    res = BenchmarkVerificationResult(
        benchmark=instance.benchmark_name,
        model_name=Path(instance.model_path).stem,
        property_name=Path(instance.property_path).stem,
    )

    if instance.property is None:
        res.result = "error"
        res.details = "Property not loaded"
        return res
    if not Path(instance.model_path).exists():
        res.result = "error"
        res.details = "Model file not found"
        return res

    max_time = timeout or instance.timeout or 300.0
    z3_result = verify_with_z3(
        onnx_path=instance.model_path,
        property=instance.property,
        timeout=max_time,
        input_shape=instance.input_shape,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
    )

    res.result = z3_result["result"]
    res.time_seconds = z3_result["time_seconds"]
    res.method = "z3"
    res.details = z3_result.get("details", "")

    if "counterexample_input" in z3_result:
        res.details += f" | CEX input: {z3_result['counterexample_input'][:5]}"

    return res


def _verify_with_smt_portfolio(
    instance: BenchmarkInstance,
    timeout: Optional[float] = None,
    method: str = "smt",
    n_workers: int = 0,
    threads_per_worker: int = 0,
) -> BenchmarkVerificationResult:
    """Dispatch to multi-solver SMT portfolio."""
    from pathlib import Path
    from .smt_solver import verify_with_smt

    res = BenchmarkVerificationResult(
        benchmark=instance.benchmark_name,
        model_name=Path(instance.model_path).stem,
        property_name=Path(instance.property_path).stem,
    )

    if instance.property is None:
        res.result, res.details = "error", "Property not loaded"
        return res
    if not Path(instance.model_path).exists():
        res.result, res.details = "error", "Model file not found"
        return res

    max_time = timeout or instance.timeout or 300.0
    solver_map = {
        "smt": None, "portfolio": None,
        "cvc5": ["cvc5"], "bitwuzla": ["bitwuzla"], "opensmt": ["opensmt"],
    }
    solvers = solver_map.get(method)

    smt_result = verify_with_smt(
        onnx_path=instance.model_path,
        property=instance.property,
        timeout=max_time,
        solvers=solvers,
        total_cores=n_workers or 0,
        save_formula=True,
        benchmark_name=instance.benchmark_name,
        instance_name=Path(instance.property_path).stem,
    )

    res.result = smt_result["result"]
    res.time_seconds = smt_result["time_seconds"]
    res.method = smt_result.get("solver", method)
    res.details = smt_result.get("details", "")
    return res


def _verify_with_gurobi(
    instance: BenchmarkInstance,
    timeout: Optional[float] = None,
    n_workers: int = 0,
) -> BenchmarkVerificationResult:
    """Dispatch to Gurobi MILP solver."""
    from pathlib import Path
    from .gurobi_solver import verify_with_gurobi

    res = BenchmarkVerificationResult(
        benchmark=instance.benchmark_name,
        model_name=Path(instance.model_path).stem,
        property_name=Path(instance.property_path).stem,
    )
    if instance.property is None:
        res.result, res.details = "error", "Property not loaded"
        return res
    if not Path(instance.model_path).exists():
        res.result, res.details = "error", "Model file not found"
        return res

    result = verify_with_gurobi(
        onnx_path=instance.model_path,
        property=instance.property,
        timeout=timeout or instance.timeout or 300.0,
        total_cores=n_workers or 0,
        save_lp=True,
        benchmark_name=instance.benchmark_name,
        instance_name=Path(instance.property_path).stem,
    )
    res.result = result["result"]
    res.time_seconds = result["time_seconds"]
    res.method = "gurobi"
    res.details = result.get("details", "")
    return res
