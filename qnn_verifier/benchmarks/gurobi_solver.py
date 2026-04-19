"""
Gurobi MILP-based exact neural network verification.

Encodes the NN as a Mixed-Integer Linear Program (MILP) using
the Big-M formulation for ReLU neurons:
    y >= x,  y >= 0,  y <= x - M(1-z),  y <= M*z
where z is a binary indicator variable.

Gurobi is highly optimised for MILP and uses all available cores
natively via its thread pool.

Generated .lp files are saved to ./gurobi_lp/<benchmark>/<instance>.lp
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LP_DIR = _PROJECT_ROOT / "gurobi_lp"


def _extract_weights_from_onnx(onnx_path: str) -> List[Dict]:
    """Reuse the ONNX weight extractor from smt_solver."""
    from .smt_solver import _extract_weights_from_onnx as extract
    return extract(onnx_path)


def _ibp_bounds(layers, lb, ub):
    """Run IBP to get per-neuron pre-activation bounds for Big-M."""
    all_bounds = []
    cl, cu = lb.copy(), ub.copy()
    for layer in layers:
        if layer["type"] == "linear":
            W, b = layer["W"], layer["b"]
            ni = W.shape[1]
            li = cl[:ni] if len(cl) >= ni else np.pad(cl, (0, ni - len(cl)))
            ui = cu[:ni] if len(cu) >= ni else np.pad(cu, (0, ni - len(cu)))
            Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
            cl = Wp @ li + Wn @ ui + b
            cu = Wp @ ui + Wn @ li + b
            all_bounds.append({"type": "linear", "lb": cl.copy(), "ub": cu.copy()})
        elif layer["type"] == "relu":
            all_bounds.append({"type": "relu", "pre_lb": cl.copy(), "pre_ub": cu.copy()})
            cl = np.maximum(cl, 0)
            cu = np.maximum(cu, 0)
        elif layer["type"] == "flatten":
            pass
    return all_bounds


def verify_with_gurobi(
    onnx_path: str,
    property,
    timeout: float = 3600.0,
    total_cores: int = 0,
    save_lp: bool = True,
    benchmark_name: str = "",
    instance_name: str = "",
) -> Dict:
    """
    Verify a neural network property using Gurobi MILP.

    The ReLU network is encoded as:
      linear layers → equality constraints
      ReLU neurons  → Big-M with binary indicator variables
      VNNLIB output constraints → linear constraints

    Gurobi uses all cores via its internal thread pool.

    Args:
        onnx_path: Path to ONNX model.
        property: Parsed VNNLIBProperty.
        timeout: Solver timeout in seconds.
        total_cores: CPU cores (0=auto).
        save_lp: Save the .lp file locally.
        benchmark_name: For file naming.
        instance_name: For file naming.
    """
    import gurobipy as gp
    from gurobipy import GRB

    t0 = time.time()
    n_cores = total_cores or os.cpu_count() or 4

    layers = _extract_weights_from_onnx(onnx_path)
    if not any(l["type"] == "linear" for l in layers):
        return {"result": "error", "solver": "gurobi", "time_seconds": 0,
                "details": "No linear layers in ONNX"}

    n_inputs = property.n_inputs
    n_outputs = property.n_outputs

    lb_in = np.where(np.isfinite(property.input_lower), property.input_lower, -1e6)
    ub_in = np.where(np.isfinite(property.input_upper), property.input_upper, 1e6)

    # Prolog-style symbolic rewrite pre-processing
    from .symbolic_rewrite import symbolic_rewrite_preprocess
    layers, lb_in, ub_in, stable_from_rewrite, out_constraints, rw_stats = \
        symbolic_rewrite_preprocess(layers, lb_in, ub_in, property.output_constraints, n_outputs)

    if rw_stats.early_unsat:
        lp_path = ""
        if save_lp:
            d = DEFAULT_LP_DIR / (benchmark_name or "unknown")
            d.mkdir(parents=True, exist_ok=True)
            lp_path = str(d / f"{instance_name or 'instance'}_UNSAT.lp")
            with open(lp_path, "w") as f:
                f.write("\\ UNSAT by symbolic rewrite pre-processing\n")
        return {"result": "verified", "solver": "symbolic_rewrite+gurobi",
                "time_seconds": rw_stats.time_seconds, "lp_file": lp_path,
                "details": f"UNSAT by symbolic rewrite | {rw_stats.summary()}"}

    ibp = _ibp_bounds(layers, lb_in, ub_in)

    # Build Gurobi model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", n_cores)
    env.setParam("TimeLimit", timeout)
    env.start()

    model = gp.Model("nn_verify", env=env)

    # Input variables
    x_in = model.addMVar(n_inputs, lb=lb_in, ub=ub_in, name="x")

    # Encode layers
    current_vars = x_in
    relu_layer_idx = 0
    ibp_idx = 0
    n_binary = 0

    for layer in layers:
        if layer["type"] == "linear":
            W = layer["W"]
            b = layer["b"]
            n_out, n_in = W.shape

            cv_len = current_vars.shape[0] if hasattr(current_vars, 'shape') else len(current_vars)
            cv = current_vars[:n_in] if cv_len >= n_in else current_vars

            # Get IBP bounds for this layer's output
            ibp_entry = ibp[ibp_idx] if ibp_idx < len(ibp) else None
            ibp_idx += 1

            if ibp_entry and ibp_entry["type"] == "linear":
                out_lb = ibp_entry["lb"]
                out_ub = ibp_entry["ub"]
            else:
                out_lb = np.full(n_out, -1e6)
                out_ub = np.full(n_out, 1e6)

            h = model.addMVar(n_out, lb=out_lb, ub=out_ub, name=f"h_{ibp_idx}")
            # h = W @ cv + b
            cv_dim = cv.shape[0] if hasattr(cv, 'shape') else len(cv)
            W_trunc = W[:, :cv_dim]
            model.addConstr(h == W_trunc @ cv + b, name=f"linear_{ibp_idx}")
            current_vars = h

        elif layer["type"] == "relu":
            n = current_vars.shape[0] if hasattr(current_vars, 'shape') else len(current_vars)

            ibp_entry = ibp[ibp_idx] if ibp_idx < len(ibp) else None
            ibp_idx += 1

            pre_lb = ibp_entry["pre_lb"] if ibp_entry else np.full(n, -1e6)
            pre_ub = ibp_entry["pre_ub"] if ibp_entry else np.full(n, 1e6)

            relu_out_ub = np.maximum(pre_ub, 0)
            r = model.addMVar(n, lb=0.0, ub=relu_out_ub, name=f"r_{relu_layer_idx}")

            for j in range(n):
                lb_j = float(pre_lb[j])
                ub_j = float(pre_ub[j])

                if lb_j >= 0:
                    # Active: r = x
                    model.addConstr(r[j] == current_vars[j],
                                    name=f"relu_{relu_layer_idx}_{j}_active")
                elif ub_j <= 0:
                    # Inactive: r = 0
                    model.addConstr(r[j] == 0,
                                    name=f"relu_{relu_layer_idx}_{j}_inactive")
                else:
                    # Unstable: Big-M with binary indicator
                    M_pos = float(ub_j)
                    M_neg = float(-lb_j)
                    z = model.addVar(vtype=GRB.BINARY,
                                     name=f"z_{relu_layer_idx}_{j}")
                    n_binary += 1
                    # r >= x
                    model.addConstr(r[j] >= current_vars[j],
                                    name=f"relu_{relu_layer_idx}_{j}_ge")
                    # r >= 0 (already by lb)
                    # r <= x + M_neg * (1 - z)  →  r <= x + M_neg - M_neg*z
                    model.addConstr(r[j] <= current_vars[j] + M_neg * (1 - z),
                                    name=f"relu_{relu_layer_idx}_{j}_ub1")
                    # r <= M_pos * z
                    model.addConstr(r[j] <= M_pos * z,
                                    name=f"relu_{relu_layer_idx}_{j}_ub2")

            current_vars = r
            relu_layer_idx += 1

        elif layer["type"] == "flatten":
            pass

    # Output variables
    cv_final_len = current_vars.shape[0] if hasattr(current_vars, 'shape') else len(current_vars)
    n_out_actual = min(n_outputs, cv_final_len)
    y = model.addMVar(n_out_actual, lb=-GRB.INFINITY, name="y")
    model.addConstr(y == current_vars[:n_out_actual], name="output")

    # Output constraints (VNNLIB unsafe region)
    _add_gurobi_output_constraints(model, y, property.output_constraints, n_out_actual)

    # Dummy objective (feasibility problem)
    model.setObjective(0, GRB.MINIMIZE)
    model.update()

    logger.info(
        f"Gurobi MILP: {model.NumVars} vars ({n_binary} binary), "
        f"{model.NumConstrs} constraints, {n_cores} threads, timeout={timeout}s"
    )

    # Save LP file
    lp_path = ""
    if save_lp:
        d = DEFAULT_LP_DIR / (benchmark_name or "unknown")
        d.mkdir(parents=True, exist_ok=True)
        lp_path = str(d / f"{instance_name or 'instance'}.lp")
        model.write(lp_path)
        logger.info(f"Gurobi LP saved: {lp_path}")

    # Solve
    model.optimize()
    elapsed = time.time() - t0

    if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
        return {
            "result": "violated",
            "solver": "gurobi",
            "time_seconds": elapsed,
            "lp_file": lp_path,
            "details": f"SAT (feasible) by Gurobi MILP | {n_binary} binary vars | "
                       f"file: {lp_path}",
        }
    elif model.status == GRB.INFEASIBLE:
        return {
            "result": "verified",
            "solver": "gurobi",
            "time_seconds": elapsed,
            "lp_file": lp_path,
            "details": f"UNSAT (infeasible) by Gurobi MILP — exact proof | "
                       f"{n_binary} binary vars | file: {lp_path}",
        }
    elif model.status == GRB.TIME_LIMIT:
        has_solution = model.SolCount > 0
        if has_solution:
            return {
                "result": "violated",
                "solver": "gurobi",
                "time_seconds": elapsed,
                "lp_file": lp_path,
                "details": f"SAT (solution found before timeout) | {n_binary} binary | "
                           f"file: {lp_path}",
            }
        return {
            "result": "unknown",
            "solver": "gurobi",
            "time_seconds": elapsed,
            "lp_file": lp_path,
            "details": f"Timeout ({timeout}s) | {n_binary} binary vars | file: {lp_path}",
        }
    else:
        return {
            "result": "unknown",
            "solver": "gurobi",
            "time_seconds": elapsed,
            "lp_file": lp_path,
            "details": f"Gurobi status={model.status} | {n_binary} binary | file: {lp_path}",
        }


def _add_gurobi_output_constraints(model, y, constraints, n_out):
    """Add VNNLIB output constraints to Gurobi model."""
    import gurobipy as gp
    from gurobipy import GRB

    for c in constraints:
        ctype = c["type"]
        if ctype == "output_bound":
            v = c["var"]
            if v >= n_out:
                continue
            if c["op"] == ">=":
                model.addConstr(y[v] >= c["bound"], name=f"out_bound_{v}")
            else:
                model.addConstr(y[v] <= c["bound"], name=f"out_bound_{v}")

        elif ctype == "comparison":
            l, r = c["left"], c["right"]
            if l >= n_out or r >= n_out:
                continue
            if c["op"] == "<=":
                model.addConstr(y[l] <= y[r], name=f"out_cmp_{l}_{r}")
            else:
                model.addConstr(y[l] >= y[r], name=f"out_cmp_{l}_{r}")

        elif ctype == "disjunction":
            # OR: at least one clause must hold → add indicator constraints
            n_clauses = len(c["clauses"])
            clause_vars = model.addMVar(n_clauses, vtype=GRB.BINARY,
                                        name="or_clause")
            # At least one clause active
            model.addConstr(clause_vars.sum() >= 1, name="or_at_least_one")

            for ci, clause in enumerate(c["clauses"]):
                for ai, atom in enumerate(clause):
                    if "right" in atom:
                        l, r = atom["left"], atom["right"]
                        if l >= n_out or r >= n_out:
                            continue
                        if atom["op"] == "<=":
                            # y[l] <= y[r] when clause active
                            M_big = 1e6
                            model.addConstr(
                                y[l] - y[r] <= M_big * (1 - clause_vars[ci]),
                                name=f"or_{ci}_{ai}_cmp"
                            )
                        else:
                            M_big = 1e6
                            model.addConstr(
                                y[r] - y[l] <= M_big * (1 - clause_vars[ci]),
                                name=f"or_{ci}_{ai}_cmp"
                            )
                    elif "bound" in atom:
                        v = atom["var"]
                        if v >= n_out:
                            continue
                        M_big = 1e6
                        if atom["op"] == ">=":
                            model.addConstr(
                                atom["bound"] - y[v] <= M_big * (1 - clause_vars[ci]),
                                name=f"or_{ci}_{ai}_bnd"
                            )
                        else:
                            model.addConstr(
                                y[v] - atom["bound"] <= M_big * (1 - clause_vars[ci]),
                                name=f"or_{ci}_{ai}_bnd"
                            )


def verify_pytorch_with_gurobi(
    model_nn,
    x0: np.ndarray,
    epsilon: float,
    true_label: int,
    target_label: Optional[int] = None,
    input_shape: Optional[Tuple] = None,
    n_classes: int = 10,
    timeout: float = 3600.0,
    total_cores: int = 0,
    model_name: str = "resnet",
) -> Dict:
    """
    Verify a PyTorch model using Gurobi MILP on the last linear layer.
    Same approach as verify_pytorch_with_smt: encode only the fc layer
    with Jacobian-bounded hidden state inputs.
    """
    import torch
    import gurobipy as gp
    from gurobipy import GRB
    from .smt_solver import _extract_last_layer_from_pytorch, _compute_hidden_perturbation

    t0 = time.time()
    n_cores = total_cores or os.cpu_count() or 4

    x0_flat = x0.flatten().astype(np.float32)
    if input_shape is None:
        input_shape = (1,) + tuple(x0.shape)
    x0_tensor = torch.tensor(x0_flat.reshape(input_shape)).float()

    extracted = _extract_last_layer_from_pytorch(model_nn, x0_tensor, input_shape, n_classes)
    if extracted is None:
        return {"result": "error", "solver": "gurobi", "time_seconds": time.time() - t0,
                "details": "Could not extract final linear layer"}

    W, b, h_nominal = extracted
    n_hidden = W.shape[1]

    logger.info(f"Computing Jacobian bounds (dim={n_hidden})...")
    h_delta = _compute_hidden_perturbation(model_nn, x0_tensor, n_hidden, epsilon)
    h_lb = (h_nominal - h_delta).astype(np.float64)
    h_ub = (h_nominal + h_delta).astype(np.float64)

    n_out = min(n_classes, W.shape[0])

    # Build Gurobi LP (purely linear — no binary vars for last layer)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", n_cores)
    env.setParam("TimeLimit", timeout)
    env.start()

    m = gp.Model("resnet_verify", env=env)
    h = m.addMVar(n_hidden, lb=h_lb, ub=h_ub, name="h")
    y = m.addMVar(n_out, lb=-GRB.INFINITY, name="y")

    m.addConstr(y == W[:n_out, :] @ h + b[:n_out], name="fc")

    # Output constraint: unsafe if any target >= true_label
    targets = [target_label] if target_label is not None else [i for i in range(n_out) if i != true_label]
    clauses = []
    for t in targets:
        clauses.append([{"op": ">=", "left": t, "right": true_label}])
    _add_gurobi_output_constraints(m, y, [{"type": "disjunction", "clauses": clauses}], n_out)

    m.setObjective(0, GRB.MINIMIZE)
    m.update()

    # Save LP
    d = DEFAULT_LP_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    lp_path = str(d / f"{model_name}_eps{epsilon}_label{true_label}.lp")
    m.write(lp_path)
    logger.info(f"Gurobi LP saved: {lp_path} ({n_hidden} vars, {m.NumConstrs} constraints)")

    m.optimize()
    elapsed = time.time() - t0

    if m.status == GRB.INFEASIBLE:
        return {"result": "verified", "solver": "gurobi", "time_seconds": elapsed,
                "lp_file": lp_path,
                "details": f"INFEASIBLE — exact proof | {n_hidden} vars | file: {lp_path}"}
    elif m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or m.SolCount > 0:
        return {"result": "violated", "solver": "gurobi", "time_seconds": elapsed,
                "lp_file": lp_path,
                "details": f"FEASIBLE — adversarial region reachable | file: {lp_path}"}
    elif m.status == GRB.TIME_LIMIT:
        return {"result": "unknown", "solver": "gurobi", "time_seconds": elapsed,
                "lp_file": lp_path,
                "details": f"Timeout ({timeout}s) | file: {lp_path}"}
    else:
        return {"result": "unknown", "solver": "gurobi", "time_seconds": elapsed,
                "lp_file": lp_path,
                "details": f"Gurobi status={m.status} | file: {lp_path}"}
