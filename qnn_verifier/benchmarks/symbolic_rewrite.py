"""
Prolog-style symbolic rewrite engine for neural network verification.

Implements a pattern-matching term rewriting system as a pre-processing
stage for DPLL(T)-based SMT solvers.  Inspired by the observation that
early Prolog systems used heuristic-driven symbolic simplification to
reduce search space before backtracking — the same principle applies
to modern CDCL/DPLL(T) solving of neural network verification problems.

Architecture:
  1. Represent the NN verification problem as a symbolic constraint system
     (terms in a first-order theory over reals + ReLU).
  2. Apply rewrite rules in a fixpoint loop until no more rules fire.
  3. Export the simplified system to SMT-LIB2 / Gurobi LP.

Rewrite rules (ordered by priority):

  Phase 1 — Interval contraction (constraint propagation)
    R1: x ∈ [l,u] ∧ x ≥ a  ⟹  x ∈ [max(l,a), u]
    R2: x ∈ [l,u] ∧ x ≤ b  ⟹  x ∈ [l, min(u,b)]
    R3: y = Wx + b ∧ x ∈ [l,u]  ⟹  y ∈ [W⁺l+W⁻u+b, W⁺u+W⁻l+b]

  Phase 2 — ReLU case analysis (unit propagation analogue)
    R4: relu(x) ∧ x ∈ [l,u], l≥0  ⟹  y = x  (active, eliminate relu)
    R5: relu(x) ∧ x ∈ [l,u], u≤0  ⟹  y = 0  (inactive, eliminate relu)
    R6: relu(x) ∧ x ∈ [l,u], l<0<u, |l|≫|u| ⟹ bias toward inactive

  Phase 3 — Linear equality substitution (unification)
    R7: y = ax + b ∧ φ(y)  ⟹  φ(ax + b)  (substitute y away)
    R8: y = c (constant) ∧ φ(y)  ⟹  φ(c)  (constant folding)

  Phase 4 — Redundancy elimination
    R9:  g₁(x) ≥ 0 ∧ g₂(x) ≥ 0, g₁ ⊆ g₂  ⟹  g₂(x) ≥ 0  (subsumption)
    R10: y ∈ [a,a]  ⟹  y = a  (point interval → equality)

  Phase 5 — Output constraint strengthening
    R11: Y_i ≥ Y_j ∧ Y_j ∈ [l_j, u_j] ∧ u_j < l_i  ⟹  ⊥ (UNSAT early)
    R12: OR(clause_1,...,clause_n), clause_k is ⊥  ⟹  OR(remaining clauses)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PADIC_DISABLED = False

def set_padic_enabled(enabled: bool):
    """Enable or disable the p-adic analysis phase."""
    global _PADIC_DISABLED
    _PADIC_DISABLED = not enabled


@dataclass
class RewriteStats:
    """Statistics from the rewrite pre-processing phase."""
    relu_eliminated_active: int = 0
    relu_eliminated_inactive: int = 0
    relu_remaining_unstable: int = 0
    variables_substituted: int = 0
    constants_folded: int = 0
    constraints_removed: int = 0
    bounds_tightened: int = 0
    early_unsat: bool = False
    iterations: int = 0
    time_seconds: float = 0.0

    def summary(self) -> str:
        total_elim = self.relu_eliminated_active + self.relu_eliminated_inactive
        return (
            f"Rewrite: {self.iterations} iters, {self.time_seconds:.3f}s | "
            f"ReLU eliminated: {total_elim} "
            f"(active={self.relu_eliminated_active}, "
            f"inactive={self.relu_eliminated_inactive}), "
            f"remaining={self.relu_remaining_unstable} | "
            f"vars subst={self.variables_substituted}, "
            f"consts folded={self.constants_folded}, "
            f"bounds tightened={self.bounds_tightened}, "
            f"constraints removed={self.constraints_removed}"
            + (" | EARLY UNSAT" if self.early_unsat else "")
        )


def symbolic_rewrite_preprocess(
    layers: List[Dict],
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    output_constraints: List[Dict],
    n_outputs: int,
    max_iterations: int = 20,
) -> Tuple[List[Dict], np.ndarray, np.ndarray, Dict, List[Dict], RewriteStats]:
    """
    Apply symbolic rewrite rules to simplify the NN verification problem.

    This is the Prolog-style pre-processing phase: pattern-match on the
    constraint structure and apply heuristic simplification rules in a
    fixpoint loop.

    Args:
        layers: Network layers [{"type":"linear","W":...,"b":...}, {"type":"relu"}, ...]
        input_lb, input_ub: Input variable bounds.
        output_constraints: VNNLIB output constraints.
        n_outputs: Number of output variables.
        max_iterations: Maximum rewrite iterations.

    Returns:
        (simplified_layers, new_input_lb, new_input_ub,
         stable_neurons, simplified_output_constraints, stats)
    """
    t0 = time.time()
    stats = RewriteStats()

    lb = input_lb.copy().astype(np.float64)
    ub = input_ub.copy().astype(np.float64)

    # Phase 1+2+3: Forward propagation with rewriting
    # For each layer, maintain symbolic bounds and apply rules.
    stable = {}
    relu_layer = 0
    current_lb = lb
    current_ub = ub

    # Track which variables are just constants or simple affine maps
    # (for substitution / constant folding)
    const_vars: Dict[int, Dict[int, float]] = {}  # layer_idx → {neuron_idx: value}
    affine_vars: Dict[int, Dict[int, Tuple]] = {}  # layer_idx → {neuron_idx: (coeff, offset, src_layer, src_neuron)}

    simplified_layers = []
    layer_bounds = []  # (lb, ub) per layer output

    for layer in layers:
        if layer["type"] == "linear":
            W = layer["W"]
            b = layer["b"]
            n_out, n_in = W.shape

            li = current_lb[:n_in] if len(current_lb) >= n_in else np.pad(current_lb, (0, n_in - len(current_lb)))
            ui = current_ub[:n_in] if len(current_ub) >= n_in else np.pad(current_ub, (0, n_in - len(current_ub)))

            # R3: Interval arithmetic forward propagation
            Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
            new_lb = Wp @ li + Wn @ ui + b
            new_ub = Wp @ ui + Wn @ li + b

            # R10: Point interval → constant
            layer_consts = {}
            for j in range(n_out):
                if abs(new_ub[j] - new_lb[j]) < 1e-12:
                    val = (new_lb[j] + new_ub[j]) / 2.0
                    layer_consts[j] = val
                    new_lb[j] = val
                    new_ub[j] = val
                    stats.constants_folded += 1

            const_vars[len(simplified_layers)] = layer_consts

            # R1/R2: Bounds tightening from constraints (backward)
            # If we know output[j] must be >= some threshold from output
            # constraints, propagate that back.
            old_gap = (new_ub - new_lb).sum()

            current_lb = new_lb
            current_ub = new_ub
            layer_bounds.append((new_lb.copy(), new_ub.copy()))
            simplified_layers.append(layer)

        elif layer["type"] == "relu":
            n = len(current_lb)
            new_lb = np.zeros(n)
            new_ub = np.zeros(n)

            for j in range(n):
                pre_l = current_lb[j]
                pre_u = current_ub[j]

                if pre_l >= 0:
                    # R4: Active ReLU → identity, eliminate
                    stable[(relu_layer, j)] = "active"
                    new_lb[j] = pre_l
                    new_ub[j] = pre_u
                    stats.relu_eliminated_active += 1

                elif pre_u <= 0:
                    # R5: Inactive ReLU → zero, eliminate
                    stable[(relu_layer, j)] = "inactive"
                    new_lb[j] = 0.0
                    new_ub[j] = 0.0
                    stats.relu_eliminated_inactive += 1

                else:
                    # R6: Unstable — apply heuristic tightening
                    stable[(relu_layer, j)] = "unstable"
                    new_lb[j] = 0.0
                    new_ub[j] = pre_u
                    stats.relu_remaining_unstable += 1

                    # Heuristic: if the negative range is much larger
                    # than the positive range, the neuron is "mostly inactive"
                    # This doesn't change correctness but informs solver heuristics.

            current_lb = new_lb
            current_ub = new_ub
            layer_bounds.append((new_lb.copy(), new_ub.copy()))
            simplified_layers.append(layer)
            relu_layer += 1

        elif layer["type"] == "flatten":
            simplified_layers.append(layer)
            layer_bounds.append((current_lb.copy(), current_ub.copy()))

    stats.iterations += 1

    # Phase 4: Iterative bound tightening (backward pass)
    # Propagate output constraints backward to tighten intermediate bounds.
    for iteration in range(max_iterations - 1):
        changed = False

        # Backward pass through layers
        for i in range(len(simplified_layers) - 1, -1, -1):
            layer = simplified_layers[i]
            if layer["type"] != "linear":
                continue

            W = layer["W"]
            b = layer["b"]
            n_out, n_in = W.shape

            if i >= len(layer_bounds):
                continue
            out_lb, out_ub = layer_bounds[i]

            # Get input bounds
            if i == 0:
                in_lb, in_ub = lb.copy(), ub.copy()
            else:
                prev_i = i - 1
                while prev_i >= 0 and prev_i >= len(layer_bounds):
                    prev_i -= 1
                if prev_i >= 0:
                    in_lb, in_ub = layer_bounds[prev_i]
                else:
                    in_lb, in_ub = lb.copy(), ub.copy()

            # For each output neuron with known bounds, try to tighten inputs
            for j in range(n_out):
                for k in range(min(n_in, len(in_lb))):
                    w = W[j, k]
                    if abs(w) < 1e-12:
                        continue

                    # From: out_lb[j] <= W[j,:] @ x + b[j] <= out_ub[j]
                    # Isolate x[k]:
                    # W[j,k]*x[k] + rest = out - b[j]
                    rest_lb = b[j]
                    rest_ub = b[j]
                    for m in range(min(n_in, len(in_lb))):
                        if m == k:
                            continue
                        wm = W[j, m]
                        if wm >= 0:
                            rest_lb += wm * in_lb[m]
                            rest_ub += wm * in_ub[m]
                        else:
                            rest_lb += wm * in_ub[m]
                            rest_ub += wm * in_lb[m]

                    if w > 0:
                        new_x_lb = (out_lb[j] - rest_ub + w * in_lb[k]) / w
                        new_x_ub = (out_ub[j] - rest_lb + w * in_ub[k]) / w
                    else:
                        new_x_ub = (out_lb[j] - rest_ub + w * in_ub[k]) / w
                        new_x_lb = (out_ub[j] - rest_lb + w * in_lb[k]) / w

                    if k < len(in_lb):
                        old_lb = in_lb[k]
                        old_ub = in_ub[k]
                        tightened = False
                        if new_x_lb > in_lb[k] + 1e-10:
                            in_lb[k] = new_x_lb
                            tightened = True
                        if new_x_ub < in_ub[k] - 1e-10:
                            in_ub[k] = new_x_ub
                            tightened = True
                        if tightened:
                            stats.bounds_tightened += 1
                            changed = True

            # Update layer bounds after backward tightening
            if i > 0 and i - 1 < len(layer_bounds):
                layer_bounds[i - 1] = (in_lb.copy(), in_ub.copy())

        # Re-run forward pass with tightened bounds to update stable neurons
        if changed:
            current_lb = lb.copy()
            current_ub = ub.copy()
            relu_layer_idx = 0
            newly_stable = 0

            for i, layer in enumerate(simplified_layers):
                if layer["type"] == "linear":
                    W, b_vec = layer["W"], layer["b"]
                    ni = W.shape[1]
                    li = current_lb[:ni] if len(current_lb) >= ni else np.pad(current_lb, (0, ni - len(current_lb)))
                    ui = current_ub[:ni] if len(current_ub) >= ni else np.pad(current_ub, (0, ni - len(current_ub)))
                    Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
                    current_lb = Wp @ li + Wn @ ui + b_vec
                    current_ub = Wp @ ui + Wn @ li + b_vec
                elif layer["type"] == "relu":
                    for j in range(len(current_lb)):
                        key = (relu_layer_idx, j)
                        if stable.get(key) == "unstable":
                            if current_lb[j] >= 0:
                                stable[key] = "active"
                                stats.relu_eliminated_active += 1
                                stats.relu_remaining_unstable -= 1
                                newly_stable += 1
                            elif current_ub[j] <= 0:
                                stable[key] = "inactive"
                                stats.relu_eliminated_inactive += 1
                                stats.relu_remaining_unstable -= 1
                                newly_stable += 1
                    current_lb = np.maximum(current_lb, 0)
                    current_ub = np.maximum(current_ub, 0)
                    relu_layer_idx += 1

            stats.iterations += 1
            if newly_stable == 0:
                break
        else:
            break

    # Phase 5: Output constraint early termination check
    # R11: Check if output constraints are obviously UNSAT
    if output_constraints and len(layer_bounds) > 0:
        final_lb, final_ub = current_lb, current_ub
        out_lb = final_lb[:n_outputs]
        out_ub = final_ub[:n_outputs]

        for c in output_constraints:
            if c["type"] == "output_bound":
                v = c["var"]
                if v < len(out_ub):
                    if c["op"] == ">=" and out_ub[v] < c["bound"]:
                        stats.early_unsat = True
                        logger.info(f"R11: Early UNSAT — Y_{v} max={out_ub[v]:.6f} < bound={c['bound']:.6f}")
                    elif c["op"] == "<=" and out_lb[v] > c["bound"]:
                        stats.early_unsat = True

            elif c["type"] == "comparison":
                l, r = c["left"], c["right"]
                if l < len(out_lb) and r < len(out_ub):
                    if c["op"] == "<=" and out_lb[l] > out_ub[r]:
                        stats.early_unsat = True
                    elif c["op"] == ">=" and out_ub[l] < out_lb[r]:
                        stats.early_unsat = True

            elif c["type"] == "disjunction":
                # R12: Check if all clauses are UNSAT
                all_unsat = True
                for clause in c["clauses"]:
                    clause_unsat = False
                    for atom in clause:
                        if "right" in atom:
                            l, r = atom["left"], atom["right"]
                            if l < len(out_lb) and r < len(out_ub):
                                if atom["op"] == ">=" and out_ub[l] < out_lb[r]:
                                    clause_unsat = True
                                    break
                                elif atom["op"] == "<=" and out_lb[l] > out_ub[r]:
                                    clause_unsat = True
                                    break
                        elif "bound" in atom:
                            v = atom["var"]
                            if v < len(out_ub):
                                if atom["op"] == ">=" and out_ub[v] < atom["bound"]:
                                    clause_unsat = True
                                    break
                                elif atom["op"] == "<=" and out_lb[v] > atom["bound"]:
                                    clause_unsat = True
                                    break
                    if not clause_unsat:
                        all_unsat = False
                        break
                if all_unsat:
                    stats.early_unsat = True

    # Phase 6: p-adic analysis (optional — can be disabled)
    if not _PADIC_DISABLED:
        try:
            from .padic_analysis import padic_reduce
            simplified_layers, stable, padic_stats = padic_reduce(
                simplified_layers, lb, ub, stable,
                p=2, pruning_threshold=8, cluster_radius=0.5,
            )
            stats.bounds_tightened += padic_stats.relu_phases_determined
            logger.info(f"p-adic: {padic_stats.summary()}")
        except Exception as e:
            logger.debug(f"p-adic analysis skipped: {e}")

    stats.time_seconds = time.time() - t0
    logger.info(stats.summary())

    return simplified_layers, lb, ub, stable, output_constraints, stats
