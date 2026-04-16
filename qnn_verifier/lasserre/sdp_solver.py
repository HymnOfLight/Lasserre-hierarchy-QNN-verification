"""
SDP solver interface for the Lasserre hierarchy relaxation.

Uses a two-tier solver strategy:
  - LP / QP sub-problems  → Gurobi (via gurobipy native API)
  - SDP problems (PSD constraints) → CVXPY + Clarabel interior-point solver
    (Gurobi does not natively support semidefinite constraints)

Gurobi is the primary solver for all problems that do not require
PSD matrix constraints.  Clarabel replaces SCS as the SDP backend
for better accuracy and reliability.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gurobi LP / QP solver  (used for last-layer verification and bound
# tightening – the performance-critical path for large networks)
# ---------------------------------------------------------------------------

class GurobiLPSolver:
    """
    Solves LP / QP problems directly through the gurobipy API.

    Typical usage: minimise a linear objective over a polytope
    (box-constrained pre-logit space of the last classification layer).
    """

    def __init__(self, verbose: bool = False, time_limit: float = 30.0):
        self.verbose = verbose
        self.time_limit = time_limit

    def minimize_linear(
        self,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Solve:  min  c^T x
                s.t. lb <= x <= ub
                     A_eq x = b_eq  (optional)

        Returns dict with 'optimal_value', 'solution', 'status'.
        """
        import gurobipy as gp
        from gurobipy import GRB

        n = len(c)
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 1 if self.verbose else 0)
            env.start()
            model = gp.Model(env=env)
            model.setParam("TimeLimit", self.time_limit)

            x = model.addMVar(n, lb=lb.astype(np.float64),
                              ub=ub.astype(np.float64), name="x")
            model.setObjective(c.astype(np.float64) @ x, GRB.MINIMIZE)

            if A_eq is not None and b_eq is not None:
                model.addMConstr(A_eq.astype(np.float64), x, "=",
                                 b_eq.astype(np.float64))

            model.optimize()

            if model.status == GRB.OPTIMAL:
                return {
                    "optimal_value": float(model.ObjVal),
                    "solution": x.X.copy(),
                    "status": "optimal",
                }
            elif model.status == GRB.INFEASIBLE:
                return {"optimal_value": np.inf, "solution": None,
                        "status": "infeasible"}
            else:
                return {
                    "optimal_value": float(model.ObjVal)
                    if model.SolCount > 0 else -np.inf,
                    "solution": x.X.copy() if model.SolCount > 0 else None,
                    "status": f"gurobi_status_{model.status}",
                }
        except Exception as e:
            logger.error(f"Gurobi LP solve failed: {e}")
            return {"optimal_value": -np.inf, "solution": None,
                    "status": f"error: {e}"}

    def solve_margin_lp(
        self,
        diff_w: np.ndarray,
        diff_b: float,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> Dict:
        """
        Compute the margin lower bound for last-layer verification:
            min  diff_w^T h + diff_b
            s.t. lb <= h <= ub

        This is the core LP that decides certified robustness.
        """
        result = self.minimize_linear(diff_w, lb, ub)
        if result["status"] == "optimal":
            result["optimal_value"] += diff_b
        elif np.isfinite(result["optimal_value"]):
            result["optimal_value"] += diff_b
        return result


# ---------------------------------------------------------------------------
# SDP solver  (Lasserre hierarchy – uses CVXPY + Clarabel)
# ---------------------------------------------------------------------------

class SDPSolver:
    """
    Solves the SDP relaxation of the POP formulated by SOSRelaxation.

    Uses Clarabel (interior-point) as the default SDP backend.
    Falls back to SCS if Clarabel is unavailable.  Gurobi cannot be
    used here because it does not support PSD (>> 0) constraints.
    """

    def __init__(
        self,
        solver: str = "GUROBI",
        verbose: bool = False,
        max_iters: int = 10000,
        eps: float = 1e-7,
    ):
        self.solver_name = solver
        self.verbose = verbose
        self.max_iters = max_iters
        self.eps = eps

    def _pick_sdp_backend(self):
        """Select the best available SDP-capable backend for CVXPY."""
        import cvxpy as cp
        available = cp.installed_solvers()
        # Clarabel is a modern, accurate interior-point SDP solver
        if "CLARABEL" in available:
            return cp.CLARABEL
        if "MOSEK" in available:
            return cp.MOSEK
        if "SCS" in available:
            return cp.SCS
        raise RuntimeError("No SDP-capable solver found (need Clarabel, MOSEK, or SCS)")

    def solve(self, sdp_data: Dict) -> Dict:
        """
        Solve the SDP problem specified by sdp_data from SOSRelaxation.

        Returns:
            Dictionary with:
            - 'lower_bound': float, the global lower bound on the objective
            - 'status': str, solver status
            - 'moments': dict, extracted moment values
            - 'moment_matrix': np.ndarray, the optimal moment matrix
        """
        import cvxpy as cp

        n_moments = sdp_data["n_moments"]
        moment_indices = sdp_data["moment_indices"]
        obj_vec = sdp_data["objective"]
        mm_size = sdp_data["moment_matrix_size"]
        mm_pattern = sdp_data["moment_matrix_pattern"]
        moment_to_idx = sdp_data["moment_to_idx"]
        loc_matrices = sdp_data["localizing_matrices"]

        y = cp.Variable(n_moments)
        constraints = []

        # y_0 = 1 (probability measure normalization)
        zero_mono = tuple(0 for _ in range(len(moment_indices[0])))
        if zero_mono in moment_to_idx:
            constraints.append(y[moment_to_idx[zero_mono]] == 1.0)

        # Moment matrix M_d(y) >> 0
        M = cp.Variable((mm_size, mm_size), symmetric=True)
        for (i, j), mono in mm_pattern.items():
            if mono in moment_to_idx:
                if i <= j:
                    constraints.append(M[i, j] == y[moment_to_idx[mono]])
        constraints.append(M >> 0)

        # Localizing matrices M_d(g_k * y) >> 0
        for k, loc_info in enumerate(loc_matrices):
            if loc_info is None:
                continue
            loc_size = loc_info["size"]
            pattern = loc_info["pattern"]
            if loc_size <= 0:
                continue

            L_k = cp.Variable((loc_size, loc_size), symmetric=True)
            for (i, j), terms in pattern.items():
                if i <= j:
                    expr = 0
                    for moment_idx, coeff in terms.items():
                        expr = expr + coeff * y[moment_idx]
                    constraints.append(L_k[i, j] == expr)
            constraints.append(L_k >> 0)

        objective = cp.Minimize(obj_vec @ y)
        problem = cp.Problem(objective, constraints)

        sdp_backend = self._pick_sdp_backend()
        backend_name = sdp_backend.__name__ if hasattr(sdp_backend, '__name__') else str(sdp_backend)

        solver_kwargs: Dict = {"verbose": self.verbose}
        if sdp_backend == cp.SCS:
            solver_kwargs["max_iters"] = self.max_iters
            solver_kwargs["eps"] = self.eps
        elif sdp_backend == cp.CLARABEL:
            solver_kwargs["max_iter"] = self.max_iters
            solver_kwargs["tol_gap_abs"] = self.eps
            solver_kwargs["tol_gap_rel"] = self.eps

        try:
            problem.solve(solver=sdp_backend, **solver_kwargs)
        except Exception as e:
            logger.error(f"SDP solve failed with {backend_name}: {e}")
            return {
                "lower_bound": -np.inf,
                "status": f"error ({backend_name})",
                "error": str(e),
                "moments": {},
                "moment_matrix": None,
            }

        if problem.status in ("infeasible", "infeasible_inaccurate"):
            return {
                "lower_bound": np.inf,
                "status": problem.status,
                "moments": {},
                "moment_matrix": None,
            }

        moment_values = {}
        if y.value is not None:
            for mono, idx in moment_to_idx.items():
                moment_values[mono] = float(y.value[idx])

        return {
            "lower_bound": float(problem.value) if problem.value is not None else -np.inf,
            "status": problem.status,
            "moments": moment_values,
            "moment_matrix": M.value if M.value is not None else None,
            "solve_time": problem.solver_stats.solve_time if problem.solver_stats else None,
            "solver_backend": backend_name,
        }

    def solve_layer_subproblem(
        self,
        n_vars: int,
        objective_poly: Dict[Tuple[int, ...], float],
        constraints: List[Dict[Tuple[int, ...], float]],
        order: int = 1,
    ) -> Dict:
        from .sos_relaxation import SOSRelaxation
        sos = SOSRelaxation(n_vars, order)
        sos.set_objective(objective_poly)
        for g in constraints:
            sos.add_inequality(g)
        sdp_data = sos.get_sdp_data()
        return self.solve(sdp_data)
