"""
SDP solver interface for the Lasserre hierarchy relaxation.

Wraps CVXPY to solve the semidefinite programming problems arising
from the SOS relaxation, extracting global optimal lower bounds
that serve as deterministic safety certificates.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SDPSolver:
    """
    Solves the SDP relaxation of the polynomial optimization problem
    formulated by the SOSRelaxation module.
    """

    def __init__(
        self,
        solver: str = "SCS",
        verbose: bool = False,
        max_iters: int = 10000,
        eps: float = 1e-6,
    ):
        self.solver_name = solver
        self.verbose = verbose
        self.max_iters = max_iters
        self.eps = eps

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

        # Decision variable: moment vector y
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

        # Objective: minimize c^T y
        objective = cp.Minimize(obj_vec @ y)

        problem = cp.Problem(objective, constraints)

        solver_kwargs = {
            "verbose": self.verbose,
            "max_iters": self.max_iters,
        }

        if self.solver_name == "SCS":
            solver_kwargs["eps"] = self.eps
            solver = cp.SCS
        elif self.solver_name == "MOSEK":
            solver = cp.MOSEK
            solver_kwargs = {"verbose": self.verbose}
        else:
            solver = cp.SCS
            solver_kwargs["eps"] = self.eps

        try:
            problem.solve(solver=solver, **solver_kwargs)
        except Exception as e:
            logger.error(f"SDP solver failed: {e}")
            return {
                "lower_bound": -np.inf,
                "status": "error",
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
        }

    def solve_layer_subproblem(
        self,
        n_vars: int,
        objective_poly: Dict[Tuple[int, ...], float],
        constraints: List[Dict[Tuple[int, ...], float]],
        order: int = 1,
    ) -> Dict:
        """
        Convenience method for solving a small sub-problem arising
        from a single layer or neuron verification.

        Uses a lightweight SDP formulation when the problem is small enough.
        """
        from .sos_relaxation import SOSRelaxation

        sos = SOSRelaxation(n_vars, order)
        sos.set_objective(objective_poly)
        for g in constraints:
            sos.add_inequality(g)

        sdp_data = sos.get_sdp_data()
        return self.solve(sdp_data)
