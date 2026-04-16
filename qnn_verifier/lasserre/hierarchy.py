"""
Lasserre hierarchy controller.

Manages the progressive tightening of SOS/SDP relaxations through
increasing hierarchy levels, providing asymptotically convergent
global lower bounds for the verification problem.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from .sos_relaxation import SOSRelaxation
from .sdp_solver import SDPSolver
from .moment_matrix import MomentMatrix

logger = logging.getLogger(__name__)


class LasserreHierarchy:
    """
    Implements the Lasserre hierarchy for polynomial optimization:

    At each level d, solve a tighter SDP relaxation to get an improving
    sequence of lower bounds:
        LB_1 <= LB_2 <= ... <= LB_d <= ... <= OPT

    Convergence is guaranteed for compact semi-algebraic feasible sets.
    """

    def __init__(
        self,
        n_vars: int,
        max_order: int = 3,
        solver_name: str = "SCS",
        verbose: bool = False,
        convergence_tol: float = 1e-4,
    ):
        self.n_vars = n_vars
        self.max_order = max_order
        self.solver = SDPSolver(solver=solver_name, verbose=verbose)
        self.convergence_tol = convergence_tol

        self.objective: Dict[Tuple[int, ...], float] = {}
        self.inequalities: List[Dict[Tuple[int, ...], float]] = []
        self.equalities: List[Dict[Tuple[int, ...], float]] = []

        self.results: List[Dict] = []

    def set_objective(self, poly: Dict[Tuple[int, ...], float]):
        self.objective = poly

    def add_inequality(self, poly: Dict[Tuple[int, ...], float]):
        self.inequalities.append(poly)

    def add_equality(self, poly: Dict[Tuple[int, ...], float]):
        self.equalities.append(poly)

    def add_box_constraints(self, bounds: List[Tuple[float, float]]):
        """Add box constraints for each variable."""
        for i, (lb, ub) in enumerate(bounds):
            ei = tuple(1 if j == i else 0 for j in range(self.n_vars))
            zero = tuple(0 for _ in range(self.n_vars))
            self.add_inequality({ei: 1.0, zero: -lb})
            self.add_inequality({ei: -1.0, zero: ub})

    def minimum_feasible_order(self) -> int:
        """
        Compute the minimum relaxation order needed based on constraint degrees.
        """
        max_constraint_deg = 0
        for g in self.inequalities + self.equalities:
            if g:
                deg = max(sum(m) for m in g.keys())
                max_constraint_deg = max(max_constraint_deg, deg)

        obj_deg = 0
        if self.objective:
            obj_deg = max(sum(m) for m in self.objective.keys())

        return max(1, int(np.ceil(max(max_constraint_deg, obj_deg) / 2.0)))

    def solve_at_order(self, order: int) -> Dict:
        """Solve the SDP relaxation at a specific order."""
        logger.info(f"Solving Lasserre hierarchy at order {order}")

        sos = SOSRelaxation(self.n_vars, order)
        sos.set_objective(self.objective)

        for g in self.inequalities:
            sos.add_inequality(g)
        for h in self.equalities:
            sos.add_equality(h)

        sdp_data = sos.get_sdp_data()
        logger.info(
            f"SDP problem: {sdp_data['n_moments']} moments, "
            f"moment matrix {sdp_data['moment_matrix_size']}x{sdp_data['moment_matrix_size']}"
        )

        result = self.solver.solve(sdp_data)
        result["order"] = order
        result["problem_info"] = sos.problem_size_info
        return result

    def solve_adaptive(
        self,
        start_order: Optional[int] = None,
        target_bound: Optional[float] = None,
    ) -> Dict:
        """
        Solve with adaptive order selection.

        Starts at the minimum feasible order and increases until:
        1. Convergence (bound improvement < tolerance), or
        2. Max order reached, or
        3. Target bound achieved (for verification: bound > 0 means safe).
        """
        if start_order is None:
            start_order = self.minimum_feasible_order()

        self.results = []
        best_bound = -np.inf

        for order in range(start_order, self.max_order + 1):
            try:
                result = self.solve_at_order(order)
            except Exception as e:
                logger.warning(f"Order {order} failed: {e}")
                result = {"lower_bound": -np.inf, "status": "error", "order": order}

            self.results.append(result)
            current_bound = result.get("lower_bound", -np.inf)

            if np.isfinite(current_bound):
                improvement = current_bound - best_bound
                best_bound = max(best_bound, current_bound)

                logger.info(
                    f"Order {order}: bound = {current_bound:.6f}, "
                    f"improvement = {improvement:.6f}"
                )

                if target_bound is not None and current_bound >= target_bound:
                    logger.info(f"Target bound {target_bound} achieved!")
                    break

                if order > start_order and abs(improvement) < self.convergence_tol:
                    logger.info("Converged (improvement below tolerance)")
                    break

        return {
            "best_bound": best_bound,
            "best_order": self.results[-1]["order"] if self.results else start_order,
            "n_levels_solved": len(self.results),
            "history": self.results,
            "converged": len(self.results) > 1
            and abs(self.results[-1].get("lower_bound", -np.inf) - best_bound) < self.convergence_tol,
        }

    def get_verification_certificate(self) -> Dict:
        """
        Extract a verification certificate from the hierarchy solution.
        If the best lower bound is > 0 for the adversarial margin objective,
        this constitutes a formal proof of robustness.
        """
        if not self.results:
            return {"certified": False, "reason": "No solution computed"}

        best = max(self.results, key=lambda r: r.get("lower_bound", -np.inf))
        bound = best.get("lower_bound", -np.inf)

        return {
            "certified": bound > 0,
            "lower_bound": bound,
            "order": best.get("order", -1),
            "solver_status": best.get("status", "unknown"),
            "reason": "lower_bound > 0 implies robustness"
            if bound > 0
            else "lower_bound <= 0, cannot certify",
        }
