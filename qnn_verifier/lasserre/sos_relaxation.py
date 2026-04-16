"""
Sum-of-Squares (SOS) relaxation for polynomial optimization.

Implements the Positivstellensatz-based SOS relaxation that converts
non-convex polynomial optimization problems into semidefinite programs.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .moment_matrix import MonomialBasis, MomentMatrix


class SOSRelaxation:
    """
    Formulates the SOS/SDP relaxation of a polynomial optimization problem:

        min  f(x)
        s.t. g_i(x) >= 0, i=1,...,m
             h_j(x) = 0,  j=1,...,p

    Using the Lasserre hierarchy at a given relaxation order d.
    """

    def __init__(self, n_vars: int, order: int):
        self.n_vars = n_vars
        self.order = order
        self.moment_mat = MomentMatrix(n_vars, order)

        self.objective: Dict[Tuple[int, ...], float] = {}
        self.inequality_constraints: List[Dict[Tuple[int, ...], float]] = []
        self.equality_constraints: List[Dict[Tuple[int, ...], float]] = []

    def set_objective(self, poly: Dict[Tuple[int, ...], float]):
        """Set the polynomial objective to minimize."""
        self.objective = poly

    def add_inequality(self, poly: Dict[Tuple[int, ...], float]):
        """Add g(x) >= 0 constraint."""
        self.inequality_constraints.append(poly)

    def add_equality(self, poly: Dict[Tuple[int, ...], float]):
        """Add h(x) = 0 constraint."""
        self.equality_constraints.append(poly)

    def add_box_constraint(self, var_idx: int, lb: float, ub: float):
        """Add box constraint lb <= x[var_idx] <= ub."""
        ei = tuple(1 if j == var_idx else 0 for j in range(self.n_vars))
        zero = tuple(0 for _ in range(self.n_vars))
        self.add_inequality({ei: 1.0, zero: -lb})
        self.add_inequality({ei: -1.0, zero: ub})

    def add_polynomial_bound(
        self,
        var_idx: int,
        output_var_idx: int,
        coefficients: np.ndarray,
        is_upper: bool = True,
    ):
        """
        Add polynomial bound constraint from activation envelope.
        If is_upper: p(x_in) - x_out >= 0  (output <= upper bound poly)
        If not is_upper: x_out - p(x_in) >= 0  (output >= lower bound poly)
        """
        poly: Dict[Tuple[int, ...], float] = {}

        for k, c in enumerate(coefficients):
            if abs(c) < 1e-15:
                continue
            mono = [0] * self.n_vars
            mono[var_idx] = k
            poly[tuple(mono)] = c if is_upper else -c

        out_mono = [0] * self.n_vars
        out_mono[output_var_idx] = 1
        poly[tuple(out_mono)] = poly.get(tuple(out_mono), 0.0) + (-1.0 if is_upper else 1.0)

        self.add_inequality(poly)

    def get_sdp_data(self) -> Dict:
        """
        Construct the SDP data for this relaxation.
        Returns moment indices, objective coefficients, and constraint
        specifications needed by the SDP solver.
        """
        moment_indices = self.moment_mat.get_moment_indices()
        moment_to_idx = {m: i for i, m in enumerate(moment_indices)}
        n_moments = len(moment_indices)

        # Objective: linear in moments
        obj_vec = np.zeros(n_moments)
        for mono, coeff in self.objective.items():
            if mono in moment_to_idx:
                obj_vec[moment_to_idx[mono]] += coeff

        # Moment matrix pattern
        mm_pattern = self.moment_mat.build_matrix_pattern()
        mm_size = self.moment_mat.matrix_size

        # Localizing matrix patterns for each inequality
        loc_patterns = []
        for g in self.inequality_constraints:
            g_degree = max(sum(m) for m in g.keys()) if g else 0
            loc_order = self.order - int(np.ceil(g_degree / 2.0))
            if loc_order < 0:
                loc_patterns.append(None)
                continue

            loc_basis = MonomialBasis(self.n_vars, loc_order)
            pattern = {}
            for i, alpha in enumerate(loc_basis.monomials):
                for j, beta in enumerate(loc_basis.monomials):
                    ab = tuple(a + b for a, b in zip(alpha, beta))
                    terms = {}
                    for gamma, coeff in g.items():
                        combined = tuple(a + b for a, b in zip(ab, gamma))
                        if combined in moment_to_idx:
                            terms[moment_to_idx[combined]] = coeff
                    pattern[(i, j)] = terms

            loc_patterns.append({
                "size": loc_basis.size,
                "pattern": pattern,
            })

        return {
            "n_moments": n_moments,
            "moment_indices": moment_indices,
            "objective": obj_vec,
            "moment_matrix_size": mm_size,
            "moment_matrix_pattern": mm_pattern,
            "moment_to_idx": moment_to_idx,
            "localizing_matrices": loc_patterns,
            "n_inequalities": len(self.inequality_constraints),
            "n_equalities": len(self.equality_constraints),
        }

    @property
    def problem_size_info(self) -> Dict[str, int]:
        return {
            "n_vars": self.n_vars,
            "order": self.order,
            "n_moments": self.moment_mat.n_moments,
            "moment_matrix_size": self.moment_mat.matrix_size,
            "n_inequalities": len(self.inequality_constraints),
            "n_equalities": len(self.equality_constraints),
        }
