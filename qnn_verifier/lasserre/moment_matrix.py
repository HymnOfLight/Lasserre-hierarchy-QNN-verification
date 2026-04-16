"""
Moment matrix construction for the Lasserre hierarchy.

Builds the moment/localizing matrices required for the SOS-SDP relaxation
of polynomial optimization problems arising in neural network verification.
"""

import numpy as np
from itertools import product as iter_product
from typing import Dict, List, Optional, Tuple
from functools import lru_cache


class MonomialBasis:
    """Manages monomial multi-index generation and ordering."""

    def __init__(self, n_vars: int, max_degree: int):
        self.n_vars = n_vars
        self.max_degree = max_degree
        self._monomials = None

    @property
    def monomials(self) -> List[Tuple[int, ...]]:
        if self._monomials is None:
            self._monomials = self._generate_monomials()
        return self._monomials

    def _generate_monomials(self) -> List[Tuple[int, ...]]:
        """Generate all monomials of total degree <= max_degree in graded
        lexicographic order."""
        result = []
        self._gen_recursive([], 0, self.max_degree, result)
        result.sort(key=lambda m: (sum(m), m))
        return result

    def _gen_recursive(
        self, current: list, var_idx: int, remaining_degree: int, result: list
    ):
        if var_idx == self.n_vars:
            result.append(tuple(current + [0] * (self.n_vars - len(current))))
            return
        for d in range(remaining_degree + 1):
            self._gen_recursive(
                current + [d], var_idx + 1, remaining_degree - d, result
            )

    @property
    def size(self) -> int:
        return len(self.monomials)

    def index_of(self, mono: Tuple[int, ...]) -> int:
        """Get the index of a monomial in the ordered basis."""
        return self.monomials.index(mono)

    def multiply(
        self, m1: Tuple[int, ...], m2: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        return tuple(a + b for a, b in zip(m1, m2))


class MomentMatrix:
    """
    Constructs the moment matrix M_d(y) for the Lasserre hierarchy at
    relaxation order d.

    The moment matrix is indexed by monomials alpha, beta of degree <= d,
    with entries y_{alpha+beta} where y is the moment sequence.
    """

    def __init__(self, n_vars: int, order: int):
        self.n_vars = n_vars
        self.order = order  # relaxation order d
        self.basis = MonomialBasis(n_vars, order)
        self._moment_basis = MonomialBasis(n_vars, 2 * order)

    @property
    def matrix_size(self) -> int:
        return self.basis.size

    @property
    def n_moments(self) -> int:
        return self._moment_basis.size

    def get_moment_indices(self) -> List[Tuple[int, ...]]:
        """All moment multi-indices up to degree 2*order."""
        return self._moment_basis.monomials

    def build_matrix_pattern(self) -> Dict[Tuple[int, int], Tuple[int, ...]]:
        """
        Returns a mapping from (i, j) matrix position to the moment
        multi-index y_{alpha_i + alpha_j}.
        """
        basis = self.basis.monomials
        pattern = {}
        for i, alpha in enumerate(basis):
            for j, beta in enumerate(basis):
                combined = tuple(a + b for a, b in zip(alpha, beta))
                pattern[(i, j)] = combined
        return pattern

    def build_numeric_matrix(
        self, moment_values: Dict[Tuple[int, ...], float]
    ) -> np.ndarray:
        """
        Given a dictionary of moment values y_alpha, construct the
        numeric moment matrix.
        """
        n = self.matrix_size
        M = np.zeros((n, n))
        basis = self.basis.monomials

        for i in range(n):
            for j in range(i, n):
                combined = tuple(
                    a + b for a, b in zip(basis[i], basis[j])
                )
                val = moment_values.get(combined, 0.0)
                M[i, j] = val
                M[j, i] = val
        return M

    def build_localizing_matrix(
        self,
        constraint_poly: Dict[Tuple[int, ...], float],
        moment_values: Dict[Tuple[int, ...], float],
    ) -> np.ndarray:
        """
        Build the localizing matrix M_d(g * y) for a constraint g(x) >= 0.

        The constraint polynomial g is given as {multi-index: coefficient}.
        Localizing matrix order is d - ceil(deg(g)/2).
        """
        g_degree = max(sum(m) for m in constraint_poly.keys()) if constraint_poly else 0
        loc_order = self.order - int(np.ceil(g_degree / 2.0))

        if loc_order < 0:
            return np.array([[0.0]])

        loc_basis = MonomialBasis(self.n_vars, loc_order)
        n = loc_basis.size
        M = np.zeros((n, n))

        for i, alpha in enumerate(loc_basis.monomials):
            for j, beta in enumerate(loc_basis.monomials):
                ab = tuple(a + b for a, b in zip(alpha, beta))
                val = 0.0
                for gamma, coeff in constraint_poly.items():
                    combined = tuple(a + b for a, b in zip(ab, gamma))
                    val += coeff * moment_values.get(combined, 0.0)
                M[i, j] = val
                M[j, i] = val

        return M

    def get_objective_linear_map(
        self, obj_poly: Dict[Tuple[int, ...], float]
    ) -> Dict[Tuple[int, ...], float]:
        """
        Express the objective function as a linear function of moments.
        obj = sum_alpha c_alpha * y_alpha
        """
        return dict(obj_poly)

    def sparsity_ratio(self) -> float:
        """
        Fraction of moment matrix entries that are distinct moments.
        Lower means more structure to exploit.
        """
        pattern = self.build_matrix_pattern()
        total = len(pattern)
        distinct = len(set(pattern.values()))
        return distinct / total if total > 0 else 1.0
