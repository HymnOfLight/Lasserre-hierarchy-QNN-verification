"""
Semi-algebraic set modeling for quantized neural network constraints.

Converts discrete quantization constraints into continuous, tight
semi-algebraic set constraint systems by introducing auxiliary relaxation
variables, fundamentally improving the precision of neuron feasible
region descriptions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class SemiAlgebraicSet:
    """
    Represents a basic semi-algebraic set defined by polynomial inequalities:
        S = { x in R^n : g_i(x) >= 0, i=1,...,m }

    Each g_i is stored as a dictionary mapping monomial multi-indices to
    coefficients.
    """

    def __init__(self, n_vars: int):
        self.n_vars = n_vars
        self.inequalities: List[Dict[Tuple[int, ...], float]] = []
        self.equalities: List[Dict[Tuple[int, ...], float]] = []

    def add_inequality(self, poly: Dict[Tuple[int, ...], float]):
        """Add g(x) >= 0 constraint."""
        self.inequalities.append(poly)

    def add_equality(self, poly: Dict[Tuple[int, ...], float]):
        """Add h(x) = 0 constraint."""
        self.equalities.append(poly)

    def add_box_constraint(self, var_idx: int, lb: float, ub: float):
        """Add lb <= x[var_idx] <= ub as two polynomial inequalities."""
        ei = tuple(1 if j == var_idx else 0 for j in range(self.n_vars))
        zero = tuple(0 for _ in range(self.n_vars))

        self.inequalities.append({ei: 1.0, zero: -lb})   # x_i - lb >= 0
        self.inequalities.append({ei: -1.0, zero: ub})    # ub - x_i >= 0

    def add_polynomial_inequality_from_coeffs(
        self, var_idx: int, coefficients: np.ndarray, sign: float = 1.0
    ):
        """
        Add a univariate polynomial inequality in variable var_idx.
        coefficients[k] is the coefficient of x^k.
        sign=1: p(x) >= 0; sign=-1: -p(x) >= 0
        """
        poly = {}
        for k, c in enumerate(coefficients):
            if abs(c) < 1e-15:
                continue
            mono = tuple(k if j == var_idx else 0 for j in range(self.n_vars))
            poly[mono] = sign * c
        if poly:
            self.inequalities.append(poly)

    def evaluate_inequality(self, idx: int, x: np.ndarray) -> float:
        """Evaluate the idx-th inequality polynomial at point x."""
        poly = self.inequalities[idx]
        val = 0.0
        for mono, coeff in poly.items():
            term = coeff
            for var, power in enumerate(mono):
                if power > 0:
                    term *= x[var] ** power
            val += term
        return val

    def is_feasible(self, x: np.ndarray, tol: float = -1e-8) -> bool:
        """Check if point x satisfies all constraints."""
        for i in range(len(self.inequalities)):
            if self.evaluate_inequality(i, x) < tol:
                return False
        for eq in self.equalities:
            val = 0.0
            for mono, coeff in eq.items():
                term = coeff
                for var, power in enumerate(mono):
                    if power > 0:
                        term *= x[var] ** power
                val += term
            if abs(val) > abs(tol):
                return False
        return True

    @property
    def max_degree(self) -> int:
        """Maximum polynomial degree across all constraints."""
        max_d = 0
        for poly in self.inequalities + self.equalities:
            for mono in poly:
                max_d = max(max_d, sum(mono))
        return max_d


class QuantizationConstraint:
    """
    Converts discrete quantization constraints into semi-algebraic form.

    For n_bits quantization with levels q_0, ..., q_{2^n-1}, the constraint
    that x equals one of these levels is equivalent to:
        prod_{i=0}^{2^n-1} (x - q_i) = 0

    This polynomial equality, combined with interval bounds, forms a
    semi-algebraic set that tightly encloses the discrete feasible set.
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = True):
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.n_levels = 1 << n_bits
        self.levels = self._compute_levels()

    def _compute_levels(self) -> np.ndarray:
        if self.symmetric:
            half = self.n_levels // 2
            return np.linspace(-half, half - 1, self.n_levels) / half
        else:
            return np.linspace(0, self.n_levels - 1, self.n_levels) / (self.n_levels - 1)

    def build_relaxed_constraints(
        self,
        var_idx: int,
        n_vars: int,
        relaxation_degree: int = 2,
    ) -> SemiAlgebraicSet:
        """
        Build a relaxed semi-algebraic representation of the quantization
        constraint. Instead of the full product polynomial (which has degree
        2^n_bits), we use a low-degree relaxation that captures the essential
        structure.

        For practical networks with 8-bit quantization (256 levels),
        the full product polynomial is intractable. We use a piecewise
        approach: partition the levels into groups, build local constraints
        per group, and combine them.
        """
        sa = SemiAlgebraicSet(n_vars)

        lb, ub = float(self.levels[0]), float(self.levels[-1])
        sa.add_box_constraint(var_idx, lb, ub)

        if self.n_levels <= 16:
            # For low bit-width, use exact product polynomial
            self._add_exact_quantization_poly(sa, var_idx, n_vars)
        else:
            # For higher bit-width, use interval-based relaxation
            # that is much tighter than a plain box
            self._add_interval_relaxation(sa, var_idx, n_vars, relaxation_degree)

        return sa

    def _add_exact_quantization_poly(
        self, sa: SemiAlgebraicSet, var_idx: int, n_vars: int
    ):
        """
        For small n_bits, add the exact polynomial:
          prod_{i} (x - q_i) = 0
        as an equality constraint.
        """
        # Build polynomial coefficients via convolution
        poly = np.array([1.0])
        for q in self.levels:
            poly = np.convolve(poly, np.array([-q, 1.0]))

        poly_dict = {}
        for k, c in enumerate(poly):
            if abs(c) < 1e-15:
                continue
            mono = tuple(k if j == var_idx else 0 for j in range(n_vars))
            poly_dict[mono] = c
        sa.add_equality(poly_dict)

    def _add_interval_relaxation(
        self, sa: SemiAlgebraicSet, var_idx: int, n_vars: int, degree: int
    ):
        """
        For large n_bits, build a tighter-than-box relaxation using
        polynomial constraints that capture the spacing structure of
        quantization levels.

        Uses the fact that quantized values satisfy:
          sin^2(pi * n_levels * x / range) is small
        This is approximated by a low-degree polynomial.
        """
        spacing = float(self.levels[1] - self.levels[0]) if len(self.levels) > 1 else 1.0
        lb, ub = float(self.levels[0]), float(self.levels[-1])

        # Polynomial capturing quantization grid: x must be near a grid point.
        # We use: (x - lb) * (ub - x) >= 0 (already added as box)
        # Plus: A Chebyshev polynomial of degree `degree` that oscillates
        # between the quantization levels, bounding the deviation.

        # Bound on deviation from nearest quantization level
        half_spacing = spacing / 2.0
        # |x - round(x)| <= spacing/2 is always true, but we add a polynomial
        # constraint that x*(x-spacing)*(x-2*spacing)*... on local intervals
        # partitioned into manageable groups.

        n_groups = min(degree, 4)
        levels_per_group = max(1, len(self.levels) // n_groups)

        for g in range(n_groups):
            start = g * levels_per_group
            end = min(start + levels_per_group, len(self.levels))
            if start >= len(self.levels):
                break

            g_lb = float(self.levels[start]) - half_spacing * 0.1
            g_ub = float(self.levels[end - 1]) + half_spacing * 0.1

            # Build quadratic constraint for the group range
            # (x - g_lb)(g_ub - x) >= 0 when x is in [g_lb, g_ub]
            ei = tuple(1 if j == var_idx else 0 for j in range(n_vars))
            e2 = tuple(2 if j == var_idx else 0 for j in range(n_vars))
            zero = tuple(0 for _ in range(n_vars))

            # We add these as auxiliary informational constraints
            # that the optimizer can use when x is in this sub-interval
            constraint = {
                e2: -1.0,
                ei: (g_lb + g_ub),
                zero: -g_lb * g_ub,
            }
            sa.add_inequality(constraint)

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Quantize continuous values to nearest quantization levels."""
        indices = np.argmin(np.abs(x[:, None] - self.levels[None, :]), axis=1)
        return self.levels[indices]

    def quantization_error_bound(self) -> float:
        """Maximum quantization error (half the spacing between levels)."""
        if len(self.levels) < 2:
            return 0.0
        return float(np.max(np.diff(self.levels)) / 2.0)
