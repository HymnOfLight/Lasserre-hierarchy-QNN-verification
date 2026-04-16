"""
Chebyshev polynomial approximation for non-smooth activation functions.

Provides minimax-optimal polynomial approximations on bounded intervals,
overcoming the locality limitations of Taylor expansion for quantized
activation functions with step-like discrete characteristics.
"""

import numpy as np
from typing import Callable, Optional, Tuple
from functools import lru_cache


class ChebyshevApproximator:
    """
    Constructs Chebyshev polynomial envelopes for activation functions
    on bounded intervals [a, b].

    Uses Chebyshev interpolation and minimax approximation to build
    tight polynomial upper/lower bounds that cover the input-output
    relationship of non-linear units.
    """

    def __init__(self, degree: int = 4, interval: Tuple[float, float] = (-1.0, 1.0)):
        self.degree = degree
        self.a, self.b = interval
        self._nodes = None
        self._coefficients = None

    @property
    def nodes(self) -> np.ndarray:
        """Chebyshev nodes on [-1, 1] mapped to [a, b]."""
        if self._nodes is None:
            k = np.arange(self.degree + 1)
            nodes_std = np.cos((2 * k + 1) * np.pi / (2 * (self.degree + 1)))
            self._nodes = 0.5 * (self.b - self.a) * nodes_std + 0.5 * (self.a + self.b)
        return self._nodes

    def _to_standard(self, x: np.ndarray) -> np.ndarray:
        """Map from [a, b] to [-1, 1]."""
        return (2.0 * x - (self.a + self.b)) / (self.b - self.a)

    def fit(self, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Compute Chebyshev coefficients for the given function using
        discrete cosine transform on Chebyshev nodes.
        """
        f_vals = func(self.nodes)
        n = self.degree + 1
        coeffs = np.zeros(n)
        nodes_std = self._to_standard(self.nodes)

        for j in range(n):
            T_j = np.cos(j * np.arccos(nodes_std))
            coeffs[j] = (2.0 / n) * np.sum(f_vals * T_j)
        coeffs[0] /= 2.0

        self._coefficients = coeffs
        return coeffs

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Chebyshev polynomial at given points."""
        if self._coefficients is None:
            raise RuntimeError("Must call fit() before evaluate()")

        t = self._to_standard(np.asarray(x, dtype=np.float64))
        n = len(self._coefficients)

        if n == 1:
            return np.full_like(t, self._coefficients[0])

        # Clenshaw recurrence
        b_next = np.zeros_like(t)
        b_curr = np.zeros_like(t)

        for j in range(n - 1, 0, -1):
            b_prev = 2.0 * t * b_curr - b_next + self._coefficients[j]
            b_next = b_curr
            b_curr = b_prev

        return self._coefficients[0] + t * b_curr - b_next

    def upper_bound_coefficients(
        self, func: Callable, n_test: int = 1000, margin: float = 0.0
    ) -> np.ndarray:
        """
        Compute coefficients for an upper-bounding polynomial.
        Shifts the approximation upward so it lies entirely above func
        on [a, b] plus a safety margin.
        """
        coeffs = self.fit(func).copy()
        x_test = np.linspace(self.a, self.b, n_test)
        f_true = func(x_test)
        f_approx = self.evaluate(x_test)
        max_deficit = np.max(f_true - f_approx)
        if max_deficit > 0:
            coeffs[0] += max_deficit + margin
        else:
            coeffs[0] += margin
        self._coefficients = coeffs
        return coeffs

    def lower_bound_coefficients(
        self, func: Callable, n_test: int = 1000, margin: float = 0.0
    ) -> np.ndarray:
        """
        Compute coefficients for a lower-bounding polynomial.
        """
        coeffs = self.fit(func).copy()
        x_test = np.linspace(self.a, self.b, n_test)
        f_true = func(x_test)
        f_approx = self.evaluate(x_test)
        max_excess = np.max(f_approx - f_true)
        if max_excess > 0:
            coeffs[0] -= max_excess + margin
        else:
            coeffs[0] -= margin
        self._coefficients = coeffs
        return coeffs

    def get_polynomial_coefficients_standard(self) -> np.ndarray:
        """
        Convert Chebyshev coefficients to standard polynomial coefficients
        (monomial basis) on the original interval [a, b].

        Returns array c where p(x) = c[0] + c[1]*x + c[2]*x^2 + ...
        """
        if self._coefficients is None:
            raise RuntimeError("Must call fit() before conversion")

        n = len(self._coefficients)
        # First build Chebyshev-to-monomial on [-1,1]
        T = np.zeros((n, n))
        T[0, 0] = 1.0
        if n > 1:
            T[1, 1] = 1.0
        for k in range(2, n):
            T[k, 1:] += 2.0 * T[k - 1, :-1]
            T[k, :] -= T[k - 2, :]

        mono_std = np.zeros(n)
        for k in range(n):
            mono_std += self._coefficients[k] * T[k]

        # Transform from t-domain [-1,1] to x-domain [a,b]
        # t = (2x - (a+b))/(b-a), so x = ((b-a)*t + (a+b))/2
        scale = 2.0 / (self.b - self.a)
        shift = -(self.a + self.b) / (self.b - self.a)

        mono_x = np.zeros(n)
        for k in range(n):
            # (scale*x + shift)^k expanded via binomial
            for j in range(k + 1):
                binom = _binom_coeff(k, j)
                mono_x[j] += mono_std[k] * binom * (scale ** j) * (shift ** (k - j))

        return mono_x

    def approximation_error(
        self, func: Callable, n_test: int = 1000
    ) -> Tuple[float, float]:
        """
        Compute max absolute error and RMS error of the approximation.
        """
        x_test = np.linspace(self.a, self.b, n_test)
        f_true = func(x_test)
        f_approx = self.evaluate(x_test)
        err = np.abs(f_true - f_approx)
        return float(np.max(err)), float(np.sqrt(np.mean(err ** 2)))


def _binom_coeff(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    result = 1.0
    for i in range(min(k, n - k)):
        result = result * (n - i) / (i + 1)
    return result


def approximate_relu(interval: Tuple[float, float], degree: int = 4) -> ChebyshevApproximator:
    """Convenience: Chebyshev approximation of ReLU on a given interval."""
    approx = ChebyshevApproximator(degree=degree, interval=interval)
    approx.fit(lambda x: np.maximum(x, 0.0))
    return approx


def approximate_sigmoid(interval: Tuple[float, float], degree: int = 4) -> ChebyshevApproximator:
    """Convenience: Chebyshev approximation of sigmoid on a given interval."""
    approx = ChebyshevApproximator(degree=degree, interval=interval)
    approx.fit(lambda x: 1.0 / (1.0 + np.exp(-x)))
    return approx


def approximate_quantized_relu(
    interval: Tuple[float, float], n_bits: int = 8, degree: int = 4
) -> ChebyshevApproximator:
    """
    Chebyshev approximation of a quantized ReLU.
    Quantized ReLU clips to [0, 2^n_bits - 1] and rounds to integer levels.
    """
    qmax = (1 << n_bits) - 1
    scale = qmax / (interval[1] - interval[0]) if interval[1] > interval[0] else 1.0

    def quantized_relu(x):
        y = np.maximum(x, 0.0) * scale
        y = np.clip(np.round(y), 0, qmax)
        return y / scale

    approx = ChebyshevApproximator(degree=degree, interval=interval)
    approx.fit(quantized_relu)
    return approx
