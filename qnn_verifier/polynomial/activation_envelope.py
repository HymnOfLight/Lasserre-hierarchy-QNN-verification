"""
Activation function polynomial envelope construction.

Builds tight convex polynomial upper and lower envelopes for various
activation functions (ReLU, Sigmoid, quantized step functions) to
replace traditional linear relaxation with higher-order bounds.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from .chebyshev import ChebyshevApproximator


class ActivationEnvelope:
    """
    Manages polynomial upper and lower envelopes for an activation function,
    providing tighter-than-linear convex relaxation of the neuron's feasible
    output region.
    """

    def __init__(
        self,
        activation_type: str = "relu",
        degree: int = 4,
        n_bits: Optional[int] = None,
    ):
        self.activation_type = activation_type
        self.degree = degree
        self.n_bits = n_bits
        self._func = self._get_activation_func()

        self._upper_approx: Optional[ChebyshevApproximator] = None
        self._lower_approx: Optional[ChebyshevApproximator] = None
        self._interval: Optional[Tuple[float, float]] = None

    def _get_activation_func(self) -> Callable:
        if self.activation_type == "relu":
            if self.n_bits is not None:
                qmax = (1 << self.n_bits) - 1
                return lambda x: np.clip(np.round(np.maximum(x, 0.0) * qmax) / qmax, 0, 1)
            return lambda x: np.maximum(x, 0.0)
        elif self.activation_type == "sigmoid":
            return lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation_type == "tanh":
            return lambda x: np.tanh(x)
        elif self.activation_type == "hardswish":
            return lambda x: x * np.clip(x + 3, 0, 6) / 6.0
        else:
            raise ValueError(f"Unknown activation: {self.activation_type}")

    def build_envelope(
        self,
        interval: Tuple[float, float],
        margin: float = 1e-6,
    ) -> Dict[str, np.ndarray]:
        """
        Build upper and lower polynomial envelopes on the given interval.

        Returns dict with 'upper_coeffs' and 'lower_coeffs' in monomial basis,
        plus 'upper_chebyshev' and 'lower_chebyshev' in Chebyshev basis.
        """
        self._interval = interval

        self._upper_approx = ChebyshevApproximator(
            degree=self.degree, interval=interval
        )
        self._upper_approx.upper_bound_coefficients(self._func, margin=margin)
        upper_mono = self._upper_approx.get_polynomial_coefficients_standard()

        self._lower_approx = ChebyshevApproximator(
            degree=self.degree, interval=interval
        )
        self._lower_approx.lower_bound_coefficients(self._func, margin=margin)
        lower_mono = self._lower_approx.get_polynomial_coefficients_standard()

        return {
            "upper_coeffs": upper_mono,
            "lower_coeffs": lower_mono,
            "upper_chebyshev": self._upper_approx._coefficients.copy(),
            "lower_chebyshev": self._lower_approx._coefficients.copy(),
            "interval": interval,
        }

    def evaluate_upper(self, x: np.ndarray) -> np.ndarray:
        if self._upper_approx is None:
            raise RuntimeError("Must call build_envelope() first")
        return self._upper_approx.evaluate(x)

    def evaluate_lower(self, x: np.ndarray) -> np.ndarray:
        if self._lower_approx is None:
            raise RuntimeError("Must call build_envelope() first")
        return self._lower_approx.evaluate(x)

    def evaluate_true(self, x: np.ndarray) -> np.ndarray:
        return self._func(x)

    def tightness_metrics(self, n_test: int = 1000) -> Dict[str, float]:
        """
        Measure how tight the envelope is compared to linear relaxation.
        Returns envelope gap area, max gap, etc.
        """
        if self._interval is None:
            raise RuntimeError("Must call build_envelope() first")

        x = np.linspace(self._interval[0], self._interval[1], n_test)
        f_true = self._func(x)
        f_upper = self.evaluate_upper(x)
        f_lower = self.evaluate_lower(x)

        gap = f_upper - f_lower
        dx = (self._interval[1] - self._interval[0]) / (n_test - 1)
        area = np.sum(gap) * dx

        # Linear relaxation gap for comparison
        f_min, f_max = np.min(f_true), np.max(f_true)
        linear_area = (f_max - f_min) * (self._interval[1] - self._interval[0])

        return {
            "envelope_area": float(area),
            "linear_area": float(linear_area),
            "tightness_ratio": float(area / max(linear_area, 1e-12)),
            "max_gap": float(np.max(gap)),
            "mean_gap": float(np.mean(gap)),
        }

    def get_polynomial_constraints(self) -> List[Dict]:
        """
        Return polynomial constraints suitable for the SOS/SDP formulation.
        Each constraint is of the form: g(x) >= 0, where
          g_upper(x) = p_upper(x) - y >= 0  (y <= upper bound)
          g_lower(x) = y - p_lower(x) >= 0  (y >= lower bound)
        """
        if self._upper_approx is None or self._lower_approx is None:
            raise RuntimeError("Must call build_envelope() first")

        upper_mono = self._upper_approx.get_polynomial_coefficients_standard()
        lower_mono = self._lower_approx.get_polynomial_coefficients_standard()

        return [
            {
                "type": "upper_bound",
                "coefficients": upper_mono,
                "description": "p_upper(x) - y >= 0",
            },
            {
                "type": "lower_bound",
                "coefficients": lower_mono,
                "description": "y - p_lower(x) >= 0",
            },
        ]
