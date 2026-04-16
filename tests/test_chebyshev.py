"""Tests for the Chebyshev polynomial approximation module."""

import numpy as np
import pytest

from qnn_verifier.polynomial.chebyshev import (
    ChebyshevApproximator,
    approximate_relu,
    approximate_sigmoid,
    approximate_quantized_relu,
)


class TestChebyshevApproximator:
    def test_fit_polynomial(self):
        approx = ChebyshevApproximator(degree=3, interval=(-1.0, 1.0))
        coeffs = approx.fit(lambda x: x ** 2)
        assert len(coeffs) == 4

    def test_evaluate_quadratic(self):
        approx = ChebyshevApproximator(degree=4, interval=(-1.0, 1.0))
        approx.fit(lambda x: x ** 2)
        x = np.array([0.0, 0.5, -0.5, 1.0])
        y = approx.evaluate(x)
        expected = x ** 2
        np.testing.assert_allclose(y, expected, atol=1e-10)

    def test_evaluate_linear(self):
        approx = ChebyshevApproximator(degree=2, interval=(0.0, 1.0))
        approx.fit(lambda x: 2 * x + 1)
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y = approx.evaluate(x)
        np.testing.assert_allclose(y, 2 * x + 1, atol=1e-10)

    def test_upper_bound(self):
        approx = ChebyshevApproximator(degree=4, interval=(-2.0, 2.0))
        func = lambda x: np.maximum(x, 0.0)
        approx.upper_bound_coefficients(func, margin=0.01)

        x_test = np.linspace(-2.0, 2.0, 100)
        f_true = func(x_test)
        f_upper = approx.evaluate(x_test)
        assert np.all(f_upper >= f_true - 1e-10)

    def test_lower_bound(self):
        approx = ChebyshevApproximator(degree=4, interval=(-2.0, 2.0))
        func = lambda x: np.maximum(x, 0.0)
        approx.lower_bound_coefficients(func, margin=0.01)

        x_test = np.linspace(-2.0, 2.0, 100)
        f_true = func(x_test)
        f_lower = approx.evaluate(x_test)
        assert np.all(f_lower <= f_true + 1e-10)

    def test_standard_coefficients(self):
        approx = ChebyshevApproximator(degree=3, interval=(-1.0, 1.0))
        approx.fit(lambda x: x ** 2)
        mono = approx.get_polynomial_coefficients_standard()
        assert len(mono) == 4
        # For x^2, mono should be approximately [0, 0, 1, 0]
        assert abs(mono[2] - 1.0) < 1e-6
        assert abs(mono[0]) < 1e-6
        assert abs(mono[1]) < 1e-6

    def test_approximation_error(self):
        approx = ChebyshevApproximator(degree=6, interval=(-2.0, 2.0))
        approx.fit(lambda x: np.maximum(x, 0.0))
        max_err, rms_err = approx.approximation_error(
            lambda x: np.maximum(x, 0.0)
        )
        assert max_err < 0.5  # Reasonable for degree 6 on [-2, 2]
        assert rms_err < max_err


class TestConvenienceFunctions:
    def test_approximate_relu(self):
        approx = approximate_relu((-2.0, 2.0), degree=4)
        x = np.array([1.0, -1.0])
        y = approx.evaluate(x)
        max_err, _ = approx.approximation_error(lambda x: np.maximum(x, 0.0))
        assert max_err < 1.0

    def test_approximate_sigmoid(self):
        approx = approximate_sigmoid((-3.0, 3.0), degree=6)
        max_err, _ = approx.approximation_error(
            lambda x: 1.0 / (1.0 + np.exp(-x))
        )
        assert max_err < 0.1

    def test_approximate_quantized_relu(self):
        approx = approximate_quantized_relu((-2.0, 2.0), n_bits=8, degree=4)
        assert approx._coefficients is not None

    def test_higher_degree_improves(self):
        func = lambda x: np.maximum(x, 0.0)
        interval = (-2.0, 2.0)
        errors = []
        for deg in [2, 4, 6, 8]:
            approx = ChebyshevApproximator(degree=deg, interval=interval)
            approx.fit(func)
            max_err, _ = approx.approximation_error(func)
            errors.append(max_err)
        # Generally non-increasing (higher degree should be at least as good)
        # Due to Runge-like effects and non-smoothness, may not be strict
        assert errors[-1] <= errors[0] * 1.1  # Allow small tolerance
