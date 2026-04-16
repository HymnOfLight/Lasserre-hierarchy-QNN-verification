"""Tests for the activation envelope module."""

import numpy as np
import pytest

from qnn_verifier.polynomial.activation_envelope import ActivationEnvelope


class TestActivationEnvelope:
    def test_relu_envelope(self):
        env = ActivationEnvelope(activation_type="relu", degree=4)
        result = env.build_envelope((-2.0, 2.0))
        assert "upper_coeffs" in result
        assert "lower_coeffs" in result

    def test_sigmoid_envelope(self):
        env = ActivationEnvelope(activation_type="sigmoid", degree=4)
        result = env.build_envelope((-3.0, 3.0))
        assert result["upper_coeffs"] is not None
        assert result["lower_coeffs"] is not None

    def test_envelope_bounds_relu(self):
        env = ActivationEnvelope(activation_type="relu", degree=4)
        env.build_envelope((-2.0, 2.0), margin=0.01)

        x = np.linspace(-2.0, 2.0, 100)
        f_true = np.maximum(x, 0.0)
        f_upper = env.evaluate_upper(x)
        f_lower = env.evaluate_lower(x)

        assert np.all(f_upper >= f_true - 1e-6)
        assert np.all(f_lower <= f_true + 1e-6)

    def test_envelope_bounds_sigmoid(self):
        env = ActivationEnvelope(activation_type="sigmoid", degree=6)
        env.build_envelope((-3.0, 3.0), margin=0.01)

        x = np.linspace(-3.0, 3.0, 100)
        f_true = 1.0 / (1.0 + np.exp(-x))
        f_upper = env.evaluate_upper(x)
        f_lower = env.evaluate_lower(x)

        assert np.all(f_upper >= f_true - 1e-6)
        assert np.all(f_lower <= f_true + 1e-6)

    def test_tightness_metrics(self):
        env = ActivationEnvelope(activation_type="relu", degree=4)
        env.build_envelope((-2.0, 2.0))
        metrics = env.tightness_metrics()

        assert "envelope_area" in metrics
        assert "linear_area" in metrics
        assert "tightness_ratio" in metrics
        assert metrics["tightness_ratio"] >= 0
        assert metrics["tightness_ratio"] <= 2.0

    def test_polynomial_constraints(self):
        env = ActivationEnvelope(activation_type="relu", degree=4)
        env.build_envelope((-1.0, 1.0))
        constraints = env.get_polynomial_constraints()

        assert len(constraints) == 2
        assert constraints[0]["type"] == "upper_bound"
        assert constraints[1]["type"] == "lower_bound"

    def test_quantized_relu_envelope(self):
        env = ActivationEnvelope(activation_type="relu", degree=4, n_bits=8)
        env.build_envelope((-1.0, 1.0))
        assert env._upper_approx is not None
        assert env._lower_approx is not None
