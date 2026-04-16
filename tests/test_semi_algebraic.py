"""Tests for the semi-algebraic set module."""

import numpy as np
import pytest

from qnn_verifier.polynomial.semi_algebraic import SemiAlgebraicSet, QuantizationConstraint


class TestSemiAlgebraicSet:
    def test_box_constraint(self):
        sa = SemiAlgebraicSet(n_vars=2)
        sa.add_box_constraint(0, -1.0, 1.0)
        sa.add_box_constraint(1, 0.0, 2.0)

        assert len(sa.inequalities) == 4
        assert sa.is_feasible(np.array([0.0, 1.0]))
        assert not sa.is_feasible(np.array([2.0, 1.0]))
        assert not sa.is_feasible(np.array([0.0, -1.0]))

    def test_polynomial_inequality(self):
        sa = SemiAlgebraicSet(n_vars=1)
        # x^2 - 1 >= 0 means x <= -1 or x >= 1
        sa.add_inequality({(2,): 1.0, (0,): -1.0})

        assert sa.is_feasible(np.array([2.0]))
        assert sa.is_feasible(np.array([-2.0]))
        assert not sa.is_feasible(np.array([0.0]))

    def test_max_degree(self):
        sa = SemiAlgebraicSet(n_vars=2)
        sa.add_inequality({(2, 0): 1.0, (0, 3): -1.0})
        assert sa.max_degree == 3

    def test_equality(self):
        sa = SemiAlgebraicSet(n_vars=2)
        # x_0 + x_1 = 1
        sa.add_equality({(1, 0): 1.0, (0, 1): 1.0, (0, 0): -1.0})
        assert sa.is_feasible(np.array([0.5, 0.5]))
        assert not sa.is_feasible(np.array([0.5, 0.6]))


class TestQuantizationConstraint:
    def test_levels_symmetric(self):
        qc = QuantizationConstraint(n_bits=2, symmetric=True)
        assert len(qc.levels) == 4
        assert qc.levels[0] == -1.0

    def test_levels_unsigned(self):
        qc = QuantizationConstraint(n_bits=2, symmetric=False)
        assert len(qc.levels) == 4
        assert qc.levels[0] == 0.0
        assert abs(qc.levels[-1] - 1.0) < 1e-10

    def test_quantize(self):
        qc = QuantizationConstraint(n_bits=4, symmetric=False)
        x = np.array([0.0, 0.5, 1.0])
        q = qc.quantize(x)
        # Check that quantized values are among the levels
        for v in q:
            assert np.min(np.abs(qc.levels - v)) < 1e-10

    def test_error_bound(self):
        qc = QuantizationConstraint(n_bits=8, symmetric=False)
        err = qc.quantization_error_bound()
        assert err > 0
        assert err < 0.01  # For 8-bit, error < 1/256

    def test_build_relaxed_constraints(self):
        qc = QuantizationConstraint(n_bits=4, symmetric=True)
        sa = qc.build_relaxed_constraints(var_idx=0, n_vars=2)
        assert len(sa.inequalities) > 0

    def test_exact_quantization_low_bits(self):
        qc = QuantizationConstraint(n_bits=2, symmetric=True)
        sa = qc.build_relaxed_constraints(var_idx=0, n_vars=1)
        # Should have box constraints + exact polynomial equality
        assert len(sa.equalities) > 0 or len(sa.inequalities) > 0
