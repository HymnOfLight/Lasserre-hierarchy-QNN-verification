"""Tests for the Lasserre hierarchy module."""

import numpy as np
import pytest

from qnn_verifier.lasserre.moment_matrix import MonomialBasis, MomentMatrix
from qnn_verifier.lasserre.sos_relaxation import SOSRelaxation
from qnn_verifier.lasserre.sdp_solver import GurobiLPSolver
from qnn_verifier.lasserre.hierarchy import LasserreHierarchy


class TestMonomialBasis:
    def test_generation(self):
        basis = MonomialBasis(n_vars=2, max_degree=2)
        monos = basis.monomials
        # For 2 vars, degree <= 2: 1, x, y, x^2, xy, y^2 -> 6
        assert len(monos) == 6
        assert (0, 0) in monos

    def test_single_var(self):
        basis = MonomialBasis(n_vars=1, max_degree=3)
        assert len(basis.monomials) == 4  # 1, x, x^2, x^3

    def test_multiply(self):
        basis = MonomialBasis(n_vars=2, max_degree=2)
        result = basis.multiply((1, 0), (0, 1))
        assert result == (1, 1)

    def test_three_vars(self):
        basis = MonomialBasis(n_vars=3, max_degree=1)
        assert len(basis.monomials) == 4  # 1, x, y, z


class TestMomentMatrix:
    def test_construction(self):
        mm = MomentMatrix(n_vars=2, order=1)
        assert mm.matrix_size == 3  # 1, x, y

    def test_pattern(self):
        mm = MomentMatrix(n_vars=1, order=1)
        pattern = mm.build_matrix_pattern()
        # 2x2 matrix, entries indexed by y_0, y_1, y_1, y_2
        assert (0, 0) in pattern
        assert pattern[(0, 0)] == (0,)  # y_0

    def test_numeric_matrix(self):
        mm = MomentMatrix(n_vars=1, order=1)
        moments = {(0,): 1.0, (1,): 0.5, (2,): 0.5}
        M = mm.build_numeric_matrix(moments)
        assert M.shape == (2, 2)
        assert M[0, 0] == 1.0  # y_0
        assert M[0, 1] == 0.5  # y_1
        assert M[1, 1] == 0.5  # y_2

    def test_localizing_matrix(self):
        mm = MomentMatrix(n_vars=1, order=2)
        # g(x) = 1 - x^2 >= 0
        g = {(0,): 1.0, (2,): -1.0}
        moments = {(0,): 1.0, (1,): 0.0, (2,): 0.33, (3,): 0.0, (4,): 0.2}
        L = mm.build_localizing_matrix(g, moments)
        # Localizing matrix should be 2x2 (order 2 - ceil(2/2) = 1)
        assert L.shape == (2, 2)

    def test_sparsity_ratio(self):
        mm = MomentMatrix(n_vars=2, order=1)
        ratio = mm.sparsity_ratio()
        assert 0 < ratio <= 1.0


class TestSOSRelaxation:
    def test_basic_setup(self):
        sos = SOSRelaxation(n_vars=1, order=1)
        sos.set_objective({(1,): 1.0})  # minimize x
        sos.add_inequality({(0,): 1.0, (1,): 1.0})   # x + 1 >= 0
        sos.add_inequality({(0,): 1.0, (1,): -1.0})  # 1 - x >= 0

        data = sos.get_sdp_data()
        assert data["n_moments"] > 0
        assert data["moment_matrix_size"] > 0

    def test_problem_size(self):
        sos = SOSRelaxation(n_vars=2, order=2)
        info = sos.problem_size_info
        assert info["n_vars"] == 2
        assert info["order"] == 2


class TestLasserreHierarchy:
    def test_simple_optimization(self):
        """Test: minimize x subject to -1 <= x <= 1."""
        h = LasserreHierarchy(n_vars=1, max_order=2)
        h.set_objective({(1,): 1.0})
        h.add_inequality({(0,): 1.0, (1,): 1.0})   # x + 1 >= 0
        h.add_inequality({(0,): 1.0, (1,): -1.0})  # 1 - x >= 0

        result = h.solve_adaptive()
        # Minimum of x on [-1, 1] is -1
        assert result["best_bound"] > -1.5
        assert result["best_bound"] < -0.5

    def test_minimum_order(self):
        h = LasserreHierarchy(n_vars=2, max_order=3)
        h.set_objective({(2, 0): 1.0})  # minimize x^2
        h.add_box_constraints([(-1.0, 1.0), (-1.0, 1.0)])
        assert h.minimum_feasible_order() >= 1

    def test_certificate(self):
        h = LasserreHierarchy(n_vars=1, max_order=2)
        h.set_objective({(2,): 1.0, (0,): -0.5})  # x^2 - 0.5
        h.add_inequality({(0,): 1.0, (1,): 1.0})
        h.add_inequality({(0,): 1.0, (1,): -1.0})

        h.solve_adaptive()
        cert = h.get_verification_certificate()
        assert "lower_bound" in cert
        assert "certified" in cert


class TestGurobiLPSolver:
    def test_simple_lp(self):
        """min x s.t. -1 <= x <= 1  →  opt = -1."""
        solver = GurobiLPSolver(verbose=False)
        result = solver.minimize_linear(
            c=np.array([1.0]),
            lb=np.array([-1.0]),
            ub=np.array([1.0]),
        )
        assert result["status"] == "optimal"
        assert abs(result["optimal_value"] - (-1.0)) < 1e-6

    def test_margin_lp(self):
        """Margin LP: min w^T h + b, w=[1,-1], b=0, h in [0,1]^2."""
        solver = GurobiLPSolver(verbose=False)
        result = solver.solve_margin_lp(
            diff_w=np.array([1.0, -1.0]),
            diff_b=0.0,
            lb=np.array([0.0, 0.0]),
            ub=np.array([1.0, 1.0]),
        )
        assert result["status"] == "optimal"
        # min h0 - h1 = 0 - 1 = -1
        assert abs(result["optimal_value"] - (-1.0)) < 1e-6

    def test_certified_margin(self):
        """Case where margin is strictly positive → certified."""
        solver = GurobiLPSolver(verbose=False)
        result = solver.solve_margin_lp(
            diff_w=np.array([1.0, 1.0]),
            diff_b=0.5,
            lb=np.array([0.0, 0.0]),
            ub=np.array([1.0, 1.0]),
        )
        assert result["status"] == "optimal"
        # min (h0 + h1) + 0.5 = 0 + 0 + 0.5 = 0.5
        assert result["optimal_value"] > 0
