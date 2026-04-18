"""Tests for the benchmarks module."""

import numpy as np
import pytest
from pathlib import Path

from qnn_verifier.benchmarks.registry import BENCHMARKS, BenchmarkInfo, list_benchmarks
from qnn_verifier.benchmarks.vnnlib_parser import VNNLIBProperty, parse_vnnlib


class TestRegistry:
    def test_all_benchmarks_registered(self):
        expected = {"acasxu", "cgan", "nn4sys", "linearizenn", "ml4acopf",
                    "vit", "collins_aerospace", "lsnc", "cctsdb"}
        assert expected == set(BENCHMARKS.keys())

    def test_benchmark_info_fields(self):
        acas = BENCHMARKS["acasxu"]
        assert acas.name == "ACAS Xu"
        assert acas.category == "classic"
        assert acas.dir_name == "acasxu_2023"
        assert "safety-critical" in acas.tags

    def test_list_benchmarks_all(self):
        all_b = list_benchmarks()
        assert len(all_b) == 9

    def test_list_benchmarks_complex(self):
        complex_b = list_benchmarks(category="complex")
        assert len(complex_b) == 8
        for b in complex_b:
            assert b.category == "complex"

    def test_list_benchmarks_classic(self):
        classic = list_benchmarks(category="classic")
        assert len(classic) == 1
        assert classic[0].name == "ACAS Xu"


class TestVNNLIBParser:
    def _make_vnnlib(self, tmp_path, content):
        f = tmp_path / "test.vnnlib"
        f.write_text(content)
        return str(f)

    def test_parse_basic(self, tmp_path):
        content = """
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -0.5))
(assert (<= X_0 0.5))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))
(assert (<= Y_0 Y_1))
"""
        path = self._make_vnnlib(tmp_path, content)
        prop = parse_vnnlib(path)
        assert prop.n_inputs == 2
        assert prop.n_outputs == 2
        assert prop.input_lower[0] == pytest.approx(-0.5)
        assert prop.input_upper[0] == pytest.approx(0.5)
        assert prop.input_lower[1] == pytest.approx(0.0)
        assert prop.input_upper[1] == pytest.approx(1.0)

    def test_parse_output_constraints(self, tmp_path):
        content = """
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (<= Y_0 Y_1))
"""
        path = self._make_vnnlib(tmp_path, content)
        prop = parse_vnnlib(path)
        assert len(prop.output_constraints) >= 1
        assert prop.output_constraints[0]["type"] == "comparison"

    def test_parse_empty_file(self, tmp_path):
        path = self._make_vnnlib(tmp_path, "")
        prop = parse_vnnlib(path)
        assert prop.n_inputs == 0
        assert prop.n_outputs == 0

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_vnnlib("/nonexistent/path.vnnlib")
