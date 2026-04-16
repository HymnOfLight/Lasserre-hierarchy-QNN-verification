"""Tests for the sparsity module."""

import numpy as np
import pytest

from qnn_verifier.network.quantized_network import QuantizedNetwork, QuantizedLayer
from qnn_verifier.network.model_loader import create_small_quantized_model
from qnn_verifier.sparsity.correlative_sparsity import CorrelativeSparsityAnalyzer
from qnn_verifier.sparsity.term_sparsity import TermSparsityExploiter
from qnn_verifier.sparsity.adaptive_order import AdaptiveOrderSelector


class TestCorrelativeSparsity:
    def test_build_graph(self):
        _, network = create_small_quantized_model(
            n_inputs=4, hidden_sizes=[4, 4], n_classes=2
        )
        analyzer = CorrelativeSparsityAnalyzer(network)
        G = analyzer.build_coupling_graph()
        assert G.number_of_nodes() >= 0

    def test_sparsity_summary(self):
        _, network = create_small_quantized_model(
            n_inputs=4, hidden_sizes=[4, 4], n_classes=2
        )
        analyzer = CorrelativeSparsityAnalyzer(network)
        summary = analyzer.sparsity_summary()
        assert "n_variables" in summary
        assert "density" in summary

    def test_block_structure(self):
        _, network = create_small_quantized_model(
            n_inputs=4, hidden_sizes=[4, 4], n_classes=2
        )
        analyzer = CorrelativeSparsityAnalyzer(network)
        blocks = analyzer.get_block_structure()
        assert "n_blocks" in blocks
        assert "cliques" in blocks


class TestTermSparsity:
    def test_basic(self):
        tse = TermSparsityExploiter(n_vars=2)
        tse.set_objective({(1, 0): 1.0, (0, 1): -1.0})
        tse.add_constraint({(2, 0): 1.0, (0, 0): -1.0})
        pattern = tse.build_term_sparsity_pattern()
        assert "monomials" in pattern

    def test_cliques(self):
        tse = TermSparsityExploiter(n_vars=2)
        tse.set_objective({(1, 0): 1.0})
        tse.add_constraint({(2, 0): 1.0, (0, 0): -1.0})
        cliques = tse.find_term_cliques()
        assert isinstance(cliques, list)

    def test_compression(self):
        tse = TermSparsityExploiter(n_vars=3)
        tse.set_objective({(1, 0, 0): 1.0})
        tse.add_constraint({(0, 2, 0): 1.0, (0, 0, 0): -1.0})
        info = tse.compression_info()
        assert "support_size" in info
        assert "reduction_ratio" in info


class TestAdaptiveOrder:
    def test_select_order(self):
        selector = AdaptiveOrderSelector(min_order=1, max_order=4)
        order = selector.select_order_for_layer(
            layer_idx=0,
            n_unstable_neurons=5,
            bound_gap=2.0,
            n_variables=10,
        )
        assert 1 <= order <= 4

    def test_stable_layer(self):
        selector = AdaptiveOrderSelector(min_order=1, max_order=4)
        order = selector.select_order_for_layer(
            layer_idx=0,
            n_unstable_neurons=0,
            bound_gap=0.001,
            n_variables=10,
        )
        assert order == 1

    def test_layer_plan(self):
        selector = AdaptiveOrderSelector(min_order=1, max_order=3)
        plan = selector.get_layer_plan([
            {"layer_idx": 0, "n_unstable": 10, "bound_gap": 5.0, "n_variables": 20},
            {"layer_idx": 1, "n_unstable": 0, "bound_gap": 0.1, "n_variables": 10},
        ])
        assert 0 in plan
        assert 1 in plan
        assert plan[1] <= plan[0]  # Easier layer should get lower order
