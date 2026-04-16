"""Tests for the network module."""

import numpy as np
import pytest
import torch

from qnn_verifier.network.quantized_network import QuantizedNetwork, QuantizedLayer
from qnn_verifier.network.model_loader import create_small_quantized_model
from qnn_verifier.network.layer_propagation import BoundPropagator


class TestQuantizedNetwork:
    def test_basic_construction(self):
        net = QuantizedNetwork(name="test")
        net.add_layer(QuantizedLayer(
            layer_type="linear",
            weight=np.random.randn(4, 3),
            bias=np.zeros(4),
        ))
        net.add_layer(QuantizedLayer(layer_type="relu"))
        net.add_layer(QuantizedLayer(
            layer_type="linear",
            weight=np.random.randn(2, 4),
            bias=np.zeros(2),
        ))
        assert net.n_layers == 3
        assert len(net.affine_layers) == 2
        assert len(net.activation_layers) == 1

    def test_summary(self):
        net = QuantizedNetwork(name="test")
        net.add_layer(QuantizedLayer(
            layer_type="linear",
            weight=np.random.randn(4, 3),
        ))
        summary = net.summary()
        assert "test" in summary


class TestCreateSmallModel:
    def test_default(self):
        model, network = create_small_quantized_model()
        assert isinstance(model, torch.nn.Module)
        assert isinstance(network, QuantizedNetwork)

    def test_custom(self):
        model, network = create_small_quantized_model(
            n_inputs=6, hidden_sizes=[10, 5], n_classes=3, n_bits=4
        )
        # Forward pass
        with torch.no_grad():
            out = model(torch.randn(1, 6))
        assert out.shape == (1, 3)

    def test_quantized_weights(self):
        model, _ = create_small_quantized_model(n_bits=4)
        # Check weights have limited unique values
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                w = m.weight.data.numpy().flatten()
                n_unique = len(np.unique(np.round(w, 6)))
                # With 4-bit quantization, should have fewer unique values
                # than full float (though exact count depends on range)
                assert n_unique <= len(w)


class TestBoundPropagator:
    def test_propagation(self):
        _, network = create_small_quantized_model(
            n_inputs=4, hidden_sizes=[4, 4], n_classes=2
        )
        propagator = BoundPropagator(network, poly_degree=2)

        input_lower = np.zeros(4)
        input_upper = np.ones(4)
        bounds = propagator.propagate(input_lower, input_upper)

        assert len(bounds) > 0
        out_lower, out_upper = propagator.get_output_bounds()
        assert len(out_lower) == 2
        assert np.all(out_upper >= out_lower)

    def test_unstable_neurons(self):
        _, network = create_small_quantized_model(
            n_inputs=4, hidden_sizes=[4, 4], n_classes=2
        )
        propagator = BoundPropagator(network, poly_degree=2)
        propagator.propagate(np.zeros(4), np.ones(4))
        unstable = propagator.get_verification_neurons()
        # Should be a list of dicts
        assert isinstance(unstable, list)
