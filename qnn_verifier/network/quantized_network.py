"""
Quantized neural network abstraction layer.

Provides a layer-by-layer representation of quantized neural networks
suitable for polynomial verification, extracting weight matrices,
bias vectors, and quantization parameters from PyTorch models.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class QuantizedLayer:
    """Represents a single layer in the quantized network abstraction."""

    layer_type: str  # 'linear', 'conv2d', 'relu', 'sigmoid', 'batchnorm', 'flatten', 'avgpool', 'add'
    weight: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None

    # Quantization parameters
    n_bits: int = 8
    scale: float = 1.0
    zero_point: int = 0
    is_quantized: bool = False

    # Activation bounds (propagated)
    input_lower: Optional[np.ndarray] = None
    input_upper: Optional[np.ndarray] = None
    output_lower: Optional[np.ndarray] = None
    output_upper: Optional[np.ndarray] = None

    # Conv parameters
    stride: Tuple[int, ...] = (1, 1)
    padding: Tuple[int, ...] = (0, 0)
    kernel_size: Tuple[int, ...] = (1, 1)

    # For residual connections
    residual_from: Optional[int] = None

    @property
    def n_inputs(self) -> int:
        if self.weight is not None:
            if self.layer_type == "conv2d":
                return int(np.prod(self.weight.shape[1:]))
            return self.weight.shape[1]
        if self.input_shape is not None:
            return int(np.prod(self.input_shape))
        return 0

    @property
    def n_outputs(self) -> int:
        if self.weight is not None:
            return self.weight.shape[0]
        if self.output_shape is not None:
            return int(np.prod(self.output_shape))
        return 0


class QuantizedNetwork:
    """
    Layer-by-layer abstraction of a quantized neural network.

    Provides the mathematical representation needed for polynomial
    verification: affine maps (W, b) + activation constraints per layer.
    """

    def __init__(self, name: str = "quantized_network"):
        self.name = name
        self.layers: List[QuantizedLayer] = []
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.n_classes: int = 0
        self.metadata: Dict = {}

    def add_layer(self, layer: QuantizedLayer):
        self.layers.append(layer)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def affine_layers(self) -> List[int]:
        """Indices of layers with weight matrices (linear or conv2d)."""
        return [
            i for i, l in enumerate(self.layers)
            if l.layer_type in ("linear", "conv2d")
        ]

    @property
    def activation_layers(self) -> List[int]:
        """Indices of activation layers."""
        return [
            i for i, l in enumerate(self.layers)
            if l.layer_type in ("relu", "sigmoid", "tanh", "hardswish")
        ]

    @property
    def total_neurons(self) -> int:
        """Total number of neurons across all activation layers."""
        total = 0
        for i in self.activation_layers:
            if self.layers[i].output_shape is not None:
                total += int(np.prod(self.layers[i].output_shape))
        return total

    def get_layer_pairs(self) -> List[Tuple[int, int]]:
        """
        Get (affine_layer_idx, activation_layer_idx) pairs for verification.
        """
        pairs = []
        affine = self.affine_layers
        activations = self.activation_layers

        for a_idx in affine:
            for act_idx in activations:
                if act_idx == a_idx + 1:
                    pairs.append((a_idx, act_idx))
                    break
        return pairs

    def get_flattened_weights(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get weight and bias for a layer, flattened for linear algebra.
        For conv layers, this im2col-style flattening depends on input shape.
        """
        layer = self.layers[layer_idx]
        if layer.weight is None:
            raise ValueError(f"Layer {layer_idx} has no weights")

        W = layer.weight
        b = layer.bias if layer.bias is not None else np.zeros(W.shape[0])

        if layer.layer_type == "conv2d" and layer.input_shape is not None:
            # For small verification sub-problems, we flatten the conv
            W_flat = self._conv_to_matrix(layer)
            return W_flat, b
        elif layer.layer_type == "linear":
            return W, b

        return W.reshape(W.shape[0], -1), b

    def _conv_to_matrix(self, layer: QuantizedLayer) -> np.ndarray:
        """
        Convert convolution to an equivalent matrix multiplication.
        Only practical for small spatial dimensions.
        """
        if layer.input_shape is None:
            raise ValueError("Input shape required for conv-to-matrix")

        C_in, H_in, W_in = layer.input_shape[-3:]
        C_out = layer.weight.shape[0]
        kH, kW = layer.kernel_size[:2] if len(layer.kernel_size) >= 2 else (layer.kernel_size[0], layer.kernel_size[0])
        sH, sW = layer.stride[:2] if len(layer.stride) >= 2 else (layer.stride[0], layer.stride[0])
        pH, pW = layer.padding[:2] if len(layer.padding) >= 2 else (layer.padding[0], layer.padding[0])

        H_out = (H_in + 2 * pH - kH) // sH + 1
        W_out = (W_in + 2 * pW - kW) // sW + 1

        n_in = C_in * H_in * W_in
        n_out = C_out * H_out * W_out

        M = np.zeros((n_out, n_in))
        weight = layer.weight.reshape(C_out, C_in, kH, kW)

        for oc in range(C_out):
            for oh in range(H_out):
                for ow in range(W_out):
                    out_idx = oc * H_out * W_out + oh * W_out + ow
                    for ic in range(C_in):
                        for fh in range(kH):
                            for fw in range(kW):
                                ih = oh * sH - pH + fh
                                iw = ow * sW - pW + fw
                                if 0 <= ih < H_in and 0 <= iw < W_in:
                                    in_idx = ic * H_in * W_in + ih * W_in + iw
                                    M[out_idx, in_idx] = weight[oc, ic, fh, fw]

        return M

    def summary(self) -> str:
        lines = [f"Network: {self.name}"]
        lines.append(f"Input shape: {self.input_shape}")
        lines.append(f"Output shape: {self.output_shape}")
        lines.append(f"Total layers: {self.n_layers}")
        lines.append(f"Total neurons: {self.total_neurons}")
        lines.append("")
        for i, layer in enumerate(self.layers):
            desc = f"  [{i}] {layer.layer_type}"
            if layer.weight is not None:
                desc += f" (weight: {layer.weight.shape})"
            if layer.input_shape:
                desc += f" in={layer.input_shape}"
            if layer.output_shape:
                desc += f" out={layer.output_shape}"
            if layer.is_quantized:
                desc += f" Q{layer.n_bits}"
            lines.append(desc)
        return "\n".join(lines)
