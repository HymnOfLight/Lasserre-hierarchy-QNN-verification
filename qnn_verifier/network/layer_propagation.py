"""
Interval bound propagation for quantized neural networks.

Computes concrete (interval arithmetic) and abstract (polynomial) bounds
on neuron activations, layer by layer, to establish the input intervals
needed by the Chebyshev approximation and Lasserre hierarchy modules.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from .quantized_network import QuantizedNetwork, QuantizedLayer
from ..polynomial.activation_envelope import ActivationEnvelope

logger = logging.getLogger(__name__)


class BoundPropagator:
    """
    Propagates bounds through a quantized neural network using a combination
    of interval bound propagation (IBP) and polynomial envelope refinement.
    """

    def __init__(
        self,
        network: QuantizedNetwork,
        poly_degree: int = 4,
        use_polynomial_refinement: bool = True,
    ):
        self.network = network
        self.poly_degree = poly_degree
        self.use_poly_refine = use_polynomial_refinement

        self.layer_bounds: List[Dict[str, np.ndarray]] = []
        self.envelopes: List[Optional[List[ActivationEnvelope]]] = []

    def propagate(
        self,
        input_lower: np.ndarray,
        input_upper: np.ndarray,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Propagate bounds through the entire network.

        Args:
            input_lower: Lower bound on network input
            input_upper: Upper bound on network input

        Returns:
            List of bound dictionaries per layer, each containing
            'pre_lower', 'pre_upper', 'post_lower', 'post_upper'.
        """
        self.layer_bounds = []
        self.envelopes = []

        current_lower = input_lower.flatten()
        current_upper = input_upper.flatten()

        self.layer_bounds.append({
            "pre_lower": current_lower.copy(),
            "pre_upper": current_upper.copy(),
            "post_lower": current_lower.copy(),
            "post_upper": current_upper.copy(),
        })

        for i, layer in enumerate(self.network.layers):
            if layer.layer_type == "batchnorm":
                new_lower, new_upper = self._propagate_batchnorm(
                    layer, current_lower, current_upper
                )
                layer.input_lower = current_lower
                layer.input_upper = current_upper
                layer.output_lower = new_lower
                layer.output_upper = new_upper

                self.layer_bounds.append({
                    "pre_lower": new_lower.copy(),
                    "pre_upper": new_upper.copy(),
                    "post_lower": new_lower.copy(),
                    "post_upper": new_upper.copy(),
                })
                self.envelopes.append(None)
                current_lower = new_lower
                current_upper = new_upper

            elif layer.layer_type in ("linear", "conv2d"):
                new_lower, new_upper = self._propagate_affine(
                    layer, current_lower, current_upper
                )
                layer.input_lower = current_lower
                layer.input_upper = current_upper
                layer.output_lower = new_lower
                layer.output_upper = new_upper

                self.layer_bounds.append({
                    "pre_lower": new_lower.copy(),
                    "pre_upper": new_upper.copy(),
                    "post_lower": new_lower.copy(),
                    "post_upper": new_upper.copy(),
                })
                self.envelopes.append(None)
                current_lower = new_lower
                current_upper = new_upper

            elif layer.layer_type in ("relu", "sigmoid", "tanh", "hardswish"):
                new_lower, new_upper, layer_envelopes = self._propagate_activation(
                    layer, current_lower, current_upper
                )
                layer.input_lower = current_lower
                layer.input_upper = current_upper
                layer.output_lower = new_lower
                layer.output_upper = new_upper

                self.layer_bounds.append({
                    "pre_lower": current_lower.copy(),
                    "pre_upper": current_upper.copy(),
                    "post_lower": new_lower.copy(),
                    "post_upper": new_upper.copy(),
                })
                self.envelopes.append(layer_envelopes)
                current_lower = new_lower
                current_upper = new_upper

            elif layer.layer_type in ("flatten", "avgpool"):
                if layer.layer_type == "avgpool" and layer.output_shape is not None:
                    out_size = int(np.prod(layer.output_shape))
                    in_size = len(current_lower)
                    if out_size < in_size and out_size > 0:
                        factor = in_size // out_size
                        current_lower = current_lower[:out_size * factor].reshape(out_size, factor).mean(axis=1)
                        current_upper = current_upper[:out_size * factor].reshape(out_size, factor).mean(axis=1)

                self.layer_bounds.append({
                    "pre_lower": current_lower.copy(),
                    "pre_upper": current_upper.copy(),
                    "post_lower": current_lower.copy(),
                    "post_upper": current_upper.copy(),
                })
                self.envelopes.append(None)

            else:
                self.layer_bounds.append({
                    "pre_lower": current_lower.copy(),
                    "pre_upper": current_upper.copy(),
                    "post_lower": current_lower.copy(),
                    "post_upper": current_upper.copy(),
                })
                self.envelopes.append(None)

        return self.layer_bounds

    def _propagate_affine(
        self,
        layer: QuantizedLayer,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interval arithmetic for affine (linear or conv2d) layers."""
        if layer.weight is None:
            return lower.copy(), upper.copy()

        if layer.layer_type == "conv2d":
            return self._propagate_conv2d(layer, lower, upper)

        W = layer.weight
        if W.ndim > 2:
            W = W.reshape(W.shape[0], -1)

        in_size = W.shape[1]
        l = lower[:in_size] if len(lower) >= in_size else np.pad(lower, (0, in_size - len(lower)))
        u = upper[:in_size] if len(upper) >= in_size else np.pad(upper, (0, in_size - len(upper)))

        W_pos = np.maximum(W, 0)
        W_neg = np.minimum(W, 0)

        new_lower = W_pos @ l + W_neg @ u
        new_upper = W_pos @ u + W_neg @ l

        if layer.bias is not None:
            b = layer.bias[:W.shape[0]] if len(layer.bias) >= W.shape[0] else np.pad(layer.bias, (0, W.shape[0] - len(layer.bias)))
            new_lower = new_lower + b
            new_upper = new_upper + b

        return new_lower, new_upper

    def _propagate_conv2d(
        self,
        layer: QuantizedLayer,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Efficient interval bound propagation for conv2d using PyTorch.
        Uses the identity: for W with pos/neg parts,
          [Wx]_lower = W_pos @ x_lower + W_neg @ x_upper
        """
        import torch
        import torch.nn.functional as F

        W = torch.from_numpy(layer.weight).float()
        in_shape = layer.input_shape

        if in_shape is None:
            # Fallback to dense matmul
            W_np = layer.weight.reshape(layer.weight.shape[0], -1)
            in_size = W_np.shape[1]
            l = lower[:in_size] if len(lower) >= in_size else np.pad(lower, (0, in_size - len(lower)))
            u = upper[:in_size] if len(upper) >= in_size else np.pad(upper, (0, in_size - len(upper)))
            W_pos = np.maximum(W_np, 0)
            W_neg = np.minimum(W_np, 0)
            new_l = W_pos @ l + W_neg @ u
            new_u = W_pos @ u + W_neg @ l
            if layer.bias is not None:
                new_l += layer.bias
                new_u += layer.bias
            return new_l, new_u

        C_in = in_shape[0] if len(in_shape) >= 3 else in_shape[0]
        spatial = int(np.prod(in_shape[1:])) if len(in_shape) >= 3 else 1
        expected_size = int(np.prod(in_shape))

        l_flat = lower[:expected_size] if len(lower) >= expected_size else np.pad(lower, (0, expected_size - len(lower)))
        u_flat = upper[:expected_size] if len(upper) >= expected_size else np.pad(upper, (0, expected_size - len(upper)))

        l_tensor = torch.from_numpy(l_flat.reshape(1, *in_shape)).float()
        u_tensor = torch.from_numpy(u_flat.reshape(1, *in_shape)).float()

        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)

        stride = layer.stride
        padding = layer.padding

        bias = torch.from_numpy(layer.bias).float() if layer.bias is not None else None

        new_l = F.conv2d(l_tensor, W_pos, None, stride, padding) + \
                F.conv2d(u_tensor, W_neg, None, stride, padding)
        new_u = F.conv2d(u_tensor, W_pos, None, stride, padding) + \
                F.conv2d(l_tensor, W_neg, None, stride, padding)

        if bias is not None:
            bias_view = bias.view(1, -1, 1, 1)
            new_l = new_l + bias_view
            new_u = new_u + bias_view

        return new_l.detach().numpy().flatten(), new_u.detach().numpy().flatten()

    def _propagate_batchnorm(
        self,
        layer: QuantizedLayer,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Efficient per-channel BN propagation: y = scale * x + shift."""
        scale = layer.weight  # 1D array (C,)
        shift = layer.bias if layer.bias is not None else np.zeros_like(scale)

        n_channels = len(scale)
        spatial = len(lower) // n_channels if n_channels > 0 else 1

        l_flat = lower.flatten()
        u_flat = upper.flatten()

        new_lower = np.empty_like(l_flat)
        new_upper = np.empty_like(u_flat)

        for c in range(n_channels):
            s = scale[c]
            b = shift[c]
            start = c * spatial
            end = start + spatial
            if end > len(l_flat):
                end = len(l_flat)
            if start >= len(l_flat):
                break
            if s >= 0:
                new_lower[start:end] = s * l_flat[start:end] + b
                new_upper[start:end] = s * u_flat[start:end] + b
            else:
                new_lower[start:end] = s * u_flat[start:end] + b
                new_upper[start:end] = s * l_flat[start:end] + b

        return new_lower, new_upper

    def _propagate_activation(
        self,
        layer: QuantizedLayer,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[ActivationEnvelope]]:
        """
        Propagate bounds through an activation layer.
        Polynomial envelopes are only built for unstable neurons
        (where the activation crosses its non-linear region) to
        keep large-network propagation fast.
        """
        n = len(lower)
        new_lower = np.zeros(n)
        new_upper = np.zeros(n)
        envelopes: List[Optional[ActivationEnvelope]] = [None] * n

        if layer.layer_type == "relu":
            # Vectorized propagation
            active = lower >= 0
            inactive = upper <= 0
            unstable = ~active & ~inactive

            new_lower[active] = lower[active]
            new_upper[active] = upper[active]
            # inactive: already 0

            new_upper[unstable] = upper[unstable]

            # Build envelopes only for unstable neurons (and only a
            # manageable subset for very wide layers)
            unstable_idx = np.where(unstable)[0]
            max_envelope_neurons = 200
            if len(unstable_idx) > max_envelope_neurons and self.use_poly_refine:
                # Prioritize neurons with the widest crossing regions
                widths = upper[unstable_idx] - lower[unstable_idx]
                top_k = np.argsort(-widths)[:max_envelope_neurons]
                envelope_idx = unstable_idx[top_k]
            else:
                envelope_idx = unstable_idx if self.use_poly_refine else np.array([], dtype=int)

            for j in envelope_idx:
                lb, ub = float(lower[j]), float(upper[j])
                env = ActivationEnvelope(
                    activation_type="relu",
                    degree=self.poly_degree,
                    n_bits=layer.n_bits if layer.is_quantized else None,
                )
                env.build_envelope((lb, ub))
                envelopes[j] = env

        elif layer.layer_type == "sigmoid":
            clipped = np.clip(lower, -500, 500)
            new_lower = 1.0 / (1.0 + np.exp(-clipped))
            clipped_u = np.clip(upper, -500, 500)
            new_upper = 1.0 / (1.0 + np.exp(-clipped_u))

        elif layer.layer_type == "tanh":
            new_lower = np.tanh(lower)
            new_upper = np.tanh(upper)

        else:
            for j in range(n):
                lb, ub = float(lower[j]), float(upper[j])
                env = ActivationEnvelope(
                    activation_type=layer.layer_type,
                    degree=self.poly_degree,
                    n_bits=layer.n_bits if layer.is_quantized else None,
                )
                interval = (lb, ub) if ub - lb > 1e-10 else (lb - 0.01, lb + 0.01)
                env.build_envelope(interval)
                new_lower[j] = float(env.evaluate_lower(np.array([lb])))
                new_upper[j] = float(env.evaluate_upper(np.array([ub])))
                envelopes[j] = env

        return new_lower, new_upper, envelopes

    def get_neuron_envelope(
        self, layer_idx: int, neuron_idx: int
    ) -> Optional[ActivationEnvelope]:
        """Get the polynomial envelope for a specific neuron."""
        if layer_idx < len(self.envelopes) and self.envelopes[layer_idx] is not None:
            envs = self.envelopes[layer_idx]
            if neuron_idx < len(envs):
                return envs[neuron_idx]
        return None

    def get_output_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the bounds on the network output."""
        if not self.layer_bounds:
            raise RuntimeError("Must call propagate() first")
        last = self.layer_bounds[-1]
        return last["post_lower"], last["post_upper"]

    def get_verification_neurons(self) -> List[Dict]:
        """
        Identify neurons that require non-trivial verification
        (those in the 'unstable' region where the activation is not
        definitively active or inactive).
        """
        unstable = []
        for layer_idx, layer in enumerate(self.network.layers):
            if layer.layer_type not in ("relu", "sigmoid", "tanh"):
                continue

            if layer_idx >= len(self.layer_bounds):
                continue

            bounds = self.layer_bounds[layer_idx]
            pre_l = bounds["pre_lower"]
            pre_u = bounds["pre_upper"]

            for j in range(len(pre_l)):
                if layer.layer_type == "relu":
                    if pre_l[j] < 0 < pre_u[j]:
                        unstable.append({
                            "layer_idx": layer_idx,
                            "neuron_idx": j,
                            "pre_lower": float(pre_l[j]),
                            "pre_upper": float(pre_u[j]),
                            "type": layer.layer_type,
                        })
                else:
                    unstable.append({
                        "layer_idx": layer_idx,
                        "neuron_idx": j,
                        "pre_lower": float(pre_l[j]),
                        "pre_upper": float(pre_u[j]),
                        "type": layer.layer_type,
                    })

        return unstable
