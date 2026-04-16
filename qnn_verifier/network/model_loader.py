"""
Model loader for quantized PyTorch models.

Loads .pth format models (including quantized ResNet variants) and
converts them into the QuantizedNetwork abstraction for verification.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import logging

from .quantized_network import QuantizedNetwork, QuantizedLayer

logger = logging.getLogger(__name__)


def load_quantized_model(
    model_path: str,
    model_arch: Optional[str] = None,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    n_classes: int = 1000,
    n_bits: int = 8,
) -> QuantizedNetwork:
    """
    Load a PyTorch model from .pth file and convert to QuantizedNetwork.

    Supports:
    - Standard PyTorch models (float32)
    - PyTorch quantized models (qint8/quint8)
    - State dict only or full model checkpoints

    Args:
        model_path: Path to .pth file
        model_arch: Architecture name (e.g., 'resnet18', 'resnet121').
                    If None, tries to load full model object.
        input_shape: Input tensor shape (batch, C, H, W)
        n_classes: Number of output classes
        n_bits: Quantization bit-width for the abstraction
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model = _build_model_from_arch(
            model_arch, n_classes, input_shape=input_shape, state_dict=state_dict
        )
        _load_state_dict_flexible(model, state_dict)
    elif isinstance(checkpoint, dict) and any(
        k.endswith(".weight") for k in checkpoint.keys()
    ):
        state_dict = checkpoint
        model = _build_model_from_arch(
            model_arch, n_classes, input_shape=input_shape, state_dict=state_dict
        )
        _load_state_dict_flexible(model, state_dict)
    elif isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        model = _build_model_from_arch(model_arch, n_classes)
        try:
            model.load_state_dict(checkpoint)
        except Exception:
            logger.warning("Direct state_dict load failed, trying flexible load")
            _load_state_dict_flexible(model, checkpoint)

    model.eval()

    network = _extract_network(model, input_shape, n_bits)
    network.metadata["model_path"] = model_path
    network.metadata["model_arch"] = model_arch
    network.metadata["n_bits"] = n_bits
    network.n_classes = n_classes

    return network


def _build_model_from_arch(
    arch: Optional[str],
    n_classes: int,
    input_shape: Optional[Tuple[int, ...]] = None,
    state_dict: Optional[dict] = None,
) -> nn.Module:
    """Build a model architecture from name, adapting for small inputs if needed."""
    import torchvision.models as models

    arch_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }

    if arch is None:
        arch = "resnet18"
        logger.warning(f"No architecture specified, defaulting to {arch}")

    arch_lower = arch.lower().replace("-", "").replace("_", "")

    if arch_lower == "resnet121":
        arch_lower = "resnet101"
        logger.info("ResNet-121 mapped to ResNet-101 (closest standard variant)")

    model = None
    for name, factory in arch_map.items():
        if name.replace("_", "") == arch_lower:
            model = factory(num_classes=n_classes)
            break

    if model is None:
        raise ValueError(
            f"Unknown architecture: {arch}. Supported: {list(arch_map.keys())}"
        )

    # Adapt conv1 for small-input models (e.g. CIFAR with 32x32)
    # Detect from state_dict if the saved conv1 has a different kernel size
    if state_dict is not None and "conv1.weight" in state_dict:
        saved_shape = state_dict["conv1.weight"].shape
        model_shape = model.conv1.weight.shape
        if saved_shape != model_shape:
            logger.info(
                f"Adapting conv1: saved {saved_shape} vs default {model_shape}"
            )
            model.conv1 = nn.Conv2d(
                saved_shape[1], saved_shape[0],
                kernel_size=saved_shape[2:],
                stride=1 if saved_shape[2] <= 3 else 2,
                padding=saved_shape[2] // 2,
                bias=False,
            )
            model.maxpool = nn.Identity()
    elif input_shape is not None and len(input_shape) == 4 and input_shape[2] <= 64:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    return model


def _load_state_dict_flexible(model: nn.Module, state_dict: dict):
    """Load state dict with flexible key matching."""
    model_keys = set(model.state_dict().keys())
    new_state_dict = {}

    for key, value in state_dict.items():
        clean_key = key.replace("module.", "").replace("model.", "")
        if clean_key in model_keys:
            new_state_dict[clean_key] = value
        elif key in model_keys:
            new_state_dict[key] = value

    if new_state_dict:
        model.load_state_dict(new_state_dict, strict=False)
    else:
        logger.warning("No matching keys found in state dict")


def _extract_network(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    n_bits: int,
) -> QuantizedNetwork:
    """
    Extract the layer-by-layer structure from a PyTorch model
    using a forward hook-based tracing approach.
    """
    network = QuantizedNetwork(name=model.__class__.__name__)
    network.input_shape = input_shape[1:]  # Remove batch dim

    traced_layers = []
    hooks = []

    def make_hook(module, name):
        def hook_fn(m, inp, out):
            info = {
                "module": m,
                "name": name,
                "input_shape": tuple(inp[0].shape) if isinstance(inp, tuple) and len(inp) > 0 else None,
                "output_shape": tuple(out.shape) if isinstance(out, torch.Tensor) else None,
            }
            traced_layers.append(info)
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d,
                               nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Hardswish,
                               nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d,
                               nn.Flatten)):
            h = module.register_forward_hook(make_hook(module, name))
            hooks.append(h)

    try:
        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            model(dummy)
    except Exception as e:
        logger.warning(f"Forward pass tracing failed: {e}")
    finally:
        for h in hooks:
            h.remove()

    for info in traced_layers:
        m = info["module"]
        in_shape = info["input_shape"]
        out_shape = info["output_shape"]

        if isinstance(m, nn.Conv2d):
            layer = QuantizedLayer(
                layer_type="conv2d",
                weight=m.weight.detach().cpu().numpy(),
                bias=m.bias.detach().cpu().numpy() if m.bias is not None else None,
                input_shape=in_shape[1:] if in_shape else None,
                output_shape=out_shape[1:] if out_shape else None,
                n_bits=n_bits,
                is_quantized=True,
                stride=m.stride,
                padding=m.padding,
                kernel_size=m.kernel_size,
            )
        elif isinstance(m, nn.Linear):
            layer = QuantizedLayer(
                layer_type="linear",
                weight=m.weight.detach().cpu().numpy(),
                bias=m.bias.detach().cpu().numpy() if m.bias is not None else None,
                input_shape=in_shape[1:] if in_shape else None,
                output_shape=out_shape[1:] if out_shape else None,
                n_bits=n_bits,
                is_quantized=True,
            )
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            gamma = m.weight.detach().cpu().numpy()
            beta = m.bias.detach().cpu().numpy()
            mean = m.running_mean.detach().cpu().numpy()
            var = m.running_var.detach().cpu().numpy()
            eps = m.eps
            scale = gamma / np.sqrt(var + eps)
            shift = beta - mean * scale

            # Store as 1D "batchnorm" layer for efficient per-channel propagation
            layer = QuantizedLayer(
                layer_type="batchnorm",
                weight=scale,
                bias=shift,
                input_shape=in_shape[1:] if in_shape else None,
                output_shape=out_shape[1:] if out_shape else None,
                n_bits=n_bits,
                is_quantized=True,
            )
        elif isinstance(m, nn.ReLU):
            layer = QuantizedLayer(
                layer_type="relu",
                input_shape=in_shape[1:] if in_shape else None,
                output_shape=out_shape[1:] if out_shape else None,
            )
        elif isinstance(m, nn.Sigmoid):
            layer = QuantizedLayer(
                layer_type="sigmoid",
                input_shape=in_shape[1:] if in_shape else None,
                output_shape=out_shape[1:] if out_shape else None,
            )
        elif isinstance(m, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
            layer = QuantizedLayer(
                layer_type="avgpool",
                input_shape=in_shape[1:] if in_shape else None,
                output_shape=out_shape[1:] if out_shape else None,
            )
        elif isinstance(m, nn.Flatten):
            layer = QuantizedLayer(
                layer_type="flatten",
                input_shape=in_shape[1:] if in_shape else None,
                output_shape=out_shape[1:] if out_shape else None,
            )
        else:
            continue

        network.add_layer(layer)

    if network.layers:
        network.output_shape = network.layers[-1].output_shape

    return network


def create_small_quantized_model(
    n_inputs: int = 4,
    hidden_sizes: List[int] = None,
    n_classes: int = 2,
    activation: str = "relu",
    n_bits: int = 8,
) -> Tuple[nn.Module, QuantizedNetwork]:
    """
    Create a small quantized MLP for testing/demonstration.
    Returns both the PyTorch model and its QuantizedNetwork abstraction.
    """
    if hidden_sizes is None:
        hidden_sizes = [8, 8]

    act_map = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }
    act_cls = act_map.get(activation, nn.ReLU)

    layers_list = []
    prev_size = n_inputs
    for h in hidden_sizes:
        layers_list.append(nn.Linear(prev_size, h))
        layers_list.append(act_cls())
        prev_size = h
    layers_list.append(nn.Linear(prev_size, n_classes))

    model = nn.Sequential(*layers_list)
    model.eval()

    # Simulate quantization: clamp weights to quantized levels
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                qmax = (1 << n_bits) - 1
                w = m.weight.data
                scale = (w.max() - w.min()) / qmax if w.max() != w.min() else 1.0
                zp = torch.round(-w.min() / scale).clamp(0, qmax)
                w_q = (torch.round(w / scale + zp) - zp) * scale
                m.weight.data = w_q

    # Build QuantizedNetwork
    network = QuantizedNetwork(name="small_mlp")
    network.input_shape = (n_inputs,)
    network.n_classes = n_classes

    for m in model:
        if isinstance(m, nn.Linear):
            layer = QuantizedLayer(
                layer_type="linear",
                weight=m.weight.detach().cpu().numpy(),
                bias=m.bias.detach().cpu().numpy() if m.bias is not None else None,
                n_bits=n_bits,
                is_quantized=True,
            )
            network.add_layer(layer)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
            act_name = {nn.ReLU: "relu", nn.Sigmoid: "sigmoid", nn.Tanh: "tanh"}
            network.add_layer(
                QuantizedLayer(layer_type=act_name.get(type(m), "relu"))
            )

    network.output_shape = (n_classes,)
    return model, network
