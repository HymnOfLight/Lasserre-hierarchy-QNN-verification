#!/usr/bin/env python3
"""
Example: Create and save a quantized ResNet model for verification.

This script creates a quantized ResNet-18 (or ResNet-101 for "ResNet-121")
model, simulates 8-bit quantization, and saves it for verification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import argparse
from pathlib import Path


def quantize_model_weights(model: nn.Module, n_bits: int = 8) -> nn.Module:
    """
    Simulate post-training weight quantization by clamping weights
    to discrete levels representable with n_bits.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                w = param.data
                w_min, w_max = w.min(), w.max()
                if w_min == w_max:
                    continue

                n_levels = (1 << n_bits) - 1
                scale = (w_max - w_min) / n_levels
                zero_point = torch.round(-w_min / scale).clamp(0, n_levels)

                w_q = torch.round(w / scale + zero_point)
                w_q = w_q.clamp(0, n_levels)
                w_dq = (w_q - zero_point) * scale

                param.data = w_dq

    return model


def create_small_test_model(n_inputs: int = 8, n_classes: int = 3, n_bits: int = 8):
    """Create a small MLP for quick testing."""
    model = nn.Sequential(
        nn.Linear(n_inputs, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, n_classes),
    )
    model = quantize_model_weights(model, n_bits)
    return model


def create_resnet_model(arch: str = "resnet18", n_classes: int = 10, n_bits: int = 8):
    """Create a quantized ResNet model."""
    arch_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,  # closest to "ResNet-121"
    }

    factory = arch_map.get(arch, models.resnet18)
    model = factory(num_classes=n_classes)
    model = quantize_model_weights(model, n_bits)
    return model


def main():
    parser = argparse.ArgumentParser(description="Create quantized model for verification")
    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "small_mlp"])
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--n-bits", type=int, default=8)
    parser.add_argument("--output", type=str, default="quantized_model.pth")
    parser.add_argument("--n-inputs", type=int, default=8,
                        help="Input size for small MLP")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.arch == "small_mlp":
        print(f"Creating small quantized MLP ({args.n_inputs} -> {args.n_classes})")
        model = create_small_test_model(args.n_inputs, args.n_classes, args.n_bits)
        input_shape = (1, args.n_inputs)
    else:
        print(f"Creating quantized {args.arch} (n_classes={args.n_classes}, {args.n_bits}-bit)")
        model = create_resnet_model(args.arch, args.n_classes, args.n_bits)
        input_shape = (1, 3, 32, 32) if args.n_classes <= 100 else (1, 3, 224, 224)

    model.eval()

    # Verify forward pass works
    dummy = torch.randn(*input_shape)
    with torch.no_grad():
        output = model(dummy)
    print(f"Model output shape: {output.shape}")
    print(f"Sample predictions: {output[0, :5].tolist()}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Save
    torch.save({
        "state_dict": model.state_dict(),
        "arch": args.arch,
        "n_classes": args.n_classes,
        "n_bits": args.n_bits,
        "input_shape": input_shape,
    }, str(output_path))
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
