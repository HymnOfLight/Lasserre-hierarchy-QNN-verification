#!/usr/bin/env python3
"""
Example: Full verification pipeline for a quantized ResNet model.

Creates a quantized ResNet, runs verification, and produces a certificate.

Usage:
    python verify_resnet.py
    python verify_resnet.py --arch resnet101 --epsilon 0.005
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from qnn_verifier.verification.pipeline import VerificationPipeline
from qnn_verifier.network.model_loader import load_quantized_model
from qnn_verifier.network.quantized_network import QuantizedNetwork
from qnn_verifier.sparsity.correlative_sparsity import CorrelativeSparsityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def quantize_weights(model: nn.Module, n_bits: int = 8) -> nn.Module:
    """Simulate post-training quantization."""
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() >= 2:
                w = param.data
                w_min, w_max = w.min(), w.max()
                if w_min == w_max:
                    continue
                n_levels = (1 << n_bits) - 1
                scale = (w_max - w_min) / n_levels
                zp = torch.round(-w_min / scale).clamp(0, n_levels)
                param.data = (torch.round(w / scale + zp).clamp(0, n_levels) - zp) * scale
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--n-bits", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--poly-degree", type=int, default=4)
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--save-model", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Quantized {args.arch.upper()} Robustness Verification")
    print("=" * 60)

    # Step 1: Create model
    import torchvision.models as models

    arch_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
    }
    factory = arch_map.get(args.arch, models.resnet18)
    model = factory(num_classes=args.n_classes)

    # Adjust for small inputs (CIFAR-like)
    if args.input_size <= 64:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    model = quantize_weights(model, args.n_bits)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.arch}, Parameters: {n_params:,}")
    print(f"Quantization: {args.n_bits}-bit")

    # Save model if requested
    model_path = args.save_model or f"/tmp/quantized_{args.arch}.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "arch": args.arch,
        "n_classes": args.n_classes,
        "n_bits": args.n_bits,
    }, model_path)
    print(f"Model saved to {model_path}")

    # Step 2: Load into verification framework
    input_shape = (1, 3, args.input_size, args.input_size)
    pipeline = VerificationPipeline(
        model_path=model_path,
        model_arch=args.arch,
        input_shape=input_shape,
        n_classes=args.n_classes,
        n_bits=args.n_bits,
        poly_degree=args.poly_degree,
        max_lasserre_order=args.max_order,
    )

    network = pipeline.network
    print(f"\nExtracted network layers: {network.n_layers}")
    print(f"Activation layers: {len(network.activation_layers)}")
    print(f"Affine layers: {len(network.affine_layers)}")

    # Step 3: Sparsity analysis (skip for very large networks)
    total_neurons = sum(
        int(np.prod(l.output_shape)) for l in network.layers
        if l.output_shape is not None and l.layer_type in ("relu", "sigmoid", "tanh")
    )
    print(f"\nTotal activation neurons: {total_neurons:,}")
    if total_neurons < 10000:
        print("\n--- Sparsity Analysis ---")
        sparsity_info = pipeline.analyze_sparsity()
        sp = sparsity_info["sparsity"]
        print(f"Variables: {sp['n_variables']}")
        print(f"Couplings: {sp['n_couplings']}")
        print(f"Density: {sp['density']:.4f}")
        print(f"Is sparse: {sp['is_sparse']}")
        bl = sparsity_info["blocks"]
        print(f"Decomposition blocks: {bl['n_blocks']}")
        print(f"Compression ratio: {bl['compression_ratio']:.2f}")
    else:
        print("(Skipping full sparsity graph analysis for large network; "
              "using layered decomposition directly)")

    # Step 4: Verification
    print(f"\n--- Verification (epsilon={args.epsilon}) ---")
    np.random.seed(42)
    x0 = np.random.rand(*input_shape[1:]).astype(np.float32)
    x0 = x0 / x0.max()  # Normalize to [0, 1]

    # Run model to get true prediction
    with torch.no_grad():
        pred = model(torch.tensor(x0).unsqueeze(0))
        true_label = pred.argmax(dim=-1).item()
    print(f"True prediction: class {true_label}")
    print(f"Output logits: {pred[0, :5].tolist()}")

    start = time.time()
    result = pipeline.verify(x0, true_label=true_label, epsilon=args.epsilon)
    elapsed = time.time() - start

    # Step 5: Report
    print(f"\n--- Results ---")
    print(f"Verified: {result.get('verified', False)}")
    print(f"Method: {result.get('method', 'unknown')}")
    print(f"Time: {elapsed:.2f}s")

    cert = result.get("certificate")
    if cert is not None:
        print(f"\n{cert.summary()}")
        cert_path = f"certificate_{args.arch}_eps{args.epsilon}.json"
        cert.to_json(cert_path)
        print(f"\nCertificate saved to {cert_path}")

    # Step 6: Multi-epsilon analysis
    print("\n--- Multi-Epsilon Analysis ---")
    for eps in [0.001, 0.005, 0.01, 0.05]:
        t0 = time.time()
        r = pipeline.verify(x0, true_label=true_label, epsilon=eps)
        dt = time.time() - t0
        v = r.get("verified", False)
        print(f"  eps={eps:.4f}: verified={v}, time={dt:.2f}s")


if __name__ == "__main__":
    main()
