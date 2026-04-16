#!/usr/bin/env python3
"""
Example: Verify adversarial robustness of a quantized neural network.

Demonstrates the full verification pipeline including:
1. Loading a quantized model (.pth)
2. Bound propagation through the network
3. Polynomial envelope construction
4. Lasserre hierarchy SDP solving
5. Certificate generation

Usage:
    # Quick demo with small MLP:
    python verify_robustness.py --demo

    # Verify a saved model:
    python verify_robustness.py --model quantized_model.pth --arch resnet18 --epsilon 0.01

    # Verify with custom parameters:
    python verify_robustness.py --demo --epsilon 0.05 --poly-degree 6 --max-order 3
"""

import argparse
import logging
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from qnn_verifier.verification.pipeline import VerificationPipeline
from qnn_verifier.network.model_loader import create_small_quantized_model
from qnn_verifier.polynomial.chebyshev import (
    ChebyshevApproximator,
    approximate_relu,
    approximate_quantized_relu,
)
from qnn_verifier.polynomial.activation_envelope import ActivationEnvelope

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def demo_polynomial_approximation():
    """Demonstrate Chebyshev polynomial approximation of activation functions."""
    print("\n" + "=" * 60)
    print("POLYNOMIAL APPROXIMATION DEMO")
    print("=" * 60)

    # ReLU approximation
    interval = (-2.0, 2.0)
    for degree in [2, 4, 6]:
        approx = approximate_relu(interval, degree=degree)
        max_err, rms_err = approx.approximation_error(
            lambda x: np.maximum(x, 0.0)
        )
        print(f"ReLU Chebyshev deg-{degree}: max_err={max_err:.6f}, rms_err={rms_err:.6f}")

    # Quantized ReLU approximation
    print("\nQuantized ReLU (8-bit):")
    for degree in [2, 4, 6]:
        approx = approximate_quantized_relu(interval, n_bits=8, degree=degree)
        qmax = 255
        scale = qmax / (interval[1] - interval[0])

        def q_relu(x):
            y = np.maximum(x, 0.0) * scale
            return np.clip(np.round(y), 0, qmax) / scale

        max_err, rms_err = approx.approximation_error(q_relu)
        print(f"  Degree {degree}: max_err={max_err:.6f}, rms_err={rms_err:.6f}")

    # Envelope tightness comparison
    print("\nPolynomial Envelope vs Linear Relaxation:")
    for act_type in ["relu", "sigmoid"]:
        env = ActivationEnvelope(activation_type=act_type, degree=4)
        env.build_envelope((-2.0, 2.0))
        metrics = env.tightness_metrics()
        print(f"  {act_type}: tightness_ratio={metrics['tightness_ratio']:.4f} "
              f"(1.0 = same as linear, lower = tighter)")


def demo_small_network_verification():
    """Demonstrate verification on a small quantized MLP."""
    print("\n" + "=" * 60)
    print("SMALL NETWORK VERIFICATION DEMO")
    print("=" * 60)

    pipeline, torch_model = VerificationPipeline.create_demo_pipeline(
        n_inputs=4, hidden_sizes=[8, 8], n_classes=2, n_bits=8
    )

    print(f"\nNetwork structure:")
    print(pipeline.network.summary())

    # Create a test input
    np.random.seed(42)
    x0 = np.random.rand(4).astype(np.float32)

    # Get model prediction
    with torch.no_grad():
        output = torch_model(torch.tensor(x0).unsqueeze(0))
        pred = output.argmax(dim=-1).item()

    print(f"\nInput: {x0}")
    print(f"Model output: {output.numpy()}")
    print(f"Predicted class: {pred}")

    # Verify at different epsilon values
    for epsilon in [0.001, 0.01, 0.05]:
        print(f"\n--- Verifying at epsilon = {epsilon} ---")
        start = time.time()
        result = pipeline.verify(x0, true_label=pred, epsilon=epsilon)
        elapsed = time.time() - start

        cert = result.get("certificate")
        if cert is not None:
            print(f"  Certified robust: {cert.certified_robust}")
            print(f"  Lower bound: {cert.lower_bound:.6f}")
            print(f"  Method: {result.get('method', 'unknown')}")
            print(f"  Time: {elapsed:.3f}s")
        else:
            print(f"  Verified: {result.get('verified', False)}")
            print(f"  Time: {elapsed:.3f}s")


def verify_saved_model(
    model_path: str,
    arch: str,
    epsilon: float,
    n_classes: int,
    n_bits: int,
    poly_degree: int,
    max_order: int,
    input_size: int = 32,
):
    """Verify a saved .pth model."""
    print("\n" + "=" * 60)
    print(f"VERIFYING MODEL: {model_path}")
    print("=" * 60)

    if arch == "small_mlp":
        input_shape = (1, 8)
    else:
        input_shape = (1, 3, input_size, input_size)

    pipeline = VerificationPipeline(
        model_path=model_path,
        model_arch=arch,
        input_shape=input_shape,
        n_classes=n_classes,
        n_bits=n_bits,
        poly_degree=poly_degree,
        max_lasserre_order=max_order,
    )

    print(f"\nNetwork structure:")
    print(pipeline.network.summary())

    # Sparsity analysis
    sparsity = pipeline.analyze_sparsity()
    print(f"\nSparsity analysis:")
    print(f"  Variables: {sparsity['sparsity']['n_variables']}")
    print(f"  Density: {sparsity['sparsity']['density']:.4f}")
    print(f"  Blocks: {sparsity['blocks']['n_blocks']}")

    # Create test input
    np.random.seed(42)
    if arch == "small_mlp":
        x0 = np.random.rand(8).astype(np.float32)
    else:
        x0 = np.random.rand(*input_shape[1:]).astype(np.float32)

    print(f"\nVerifying at epsilon = {epsilon}")
    result = pipeline.verify(x0, true_label=0, epsilon=epsilon)

    cert = result.get("certificate")
    if cert is not None:
        print(cert.summary())
        cert.to_json("verification_certificate.json")
        print("\nCertificate saved to verification_certificate.json")


def main():
    parser = argparse.ArgumentParser(
        description="Verify adversarial robustness of quantized neural networks"
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with small MLP")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .pth model file")
    parser.add_argument("--arch", type=str, default="resnet18",
                        help="Model architecture")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="L_inf perturbation radius")
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--n-bits", type=int, default=8)
    parser.add_argument("--poly-degree", type=int, default=4)
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=32,
                        help="Spatial input size (H=W)")
    args = parser.parse_args()

    if args.demo or args.model is None:
        demo_polynomial_approximation()
        demo_small_network_verification()
    else:
        verify_saved_model(
            model_path=args.model,
            arch=args.arch,
            epsilon=args.epsilon,
            n_classes=args.n_classes,
            n_bits=args.n_bits,
            poly_degree=args.poly_degree,
            max_order=args.max_order,
            input_size=args.input_size,
        )


if __name__ == "__main__":
    main()
