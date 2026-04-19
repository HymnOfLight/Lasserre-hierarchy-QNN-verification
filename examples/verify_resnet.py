#!/usr/bin/env python3
"""
Verify adversarial robustness of a quantized ResNet (including ResNet-121).

Supports multiple verification methods:
  jacobian  — Forward-pass-anchored Jacobian bounds (fast, default)
  smt       — Multi-solver SMT portfolio (Z3+CVC5+Bitwuzla, exact)
  compare   — Run both and print comparison table

Usage:
    # Default Jacobian verification
    python verify_resnet.py --arch resnet18 --epsilon 0.01

    # ResNet-101 (≈ ResNet-121) with SMT solver, 32 cores, 1 hour timeout
    python verify_resnet.py --arch resnet101 --epsilon 0.001 \
        --solver smt --cores 32 --timeout 3600

    # Compare Jacobian vs SMT on the same input
    python verify_resnet.py --arch resnet18 --epsilon 0.01 --compare --cores 32
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from qnn_verifier.verification.pipeline import VerificationPipeline
from qnn_verifier.benchmarks.smt_solver import verify_pytorch_with_smt, detect_solvers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def quantize_weights(model: nn.Module, n_bits: int = 8) -> nn.Module:
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


def create_model(arch, n_classes, input_size, n_bits):
    import torchvision.models as models
    arch_map = {
        "resnet18": models.resnet18, "resnet34": models.resnet34,
        "resnet50": models.resnet50, "resnet101": models.resnet101,
    }
    # ResNet-121 maps to ResNet-101 (closest standard variant)
    key = arch.lower().replace("-", "").replace("_", "")
    if key == "resnet121":
        key = "resnet101"
    factory = arch_map.get(key, arch_map.get(arch, models.resnet18))
    model = factory(num_classes=n_classes)

    if input_size <= 64:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    model = quantize_weights(model, n_bits)
    model.eval()
    return model


def run_jacobian(pipeline, x0, true_label, epsilons):
    """Run Jacobian-based verification."""
    results = {}
    for eps in epsilons:
        t0 = time.time()
        r = pipeline.verify(x0, true_label=true_label, epsilon=eps)
        elapsed = time.time() - t0
        verified = r.get("verified", False)
        margin = r.get("pop_result", {}).get("lower_bound", float("-inf"))
        results[eps] = {"verified": verified, "time": elapsed, "margin": margin, "method": "jacobian"}
        tag = "VERIFIED" if verified else "UNKNOWN "
        print(f"    eps={eps:.4f}: [{tag}]  margin={margin:+.6f}  time={elapsed:.2f}s")
    return results


def run_smt(model, x0, true_label, epsilons, input_shape, n_classes, timeout, n_cores, arch):
    """Run SMT portfolio verification."""
    results = {}
    for eps in epsilons:
        print(f"    eps={eps:.4f}: running SMT portfolio (timeout={timeout}s, cores={n_cores})...")
        r = verify_pytorch_with_smt(
            model=model, x0=x0, epsilon=eps,
            true_label=true_label, input_shape=input_shape,
            n_classes=n_classes, timeout=timeout,
            total_cores=n_cores, model_name=arch,
        )
        elapsed = r["time_seconds"]
        verified = r["result"] == "verified"
        tag = {"verified": "VERIFIED", "violated": "VIOLATED", "unknown": "UNKNOWN "}.get(r["result"], r["result"])
        smt2_file = r.get("smt2_file", "")
        print(f"    eps={eps:.4f}: [{tag}]  solver={r.get('solver','?')}  "
              f"time={elapsed:.2f}s  file={smt2_file}")
        results[eps] = {"verified": verified, "result": r["result"],
                        "time": elapsed, "method": "smt", "file": smt2_file}
    return results


def main():
    parser = argparse.ArgumentParser(description="Quantized ResNet verification")
    parser.add_argument("--arch", type=str, default="resnet18",
                        help="resnet18 / resnet34 / resnet50 / resnet101 / resnet121")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Single epsilon (default: multi-epsilon sweep)")
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--n-bits", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--solver", type=str, default="jacobian",
                        choices=["jacobian", "smt", "compare"],
                        help="jacobian (fast), smt (exact), compare (both)")
    parser.add_argument("--cores", type=int, default=0,
                        help="CPU cores for SMT solvers (0=auto, e.g. 32)")
    parser.add_argument("--timeout", type=float, default=3600,
                        help="SMT solver timeout in seconds (default: 3600=1hr)")
    parser.add_argument("--compare", action="store_true",
                        help="Run both Jacobian and SMT, print comparison")
    args = parser.parse_args()

    n_cores = args.cores or os.cpu_count() or 4
    if args.compare:
        args.solver = "compare"

    epsilons = [args.epsilon] if args.epsilon else [0.001, 0.005, 0.01]

    # ---- Create model ----
    print("=" * 72)
    print(f"  Quantized {args.arch.upper()} Robustness Verification")
    print(f"  Solver: {args.solver} | Cores: {n_cores} | Timeout: {args.timeout}s")
    print("=" * 72)

    model = create_model(args.arch, args.n_classes, args.input_size, args.n_bits)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.arch} | Params: {n_params:,} | {args.n_bits}-bit quantized")

    if args.solver in ("smt", "compare"):
        available = detect_solvers()
        print(f"  SMT solvers: {available}")

    input_shape = (1, 3, args.input_size, args.input_size)

    # ---- Prepare input ----
    np.random.seed(42)
    x0 = np.random.rand(*input_shape[1:]).astype(np.float32)
    x0 = x0 / x0.max()

    with torch.no_grad():
        pred = model(torch.tensor(x0).unsqueeze(0))
        true_label = pred.argmax(dim=-1).item()
    print(f"  Input shape: {x0.shape} | Prediction: class {true_label}")
    print(f"  Logits: {pred[0, :5].tolist()}")

    # ---- Save model ----
    model_path = f"models/quantized_{args.arch}.pth"
    os.makedirs("models", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": args.arch,
                "n_classes": args.n_classes, "n_bits": args.n_bits}, model_path)

    # ---- Run verification ----
    if args.solver == "jacobian":
        print(f"\n--- Jacobian Bounds ---")
        pipeline = VerificationPipeline(
            model_path=model_path, model_arch=args.arch,
            input_shape=input_shape, n_classes=args.n_classes,
            n_bits=args.n_bits, poly_degree=4, max_lasserre_order=2,
        )
        run_jacobian(pipeline, x0, true_label, epsilons)

    elif args.solver == "smt":
        print(f"\n--- SMT Portfolio (cores={n_cores}, timeout={args.timeout}s) ---")
        run_smt(model, x0, true_label, epsilons, input_shape,
                args.n_classes, args.timeout, n_cores, args.arch)

    elif args.solver == "compare":
        # Phase 1: Jacobian
        print(f"\n--- Phase 1: Jacobian Bounds ---")
        pipeline = VerificationPipeline(
            model_path=model_path, model_arch=args.arch,
            input_shape=input_shape, n_classes=args.n_classes,
            n_bits=args.n_bits, poly_degree=4, max_lasserre_order=2,
        )
        jac_results = run_jacobian(pipeline, x0, true_label, epsilons)

        # Phase 2: SMT
        print(f"\n--- Phase 2: SMT Portfolio (cores={n_cores}, timeout={args.timeout}s) ---")
        smt_results = run_smt(model, x0, true_label, epsilons, input_shape,
                              args.n_classes, args.timeout, n_cores, args.arch)

        # Comparison table
        print(f"\n{'='*72}")
        print(f"  {'eps':>8}  {'Jacobian':^20}  {'SMT':^20}  {'SMT file'}")
        print(f"  {'-'*68}")
        for eps in epsilons:
            jr = jac_results.get(eps, {})
            sr = smt_results.get(eps, {})
            jv = "VERIFIED" if jr.get("verified") else "UNKNOWN"
            jt = f"{jr.get('time', 0):.2f}s"
            sv = sr.get("result", "?").upper()
            st = f"{sr.get('time', 0):.2f}s"
            sf = sr.get("file", "")
            print(f"  {eps:>8.4f}  {jv:>8} {jt:>8}    {sv:>8} {st:>8}  {sf}")
        print(f"{'='*72}")

    print("\nDone.")


if __name__ == "__main__":
    main()
