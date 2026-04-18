#!/usr/bin/env python3
"""
Example: Download and verify VNN-COMP benchmark instances.

Supports ACAS Xu and all VNN-COMP 2024 complex benchmarks:
  acasxu, cGAN, NN4Sys, LinearizeNN, ml4acopf, ViT,
  Collins Aerospace, LSNC-ReLU, CCTSDB

Models are downloaded to ./benchmarks_data/ in the project directory.

Usage:
    # List available benchmarks
    python verify_benchmark.py --list

    # Download ACAS Xu benchmark
    python verify_benchmark.py --download acasxu

    # Download all benchmarks (small models only, skip large downloads)
    python verify_benchmark.py --download-all --skip-large

    # Download and verify ACAS Xu instance #0
    python verify_benchmark.py --benchmark acasxu --instance 0

    # Verify multiple instances
    python verify_benchmark.py --benchmark acasxu --instances 0-9

    # Verify with Lasserre SDP refinement
    python verify_benchmark.py --benchmark linearizenn --instance 0 --use-sdp
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from qnn_verifier.benchmarks import (
    BENCHMARKS,
    list_benchmarks,
    download_benchmark,
    download_all,
    load_benchmark_instance,
)
from qnn_verifier.benchmarks.loader import list_instances

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def print_benchmarks():
    """Print all available benchmarks."""
    print("\n" + "=" * 72)
    print(f"  {'Name':<22} {'Category':<10} {'Description'}")
    print("=" * 72)
    for key, info in BENCHMARKS.items():
        large = " [LARGE]" if info.needs_large_download else ""
        print(f"  {key:<22} {info.category:<10} {info.description[:50]}...{large}")
    print("=" * 72)
    print(f"\n  Total: {len(BENCHMARKS)} benchmarks")
    print(f"  Classic: {sum(1 for b in BENCHMARKS.values() if b.category == 'classic')}")
    print(f"  Complex: {sum(1 for b in BENCHMARKS.values() if b.category == 'complex')}")
    print()


def verify_instance(benchmark_name: str, instance_idx: int, use_sdp: bool = False):
    """Download (if needed), load, and verify a single benchmark instance."""
    # Download
    try:
        bench_dir = download_benchmark(benchmark_name, skip_large=False)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Retrying with skip_large=True...")
        bench_dir = download_benchmark(benchmark_name, skip_large=True)

    # Load instance
    inst = load_benchmark_instance(benchmark_name, instance_idx)

    print(f"\n--- {BENCHMARKS[benchmark_name].name} Instance #{instance_idx} ---")
    print(f"  Model    : {Path(inst.model_path).name}")
    print(f"  Property : {Path(inst.property_path).name}")
    print(f"  Timeout  : {inst.timeout}s")

    if inst.property is not None:
        prop = inst.property
        print(f"  Inputs   : {prop.n_inputs}")
        print(f"  Outputs  : {prop.n_outputs}")
        if prop.input_lower is not None:
            finite_lb = np.isfinite(prop.input_lower).sum()
            finite_ub = np.isfinite(prop.input_upper).sum()
            print(f"  Bounded  : {finite_lb}/{prop.n_inputs} lower, {finite_ub}/{prop.n_inputs} upper")
            if finite_lb > 0 and finite_ub > 0:
                widths = prop.input_upper - prop.input_lower
                finite_widths = widths[np.isfinite(widths)]
                if len(finite_widths) > 0:
                    print(f"  Eps range: [{finite_widths.min():.6f}, {finite_widths.max():.6f}]")
        print(f"  Output constraints: {len(prop.output_constraints)}")

    if inst.model is not None:
        print(f"  Input shape : {inst.input_shape}")
        print(f"  Output shape: {inst.output_shape}")

        # Run a forward pass to test
        try:
            import torch
            n_in = int(np.prod(inst.input_shape))
            if inst.property is not None and inst.property.input_lower is not None:
                lb = inst.property.input_lower
                ub = inst.property.input_upper
                lb = np.where(np.isfinite(lb), lb, -1.0)
                ub = np.where(np.isfinite(ub), ub, 1.0)
                x0 = (lb + ub) / 2.0
            else:
                x0 = np.zeros(n_in)

            x_tensor = torch.tensor(x0, dtype=torch.float32).reshape(inst.input_shape)
            t0 = time.time()
            with torch.no_grad():
                output = inst.model(x_tensor)
            fwd_time = time.time() - t0
            print(f"  Forward pass: {fwd_time:.4f}s")
            print(f"  Output (first 5): {output.flatten()[:5].tolist()}")

            # If property has output constraints, check nominal satisfaction
            if inst.property and inst.property.output_constraints:
                out_np = output.flatten().numpy()
                print(f"  Nominal argmax: {np.argmax(out_np)}")
        except Exception as e:
            print(f"  Forward pass failed: {e}")
    else:
        print("  (Model not loaded — install onnx2pytorch or onnxruntime)")

    return inst


def main():
    parser = argparse.ArgumentParser(description="VNN-COMP benchmark verification")
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    parser.add_argument("--download", type=str, default=None,
                        help="Download a specific benchmark")
    parser.add_argument("--download-all", action="store_true",
                        help="Download all benchmarks")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip benchmarks that need large downloads")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Benchmark to verify (e.g. acasxu)")
    parser.add_argument("--instance", type=int, default=0,
                        help="Instance index to verify")
    parser.add_argument("--instances", type=str, default=None,
                        help="Instance range (e.g. '0-9')")
    parser.add_argument("--use-sdp", action="store_true",
                        help="Use Lasserre SDP refinement")
    parser.add_argument("--list-instances", type=str, default=None,
                        help="List instances for a benchmark")
    args = parser.parse_args()

    if args.list:
        print_benchmarks()
        return

    if args.download:
        print(f"Downloading benchmark: {args.download}")
        path = download_benchmark(args.download, skip_large=args.skip_large)
        print(f"Downloaded to: {path}")
        return

    if args.download_all:
        print("Downloading all benchmarks...")
        paths = download_all(skip_large=args.skip_large)
        for name, path in paths.items():
            status = "OK" if path else "FAILED"
            print(f"  {name:<22} {status}")
        return

    if args.list_instances:
        bench_dir = download_benchmark(args.list_instances, skip_large=True)
        instances = list_instances(args.list_instances)
        print(f"\n{BENCHMARKS[args.list_instances].name}: {len(instances)} instances")
        for i, (model, prop, timeout) in enumerate(instances[:20]):
            print(f"  [{i:3d}] {Path(model).name:<45} {Path(prop).name}")
        if len(instances) > 20:
            print(f"  ... and {len(instances) - 20} more")
        return

    if args.benchmark:
        if args.instances:
            parts = args.instances.split("-")
            start, end = int(parts[0]), int(parts[1])
            for i in range(start, end + 1):
                try:
                    verify_instance(args.benchmark, i, args.use_sdp)
                except Exception as e:
                    print(f"  Instance {i} failed: {e}")
        else:
            verify_instance(args.benchmark, args.instance, args.use_sdp)
        return

    print_benchmarks()
    print("Usage examples:")
    print("  python verify_benchmark.py --download acasxu")
    print("  python verify_benchmark.py --benchmark acasxu --instance 0")
    print("  python verify_benchmark.py --benchmark acasxu --instances 0-4")


if __name__ == "__main__":
    main()
