"""
VNN-COMP benchmark suite support.

Downloads, loads, and verifies standard neural network verification
benchmarks including ACAS Xu and VNN-COMP 2024 complex benchmarks
(cGAN, NN4Sys, LinearizeNN, ml4acopf, ViT, Collins Aerospace,
LSNC-ReLU, CCTSDB).

Models are stored in the project-local ./benchmarks_data/ directory.
"""

from .registry import BENCHMARKS, BenchmarkInfo, list_benchmarks
from .downloader import download_benchmark, download_all
from .loader import load_benchmark_instance, load_onnx_model
from .vnnlib_parser import parse_vnnlib
from .verifier import verify_instance, BenchmarkVerificationResult

__all__ = [
    "BENCHMARKS",
    "BenchmarkInfo",
    "list_benchmarks",
    "download_benchmark",
    "download_all",
    "load_benchmark_instance",
    "load_onnx_model",
    "parse_vnnlib",
    "verify_instance",
    "BenchmarkVerificationResult",
]
