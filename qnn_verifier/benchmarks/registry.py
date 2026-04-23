"""
Benchmark registry — metadata for all supported verification benchmarks.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "benchmarks_data"

VNNCOMP_REPO = "https://github.com/ChristopherBrix/vnncomp2024_benchmarks.git"
VNNCOMP_LARGE_MODELS_URL = (
    "https://rwth-aachen.sciebo.de/s/RapAoed1dxG1PMs/download"
)


@dataclass
class BenchmarkInfo:
    name: str
    category: str  # "classic" or "complex"
    description: str
    dir_name: str  # directory name inside benchmarks repo
    n_instances: int = 0
    input_format: str = "onnx"
    property_format: str = "vnnlib"
    needs_large_download: bool = False
    tags: List[str] = field(default_factory=list)


BENCHMARKS: Dict[str, BenchmarkInfo] = {
    "acasxu": BenchmarkInfo(
        name="ACAS Xu",
        category="classic",
        description="Airborne Collision Avoidance System for unmanned aircraft. "
                    "45 networks (5 inputs, 5 outputs, 6 hidden layers of 50 neurons). "
                    "Standard safety properties from Reluplex.",
        dir_name="acasxu_2023",
        n_instances=180,
        tags=["safety-critical", "control", "ReLU"],
    ),
    "cgan": BenchmarkInfo(
        name="cGAN",
        category="complex",
        description="Conditional Generative Adversarial Network verification. "
                    "Image generation robustness with transposed convolutions.",
        dir_name="cgan_2023",
        n_instances=54,
        needs_large_download=True,
        tags=["generative", "convolution", "image"],
    ),
    "nn4sys": BenchmarkInfo(
        name="NN4Sys",
        category="complex",
        description="Neural networks for computer systems (learned index, "
                    "cardinality estimation, congestion control).",
        dir_name="nn4sys_2023",
        n_instances=45,
        needs_large_download=True,
        tags=["systems", "ReLU", "regression"],
    ),
    "linearizenn": BenchmarkInfo(
        name="LinearizeNN",
        category="complex",
        description="Verification of linearised neural network controllers "
                    "for autonomous systems.",
        dir_name="linearizenn",
        n_instances=24,
        tags=["control", "linearisation", "ReLU"],
    ),
    "ml4acopf": BenchmarkInfo(
        name="ml4acopf",
        category="complex",
        description="Machine learning for AC Optimal Power Flow. "
                    "Verifying safety constraints in power grid operation.",
        dir_name="ml4acopf_2024",
        n_instances=36,
        tags=["power-systems", "safety", "ReLU"],
    ),
    "vit": BenchmarkInfo(
        name="ViT",
        category="complex",
        description="Vision Transformer robustness verification. "
                    "Adversarial perturbation of image patches.",
        dir_name="vit_2023",
        n_instances=48,
        needs_large_download=True,
        tags=["transformer", "vision", "attention"],
    ),
    "collins_aerospace": BenchmarkInfo(
        name="Collins Aerospace",
        category="complex",
        description="Industrial aerospace neural network verification "
                    "benchmarks from Collins Aerospace.",
        dir_name="collins_aerospace_benchmark",
        n_instances=36,
        tags=["aerospace", "safety-critical", "industrial"],
    ),
    "lsnc": BenchmarkInfo(
        name="LSNC-ReLU",
        category="complex",
        description="Lyapunov-stable Neural Control verification. "
                    "Certifying stability of learned controllers.",
        dir_name="lsnc",
        n_instances=60,
        tags=["control", "stability", "Lyapunov", "ReLU"],
    ),
    "cctsdb": BenchmarkInfo(
        name="CCTSDB",
        category="complex",
        description="China Traffic Sign Detection Benchmark with YOLO-based "
                    "object detection networks.",
        dir_name="cctsdb_yolo_2023",
        n_instances=24,
        needs_large_download=True,
        tags=["detection", "YOLO", "traffic-signs", "convolution"],
    ),
    "vggnet16": BenchmarkInfo(
        name="VGGNet16",
        category="complex",
        description="VGG-16 image classification robustness verification. "
                    "18 ImageNet instances with adversarial perturbation.",
        dir_name="vggnet16_2023",
        n_instances=18,
        needs_large_download=True,
        tags=["classification", "convolution", "ImageNet", "VGG"],
    ),
    "yolo": BenchmarkInfo(
        name="YOLO",
        category="complex",
        description="TinyYOLO object detection robustness verification. "
                    "72 instances with L_inf perturbation eps=1/255.",
        dir_name="yolo_2023",
        n_instances=72,
        needs_large_download=True,
        tags=["detection", "YOLO", "convolution", "object-detection"],
    ),
    "cifar100": BenchmarkInfo(
        name="CIFAR100",
        category="complex",
        description="CIFAR-100 ResNet classification robustness. "
                    "200 instances with small/medium/large models.",
        dir_name="cifar100",
        n_instances=200,
        needs_large_download=True,
        tags=["classification", "ResNet", "CIFAR", "convolution"],
    ),
}


def list_benchmarks(category: Optional[str] = None) -> List[BenchmarkInfo]:
    """List available benchmarks, optionally filtered by category."""
    benchmarks = list(BENCHMARKS.values())
    if category:
        benchmarks = [b for b in benchmarks if b.category == category]
    return benchmarks
