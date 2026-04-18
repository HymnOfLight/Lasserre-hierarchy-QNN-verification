"""
Benchmark instance loader — loads ONNX models and VNNLIB properties
into PyTorch models ready for verification.
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .registry import BENCHMARKS, DEFAULT_DATA_DIR
from .vnnlib_parser import VNNLIBProperty, parse_vnnlib

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkInstance:
    """A single verification instance: model + property."""
    benchmark_name: str
    model_path: str
    property_path: str
    timeout: int = 300
    model: Optional[torch.nn.Module] = None
    property: Optional[VNNLIBProperty] = None
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None


def load_onnx_model(onnx_path: str) -> Tuple[torch.nn.Module, Tuple, Tuple]:
    """
    Load an ONNX model and convert to PyTorch.

    Returns (model, input_shape, output_shape).
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        raise ImportError("pip install onnx onnx2pytorch  # required for ONNX loading")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Get input/output shapes
    input_shape = tuple(
        d.dim_value if d.dim_value > 0 else 1
        for d in onnx_model.graph.input[0].type.tensor_type.shape.dim
    )
    output_shape = tuple(
        d.dim_value if d.dim_value > 0 else 1
        for d in onnx_model.graph.output[0].type.tensor_type.shape.dim
    )

    # Convert to PyTorch
    try:
        from onnx2pytorch import ConvertModel
        pytorch_model = ConvertModel(onnx_model, experimental=True)
        pytorch_model.eval()
        return pytorch_model, input_shape, output_shape
    except ImportError:
        pass

    # Fallback: use torch.onnx or manual conversion for simple networks
    try:
        import onnxruntime as ort
        return _OnnxRuntimeWrapper(onnx_path, input_shape, output_shape), input_shape, output_shape
    except ImportError:
        raise ImportError(
            "Install one of: onnx2pytorch, onnxruntime\n"
            "  pip install onnx2pytorch  # preferred\n"
            "  pip install onnxruntime   # fallback"
        )


class _OnnxRuntimeWrapper(torch.nn.Module):
    """Wraps ONNX Runtime session as a PyTorch-compatible module."""

    def __init__(self, onnx_path: str, input_shape, output_shape):
        super().__init__()
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self._input_shape = input_shape
        self._output_shape = output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().float().numpy()
        outputs = self.session.run(None, {self.input_name: x_np})
        return torch.from_numpy(outputs[0]).to(x.device)


def load_benchmark_instance(
    benchmark_name: str,
    instance_idx: int = 0,
    data_dir: Optional[str] = None,
    load_model: bool = True,
) -> BenchmarkInstance:
    """
    Load a specific benchmark instance by index.

    Args:
        benchmark_name: Key from BENCHMARKS registry
        instance_idx: 0-based index into instances.csv
        data_dir: Override data directory
        load_model: Whether to load the ONNX model into memory

    Returns:
        BenchmarkInstance with model and property loaded.
    """
    if benchmark_name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    info = BENCHMARKS[benchmark_name]
    base_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    bench_dir = base_dir / "_vnncomp_repo" / "benchmarks" / info.dir_name

    if not bench_dir.exists():
        raise FileNotFoundError(
            f"Benchmark not downloaded: {bench_dir}\n"
            f"Run: from qnn_verifier.benchmarks import download_benchmark; "
            f"download_benchmark('{benchmark_name}')"
        )

    # Parse instances.csv
    instances_csv = bench_dir / "instances.csv"
    if not instances_csv.exists():
        raise FileNotFoundError(f"instances.csv not found in {bench_dir}")

    instances = []
    with open(instances_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                instances.append(row)

    if instance_idx >= len(instances):
        raise IndexError(
            f"Instance index {instance_idx} out of range "
            f"(benchmark has {len(instances)} instances)"
        )

    row = instances[instance_idx]
    model_rel = row[0].strip()
    prop_rel = row[1].strip()
    timeout = int(row[2].strip()) if len(row) > 2 else 300

    model_path = str(bench_dir / model_rel)
    prop_path = str(bench_dir / prop_rel)

    inst = BenchmarkInstance(
        benchmark_name=benchmark_name,
        model_path=model_path,
        property_path=prop_path,
        timeout=timeout,
    )

    # Load property
    if Path(prop_path).exists():
        inst.property = parse_vnnlib(prop_path)

    # Load model
    if load_model and Path(model_path).exists():
        try:
            model, in_shape, out_shape = load_onnx_model(model_path)
            inst.model = model
            inst.input_shape = in_shape
            inst.output_shape = out_shape
        except Exception as e:
            logger.warning(f"Failed to load ONNX model {model_path}: {e}")

    return inst


def list_instances(
    benchmark_name: str,
    data_dir: Optional[str] = None,
) -> List[Tuple[str, str, int]]:
    """
    List all instances for a benchmark.

    Returns list of (model_path, property_path, timeout) tuples.
    """
    info = BENCHMARKS[benchmark_name]
    base_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    bench_dir = base_dir / "_vnncomp_repo" / "benchmarks" / info.dir_name
    instances_csv = bench_dir / "instances.csv"

    if not instances_csv.exists():
        return []

    instances = []
    with open(instances_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                model = row[0].strip()
                prop = row[1].strip()
                timeout = int(row[2].strip()) if len(row) > 2 else 300
                instances.append((model, prop, timeout))
    return instances
