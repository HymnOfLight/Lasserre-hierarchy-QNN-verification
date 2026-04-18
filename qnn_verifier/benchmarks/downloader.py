"""
Benchmark downloader — fetches ONNX models and VNNLIB properties
from the VNN-COMP 2024 repository into the local benchmarks_data/ directory.
"""

import gzip
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from .registry import (
    BENCHMARKS,
    BenchmarkInfo,
    DEFAULT_DATA_DIR,
    VNNCOMP_REPO,
    VNNCOMP_LARGE_MODELS_URL,
)

logger = logging.getLogger(__name__)


def _run(cmd: List[str], cwd: Optional[str] = None, timeout: int = 600):
    """Run a subprocess command."""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        logger.error(f"Command failed: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def _gunzip_recursive(directory: Path):
    """Decompress all .gz files in a directory tree."""
    for gz_file in directory.rglob("*.gz"):
        out_file = gz_file.with_suffix("")
        if out_file.exists():
            gz_file.unlink()
            continue
        try:
            with gzip.open(gz_file, "rb") as f_in:
                with open(out_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            gz_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to gunzip {gz_file}: {e}")


def _clone_repo(data_dir: Path):
    """Clone the VNN-COMP 2024 benchmarks repo (sparse checkout)."""
    repo_dir = data_dir / "_vnncomp_repo"
    if repo_dir.exists():
        logger.info("VNN-COMP repo already cloned")
        return repo_dir

    os.makedirs(data_dir, exist_ok=True)
    _run(["git", "clone", "--depth", "1", VNNCOMP_REPO, str(repo_dir)], timeout=300)
    return repo_dir


def _download_large_models(data_dir: Path, repo_dir: Path):
    """Download and extract the large model archive from VNN-COMP."""
    zip_path = data_dir / "large_models.zip"
    extract_dir = data_dir / "large_models"

    if extract_dir.exists():
        logger.info("Large models already downloaded")
        return

    logger.info(f"Downloading large models from {VNNCOMP_LARGE_MODELS_URL}")
    _run(["wget", "-q", "--show-progress", VNNCOMP_LARGE_MODELS_URL,
          "-O", str(zip_path)], timeout=1800)

    logger.info("Extracting large models...")
    _run(["unzip", "-q", "-o", str(zip_path), "-d", str(extract_dir)], timeout=600)

    # Move files into the repo benchmark directories
    vnncomp_dir = extract_dir / "vnncomp2024"
    if vnncomp_dir.exists():
        for bench_dir in vnncomp_dir.iterdir():
            if not bench_dir.is_dir():
                continue
            seed_dir = bench_dir / "seed_896832480"
            if not seed_dir.exists():
                for sub in bench_dir.iterdir():
                    if sub.is_dir() and sub.name.startswith("seed_"):
                        seed_dir = sub
                        break
            if seed_dir.exists():
                target = repo_dir / "benchmarks" / bench_dir.name
                os.makedirs(target / "onnx", exist_ok=True)
                os.makedirs(target / "vnnlib", exist_ok=True)
                for f in seed_dir.rglob("*"):
                    if f.is_file():
                        dest = target / f.name
                        if f.suffix in (".onnx", ".gz") and "onnx" in f.name.lower():
                            dest = target / "onnx" / f.name
                        elif f.suffix in (".vnnlib", ".gz"):
                            dest = target / "vnnlib" / f.name
                        shutil.move(str(f), str(dest))

    # Cleanup
    if zip_path.exists():
        zip_path.unlink()
    if extract_dir.exists():
        shutil.rmtree(extract_dir, ignore_errors=True)


def download_benchmark(
    benchmark_name: str,
    data_dir: Optional[str] = None,
    skip_large: bool = False,
) -> Path:
    """
    Download a specific benchmark to the local data directory.

    Args:
        benchmark_name: Key from BENCHMARKS registry (e.g. "acasxu", "cgan")
        data_dir: Override the default data directory
        skip_large: Skip the large model download (for benchmarks that need it)

    Returns:
        Path to the benchmark directory containing onnx/ and vnnlib/ subdirs.
    """
    if benchmark_name not in BENCHMARKS:
        available = ", ".join(BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark '{benchmark_name}'. Available: {available}")

    info = BENCHMARKS[benchmark_name]
    target_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    os.makedirs(target_dir, exist_ok=True)

    repo_dir = _clone_repo(target_dir)
    bench_dir = repo_dir / "benchmarks" / info.dir_name

    if not bench_dir.exists():
        raise RuntimeError(f"Benchmark directory not found: {bench_dir}")

    # Download large models if needed
    if info.needs_large_download and not skip_large:
        onnx_dir = bench_dir / "onnx"
        has_onnx = any(onnx_dir.glob("*.onnx")) if onnx_dir.exists() else False
        if not has_onnx:
            _download_large_models(target_dir, repo_dir)

    # Decompress .gz files
    _gunzip_recursive(bench_dir)

    # Verify
    onnx_count = len(list((bench_dir / "onnx").glob("*.onnx"))) if (bench_dir / "onnx").exists() else 0
    vnnlib_count = len(list((bench_dir / "vnnlib").glob("*.vnnlib"))) if (bench_dir / "vnnlib").exists() else 0

    logger.info(
        f"Benchmark '{info.name}' ready: {onnx_count} ONNX models, "
        f"{vnnlib_count} VNNLIB properties in {bench_dir}"
    )

    return bench_dir


def download_all(
    data_dir: Optional[str] = None,
    skip_large: bool = False,
    categories: Optional[List[str]] = None,
) -> dict:
    """
    Download all benchmarks (or a subset by category).

    Returns dict mapping benchmark_name -> local Path.
    """
    paths = {}
    for name, info in BENCHMARKS.items():
        if categories and info.category not in categories:
            continue
        try:
            paths[name] = download_benchmark(name, data_dir, skip_large)
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            paths[name] = None
    return paths
