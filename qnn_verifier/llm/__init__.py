"""
LLM (Large Language Model) verification module.

Provides adversarial robustness verification for quantized large language
models (e.g., Qwen2-7B) under embedding-space perturbations, using
forward-pass-anchored Jacobian bounding and Lasserre hierarchy SDP.
"""

from .loader import load_llm, LLMWrapper
from .propagation import LLMBoundPropagator
from .verifier import LLMRobustnessVerifier

__all__ = [
    "load_llm",
    "LLMWrapper",
    "LLMBoundPropagator",
    "LLMRobustnessVerifier",
]
