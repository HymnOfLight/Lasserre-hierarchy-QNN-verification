"""
LLM model loader for Qwen2 and other causal LMs.

Supports two download sources:
  - ModelScope (魔搭社区, https://modelscope.cn) — 国内默认, 速度快
  - HuggingFace (https://huggingface.co) — 可通过 --mirror huggingface 切换

Models are cached in the project-local ./models/ directory (not the
global ~/.cache), making the project self-contained and reproducible.

Supports loading in fp16, int8, int4 (bitsandbytes), or fp32 modes,
with automatic device selection (CUDA if available, else CPU).
"""

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# Project-local model cache directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models"

# ModelScope model id mapping (HuggingFace id -> ModelScope id)
_MODELSCOPE_MAP = {
    "Qwen/Qwen2-0.5B": "qwen/Qwen2-0.5B",
    "Qwen/Qwen2-1.5B": "qwen/Qwen2-1.5B",
    "Qwen/Qwen2-7B": "qwen/Qwen2-7B",
    "Qwen/Qwen2.5-0.5B": "qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B": "qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B": "qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B": "qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B": "qwen/Qwen2.5-14B",
    "Qwen/Qwen3-8B": "qwen/Qwen3-8B",
}


class LLMWrapper:
    """
    Lightweight wrapper around a causal LM that exposes the interfaces
    needed by the verification pipeline.
    """

    def __init__(self, model, tokenizer, model_name: str, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.model.eval()

    def get_embedding_matrix(self) -> torch.Tensor:
        emb = self.model.get_input_embeddings()
        return emb.weight.detach()

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.model.get_input_embeddings()
        return emb(input_ids.to(self.device))

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    def forward_from_embeddings(
        self, embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.model(inputs_embeds=embeds, attention_mask=attention_mask)
        return out.logits

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.model(input_ids=input_ids.to(self.device), **kwargs)
        return out.logits

    def tokenize(self, text: str) -> dict:
        return self.tokenizer(text, return_tensors="pt")

    def decode(self, token_ids) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def predict_next_token(self, text: str, top_k: int = 5):
        inputs = self.tokenize(text)
        input_ids = inputs["input_ids"].to(self.device)
        with torch.no_grad():
            logits = self.forward(input_ids)
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        topk = torch.topk(probs, top_k)
        results = []
        for i in range(top_k):
            tid = topk.indices[i].item()
            results.append({
                "token_id": tid,
                "token": self.decode([tid]),
                "probability": topk.values[i].item(),
                "logit": last_logits[tid].item(),
            })
        return results


# ------------------------------------------------------------------
# Download helpers
# ------------------------------------------------------------------

def _download_from_modelscope(model_id: str, cache_dir: str) -> str:
    """Download a model from ModelScope and return the local path."""
    from modelscope import snapshot_download
    local_path = snapshot_download(model_id, cache_dir=cache_dir)
    logger.info(f"ModelScope download complete: {local_path}")
    return local_path


def _resolve_model_path(
    model_name: str,
    mirror: str,
    cache_dir: Optional[str],
) -> str:
    """
    Resolve the model to a local path, downloading if necessary.

    For 'modelscope' mirror: download to cache_dir via ModelScope SDK.
    For 'huggingface' mirror: let transformers handle caching into cache_dir.
    For a local path that already exists: use it directly.
    """
    # Already a local directory
    if os.path.isdir(model_name):
        logger.info(f"Using local model directory: {model_name}")
        return model_name

    if cache_dir is None:
        cache_dir = str(DEFAULT_MODEL_DIR)
    os.makedirs(cache_dir, exist_ok=True)

    if mirror == "modelscope":
        ms_id = _MODELSCOPE_MAP.get(model_name, model_name)
        logger.info(f"Downloading from ModelScope: {ms_id} -> {cache_dir}")
        return _download_from_modelscope(ms_id, cache_dir)
    else:
        # HuggingFace: set cache env so transformers downloads here
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        return model_name


# ------------------------------------------------------------------
# Main loader
# ------------------------------------------------------------------

def load_llm(
    model_name: str = "Qwen/Qwen2-0.5B",
    quantization: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    trust_remote_code: bool = True,
    max_memory: Optional[dict] = None,
    mirror: str = "modelscope",
    cache_dir: Optional[str] = None,
) -> LLMWrapper:
    """
    Load a causal LM for verification.

    Args:
        model_name: Model id or local path.
            Examples: "Qwen/Qwen2-0.5B", "Qwen/Qwen2-7B", "./models/my_model"
        quantization: None (auto), "int8", "int4", "fp16", "fp32"
        device: "cuda", "cpu", or None (auto)
        torch_dtype: Override dtype ("float16", "bfloat16", "float32")
        trust_remote_code: Allow model code execution
        max_memory: Memory budget per device
        mirror: Download source — "modelscope" (国内默认) or "huggingface"
        cache_dir: Model cache directory (default: ./models/)

    Returns:
        LLMWrapper instance ready for verification.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if cache_dir is None:
        cache_dir = str(DEFAULT_MODEL_DIR)

    logger.info(
        f"Loading LLM: {model_name} "
        f"(mirror={mirror}, quantization={quantization}, device={device}, "
        f"cache_dir={cache_dir})"
    )

    # Resolve to local path (download if needed)
    local_path = _resolve_model_path(model_name, mirror, cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        local_path, trust_remote_code=trust_remote_code, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "cache_dir": cache_dir,
    }

    if torch_dtype is not None:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model_kwargs["torch_dtype"] = dtype_map.get(torch_dtype, torch.float32)
    elif device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    if quantization == "int8":
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"
    elif quantization == "int4":
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"
    elif device == "cuda":
        model_kwargs["device_map"] = "auto"

    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory

    model = AutoModelForCausalLM.from_pretrained(local_path, **model_kwargs)

    if "device_map" not in model_kwargs and device != "cpu":
        model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Loaded {model_name}: {n_params/1e9:.2f}B params, "
        f"{model.config.num_hidden_layers} layers, "
        f"hidden={model.config.hidden_size}, vocab={model.config.vocab_size}"
    )

    return LLMWrapper(model, tokenizer, model_name, device)
