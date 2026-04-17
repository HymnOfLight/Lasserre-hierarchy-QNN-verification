"""
LLM model loader for Qwen2 and other HuggingFace causal LMs.

Supports loading in fp16, int8, int4 (bitsandbytes), or fp32 modes,
with automatic device selection (CUDA if available, else CPU).
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class LLMWrapper:
    """
    Lightweight wrapper around a HuggingFace causal LM that exposes
    the interfaces needed by the verification pipeline.
    """

    def __init__(self, model, tokenizer, model_name: str, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.model.eval()

    # ---- embedding access ------------------------------------------------

    def get_embedding_matrix(self) -> torch.Tensor:
        """Return the token embedding weight matrix (vocab_size, hidden_dim)."""
        emb = self.model.get_input_embeddings()
        return emb.weight.detach()

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token ids to embedding vectors."""
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

    # ---- forward pass helpers -------------------------------------------

    def forward_from_embeddings(
        self, embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Run the model from embedding tensors (bypassing the embedding layer).
        Returns logits of shape (batch, seq_len, vocab_size).
        """
        out = self.model(inputs_embeds=embeds, attention_mask=attention_mask)
        return out.logits

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Standard forward pass from token ids. Returns logits."""
        out = self.model(input_ids=input_ids.to(self.device), **kwargs)
        return out.logits

    def tokenize(self, text: str) -> dict:
        """Tokenize text and return dict with input_ids, attention_mask."""
        return self.tokenizer(text, return_tensors="pt")

    def decode(self, token_ids) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def predict_next_token(self, text: str, top_k: int = 5):
        """Get top-k next token predictions for a prompt."""
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


def load_llm(
    model_name: str = "Qwen/Qwen2-0.5B",
    quantization: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    trust_remote_code: bool = True,
    max_memory: Optional[dict] = None,
) -> LLMWrapper:
    """
    Load a HuggingFace causal LM for verification.

    Args:
        model_name: HuggingFace model id or local path.
            Examples: "Qwen/Qwen2-0.5B", "Qwen/Qwen2-1.5B",
                      "Qwen/Qwen2-7B", "Qwen/Qwen2.5-7B"
        quantization: None (auto), "int8", "int4", "fp16", "fp32"
        device: "cuda", "cpu", or None (auto)
        torch_dtype: Override dtype ("float16", "bfloat16", "float32")
        trust_remote_code: Allow model code from HuggingFace
        max_memory: Memory budget per device, e.g. {"cuda:0": "28GiB", "cpu": "80GiB"}

    Returns:
        LLMWrapper instance ready for verification.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading LLM: {model_name} (quantization={quantization}, device={device})")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
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

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if "device_map" not in model_kwargs and device != "cpu":
        model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Loaded {model_name}: {n_params/1e9:.2f}B params, "
        f"{model.config.num_hidden_layers} layers, "
        f"hidden={model.config.hidden_size}, vocab={model.config.vocab_size}"
    )

    return LLMWrapper(model, tokenizer, model_name, device)
