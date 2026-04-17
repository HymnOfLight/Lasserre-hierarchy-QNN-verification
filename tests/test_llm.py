"""Tests for the LLM verification module."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from qnn_verifier.llm.loader import LLMWrapper
from qnn_verifier.llm.propagation import LLMBoundPropagator
from qnn_verifier.llm.verifier import LLMRobustnessVerifier, LLMVerificationResult


class FakeLMConfig:
    hidden_size = 32
    vocab_size = 100
    num_hidden_layers = 2


class FakeLMHead(nn.Module):
    def __init__(self, hidden, vocab):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, hidden) * 0.1)
        self.bias = nn.Parameter(torch.zeros(vocab))

    def forward(self, x):
        return x @ self.weight.T + self.bias


class FakeCausalLM(nn.Module):
    """Minimal causal LM for unit testing (no real transformer blocks)."""
    def __init__(self, hidden=32, vocab=100, layers=2):
        super().__init__()
        self.config = FakeLMConfig()
        self.config.hidden_size = hidden
        self.config.vocab_size = vocab
        self.config.num_hidden_layers = layers
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.lm_head = FakeLMHead(hidden, vocab)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        h = self.fc(inputs_embeds)
        logits = self.lm_head(h)
        return type("Out", (), {"logits": logits})()


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    _vocab = {f"tok{i}": i for i in range(100)}

    def __call__(self, text, return_tensors="pt", **kw):
        ids = [hash(c) % 100 for c in text.split()]
        if not ids:
            ids = [0]
        t = torch.tensor([ids])
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def decode(self, ids, **kw):
        if isinstance(ids, (list, tuple)):
            return " ".join(f"tok{i}" for i in ids)
        return f"tok{ids}"


def _make_llm(hidden=32, vocab=100):
    model = FakeCausalLM(hidden=hidden, vocab=vocab)
    model.eval()
    tok = FakeTokenizer()
    return LLMWrapper(model, tok, "fake-lm", "cpu")


class TestLLMWrapper:
    def test_embedding_matrix(self):
        llm = _make_llm()
        E = llm.get_embedding_matrix()
        assert E.shape == (100, 32)

    def test_forward(self):
        llm = _make_llm()
        ids = torch.tensor([[1, 2, 3]])
        logits = llm.forward(ids)
        assert logits.shape == (1, 3, 100)

    def test_forward_from_embeddings(self):
        llm = _make_llm()
        embeds = torch.randn(1, 4, 32)
        logits = llm.forward_from_embeddings(embeds)
        assert logits.shape == (1, 4, 100)

    def test_predict_next_token(self):
        llm = _make_llm()
        preds = llm.predict_next_token("hello world", top_k=5)
        assert len(preds) == 5
        assert "token_id" in preds[0]
        assert "probability" in preds[0]
        assert preds[0]["probability"] >= preds[1]["probability"]

    def test_properties(self):
        llm = _make_llm(hidden=64, vocab=200)
        assert llm.hidden_size == 64
        assert llm.vocab_size == 200
        assert llm.num_layers == 2


class TestLLMBoundPropagator:
    def test_jacobian_bounds(self):
        llm = _make_llm()
        propagator = LLMBoundPropagator(llm, position=-1)
        ids = torch.tensor([1, 2, 3])
        bounds = propagator.compute_jacobian_bounds(ids, epsilon=0.01, top_k=5)

        assert "nominal_logits" in bounds
        assert "logit_lower" in bounds
        assert "logit_upper" in bounds
        assert len(bounds["token_ids"]) == 5
        assert np.all(bounds["logit_lower"] <= bounds["nominal_logits"] + 1e-6)
        assert np.all(bounds["logit_upper"] >= bounds["nominal_logits"] - 1e-6)

    def test_smaller_epsilon_tighter(self):
        llm = _make_llm()
        propagator = LLMBoundPropagator(llm, position=-1)
        ids = torch.tensor([1, 2, 3])

        b1 = propagator.compute_jacobian_bounds(ids, epsilon=0.001, top_k=5)
        b2 = propagator.compute_jacobian_bounds(ids, epsilon=0.01, top_k=5)

        gap1 = b1["logit_upper"] - b1["logit_lower"]
        gap2 = b2["logit_upper"] - b2["logit_lower"]
        assert np.all(gap1 <= gap2 + 1e-6)


class TestLLMRobustnessVerifier:
    def test_verify_next_token(self):
        llm = _make_llm()
        verifier = LLMRobustnessVerifier(llm)
        result = verifier.verify_next_token("hello world", epsilon=0.001)

        assert isinstance(result, LLMVerificationResult)
        assert result.epsilon == 0.001
        assert result.nominal_token_id >= 0
        assert isinstance(result.margin, float)

    def test_small_epsilon_verifies(self):
        llm = _make_llm()
        verifier = LLMRobustnessVerifier(llm)
        # Very small epsilon should verify for most random models
        result = verifier.verify_next_token("hello world", epsilon=1e-6)
        assert result.verified is True
        assert result.margin > 0

    def test_large_epsilon_fails(self):
        llm = _make_llm()
        verifier = LLMRobustnessVerifier(llm)
        result = verifier.verify_next_token("hello world", epsilon=10.0)
        assert result.verified is False

    def test_multi_epsilon(self):
        llm = _make_llm()
        verifier = LLMRobustnessVerifier(llm)
        results = verifier.verify_multi_epsilon(
            "test", [1e-6, 1e-3, 1.0]
        )
        assert len(results) == 3
        # Monotone: if small eps verifies, margin should be larger
        assert results[0].margin >= results[2].margin - 1e-6

    def test_find_certified_radius(self):
        llm = _make_llm()
        verifier = LLMRobustnessVerifier(llm)
        radius = verifier.find_certified_radius("hello", n_steps=5)
        assert radius >= 0

    def test_result_summary(self):
        result = LLMVerificationResult(
            verified=True, margin=0.5, prompt="test",
            nominal_token="hello", epsilon=0.01,
        )
        s = result.summary()
        assert "CERTIFIED ROBUST" in s
        assert "test" in s
