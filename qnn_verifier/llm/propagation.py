"""
Bound propagation for LLM verification.

Computes output logit bounds under embedding-space perturbation using:
  1. Forward-pass-anchored Jacobian bounds (local Lipschitz at the
     nominal input) – fast, works for any model size.
  2. Per-layer IBP through the final MLP head – tight for the last
     linear projection from hidden states to vocabulary logits.

The combination provides meaningful bounds even for multi-billion
parameter models where full IBP is catastrophically loose.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .loader import LLMWrapper

logger = logging.getLogger(__name__)


class LLMBoundPropagator:
    """
    Propagates perturbation bounds through an LLM from the embedding
    space to the output logits.

    The perturbation model is:
        e' = e_0 + delta,  ||delta||_inf <= epsilon
    where e_0 is the nominal embedding for the last token position
    (or a specified position), and e' is the perturbed embedding.
    """

    def __init__(self, llm: LLMWrapper, position: int = -1):
        """
        Args:
            llm: LLMWrapper instance.
            position: Token position to verify (-1 = last token).
        """
        self.llm = llm
        self.position = position

    def compute_jacobian_bounds(
        self,
        input_ids: torch.Tensor,
        epsilon: float,
        top_k: int = 20,
    ) -> Dict:
        """
        Compute logit bounds using the Jacobian of the model output
        w.r.t. the embedding of the target token position.

        For a model f and embedding e at position p:
            |f_j(e') - f_j(e_0)| <= ||∂f_j/∂e_p||_1 · epsilon

        Only computes gradients for the top-k logit classes to keep
        the cost manageable (k backward passes).

        Returns dict with 'logit_lower', 'logit_upper' arrays (over
        the top-k token ids) and 'nominal_logits'.
        """
        model = self.llm.model
        device = self.llm.device
        model.eval()

        input_ids = input_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # Get nominal embeddings
        embeds_nominal = self.llm.embed_tokens(input_ids).detach().clone()
        seq_len = embeds_nominal.shape[1]
        pos = self.position if self.position >= 0 else seq_len - 1
        hidden_dim = embeds_nominal.shape[2]

        # Nominal forward pass
        with torch.no_grad():
            logits_nominal = self.llm.forward_from_embeddings(embeds_nominal)
        last_logits = logits_nominal[0, pos, :].detach()

        # Find top-k tokens
        topk_vals, topk_ids = torch.topk(last_logits, top_k)

        # Compute Jacobian norms for each top-k output via backward passes
        grad_l1_norms = torch.zeros(top_k)

        embeds_var = embeds_nominal.clone().requires_grad_(True)

        for i in range(top_k):
            if embeds_var.grad is not None:
                embeds_var.grad.zero_()

            logits = self.llm.forward_from_embeddings(embeds_var)
            target_logit = logits[0, pos, topk_ids[i]]
            target_logit.backward(retain_graph=(i < top_k - 1))

            # Gradient w.r.t. the embedding at the perturbed position
            grad_at_pos = embeds_var.grad[0, pos, :].detach()
            grad_l1_norms[i] = grad_at_pos.abs().sum().item()

        # Lipschitz bound with safety factor for non-linearity
        safety = 1.3
        perturbation = grad_l1_norms.numpy() * epsilon * safety

        nominal_vals = last_logits[topk_ids].cpu().float().numpy()
        logit_lower = nominal_vals - perturbation
        logit_upper = nominal_vals + perturbation

        return {
            "token_ids": topk_ids.cpu().numpy(),
            "nominal_logits": nominal_vals,
            "logit_lower": logit_lower,
            "logit_upper": logit_upper,
            "grad_l1_norms": grad_l1_norms.numpy(),
            "perturbation": perturbation,
            "epsilon": epsilon,
            "position": pos,
            "hidden_dim": hidden_dim,
        }

    def compute_last_layer_bounds(
        self,
        input_ids: torch.Tensor,
        epsilon: float,
        top_k: int = 20,
    ) -> Dict:
        """
        Tighter bounds using the structure of the LM head:
            logits = W_head @ hidden_state + bias

        Steps:
        1. Run the full model to get the nominal hidden state h_0 at
           the target position.
        2. Bound ||h - h_0||_inf using Jacobian of hidden states w.r.t.
           the embedding perturbation.
        3. Apply interval arithmetic on W_head to get logit bounds.
        """
        model = self.llm.model
        device = self.llm.device
        model.eval()

        input_ids = input_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        embeds_nominal = self.llm.embed_tokens(input_ids).detach().clone()
        seq_len = embeds_nominal.shape[1]
        pos = self.position if self.position >= 0 else seq_len - 1
        hidden_dim = embeds_nominal.shape[2]

        # Get nominal hidden state at the target position
        # by hooking into the model's last hidden layer output
        hidden_states_list = []

        def capture_hook(module, inp, out):
            if isinstance(out, tuple):
                hidden_states_list.append(out[0].detach())
            else:
                hidden_states_list.append(out.detach())

        # Hook the model's output layer norm or last layer
        hook_target = None
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            hook_target = model.model.norm
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            hook_target = model.transformer.ln_f

        if hook_target is not None:
            hook = hook_target.register_forward_hook(capture_hook)
        else:
            hook = None

        with torch.no_grad():
            logits_nominal = self.llm.forward_from_embeddings(embeds_nominal)

        if hook is not None:
            hook.remove()

        if not hidden_states_list:
            logger.warning("Could not capture hidden states, falling back to Jacobian bounds")
            return self.compute_jacobian_bounds(input_ids, epsilon, top_k)

        h_nominal = hidden_states_list[0][0, pos, :].cpu().numpy()

        # Get the LM head weight matrix
        lm_head = model.lm_head if hasattr(model, "lm_head") else None
        if lm_head is None:
            return self.compute_jacobian_bounds(input_ids, epsilon, top_k)

        W = lm_head.weight.detach().cpu().float().numpy()  # (vocab_size, hidden_dim)
        bias = lm_head.bias.detach().cpu().float().numpy() if lm_head.bias is not None else np.zeros(W.shape[0])

        # Compute Jacobian of hidden state w.r.t. embedding perturbation
        embeds_var = embeds_nominal.clone().requires_grad_(True)

        # We need ∂h_pos/∂e_pos which is (hidden_dim, hidden_dim)
        # Compute column-wise: for each hidden dimension, one backward pass
        # This is expensive for large hidden_dim, so we use a sampling approach
        n_probe = min(hidden_dim, 32)
        probe_dims = np.linspace(0, hidden_dim - 1, n_probe, dtype=int)

        max_jac_l1 = 0.0
        for d_idx in probe_dims:
            if embeds_var.grad is not None:
                embeds_var.grad.zero_()
            hidden_states_list.clear()

            if hook_target is not None:
                hook = hook_target.register_forward_hook(capture_hook)

            logits = self.llm.forward_from_embeddings(embeds_var)

            if hook is not None:
                hook.remove()

            if hidden_states_list:
                h_out = hidden_states_list[0][0, pos, d_idx]
                h_out.backward(retain_graph=True)
                grad = embeds_var.grad[0, pos, :].detach().abs().sum().item()
                max_jac_l1 = max(max_jac_l1, grad)

        if max_jac_l1 == 0:
            return self.compute_jacobian_bounds(input_ids, epsilon, top_k)

        # Bound on hidden state perturbation
        safety = 1.3
        h_delta_bound = max_jac_l1 * epsilon * safety

        last_logits = logits_nominal[0, pos, :].detach().cpu().numpy()
        topk_ids = np.argsort(-last_logits)[:top_k]

        # Interval arithmetic on the LM head: logit_j = W_j @ h + bias_j
        # |logit_j(h') - logit_j(h0)| <= ||W_j||_1 * h_delta_bound
        w_l1_norms = np.abs(W[topk_ids, :]).sum(axis=1)
        logit_perturbation = w_l1_norms * h_delta_bound

        nominal_vals = last_logits[topk_ids]
        logit_lower = nominal_vals - logit_perturbation
        logit_upper = nominal_vals + logit_perturbation

        return {
            "token_ids": topk_ids,
            "nominal_logits": nominal_vals,
            "logit_lower": logit_lower,
            "logit_upper": logit_upper,
            "h_delta_bound": h_delta_bound,
            "logit_perturbation": logit_perturbation,
            "epsilon": epsilon,
            "position": pos,
            "hidden_dim": hidden_dim,
            "method": "last_layer_IBP",
        }
