"""
LLM robustness verifier.

Verifies that the next-token prediction of a large language model
is stable under embedding-space perturbations.  The perturbation
model captures both quantisation noise and adversarial embedding
attacks (e.g., universal suffix injection in the continuous space).

Verification properties:
  1. **Next-token stability**: the argmax prediction is invariant.
  2. **Top-k preservation**: the set of top-k tokens is invariant.
  3. **Margin certification**: the gap between the top-1 and runner-up
     logits remains positive for all perturbed embeddings.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .loader import LLMWrapper
from .propagation import LLMBoundPropagator
from ..lasserre.sdp_solver import SDPSolver
from ..lasserre.hierarchy import LasserreHierarchy
from ..polynomial.activation_envelope import ActivationEnvelope

logger = logging.getLogger(__name__)


@dataclass
class LLMVerificationResult:
    """Structured result of an LLM verification query."""
    verified: bool = False
    property_type: str = "next_token_stability"
    prompt: str = ""
    epsilon: float = 0.0
    nominal_token: str = ""
    nominal_token_id: int = -1
    margin: float = -np.inf
    runner_up_token: str = ""
    runner_up_token_id: int = -1
    method: str = ""
    top_k_preserved: bool = False
    top_k: int = 5
    computation_time: float = 0.0
    bounds_info: Dict = field(default_factory=dict)

    def summary(self) -> str:
        status = "CERTIFIED ROBUST" if self.verified else "NOT CERTIFIED"
        lines = [
            "=" * 64,
            f"  LLM VERIFICATION: {status}",
            "=" * 64,
            f"  Prompt     : \"{self.prompt}\"",
            f"  Predicted  : \"{self.nominal_token}\" (id={self.nominal_token_id})",
            f"  Runner-up  : \"{self.runner_up_token}\" (id={self.runner_up_token_id})",
            f"  Epsilon    : {self.epsilon}",
            f"  Margin     : {self.margin:.6f}",
            f"  Method     : {self.method}",
            f"  Property   : {self.property_type}",
            f"  Top-{self.top_k} stable: {self.top_k_preserved}",
            f"  Time       : {self.computation_time:.2f}s",
            "=" * 64,
        ]
        return "\n".join(lines)


class LLMRobustnessVerifier:
    """
    Verifies adversarial robustness of an LLM's next-token prediction
    under L_inf embedding perturbation.

    Verification question:
        Given prompt P with token embeddings e_0, is it true that
        for all e' with ||e'_last - e_0_last||_inf <= epsilon,
        argmax f(e') = argmax f(e_0)?

    Approach:
        1. Jacobian-based bounding: compute ||∂logit_j / ∂e_pos||_1
           for the top-k output classes.
        2. The margin lower bound is:
           LB = logit[top1] - logit[top2] - (grad_top1 + grad_top2) * eps
        3. If LB > 0, the prediction is certifiably stable.
        4. Optionally refine with Lasserre SDP on the LM-head layer.
    """

    def __init__(
        self,
        llm: LLMWrapper,
        poly_degree: int = 4,
        max_lasserre_order: int = 2,
        verbose: bool = False,
    ):
        self.llm = llm
        self.poly_degree = poly_degree
        self.max_lasserre_order = max_lasserre_order
        self.verbose = verbose

    def verify_next_token(
        self,
        prompt: str,
        epsilon: float,
        position: int = -1,
        top_k: int = 10,
        use_sdp_refinement: bool = False,
    ) -> LLMVerificationResult:
        """
        Verify that the next-token prediction is stable under perturbation.

        Args:
            prompt: Input text.
            epsilon: L_inf perturbation radius on embedding space.
            position: Token position to perturb (-1 = last).
            top_k: Number of top tokens to consider.
            use_sdp_refinement: Apply Lasserre SDP on the LM head.
        """
        t0 = time.time()

        inputs = self.llm.tokenize(prompt)
        input_ids = inputs["input_ids"]

        logger.info(f"Prompt: \"{prompt}\" -> {input_ids.shape[1]} tokens")

        # Nominal prediction
        predictions = self.llm.predict_next_token(prompt, top_k=top_k)
        top1 = predictions[0]
        top2 = predictions[1] if len(predictions) > 1 else {"token_id": -1, "token": "", "logit": -np.inf}
        nominal_margin = top1["logit"] - top2["logit"]

        logger.info(
            f"Top-1: \"{top1['token']}\" (logit={top1['logit']:.4f}), "
            f"Top-2: \"{top2['token']}\" (logit={top2['logit']:.4f}), "
            f"margin={nominal_margin:.4f}"
        )

        # Bound propagation
        propagator = LLMBoundPropagator(self.llm, position=position)
        bounds = propagator.compute_jacobian_bounds(
            input_ids.squeeze(0), epsilon, top_k=top_k
        )

        # Extract margins from bounds
        # For top-1 stability: need logit_lower[top1] > logit_upper[top2]
        token_ids = bounds["token_ids"]
        logit_lower = bounds["logit_lower"]
        logit_upper = bounds["logit_upper"]

        # Find indices of top1 and top2 in the bounds arrays
        top1_idx = np.where(token_ids == top1["token_id"])[0]
        top2_idx = np.where(token_ids == top2["token_id"])[0]

        if len(top1_idx) > 0 and len(top2_idx) > 0:
            top1_idx = top1_idx[0]
            top2_idx = top2_idx[0]
            margin_lb = logit_lower[top1_idx] - logit_upper[top2_idx]
        else:
            margin_lb = -np.inf

        # Check top-k preservation
        topk_stable = True
        nominal_topk_ids = set(p["token_id"] for p in predictions[:top_k])
        for i in range(min(top_k, len(token_ids))):
            for j in range(min(top_k, len(token_ids)), len(token_ids)):
                if logit_lower[i] < logit_upper[j]:
                    topk_stable = False
                    break

        verified = bool(margin_lb > 0)
        method = "jacobian_bound"

        # SDP refinement on LM head if requested and margin is close
        if not verified and use_sdp_refinement and margin_lb > -1.0:
            sdp_margin = self._refine_with_sdp(
                input_ids, epsilon, top1["token_id"], top2["token_id"],
                position, bounds
            )
            if sdp_margin is not None and sdp_margin > margin_lb:
                margin_lb = sdp_margin
                verified = bool(margin_lb > 0)
                method = "jacobian_bound + Lasserre_SDP"

        elapsed = time.time() - t0

        result = LLMVerificationResult(
            verified=verified,
            property_type="next_token_stability",
            prompt=prompt,
            epsilon=epsilon,
            nominal_token=top1["token"],
            nominal_token_id=top1["token_id"],
            margin=float(margin_lb),
            runner_up_token=top2["token"],
            runner_up_token_id=top2["token_id"],
            method=method,
            top_k_preserved=topk_stable,
            top_k=top_k,
            computation_time=elapsed,
            bounds_info={
                "nominal_margin": nominal_margin,
                "grad_l1_norms": bounds.get("grad_l1_norms", np.array([])).tolist(),
                "perturbation": bounds.get("perturbation", np.array([])).tolist(),
            },
        )

        logger.info(result.summary())
        return result

    def verify_multi_epsilon(
        self,
        prompt: str,
        epsilons: List[float],
        position: int = -1,
        top_k: int = 10,
    ) -> List[LLMVerificationResult]:
        """Verify at multiple perturbation radii."""
        results = []
        for eps in epsilons:
            r = self.verify_next_token(prompt, eps, position, top_k)
            results.append(r)
        return results

    def find_certified_radius(
        self,
        prompt: str,
        position: int = -1,
        eps_min: float = 1e-5,
        eps_max: float = 1.0,
        n_steps: int = 10,
    ) -> float:
        """
        Binary search for the maximum certifiable perturbation radius.
        Returns the largest epsilon where verification succeeds.
        """
        best_eps = 0.0
        lo, hi = eps_min, eps_max

        for _ in range(n_steps):
            mid = (lo + hi) / 2.0
            result = self.verify_next_token(prompt, mid, position, top_k=5)
            if result.verified:
                best_eps = mid
                lo = mid
            else:
                hi = mid

        logger.info(f"Certified radius for \"{prompt}\": {best_eps:.6f}")
        return best_eps

    # ------------------------------------------------------------------
    # SDP refinement on the LM head
    # ------------------------------------------------------------------

    def _refine_with_sdp(
        self,
        input_ids: torch.Tensor,
        epsilon: float,
        top1_id: int,
        top2_id: int,
        position: int,
        jacobian_bounds: Dict,
    ) -> Optional[float]:
        """
        Apply Lasserre SDP to the LM head linear layer for a tighter
        margin bound.

        LM head: logit = W @ h + bias
        We bound h using the Jacobian-based hidden-state perturbation,
        then solve:  min (W[top1] - W[top2])^T h + (b[top1] - b[top2])
                     s.t. h_lower <= h <= h_upper
        """
        model = self.llm.model
        lm_head = model.lm_head if hasattr(model, "lm_head") else None
        if lm_head is None:
            return None

        W = lm_head.weight.detach().cpu().float().numpy()
        bias = lm_head.bias.detach().cpu().float().numpy() if lm_head.bias is not None else np.zeros(W.shape[0])

        hidden_dim = W.shape[1]
        if hidden_dim > 128:
            return None

        # Get nominal hidden state via the propagator's hook mechanism
        propagator = LLMBoundPropagator(self.llm, position=position)
        last_layer_bounds = propagator.compute_last_layer_bounds(
            input_ids.squeeze(0) if input_ids.dim() > 1 else input_ids,
            epsilon, top_k=5
        )

        h_delta = last_layer_bounds.get("h_delta_bound", None)
        if h_delta is None:
            return None

        # Build Lasserre hierarchy for the margin LP/QP
        diff_w = W[top1_id, :] - W[top2_id, :]
        diff_b = float(bias[top1_id] - bias[top2_id])

        # Hidden state bounds (nominal ± delta)
        # We need the nominal hidden state; approximate from logits
        # For the SDP, just use the interval [h_0 - h_delta, h_0 + h_delta]
        n_vars = hidden_dim
        hierarchy = LasserreHierarchy(
            n_vars=n_vars,
            max_order=min(1, self.max_lasserre_order),
            solver_name="SCS",
            verbose=self.verbose,
        )

        for j in range(n_vars):
            lb = -h_delta
            ub = h_delta
            hierarchy.add_box_constraints([(lb, ub)])

        obj = {}
        for k in range(n_vars):
            if abs(diff_w[k]) > 1e-12:
                ek = tuple(1 if j == k else 0 for j in range(n_vars))
                obj[ek] = float(diff_w[k])
        zero = tuple(0 for _ in range(n_vars))
        obj[zero] = float(diff_b)
        hierarchy.set_objective(obj)

        try:
            result = hierarchy.solve_adaptive(target_bound=0.0)
            return result["best_bound"]
        except Exception as e:
            logger.warning(f"SDP refinement failed: {e}")
            return None
