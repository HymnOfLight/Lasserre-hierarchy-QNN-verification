"""
Adaptive relaxation order selection for layered verification.

Controls the Lasserre hierarchy order per layer/block based on
approximation error, quantization truncation error, and the
desired verification precision, balancing accuracy vs. computation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AdaptiveOrderSelector:
    """
    Selects the relaxation order for each sub-problem (layer or block)
    in the hierarchical verification, adapting to the local difficulty.

    Key idea: layers with more unstable neurons or wider bound gaps
    need higher relaxation orders, while "easy" layers can use lower
    orders for efficiency.
    """

    def __init__(
        self,
        min_order: int = 1,
        max_order: int = 4,
        target_precision: float = 1e-3,
        budget_factor: float = 1.0,
    ):
        self.min_order = min_order
        self.max_order = max_order
        self.target_precision = target_precision
        self.budget_factor = budget_factor

        self._layer_orders: Dict[int, int] = {}
        self._error_history: Dict[int, List[float]] = {}

    def select_order_for_layer(
        self,
        layer_idx: int,
        n_unstable_neurons: int,
        bound_gap: float,
        n_variables: int,
        poly_approx_error: float = 0.0,
        quant_error: float = 0.0,
    ) -> int:
        """
        Select the relaxation order for a specific layer.

        Higher order when:
        - Many unstable neurons (wide ReLU crossing region)
        - Large bound gaps (loose propagation)
        - High polynomial approximation error
        - Small quantization error relative to bound gap (more to gain)

        Lower order when:
        - Few variables (small problem anyway)
        - All neurons are stable (linear behavior)
        - Combined error already below target
        """
        combined_error = poly_approx_error + quant_error

        if combined_error < self.target_precision and n_unstable_neurons == 0:
            order = self.min_order
        elif n_unstable_neurons == 0:
            order = self.min_order
        else:
            difficulty = self._compute_difficulty(
                n_unstable_neurons, bound_gap, n_variables, combined_error
            )
            order = self.min_order + int(np.ceil(difficulty * (self.max_order - self.min_order)))
            order = min(order, self.max_order)

        # Computational budget constraint: higher-order SDP scales as O(n^(2d))
        max_feasible = self._max_feasible_order(n_variables)
        order = min(order, max_feasible)
        order = max(order, self.min_order)

        self._layer_orders[layer_idx] = order
        return order

    def _compute_difficulty(
        self,
        n_unstable: int,
        bound_gap: float,
        n_variables: int,
        combined_error: float,
    ) -> float:
        """
        Compute a difficulty score in [0, 1] for the layer.
        """
        # Normalize factors
        instability_score = min(n_unstable / max(n_variables, 1), 1.0)
        gap_score = min(bound_gap / 10.0, 1.0)
        error_score = min(combined_error / max(self.target_precision, 1e-10), 1.0)

        difficulty = 0.4 * instability_score + 0.3 * gap_score + 0.3 * error_score
        return min(difficulty, 1.0)

    def _max_feasible_order(self, n_variables: int) -> int:
        """
        Estimate the maximum feasible order given computational constraints.
        SDP size ~= C(n+d, d) which grows polynomially in n for fixed d.
        """
        for d in range(self.max_order, self.min_order - 1, -1):
            # Moment matrix size ~ C(n+d, d)
            size = 1
            for i in range(d):
                size = size * (n_variables + i + 1) // (i + 1)

            # SDP with matrix variable of this size is feasible if
            # size^2 * budget_factor < threshold
            if size ** 2 * self.budget_factor < 1e8:
                return d

        return self.min_order

    def update_error(self, layer_idx: int, error: float):
        """Record verification error for adaptive refinement."""
        if layer_idx not in self._error_history:
            self._error_history[layer_idx] = []
        self._error_history[layer_idx].append(error)

    def should_increase_order(self, layer_idx: int) -> bool:
        """Check if the order should be increased based on error history."""
        history = self._error_history.get(layer_idx, [])
        if len(history) < 2:
            return True

        # Increase if error is not converging
        recent = history[-2:]
        improvement = recent[0] - recent[1]
        return improvement > self.target_precision

    def get_layer_plan(
        self, layer_infos: List[Dict]
    ) -> Dict[int, int]:
        """
        Compute the full layer-wise order plan.

        Args:
            layer_infos: List of dicts with keys:
                - 'layer_idx', 'n_unstable', 'bound_gap',
                  'n_variables', 'poly_error', 'quant_error'
        """
        plan = {}
        for info in layer_infos:
            order = self.select_order_for_layer(
                layer_idx=info["layer_idx"],
                n_unstable_neurons=info.get("n_unstable", 0),
                bound_gap=info.get("bound_gap", 1.0),
                n_variables=info.get("n_variables", 10),
                poly_approx_error=info.get("poly_error", 0.0),
                quant_error=info.get("quant_error", 0.0),
            )
            plan[info["layer_idx"]] = order

        return plan

    def summary(self) -> str:
        lines = ["Adaptive Order Plan:"]
        for layer_idx, order in sorted(self._layer_orders.items()):
            lines.append(f"  Layer {layer_idx}: order={order}")
        return "\n".join(lines)
