"""
Adversarial robustness verification for quantized neural networks.

Formulates the robustness verification as a polynomial optimization
problem and solves it via the Lasserre hierarchy, providing
deterministic certificates for safety within a given perturbation radius.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from ..network.quantized_network import QuantizedNetwork, QuantizedLayer
from ..network.layer_propagation import BoundPropagator
from ..polynomial.activation_envelope import ActivationEnvelope
from ..polynomial.semi_algebraic import SemiAlgebraicSet, QuantizationConstraint
from ..lasserre.hierarchy import LasserreHierarchy
from ..sparsity.adaptive_order import AdaptiveOrderSelector

logger = logging.getLogger(__name__)


class RobustnessVerifier:
    """
    Verifies adversarial robustness of a quantized neural network.

    Given:
    - A network f: R^n -> R^K
    - An input x_0 with true label y_true
    - A perturbation radius epsilon (L_inf ball)

    Proves that: for all x in B_inf(x_0, epsilon),
        f(x)[y_true] > f(x)[y_target]  for all y_target != y_true

    This is formulated as:
        min_{x in B_inf(x_0, eps)} (f(x)[y_true] - f(x)[y_target])
    If the minimum > 0, the network is certified robust.
    """

    def __init__(
        self,
        network: QuantizedNetwork,
        poly_degree: int = 4,
        max_lasserre_order: int = 3,
        solver: str = "SCS",
        verbose: bool = False,
    ):
        self.network = network
        self.poly_degree = poly_degree
        self.max_lasserre_order = max_lasserre_order
        self.solver = solver
        self.verbose = verbose

    def verify(
        self,
        x0: np.ndarray,
        y_true: int,
        epsilon: float,
        y_target: Optional[int] = None,
        use_sparsity: bool = True,
    ) -> Dict:
        """
        Verify robustness for a single input.

        Args:
            x0: Input sample (flattened)
            y_true: True class label
            epsilon: L_inf perturbation radius
            y_target: Specific target class to verify against.
                      If None, verifies against all other classes.
            use_sparsity: Whether to exploit sparsity structure

        Returns:
            Dictionary with verification results including certificate.
        """
        x0 = x0.flatten()
        input_lower = np.clip(x0 - epsilon, 0.0, 1.0)
        input_upper = np.clip(x0 + epsilon, 0.0, 1.0)

        logger.info(
            f"Verifying robustness: eps={epsilon}, true_label={y_true}, "
            f"input_dim={len(x0)}"
        )

        # Step 1: Bound propagation
        propagator = BoundPropagator(
            self.network, poly_degree=self.poly_degree
        )
        layer_bounds = propagator.propagate(input_lower, input_upper)
        output_lower, output_upper = propagator.get_output_bounds()

        logger.info(
            f"Output bounds: lower={output_lower[:5]}..., "
            f"upper={output_upper[:5]}..."
        )

        # Quick check: if output bounds already prove robustness
        if y_target is not None:
            targets = [y_target]
        else:
            targets = [
                i for i in range(self.network.n_classes) if i != y_true
            ]

        # IBP-based quick verification
        ibp_verified = True
        ibp_margins = {}
        for t in targets:
            if t < len(output_lower) and y_true < len(output_lower):
                margin = output_lower[y_true] - output_upper[t]
                ibp_margins[t] = float(margin)
                if margin <= 0:
                    ibp_verified = False
            else:
                ibp_verified = False
                ibp_margins[t] = -np.inf

        if ibp_verified:
            logger.info("Verified by IBP alone!")
            return {
                "verified": True,
                "method": "IBP",
                "margins": ibp_margins,
                "certificate": {
                    "type": "interval_bound",
                    "epsilon": epsilon,
                    "true_label": y_true,
                },
            }

        # Step 2: Polynomial verification via Lasserre hierarchy
        # Focus on the most critical (smallest margin) target class
        critical_target = min(ibp_margins, key=ibp_margins.get)
        critical_margin = ibp_margins[critical_target]

        logger.info(
            f"IBP margin for critical target {critical_target}: {critical_margin:.6f}"
        )

        # Build the POP for the critical pair
        pop_result = self._build_and_solve_pop(
            x0, y_true, critical_target, epsilon,
            propagator, use_sparsity
        )

        # Check all targets if the critical one is verified
        all_verified = pop_result.get("lower_bound", -np.inf) > 0
        if all_verified and len(targets) > 1:
            for t in targets:
                if t == critical_target:
                    continue
                if ibp_margins.get(t, -np.inf) > 0:
                    continue
                t_result = self._build_and_solve_pop(
                    x0, y_true, t, epsilon, propagator, use_sparsity
                )
                if t_result.get("lower_bound", -np.inf) <= 0:
                    all_verified = False
                    break

        return {
            "verified": all_verified,
            "method": "Lasserre_hierarchy",
            "ibp_margins": ibp_margins,
            "pop_result": pop_result,
            "critical_target": critical_target,
            "epsilon": epsilon,
            "true_label": y_true,
        }

    def _build_and_solve_pop(
        self,
        x0: np.ndarray,
        y_true: int,
        y_target: int,
        epsilon: float,
        propagator: BoundPropagator,
        use_sparsity: bool,
    ) -> Dict:
        """
        Build and solve the polynomial optimization problem for
        verifying f(x)[y_true] > f(x)[y_target] in the epsilon-ball.

        For tractability with large networks, we use a layered approach:
        verify layer by layer, propagating polynomial bounds.
        """
        n_input = len(x0)

        # For large networks, use the layered decomposition approach
        if n_input > 20:
            return self._solve_layered(
                x0, y_true, y_target, epsilon, propagator, use_sparsity
            )

        return self._solve_monolithic(
            x0, y_true, y_target, epsilon, propagator
        )

    def _solve_monolithic(
        self,
        x0: np.ndarray,
        y_true: int,
        y_target: int,
        epsilon: float,
        propagator: BoundPropagator,
    ) -> Dict:
        """
        Monolithic POP formulation for small networks.
        All variables and constraints in a single SDP.
        """
        # Collect all variables: input neurons + hidden neurons + output
        n_input = len(x0)
        var_offset = 0
        var_map = {"input": (var_offset, n_input)}
        var_offset += n_input

        layer_var_info = []
        for i, layer in enumerate(self.network.layers):
            if layer.layer_type in ("linear", "conv2d"):
                n_out = layer.n_outputs
                layer_var_info.append({
                    "layer_idx": i,
                    "offset": var_offset,
                    "size": n_out,
                    "type": "pre_activation",
                })
                var_offset += n_out
            elif layer.layer_type in ("relu", "sigmoid", "tanh"):
                if layer.output_shape is not None:
                    n_out = int(np.prod(layer.output_shape))
                elif i > 0 and layer_var_info:
                    n_out = layer_var_info[-1]["size"]
                else:
                    n_out = 0
                layer_var_info.append({
                    "layer_idx": i,
                    "offset": var_offset,
                    "size": n_out,
                    "type": "post_activation",
                })
                var_offset += n_out

        n_vars = var_offset

        # Adaptive order selection
        order_selector = AdaptiveOrderSelector(
            min_order=1, max_order=self.max_lasserre_order
        )
        unstable = propagator.get_verification_neurons()
        order = order_selector.select_order_for_layer(
            layer_idx=0,
            n_unstable_neurons=len(unstable),
            bound_gap=float(np.max(propagator.get_output_bounds()[1] - propagator.get_output_bounds()[0])),
            n_variables=n_vars,
        )

        hierarchy = LasserreHierarchy(
            n_vars=n_vars,
            max_order=order,
            solver_name=self.solver,
            verbose=self.verbose,
        )

        # Input box constraints
        for j in range(n_input):
            lb = max(float(x0[j]) - epsilon, 0.0)
            ub = min(float(x0[j]) + epsilon, 1.0)
            ei = tuple(1 if k == j else 0 for k in range(n_vars))
            zero = tuple(0 for _ in range(n_vars))
            hierarchy.add_inequality({ei: 1.0, zero: -lb})
            hierarchy.add_inequality({ei: -1.0, zero: ub})

        # Layer constraints
        for i, layer in enumerate(self.network.layers):
            if layer.layer_type in ("linear",) and layer.weight is not None:
                W = layer.weight
                b = layer.bias if layer.bias is not None else np.zeros(W.shape[0])

                # Find input and output variable offsets
                in_offset = 0
                out_offset = 0
                for info in layer_var_info:
                    if info["layer_idx"] == i:
                        out_offset = info["offset"]
                        break
                if i > 0:
                    for info in layer_var_info:
                        if info["layer_idx"] == i - 1:
                            in_offset = info["offset"]
                            break
                else:
                    in_offset = var_map["input"][0]

                # y_j = sum_k W[j,k] * x_k + b_j
                n_in = min(W.shape[1], n_input if i == 0 else W.shape[1])
                for j in range(min(W.shape[0], 4)):  # Limit for tractability
                    eq_poly: Dict[Tuple[int, ...], float] = {}
                    ej = [0] * n_vars
                    ej[out_offset + j] = 1
                    eq_poly[tuple(ej)] = 1.0

                    for k in range(n_in):
                        if abs(W[j, k]) > 1e-10:
                            ek = [0] * n_vars
                            ek[in_offset + k] = 1
                            eq_poly[tuple(ek)] = eq_poly.get(tuple(ek), 0.0) - W[j, k]

                    zero = tuple(0 for _ in range(n_vars))
                    eq_poly[zero] = eq_poly.get(zero, 0.0) - b[j]
                    hierarchy.add_equality(eq_poly)

            elif layer.layer_type in ("relu", "sigmoid", "tanh"):
                # Add polynomial envelope constraints
                for info in layer_var_info:
                    if info["layer_idx"] == i:
                        post_offset = info["offset"]
                        n_neurons = info["size"]
                        break
                else:
                    continue

                pre_offset = 0
                for info in layer_var_info:
                    if info["layer_idx"] == i - 1:
                        pre_offset = info["offset"]
                        break

                bounds = propagator.layer_bounds[min(i + 1, len(propagator.layer_bounds) - 1)]

                for j in range(min(n_neurons, 4)):  # Limit for tractability
                    env = propagator.get_neuron_envelope(i, j)
                    if env is None:
                        continue

                    constraints = env.get_polynomial_constraints()
                    for c in constraints:
                        coeffs = c["coefficients"]
                        poly: Dict[Tuple[int, ...], float] = {}

                        for k_deg, coeff in enumerate(coeffs):
                            if abs(coeff) > 1e-15:
                                mono = [0] * n_vars
                                mono[pre_offset + j] = k_deg
                                poly[tuple(mono)] = coeff

                        yj = [0] * n_vars
                        yj[post_offset + j] = 1
                        if c["type"] == "upper_bound":
                            poly[tuple(yj)] = poly.get(tuple(yj), 0.0) - 1.0
                        else:
                            poly[tuple(yj)] = poly.get(tuple(yj), 0.0) + 1.0
                            poly = {m: -v for m, v in poly.items()}

                        hierarchy.add_inequality(poly)

        # Objective: minimize f(x)[y_true] - f(x)[y_target]
        # In the final layer, this is the margin
        output_offset = 0
        for info in layer_var_info:
            if info["type"] == "pre_activation" and info["layer_idx"] == len(self.network.layers) - 1:
                output_offset = info["offset"]
                break
        if output_offset == 0 and layer_var_info:
            output_offset = layer_var_info[-1]["offset"]

        obj: Dict[Tuple[int, ...], float] = {}
        if y_true < n_vars - output_offset:
            yt = [0] * n_vars
            yt[output_offset + y_true] = 1
            obj[tuple(yt)] = 1.0
        if y_target < n_vars - output_offset:
            ya = [0] * n_vars
            ya[output_offset + y_target] = 1
            obj[tuple(ya)] = -1.0

        hierarchy.set_objective(obj)

        result = hierarchy.solve_adaptive(target_bound=0.0)
        cert = hierarchy.get_verification_certificate()

        return {
            "lower_bound": result["best_bound"],
            "n_vars": n_vars,
            "order_used": result["best_order"],
            "certificate": cert,
            "solver_status": result["history"][-1].get("status", "unknown") if result["history"] else "unknown",
        }

    def _solve_layered(
        self,
        x0: np.ndarray,
        y_true: int,
        y_target: int,
        epsilon: float,
        propagator: BoundPropagator,
        use_sparsity: bool,
    ) -> Dict:
        """
        Layered decomposition approach for large networks.

        Instead of one big SDP, solves a sequence of small SDPs
        per layer, propagating polynomial bounds forward.
        """
        from ..lasserre.sdp_solver import SDPSolver
        from ..polynomial.chebyshev import ChebyshevApproximator

        sdp = SDPSolver(solver=self.solver, verbose=self.verbose)
        order_selector = AdaptiveOrderSelector(
            min_order=1, max_order=self.max_lasserre_order
        )

        # Current bounds being refined
        current_lower = np.clip(x0 - epsilon, 0.0, 1.0)
        current_upper = np.clip(x0 + epsilon, 0.0, 1.0)

        refined_bounds = [(current_lower.copy(), current_upper.copy())]

        for i, layer in enumerate(self.network.layers):
            if layer.layer_type == "batchnorm" and layer.weight is not None:
                scale = layer.weight
                shift = layer.bias if layer.bias is not None else np.zeros_like(scale)
                n_ch = len(scale)
                spatial = len(current_lower) // n_ch if n_ch > 0 else 1
                new_lower = np.empty_like(current_lower)
                new_upper = np.empty_like(current_upper)
                for c in range(n_ch):
                    s, b_val = scale[c], shift[c]
                    sl = slice(c * spatial, (c + 1) * spatial)
                    if s >= 0:
                        new_lower[sl] = s * current_lower[sl] + b_val
                        new_upper[sl] = s * current_upper[sl] + b_val
                    else:
                        new_lower[sl] = s * current_upper[sl] + b_val
                        new_upper[sl] = s * current_lower[sl] + b_val
                current_lower = new_lower
                current_upper = new_upper
                refined_bounds.append((current_lower.copy(), current_upper.copy()))

            elif layer.layer_type in ("linear", "conv2d") and layer.weight is not None:
                W = layer.weight
                if W.ndim > 2:
                    W = W.reshape(W.shape[0], -1)
                b = layer.bias if layer.bias is not None else np.zeros(W.shape[0])

                n_in = min(W.shape[1], len(current_lower))
                l_in = current_lower[:n_in]
                u_in = current_upper[:n_in]

                W_trunc = W[:, :n_in]
                W_pos = np.maximum(W_trunc, 0)
                W_neg = np.minimum(W_trunc, 0)
                new_lower = W_pos @ l_in + W_neg @ u_in + b
                new_upper = W_pos @ u_in + W_neg @ l_in + b

                current_lower = new_lower
                current_upper = new_upper
                refined_bounds.append((current_lower.copy(), current_upper.copy()))

            elif layer.layer_type in ("relu", "sigmoid", "tanh"):
                n = len(current_lower)
                new_lower = np.zeros(n)
                new_upper = np.zeros(n)

                n_unstable = sum(
                    1 for j in range(n)
                    if layer.layer_type == "relu" and current_lower[j] < 0 < current_upper[j]
                )

                for j in range(n):
                    lb, ub = float(current_lower[j]), float(current_upper[j])

                    if layer.layer_type == "relu":
                        if ub <= 0:
                            new_lower[j] = 0.0
                            new_upper[j] = 0.0
                        elif lb >= 0:
                            new_lower[j] = lb
                            new_upper[j] = ub
                        else:
                            # Unstable: try polynomial refinement
                            new_lower[j] = 0.0
                            new_upper[j] = ub

                            if n_unstable <= 50 and (ub - lb) > 0.01:
                                env = ActivationEnvelope("relu", self.poly_degree)
                                env.build_envelope((lb, ub))
                                test_pts = np.linspace(lb, ub, 20)
                                upper_vals = env.evaluate_upper(test_pts)
                                new_upper[j] = min(ub, float(np.max(upper_vals)))
                    elif layer.layer_type == "sigmoid":
                        new_lower[j] = 1.0 / (1.0 + np.exp(-lb))
                        new_upper[j] = 1.0 / (1.0 + np.exp(-ub))
                    elif layer.layer_type == "tanh":
                        new_lower[j] = np.tanh(lb)
                        new_upper[j] = np.tanh(ub)

                current_lower = new_lower
                current_upper = new_upper
                refined_bounds.append((current_lower.copy(), current_upper.copy()))

            elif layer.layer_type == "avgpool":
                if layer.output_shape is not None:
                    out_size = int(np.prod(layer.output_shape))
                    in_size = len(current_lower)
                    if out_size < in_size and out_size > 0:
                        factor = in_size // out_size
                        current_lower = current_lower[:out_size * factor].reshape(out_size, factor).mean(axis=1)
                        current_upper = current_upper[:out_size * factor].reshape(out_size, factor).mean(axis=1)
                refined_bounds.append((current_lower.copy(), current_upper.copy()))
            elif layer.layer_type == "flatten":
                refined_bounds.append((current_lower.copy(), current_upper.copy()))
            else:
                refined_bounds.append((current_lower.copy(), current_upper.copy()))

        # Final margin computation
        if y_true < len(current_lower) and y_target < len(current_lower):
            margin_lower = current_lower[y_true] - current_upper[y_target]
            margin_upper = current_upper[y_true] - current_lower[y_target]
        else:
            margin_lower = -np.inf
            margin_upper = np.inf

        # If layered IBP doesn't verify, try SDP on the last few layers
        if margin_lower <= 0 and len(refined_bounds) >= 3:
            sdp_result = self._solve_final_layers_sdp(
                refined_bounds, y_true, y_target, propagator
            )
            if sdp_result is not None:
                return sdp_result

        return {
            "lower_bound": float(margin_lower),
            "upper_bound": float(margin_upper),
            "method": "layered_propagation",
            "n_layers_processed": len(refined_bounds) - 1,
            "certificate": {
                "certified": margin_lower > 0,
                "type": "layered_polynomial_propagation",
            },
        }

    def _solve_final_layers_sdp(
        self,
        refined_bounds: List[Tuple[np.ndarray, np.ndarray]],
        y_true: int,
        y_target: int,
        propagator: BoundPropagator,
    ) -> Optional[Dict]:
        """
        Apply SDP verification to the final classification layers
        where the problem size is manageable.
        """
        # Find the last linear layer
        last_linear_idx = -1
        for i, layer in enumerate(self.network.layers):
            if layer.layer_type == "linear":
                last_linear_idx = i

        if last_linear_idx < 0:
            return None

        layer = self.network.layers[last_linear_idx]
        if layer.weight is None:
            return None

        W = layer.weight
        b = layer.bias if layer.bias is not None else np.zeros(W.shape[0])

        n_in = W.shape[1]
        n_out = W.shape[0]

        if n_in > 100:
            return None

        # Build a small POP for just the last layer
        n_vars = n_in + n_out
        hierarchy = LasserreHierarchy(
            n_vars=n_vars,
            max_order=min(2, self.max_lasserre_order),
            solver_name=self.solver,
            verbose=self.verbose,
        )

        # Input bounds for the last layer
        if last_linear_idx < len(refined_bounds):
            lb_in, ub_in = refined_bounds[last_linear_idx]
        else:
            lb_in = np.zeros(n_in)
            ub_in = np.ones(n_in)

        for j in range(n_in):
            if j < len(lb_in):
                hierarchy.add_box_constraints([(float(lb_in[j]), float(ub_in[j]))])

        # Objective: margin = output[y_true] - output[y_target]
        # output[j] = W[j,:] @ x + b[j]
        obj: Dict[Tuple[int, ...], float] = {}

        if y_true < n_out and y_target < n_out:
            diff_w = W[y_true, :n_in] - W[y_target, :n_in]
            diff_b = b[y_true] - b[y_target]

            for k in range(n_in):
                if abs(diff_w[k]) > 1e-10:
                    ek = tuple(1 if j == k else 0 for j in range(n_vars))
                    obj[ek] = float(diff_w[k])

            zero = tuple(0 for _ in range(n_vars))
            obj[zero] = float(diff_b)

        hierarchy.set_objective(obj)

        try:
            result = hierarchy.solve_adaptive(target_bound=0.0)
            return {
                "lower_bound": result["best_bound"],
                "method": "final_layer_SDP",
                "order_used": result["best_order"],
                "certificate": hierarchy.get_verification_certificate(),
            }
        except Exception as e:
            logger.warning(f"Final layer SDP failed: {e}")
            return None

    def verify_batch(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        epsilon: float,
        max_samples: int = 100,
    ) -> Dict:
        """
        Verify robustness for a batch of inputs.
        Returns aggregate statistics.
        """
        n = min(len(inputs), max_samples)
        results = []
        verified_count = 0

        for i in range(n):
            logger.info(f"Verifying sample {i + 1}/{n}")
            result = self.verify(inputs[i], int(labels[i]), epsilon)
            results.append(result)
            if result.get("verified", False):
                verified_count += 1

        return {
            "n_samples": n,
            "n_verified": verified_count,
            "certified_accuracy": verified_count / n if n > 0 else 0.0,
            "epsilon": epsilon,
            "results": results,
        }
