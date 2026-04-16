"""
End-to-end verification pipeline.

Orchestrates the full verification workflow: model loading, bound
propagation, polynomial approximation, sparsity analysis, and
Lasserre hierarchy solving.
"""

import numpy as np
import torch
import logging
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..network.model_loader import load_quantized_model, create_small_quantized_model
from ..network.quantized_network import QuantizedNetwork
from ..network.layer_propagation import BoundPropagator
from ..polynomial.activation_envelope import ActivationEnvelope
from ..sparsity.correlative_sparsity import CorrelativeSparsityAnalyzer
from ..sparsity.adaptive_order import AdaptiveOrderSelector
from .robustness import RobustnessVerifier
from .certificate import VerificationCertificate

logger = logging.getLogger(__name__)


class VerificationPipeline:
    """
    Complete pipeline for quantized neural network robustness verification.

    Usage:
        pipeline = VerificationPipeline(model_path="model.pth", model_arch="resnet18")
        result = pipeline.verify(input_tensor, true_label=0, epsilon=0.01)
        print(result.certificate.summary())
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_arch: Optional[str] = None,
        network: Optional[QuantizedNetwork] = None,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        n_classes: int = 1000,
        n_bits: int = 8,
        poly_degree: int = 4,
        max_lasserre_order: int = 3,
        solver: str = "GUROBI",
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.model_arch = model_arch
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_bits = n_bits
        self.poly_degree = poly_degree
        self.max_lasserre_order = max_lasserre_order
        self.solver = solver
        self.verbose = verbose

        self._network: Optional[QuantizedNetwork] = network
        self._torch_model = None
        self._sparsity_analyzer: Optional[CorrelativeSparsityAnalyzer] = None

    @property
    def network(self) -> QuantizedNetwork:
        if self._network is None:
            self._load_network()
        return self._network

    def _load_network(self):
        """Load and convert the neural network model."""
        if self.model_path is not None:
            logger.info(f"Loading model from {self.model_path}")
            self._network = load_quantized_model(
                model_path=self.model_path,
                model_arch=self.model_arch,
                input_shape=self.input_shape,
                n_classes=self.n_classes,
                n_bits=self.n_bits,
            )
            logger.info(f"Loaded network:\n{self._network.summary()}")
        else:
            raise ValueError("Either model_path or network must be provided")

    def analyze_sparsity(self) -> Dict:
        """Analyze the network's sparsity structure."""
        self._sparsity_analyzer = CorrelativeSparsityAnalyzer(self.network)
        self._sparsity_analyzer.build_coupling_graph()

        summary = self._sparsity_analyzer.sparsity_summary()
        block_structure = self._sparsity_analyzer.get_block_structure()

        logger.info(f"Sparsity analysis: {summary}")
        logger.info(f"Block structure: {block_structure['n_blocks']} blocks")

        return {
            "sparsity": summary,
            "blocks": block_structure,
        }

    def propagate_bounds(
        self,
        input_lower: np.ndarray,
        input_upper: np.ndarray,
    ) -> BoundPropagator:
        """Run bound propagation on the network."""
        propagator = BoundPropagator(
            self.network,
            poly_degree=self.poly_degree,
            use_polynomial_refinement=True,
        )
        propagator.propagate(input_lower, input_upper)
        return propagator

    def verify(
        self,
        x0: np.ndarray,
        true_label: int,
        epsilon: float,
        target_label: Optional[int] = None,
    ) -> Dict:
        """
        Run the full verification pipeline on a single input.

        Args:
            x0: Input tensor (numpy array)
            true_label: True class label
            epsilon: L_inf perturbation radius
            target_label: Specific adversarial target (None for all)

        Returns:
            Dictionary with verification result and certificate.
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("VERIFICATION PIPELINE START")
        logger.info(f"Input shape: {x0.shape}, True label: {true_label}")
        logger.info(f"Epsilon: {epsilon}")
        logger.info("=" * 60)

        # Step 1: Ensure network is loaded
        network = self.network

        # Step 2: Create verifier
        verifier = RobustnessVerifier(
            network=network,
            poly_degree=self.poly_degree,
            max_lasserre_order=self.max_lasserre_order,
            solver=self.solver,
            verbose=self.verbose,
        )

        # Step 3: Run verification
        result = verifier.verify(
            x0=x0,
            y_true=true_label,
            epsilon=epsilon,
            y_target=target_label,
        )

        elapsed = time.time() - start_time
        result["computation_time"] = elapsed

        # Step 4: Generate certificate
        import hashlib

        network_hash = hashlib.sha256(
            network.summary().encode()
        ).hexdigest()[:16]

        certificate = VerificationCertificate.from_verification_result(
            result=result,
            network_hash=network_hash,
            input_data=x0,
            epsilon=epsilon,
            true_label=true_label,
            target_label=target_label if target_label is not None else -1,
        )
        certificate.computation_time_seconds = elapsed

        result["certificate"] = certificate

        logger.info(certificate.summary())
        logger.info(f"Total time: {elapsed:.2f}s")

        return result

    def verify_with_torch_model(
        self,
        torch_model: torch.nn.Module,
        x0: torch.Tensor,
        true_label: int,
        epsilon: float,
        target_label: Optional[int] = None,
    ) -> Dict:
        """
        Verify using a PyTorch model directly.
        First runs the model to confirm the prediction, then verifies.
        """
        torch_model.eval()

        with torch.no_grad():
            output = torch_model(x0.unsqueeze(0) if x0.dim() == 1 else x0)
            pred = output.argmax(dim=-1).item()

        if pred != true_label:
            logger.warning(
                f"Model predicts {pred}, not {true_label}. "
                "Verification may not be meaningful."
            )

        x0_np = x0.detach().cpu().numpy()
        return self.verify(x0_np, true_label, epsilon, target_label)

    def verify_batch(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        epsilon: float,
        max_samples: int = 100,
    ) -> Dict:
        """Verify a batch of inputs."""
        verifier = RobustnessVerifier(
            network=self.network,
            poly_degree=self.poly_degree,
            max_lasserre_order=self.max_lasserre_order,
            solver=self.solver,
            verbose=self.verbose,
        )
        return verifier.verify_batch(inputs, labels, epsilon, max_samples)

    @staticmethod
    def create_demo_pipeline(
        n_inputs: int = 4,
        hidden_sizes: Optional[List[int]] = None,
        n_classes: int = 2,
        n_bits: int = 8,
    ) -> Tuple["VerificationPipeline", torch.nn.Module]:
        """
        Create a demonstration pipeline with a small quantized MLP.
        Useful for testing and tutorials.
        """
        if hidden_sizes is None:
            hidden_sizes = [8, 8]

        torch_model, q_network = create_small_quantized_model(
            n_inputs=n_inputs,
            hidden_sizes=hidden_sizes,
            n_classes=n_classes,
            n_bits=n_bits,
        )

        pipeline = VerificationPipeline(
            network=q_network,
            input_shape=(1, n_inputs),
            n_classes=n_classes,
            n_bits=n_bits,
            poly_degree=4,
            max_lasserre_order=2,
        )

        return pipeline, torch_model
