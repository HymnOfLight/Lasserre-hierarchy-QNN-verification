"""Tests for the verification pipeline."""

import numpy as np
import pytest
import torch

from qnn_verifier.verification.pipeline import VerificationPipeline
from qnn_verifier.verification.certificate import VerificationCertificate
from qnn_verifier.verification.robustness import RobustnessVerifier
from qnn_verifier.network.model_loader import create_small_quantized_model


class TestVerificationPipeline:
    def test_demo_pipeline_creation(self):
        pipeline, model = VerificationPipeline.create_demo_pipeline(
            n_inputs=4, hidden_sizes=[4, 4], n_classes=2
        )
        assert pipeline.network is not None
        assert pipeline.network.n_layers > 0

    def test_demo_verification(self):
        pipeline, model = VerificationPipeline.create_demo_pipeline(
            n_inputs=4, hidden_sizes=[4, 4], n_classes=2
        )
        x0 = np.random.rand(4).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.tensor(x0).unsqueeze(0))
            label = pred.argmax(dim=-1).item()

        result = pipeline.verify(x0, true_label=label, epsilon=0.001)
        assert "verified" in result or "certificate" in result

    def test_bound_propagation(self):
        pipeline, _ = VerificationPipeline.create_demo_pipeline(
            n_inputs=4, hidden_sizes=[4, 4], n_classes=2
        )
        input_lower = np.zeros(4)
        input_upper = np.ones(4)
        propagator = pipeline.propagate_bounds(input_lower, input_upper)
        out_l, out_u = propagator.get_output_bounds()
        assert len(out_l) == 2
        assert np.all(out_u >= out_l)


class TestRobustnessVerifier:
    def test_small_network(self):
        _, network = create_small_quantized_model(
            n_inputs=4, hidden_sizes=[4, 4], n_classes=2
        )
        verifier = RobustnessVerifier(
            network=network, poly_degree=2, max_lasserre_order=1
        )
        x0 = np.random.rand(4).astype(np.float32)
        result = verifier.verify(x0, y_true=0, epsilon=0.001)
        assert "verified" in result

    def test_batch_verification(self):
        _, network = create_small_quantized_model(
            n_inputs=4, hidden_sizes=[4, 4], n_classes=2
        )
        verifier = RobustnessVerifier(
            network=network, poly_degree=2, max_lasserre_order=1
        )
        inputs = np.random.rand(3, 4).astype(np.float32)
        labels = np.array([0, 1, 0])
        result = verifier.verify_batch(inputs, labels, epsilon=0.001, max_samples=3)
        assert result["n_samples"] == 3
        assert "certified_accuracy" in result


class TestVerificationCertificate:
    def test_creation(self):
        cert = VerificationCertificate(
            certified_robust=True,
            lower_bound=0.5,
            epsilon=0.01,
            true_label=0,
            target_label=1,
        )
        assert cert.certified_robust
        assert cert.verify_certificate()

    def test_json_roundtrip(self):
        cert = VerificationCertificate(
            certified_robust=True,
            lower_bound=0.5,
            epsilon=0.01,
            true_label=0,
        )
        json_str = cert.to_json()
        cert2 = VerificationCertificate.from_json(json_str)
        assert cert2.certified_robust == cert.certified_robust
        assert abs(cert2.lower_bound - cert.lower_bound) < 1e-10

    def test_summary(self):
        cert = VerificationCertificate(
            certified_robust=True,
            lower_bound=0.5,
            epsilon=0.01,
        )
        summary = cert.summary()
        assert "CERTIFIED ROBUST" in summary

    def test_inconsistent_certificate(self):
        cert = VerificationCertificate(
            certified_robust=True,
            lower_bound=-0.5,
        )
        assert not cert.verify_certificate()

    def test_from_result(self):
        result = {
            "verified": True,
            "method": "Lasserre_hierarchy",
            "pop_result": {
                "lower_bound": 0.3,
                "n_vars": 10,
                "certificate": {
                    "order": 2,
                    "solver_status": "optimal",
                },
            },
        }
        cert = VerificationCertificate.from_verification_result(
            result, epsilon=0.01, true_label=0, target_label=1
        )
        assert cert.certified_robust
        assert abs(cert.lower_bound - 0.3) < 1e-10
