"""
Verification certificate generation and validation.

Produces formal mathematical certificates that can be independently
verified, proving that a quantized neural network is robust against
adversarial perturbations within a given radius.
"""

import numpy as np
import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class VerificationCertificate:
    """
    A formal certificate of adversarial robustness.

    This certificate contains all the information needed to independently
    verify the robustness claim, including:
    - The problem specification (network hash, input, epsilon)
    - The verification method and parameters
    - The lower bound and solver status
    - Sufficient information to reconstruct the SOS proof
    """

    # Problem specification
    network_hash: str = ""
    input_hash: str = ""
    true_label: int = -1
    target_label: int = -1
    epsilon: float = 0.0
    perturbation_norm: str = "Linf"

    # Verification result
    certified_robust: bool = False
    lower_bound: float = -np.inf
    upper_bound: float = np.inf

    # Method details
    verification_method: str = "Lasserre_hierarchy"
    relaxation_order: int = 1
    polynomial_degree: int = 4
    solver_used: str = "SCS"
    solver_status: str = "unknown"

    # Proof data
    moment_matrix_rank: int = -1
    sos_certificates: List[Dict] = field(default_factory=list)

    # Metadata
    timestamp: str = ""
    computation_time_seconds: float = 0.0
    n_variables: int = 0
    n_constraints: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @classmethod
    def from_verification_result(
        cls,
        result: Dict,
        network_hash: str = "",
        input_data: Optional[np.ndarray] = None,
        epsilon: float = 0.0,
        true_label: int = -1,
        target_label: int = -1,
    ) -> "VerificationCertificate":
        """Create a certificate from a RobustnessVerifier result."""
        cert = cls()
        cert.network_hash = network_hash
        cert.epsilon = epsilon
        cert.true_label = true_label
        cert.target_label = target_label

        if input_data is not None:
            cert.input_hash = hashlib.sha256(
                input_data.tobytes()
            ).hexdigest()[:16]

        cert.certified_robust = result.get("verified", False)
        cert.lower_bound = float(
            result.get("pop_result", {}).get("lower_bound", result.get("lower_bound", -np.inf))
        )

        pop_result = result.get("pop_result", {})
        sub_cert = pop_result.get("certificate", {})
        cert.relaxation_order = sub_cert.get("order", 1)
        cert.solver_status = sub_cert.get("solver_status", result.get("solver_status", "unknown"))

        cert.verification_method = result.get("method", "Lasserre_hierarchy")
        cert.n_variables = pop_result.get("n_vars", 0)

        return cert

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        d = asdict(self)
        d["lower_bound"] = float(d["lower_bound"]) if np.isfinite(d["lower_bound"]) else str(d["lower_bound"])
        d["upper_bound"] = float(d["upper_bound"]) if np.isfinite(d["upper_bound"]) else str(d["upper_bound"])
        return d

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialize to JSON."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str: str) -> "VerificationCertificate":
        """Deserialize from JSON."""
        d = json.loads(json_str)
        lb = d.get("lower_bound", -np.inf)
        if isinstance(lb, str):
            lb = -np.inf if "inf" in lb.lower() else float(lb)
        d["lower_bound"] = lb
        ub = d.get("upper_bound", np.inf)
        if isinstance(ub, str):
            ub = np.inf if "inf" in ub.lower() else float(ub)
        d["upper_bound"] = ub
        d.pop("sos_certificates", None)
        return cls(**d)

    def summary(self) -> str:
        """Human-readable summary of the certificate."""
        lines = [
            "=" * 60,
            "ROBUSTNESS VERIFICATION CERTIFICATE",
            "=" * 60,
            f"Status: {'CERTIFIED ROBUST' if self.certified_robust else 'NOT CERTIFIED'}",
            f"Perturbation: {self.perturbation_norm} ball, epsilon = {self.epsilon}",
            f"True label: {self.true_label}, Target label: {self.target_label}",
            f"Lower bound on margin: {self.lower_bound:.6f}",
            f"Method: {self.verification_method}",
            f"Relaxation order: {self.relaxation_order}",
            f"Polynomial degree: {self.polynomial_degree}",
            f"Solver: {self.solver_used} ({self.solver_status})",
            f"Variables: {self.n_variables}",
            f"Timestamp: {self.timestamp}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def verify_certificate(self) -> bool:
        """
        Basic self-consistency check on the certificate.
        A full independent verification would re-solve the SDP.
        """
        if self.certified_robust and self.lower_bound <= 0:
            return False
        if not self.certified_robust and self.lower_bound > 0:
            return False
        if self.epsilon < 0:
            return False
        return True
