# Lasserre Hierarchy based Quantized Neural Network Verification

A formal verification framework for quantized neural networks (QNNs) that uses **higher-order polynomial approximation** and **Lasserre hierarchy SOS/SDP relaxation** to provide deterministic robustness certificates against adversarial perturbations.

## Motivation

Traditional verification methods based on linear relaxation (polyhedral abstraction) suffer from severe **over-approximation accumulation** in deep networks, leading to loose bounds and high false-positive rates. This framework addresses the NP-hard verification problem by:

1. **Chebyshev polynomial envelopes** instead of linear bounds -- providing tighter feasible region descriptions for non-linear activations (ReLU, Sigmoid, quantized step functions)
2. **Lasserre hierarchy** (SOS-SDP relaxation) -- formulating verification as polynomial optimization and solving via progressively tightening semidefinite programs
3. **Sparsity exploitation** -- leveraging network topology structure (correlative + term sparsity) to decompose large SDPs into manageable blocks
4. **Semi-algebraic modeling** of quantization constraints -- converting discrete weight levels into tight continuous polynomial constraints

## Architecture

```
qnn_verifier/
├── polynomial/          # Polynomial approximation theory
│   ├── chebyshev.py        # Chebyshev interpolation & minimax approximation
│   ├── activation_envelope.py  # Upper/lower polynomial envelopes for activations
│   └── semi_algebraic.py   # Semi-algebraic set modeling of quantization
├── lasserre/            # Lasserre hierarchy & SDP solving
│   ├── moment_matrix.py    # Moment/localizing matrix construction
│   ├── sos_relaxation.py   # Positivstellensatz-based SOS formulation
│   ├── sdp_solver.py       # CVXPY SDP solver interface
│   └── hierarchy.py        # Adaptive hierarchy level controller
├── network/             # Neural network abstraction
│   ├── quantized_network.py # Layer-by-layer QNN representation
│   ├── model_loader.py     # PyTorch .pth model loader
│   └── layer_propagation.py # Interval + polynomial bound propagation
├── sparsity/            # Scalability through sparsity
│   ├── correlative_sparsity.py # Network topology graph analysis
│   ├── term_sparsity.py    # Newton polytope structure exploitation
│   └── adaptive_order.py   # Per-layer relaxation order selection
└── verification/        # End-to-end verification pipeline
    ├── robustness.py       # Adversarial robustness verifier
    ├── certificate.py      # Formal verification certificates
    └── pipeline.py         # Complete orchestration pipeline
```

## Installation

```bash
pip install -e .
```

Dependencies: PyTorch >= 2.0, CVXPY >= 1.3, NumPy, SciPy, NetworkX, SymPy.

## Quick Start

### 1. Small Network Demo

```python
from qnn_verifier.verification.pipeline import VerificationPipeline
import numpy as np

# Create demo pipeline with small quantized MLP
pipeline, torch_model = VerificationPipeline.create_demo_pipeline(
    n_inputs=4, hidden_sizes=[8, 8], n_classes=2, n_bits=8
)

# Verify robustness at epsilon=0.01
x0 = np.random.rand(4).astype(np.float32)
result = pipeline.verify(x0, true_label=0, epsilon=0.01)
print(result["certificate"].summary())
```

### 2. Verify a Saved ResNet Model (.pth)

```python
from qnn_verifier.verification.pipeline import VerificationPipeline
import numpy as np

pipeline = VerificationPipeline(
    model_path="quantized_resnet18.pth",
    model_arch="resnet18",
    input_shape=(1, 3, 32, 32),
    n_classes=10,
    n_bits=8,
    poly_degree=4,
    max_lasserre_order=2,
)

x0 = np.random.rand(3, 32, 32).astype(np.float32)
result = pipeline.verify(x0, true_label=0, epsilon=0.01)
cert = result["certificate"]
print(cert.summary())
cert.to_json("certificate.json")
```

### 3. Command-Line Examples

```bash
# Run demo with polynomial approximation visualization
python examples/verify_robustness.py --demo

# Create a quantized model
python examples/create_quantized_model.py --arch resnet18 --n-classes 10 --n-bits 8

# Verify a ResNet-18 model
python examples/verify_resnet.py --arch resnet18 --epsilon 0.01
```

## Theoretical Background

### Polynomial Approximation (Chebyshev Envelopes)

For each activation function σ(x) on interval [a, b], we construct degree-d Chebyshev polynomial upper bound p_U(x) and lower bound p_L(x) such that:

```
p_L(x) ≤ σ(x) ≤ p_U(x),  ∀x ∈ [a, b]
```

The Chebyshev basis provides minimax-optimal approximation, achieving the smallest maximum error among all polynomials of the same degree.

### Lasserre Hierarchy (SOS-SDP Relaxation)

The adversarial robustness verification is formulated as a polynomial optimization problem (POP):

```
min  f(x)[y_true] - f(x)[y_target]
s.t. x ∈ B_∞(x_0, ε)
     network constraints (polynomial envelope form)
```

This POP is relaxed via the Lasserre hierarchy into an SDP:

```
max  γ
s.t. f - γ = σ_0 + Σ_i σ_i · g_i    (SOS certificate)
     σ_i are SOS polynomials
```

If the optimal γ > 0, the network is **certified robust**.

### Sparsity Exploitation

For deep networks, the moment matrix grows combinatorially. We exploit:
- **Correlative sparsity**: neurons in distant layers are weakly coupled, enabling block-diagonal moment matrix decomposition
- **Term sparsity**: the Newton polytope of constraints is sparse, reducing SOS multiplier sizes
- **Adaptive order**: layers with more unstable neurons get higher relaxation orders

## Supported Models

- **Architectures**: ResNet-18/34/50/101 (and "ResNet-121" mapped to ResNet-101), MLPs
- **Input format**: PyTorch `.pth` files (state_dict or full model checkpoint)
- **Quantization**: Post-training weight quantization (1-16 bit)
- **Activations**: ReLU, Sigmoid, Tanh, HardSwish
- **Input**: Images (CIFAR-size 32×32, ImageNet-size 224×224) or flat vectors

## Testing

```bash
pytest tests/ -v
```

68 tests covering all modules: polynomial approximation, Lasserre hierarchy, network loading, sparsity analysis, and end-to-end verification.

## Limitations and Notes

- IBP (Interval Bound Propagation) is inherently loose for deep networks with many layers; the Lasserre hierarchy SDP refinement at the final classification layer provides tighter bounds
- Full monolithic SDP verification is only tractable for networks with ~20 input neurons; larger networks use the layered decomposition approach
- Random-weight networks will not be certifiable (bounds blow up); trained models with actual robustness will yield meaningful certificates
- For best results, use small perturbation radii (ε ≤ 0.01) on CIFAR-scale models

## License

MIT
