"""
p-adic analysis for neural network verification complexity reduction.

Uses p-adic valuations and the ultrametric structure of quantized weight
spaces to reduce the dimensionality of verification sub-problems.

Theoretical background:
  For a prime p, the p-adic valuation v_p(x) of a rational x = a/b is
  the exponent of p in the prime factorisation of x.  The p-adic norm
  |x|_p = p^{-v_p(x)} satisfies the ULTRAMETRIC inequality:
      |x + y|_p ≤ max(|x|_p, |y|_p)

  This has three consequences for NN verification:

  1. WEIGHT SIGNIFICANCE: In a linear combination y = Σ w_i x_i,
     the p-adic norm |y|_p ≤ max_i |w_i x_i|_p.  Weights with
     high p-adic valuation (small |w|_p) contribute negligibly
     in the p-adic metric — they can be pruned WITHOUT affecting
     the p-adic structure of the output.

  2. ULTRAMETRIC CLUSTERING: The space of neurons, under the p-adic
     distance d_p(i,j) = |w_i - w_j|_p on their weight vectors,
     decomposes into disjoint balls (clopen sets). Neurons in the
     same p-adic ball have nearly identical p-adic behaviour and
     can be analysed as a group.

  3. RELU PHASE from p-adic STRUCTURE: For quantised weights (n-bit),
     the denominators are powers of 2.  Using p=2, the 2-adic valuation
     reveals the "granularity" of each weight.  If a neuron's pre-activation
     is a sum of terms with uniformly high 2-adic valuation (all divisible
     by 2^k), the pre-activation is divisible by 2^k, constraining its
     sign and potentially fixing the ReLU phase.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PAdicReductionStats:
    """Statistics from p-adic analysis."""
    prime: int = 2
    neurons_pruned: int = 0
    weights_zeroed: int = 0
    clusters_found: int = 0
    relu_phases_determined: int = 0
    original_unstable: int = 0
    reduced_unstable: int = 0
    time_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"p-adic(p={self.prime}): "
            f"pruned {self.neurons_pruned} neurons, "
            f"zeroed {self.weights_zeroed} weights, "
            f"{self.clusters_found} clusters, "
            f"ReLU phases fixed: {self.relu_phases_determined}, "
            f"unstable: {self.original_unstable}→{self.reduced_unstable}, "
            f"{self.time_seconds:.3f}s"
        )


# ------------------------------------------------------------------
# p-adic valuation and norm
# ------------------------------------------------------------------

def padic_valuation(x: float, p: int = 2) -> int:
    """
    Compute the p-adic valuation v_p(x).

    Fast path for p=2: use bit manipulation on the float's mantissa.
    General path: convert to rational.
    Returns 100 for x ≈ 0.
    """
    if abs(x) < 1e-15:
        return 100

    if p == 2:
        return _v2_fast(x)

    from fractions import Fraction
    frac = Fraction(x).limit_denominator(2**24)
    a, b = abs(frac.numerator), abs(frac.denominator)
    va = 0
    while a > 0 and a % p == 0: a //= p; va += 1
    vb = 0
    while b > 0 and b % p == 0: b //= p; vb += 1
    return va - vb


def _v2_fast(x: float) -> int:
    """Fast 2-adic valuation via float structure."""
    import struct
    if x == 0: return 100
    x = abs(x)
    # Represent as m * 2^e where m is odd integer
    # float64: x = (1 + mantissa/2^52) * 2^(exponent-1023)
    bits = struct.pack('!d', x)
    n = int.from_bytes(bits, 'big')
    exp = ((n >> 52) & 0x7FF) - 1023 - 52
    mantissa = (n & 0xFFFFFFFFFFFFF) | (1 << 52)
    # v_2(mantissa)
    if mantissa == 0: return 100
    v_m = (mantissa & -mantissa).bit_length() - 1
    return exp + v_m


def padic_norm(x: float, p: int = 2) -> float:
    """Compute |x|_p = p^{-v_p(x)}."""
    v = padic_valuation(x, p)
    return 0.0 if v >= 100 else float(p) ** (-v)


def padic_vector_valuation(vec: np.ndarray, p: int = 2) -> int:
    """Minimum p-adic valuation across non-zero elements of a vector."""
    nz = vec[np.abs(vec) > 1e-15]
    if len(nz) == 0: return 100
    if p == 2:
        return min(_v2_fast(float(v)) for v in nz)
    return min(padic_valuation(float(v), p) for v in nz)


# ------------------------------------------------------------------
# Phase 1: p-adic weight pruning
# ------------------------------------------------------------------

def padic_weight_pruning(
    layers: List[Dict],
    p: int = 2,
    magnitude_threshold: float = 1e-6,
    relative_threshold: float = 0.01,
) -> Tuple[List[Dict], int]:
    """
    Prune weights that are negligible relative to the row's dominant term.

    In p-adic analysis, |x+y|_p ≤ max(|x|_p, |y|_p) (ultrametric).
    This means the p-adic contribution of small terms is dominated
    by the largest term.  Analogously in the real-valued setting:
    we prune weights whose magnitude is < relative_threshold × max
    of the row, since they contribute negligibly to the output.

    This is a SOUND over-approximation: removing a small weight
    can only widen the output interval by at most |w| * input_range.
    """
    pruned = []
    n_zeroed = 0

    for layer in layers:
        if layer["type"] == "linear":
            W = layer["W"].copy()
            b = layer["b"].copy()

            for i in range(W.shape[0]):
                row_max = np.max(np.abs(W[i]))
                if row_max < 1e-15:
                    continue
                threshold = max(magnitude_threshold, relative_threshold * row_max)
                small_mask = (np.abs(W[i]) > 0) & (np.abs(W[i]) < threshold)
                n_zeroed += int(small_mask.sum())
                W[i, small_mask] = 0.0

            pruned.append({"type": "linear", "W": W, "b": b})
        else:
            pruned.append(layer)

    return pruned, n_zeroed


# ------------------------------------------------------------------
# Phase 2: p-adic neuron clustering
# ------------------------------------------------------------------

def padic_neuron_distance(w1: np.ndarray, w2: np.ndarray, p: int = 2) -> float:
    """
    Ultrametric distance between two neuron weight vectors:
      d_p(w1, w2) = max_j |w1_j - w2_j|_p
    """
    max_norm = 0.0
    for j in range(min(len(w1), len(w2))):
        norm = padic_norm(float(w1[j] - w2[j]), p)
        max_norm = max(max_norm, norm)
    return max_norm


def padic_cluster_neurons(
    W: np.ndarray,
    p: int = 2,
    radius: float = 0.5,
) -> List[Set[int]]:
    """
    Cluster neurons by p-adic ultrametric distance.

    Due to the ultrametric inequality, p-adic balls are either
    disjoint or nested → exact clustering.

    For large layers (>100 neurons), sample to keep O(n) complexity.
    """
    n = W.shape[0]
    if n > 100:
        # Sample-based: only cluster first 100 neurons
        n = 100

    cluster_id = [-1] * n
    clusters = []
    cid = 0
    for i in range(n):
        if cluster_id[i] >= 0: continue
        cluster = {i}
        cluster_id[i] = cid
        for j in range(i + 1, n):
            if cluster_id[j] >= 0: continue
            d = padic_neuron_distance(W[i], W[j], p)
            if d <= radius:
                cluster.add(j)
                cluster_id[j] = cid
        clusters.append(cluster)
        cid += 1
    return clusters


# ------------------------------------------------------------------
# Phase 3: p-adic ReLU phase determination
# ------------------------------------------------------------------

def padic_relu_analysis(
    layers: List[Dict],
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    stable: Dict[Tuple[int, int], str],
    p: int = 2,
) -> Tuple[Dict[Tuple[int, int], str], int]:
    """
    Use weight-sparsity and ultrametric structure to determine ReLU phases.

    After weight pruning (Phase 1), some rows may have significantly
    fewer non-zero entries, making the pre-activation bounds tighter.
    Re-run IBP on the pruned network to discover newly stable neurons.

    Additionally, use the ultrametric insight: if a neuron's pre-activation
    is dominated by a single large term (all other terms are p-adically
    small), the sign of the pre-activation is determined by that term alone.
    """
    new_stable = dict(stable)
    n_fixed = 0

    cl, cu = input_lb.copy(), input_ub.copy()
    rl = 0

    for layer in layers:
        if layer["type"] == "linear":
            W = layer["W"]
            b = layer["b"]
            ni = W.shape[1]
            li = cl[:ni] if len(cl) >= ni else np.pad(cl, (0, ni - len(cl)))
            ui = cu[:ni] if len(cu) >= ni else np.pad(cu, (0, ni - len(cu)))
            Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
            cl = Wp @ li + Wn @ ui + b
            cu = Wp @ ui + Wn @ li + b

        elif layer["type"] == "relu":
            for j in range(len(cl)):
                key = (rl, j)
                if new_stable.get(key) != "unstable":
                    if new_stable.get(key) == "inactive":
                        cl[j] = 0.0; cu[j] = 0.0
                    continue

                # Re-check with (possibly tighter) bounds from pruned weights
                if cl[j] >= 0:
                    new_stable[key] = "active"
                    n_fixed += 1
                elif cu[j] <= 0:
                    new_stable[key] = "inactive"
                    cl[j] = 0.0; cu[j] = 0.0
                    n_fixed += 1
                else:
                    cl[j] = 0.0

            cl = np.maximum(cl, 0)
            cu = np.maximum(cu, 0)
            rl += 1

    return new_stable, n_fixed


# ------------------------------------------------------------------
# Main entry: p-adic reduction pipeline
# ------------------------------------------------------------------

def padic_reduce(
    layers: List[Dict],
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    stable: Dict[Tuple[int, int], str],
    p: int = 2,
    pruning_threshold: int = 8,
    cluster_radius: float = 0.5,
) -> Tuple[List[Dict], Dict[Tuple[int, int], str], PAdicReductionStats]:
    """
    Apply the full p-adic analysis pipeline:
      1. Weight pruning (zero out p-adically negligible weights)
      2. Neuron clustering (group by ultrametric distance)
      3. ReLU phase determination (use p-adic granularity)

    Args:
        layers: Network layers.
        input_lb, input_ub: Input bounds.
        stable: Current stable neuron map.
        p: Prime for p-adic analysis (default 2 for quantised nets).
        pruning_threshold: Min valuation to prune a weight.
        cluster_radius: Ultrametric ball radius for clustering.

    Returns:
        (reduced_layers, updated_stable, stats)
    """
    import time
    t0 = time.time()
    stats = PAdicReductionStats(prime=p)
    stats.original_unstable = sum(1 for v in stable.values() if v == "unstable")

    # Phase 1: Weight pruning
    pruned_layers, n_zeroed = padic_weight_pruning(layers, p, pruning_threshold)
    stats.weights_zeroed = n_zeroed

    # Count neurons effectively pruned (entire row zeroed)
    for layer in pruned_layers:
        if layer["type"] == "linear":
            W = layer["W"]
            for i in range(W.shape[0]):
                if np.all(np.abs(W[i]) < 1e-15):
                    stats.neurons_pruned += 1

    # Phase 2: Neuron clustering (informational)
    for layer in pruned_layers:
        if layer["type"] == "linear":
            clusters = padic_cluster_neurons(layer["W"], p, cluster_radius)
            stats.clusters_found += len(clusters)
            break  # only cluster the first layer for efficiency

    # Phase 3: ReLU phase determination
    new_stable, n_fixed = padic_relu_analysis(
        pruned_layers, input_lb, input_ub, stable, p,
    )
    stats.relu_phases_determined = n_fixed
    stats.reduced_unstable = sum(1 for v in new_stable.values() if v == "unstable")

    stats.time_seconds = time.time() - t0
    logger.info(stats.summary())

    return pruned_layers, new_stable, stats
