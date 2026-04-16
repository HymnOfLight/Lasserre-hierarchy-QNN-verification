"""
Term sparsity exploitation for polynomial optimization.

Analyzes and exploits the term sparsity pattern (Newton polytope
structure) in the polynomial constraints to reduce the size of
the SOS/SDP relaxation.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


class TermSparsityExploiter:
    """
    Exploits term sparsity in the polynomial optimization formulation
    by analyzing the support (set of monomials with nonzero coefficients)
    of the objective and constraint polynomials.

    Term sparsity allows using smaller SOS multipliers in the
    Positivstellensatz certificate, reducing the SDP matrix sizes.
    """

    def __init__(self, n_vars: int):
        self.n_vars = n_vars
        self.objective_support: Set[Tuple[int, ...]] = set()
        self.constraint_supports: List[Set[Tuple[int, ...]]] = []
        self._tsp_graph: Optional[Dict] = None

    def set_objective(self, poly: Dict[Tuple[int, ...], float]):
        """Record the support of the objective polynomial."""
        self.objective_support = {
            mono for mono, coeff in poly.items() if abs(coeff) > 1e-15
        }

    def add_constraint(self, poly: Dict[Tuple[int, ...], float]):
        """Record the support of a constraint polynomial."""
        support = {
            mono for mono, coeff in poly.items() if abs(coeff) > 1e-15
        }
        self.constraint_supports.append(support)

    def build_term_sparsity_pattern(self) -> Dict:
        """
        Build the term sparsity pattern (TSP) graph.

        Nodes are monomials (from the combined support).
        Two monomials alpha, beta are connected if alpha + beta
        appears in some support.
        """
        combined_support = set(self.objective_support)
        for s in self.constraint_supports:
            combined_support |= s

        all_monomials = sorted(combined_support)
        mono_to_idx = {m: i for i, m in enumerate(all_monomials)}

        adjacency = {i: set() for i in range(len(all_monomials))}

        # Two monomials are coupled if their "sum" could appear
        # in a product of SOS multiplier * constraint
        for i, alpha in enumerate(all_monomials):
            for j, beta in enumerate(all_monomials):
                if i >= j:
                    continue
                gamma = tuple(a + b for a, b in zip(alpha, beta))
                # Check if gamma or any monomial "close" to it appears in supports
                if gamma in combined_support:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        self._tsp_graph = {
            "monomials": all_monomials,
            "mono_to_idx": mono_to_idx,
            "adjacency": adjacency,
        }
        return self._tsp_graph

    def find_term_cliques(self) -> List[Set[int]]:
        """
        Find cliques in the TSP graph for block-diagonal SOS decomposition.
        """
        if self._tsp_graph is None:
            self.build_term_sparsity_pattern()

        adjacency = self._tsp_graph["adjacency"]
        n = len(self._tsp_graph["monomials"])

        if n == 0:
            return []

        # Simple greedy clique cover
        covered = set()
        cliques = []

        nodes = sorted(range(n), key=lambda x: -len(adjacency.get(x, set())))

        for node in nodes:
            if node in covered:
                continue

            clique = {node}
            candidates = adjacency.get(node, set()) - covered

            for c in sorted(candidates, key=lambda x: -len(adjacency.get(x, set()))):
                if all(c in adjacency.get(v, set()) for v in clique):
                    clique.add(c)

            cliques.append(clique)
            covered |= clique

        # Add uncovered singletons
        for node in range(n):
            if node not in covered:
                cliques.append({node})

        return cliques

    def get_reduced_basis(self, max_degree: int) -> List[Tuple[int, ...]]:
        """
        Get a reduced monomial basis using the term sparsity pattern.
        Only includes monomials that appear in the Newton polytope
        of the problem.
        """
        if self._tsp_graph is None:
            self.build_term_sparsity_pattern()

        monomials = self._tsp_graph["monomials"]
        # Include all monomials up to max_degree that are in the support
        reduced = [m for m in monomials if sum(m) <= max_degree]

        # Also include monomials needed for SOS representation
        # (half-degree monomials whose products are in the support)
        half_deg = max_degree // 2
        half_monos = [m for m in monomials if sum(m) <= half_deg]

        additional = set()
        for m in half_monos:
            for m2 in half_monos:
                prod = tuple(a + b for a, b in zip(m, m2))
                if sum(prod) <= max_degree:
                    additional.add(prod)

        result = set(tuple(m) for m in reduced)
        result |= additional
        return sorted(result, key=lambda m: (sum(m), m))

    def compression_info(self) -> Dict:
        """Report how much the term sparsity reduces the problem size."""
        if self._tsp_graph is None:
            self.build_term_sparsity_pattern()

        n_monomials = len(self._tsp_graph["monomials"])
        cliques = self.find_term_cliques()
        sum_clique_sizes = sum(len(c) for c in cliques)

        from ..lasserre.moment_matrix import MonomialBasis
        max_deg = 0
        for m in self._tsp_graph["monomials"]:
            max_deg = max(max_deg, sum(m))

        full_basis = MonomialBasis(self.n_vars, max_deg)
        full_size = full_basis.size

        return {
            "support_size": n_monomials,
            "full_basis_size": full_size,
            "n_term_cliques": len(cliques),
            "sum_clique_sizes": sum_clique_sizes,
            "reduction_ratio": n_monomials / max(full_size, 1),
        }
