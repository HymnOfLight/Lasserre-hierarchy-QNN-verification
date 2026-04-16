"""
Correlative sparsity analysis for deep network verification.

Analyzes the network topology to identify variable coupling patterns,
enabling block-diagonal decomposition of the moment matrix and
reducing the SDP problem size dramatically.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
import logging

from ..network.quantized_network import QuantizedNetwork

logger = logging.getLogger(__name__)


class CorrelativeSparsityAnalyzer:
    """
    Exploits correlative sparsity (variable coupling) patterns in the
    polynomial optimization problem arising from neural network verification.

    In deep networks, neurons in distant layers have indirect coupling
    through intermediate layers. By analyzing this structure, we can
    decompose the large moment matrix into smaller blocks.
    """

    def __init__(self, network: QuantizedNetwork):
        self.network = network
        self._coupling_graph: Optional[nx.Graph] = None
        self._cliques: Optional[List[Set[int]]] = None
        self._variable_map: Dict[str, int] = {}

    def build_coupling_graph(self) -> nx.Graph:
        """
        Build the correlative sparsity pattern (CSP) graph.

        Each variable (neuron pre/post activation value) is a node.
        An edge connects two variables if they appear together in
        at least one constraint polynomial.
        """
        G = nx.Graph()
        var_idx = 0

        # Assign variable indices to each neuron
        for layer_idx, layer in enumerate(self.network.layers):
            if layer.layer_type in ("linear", "conv2d"):
                n_out = layer.n_outputs
                for j in range(n_out):
                    name = f"L{layer_idx}_pre_{j}"
                    self._variable_map[name] = var_idx
                    G.add_node(var_idx, layer=layer_idx, neuron=j, type="pre")
                    var_idx += 1

            elif layer.layer_type in ("relu", "sigmoid", "tanh"):
                n_neurons = layer.n_outputs if layer.n_outputs > 0 else 0
                if n_neurons == 0 and layer.output_shape is not None:
                    n_neurons = int(np.prod(layer.output_shape))

                for j in range(n_neurons):
                    name = f"L{layer_idx}_post_{j}"
                    self._variable_map[name] = var_idx
                    G.add_node(var_idx, layer=layer_idx, neuron=j, type="post")
                    var_idx += 1

        # Add edges based on layer connectivity
        for layer_idx, layer in enumerate(self.network.layers):
            if layer.layer_type in ("linear", "conv2d") and layer.weight is not None:
                W = layer.weight
                if W.ndim > 2:
                    W = W.reshape(W.shape[0], -1)

                n_out, n_in = W.shape

                # Find input variable indices
                prev_vars = []
                for prev_idx in range(layer_idx - 1, -1, -1):
                    prev_layer = self.network.layers[prev_idx]
                    if prev_layer.layer_type in ("relu", "sigmoid", "tanh"):
                        for j in range(n_in):
                            key = f"L{prev_idx}_post_{j}"
                            if key in self._variable_map:
                                prev_vars.append(self._variable_map[key])
                        break

                # Output variable indices
                out_vars = []
                for j in range(n_out):
                    key = f"L{layer_idx}_pre_{j}"
                    if key in self._variable_map:
                        out_vars.append(self._variable_map[key])

                # Connect inputs to outputs based on weight sparsity
                for i, ov in enumerate(out_vars):
                    for j, iv in enumerate(prev_vars):
                        if j < n_in and abs(W[i, j]) > 1e-10:
                            G.add_edge(iv, ov)

            elif layer.layer_type in ("relu", "sigmoid", "tanh"):
                # Pre-activation to post-activation: each neuron couples to itself
                for j in range(layer.n_outputs if layer.n_outputs > 0 else 0):
                    pre_key = f"L{layer_idx - 1}_pre_{j}" if layer_idx > 0 else None
                    post_key = f"L{layer_idx}_post_{j}"
                    if pre_key and pre_key in self._variable_map and post_key in self._variable_map:
                        G.add_edge(
                            self._variable_map[pre_key],
                            self._variable_map[post_key],
                        )

        self._coupling_graph = G
        return G

    def find_maximal_cliques(self, max_clique_size: int = 50) -> List[Set[int]]:
        """
        Find maximal cliques in the coupling graph for block decomposition.

        Uses chordal completion to ensure running intersection property
        holds for the sparse SOS decomposition.
        """
        if self._coupling_graph is None:
            self.build_coupling_graph()

        G = self._coupling_graph

        if G.number_of_nodes() == 0:
            return []

        # For tree-like networks, the graph is often already chordal
        # or close to chordal. Use minimum degree ordering for
        # chordal completion.
        try:
            if nx.is_chordal(G):
                cliques = list(nx.chordal_graph_cliques(G))
            else:
                # Chordal completion via minimum fill-in heuristic
                H = G.copy()
                ordering = list(nx.algorithms.coloring.greedy_color(H, strategy="smallest_last").keys())
                for v in ordering:
                    neighbors = list(H.neighbors(v))
                    for i, n1 in enumerate(neighbors):
                        for n2 in neighbors[i + 1:]:
                            if not H.has_edge(n1, n2):
                                H.add_edge(n1, n2)
                cliques = [set(c) for c in nx.find_cliques(H)]

        except Exception:
            # Fallback: use layer-based cliques
            cliques = self._layer_based_cliques()

        # Filter cliques that are too large
        filtered = [c for c in cliques if len(c) <= max_clique_size]
        if not filtered:
            filtered = self._layer_based_cliques()

        self._cliques = filtered
        return filtered

    def _layer_based_cliques(self) -> List[Set[int]]:
        """
        Fallback: create cliques based on adjacent layers.
        Each clique contains variables from two adjacent layers.
        """
        cliques = []
        layers_with_vars: Dict[int, Set[int]] = {}

        if self._coupling_graph is None:
            return []

        for node, data in self._coupling_graph.nodes(data=True):
            layer = data.get("layer", -1)
            if layer not in layers_with_vars:
                layers_with_vars[layer] = set()
            layers_with_vars[layer].add(node)

        sorted_layers = sorted(layers_with_vars.keys())
        for i in range(len(sorted_layers) - 1):
            clique = layers_with_vars[sorted_layers[i]] | layers_with_vars[sorted_layers[i + 1]]
            cliques.append(clique)

        return cliques

    def get_block_structure(self) -> Dict:
        """
        Get the block-diagonal structure for the sparse SDP formulation.
        Returns the cliques and their overlap structure.
        """
        if self._cliques is None:
            self.find_maximal_cliques()

        cliques = self._cliques or []

        # Compute overlap between cliques
        overlaps = {}
        for i in range(len(cliques)):
            for j in range(i + 1, len(cliques)):
                overlap = cliques[i] & cliques[j]
                if overlap:
                    overlaps[(i, j)] = overlap

        total_vars = len(self._variable_map)
        sum_block_sizes = sum(len(c) for c in cliques)
        compression = sum_block_sizes / max(total_vars, 1)

        return {
            "n_blocks": len(cliques),
            "cliques": cliques,
            "overlaps": overlaps,
            "total_variables": total_vars,
            "sum_block_sizes": sum_block_sizes,
            "compression_ratio": compression,
            "variable_map": dict(self._variable_map),
        }

    def sparsity_summary(self) -> Dict:
        """Summary statistics about the sparsity structure."""
        if self._coupling_graph is None:
            self.build_coupling_graph()

        G = self._coupling_graph
        n = G.number_of_nodes()
        m = G.number_of_edges()
        max_edges = n * (n - 1) / 2 if n > 1 else 1

        return {
            "n_variables": n,
            "n_couplings": m,
            "density": m / max_edges if max_edges > 0 else 0.0,
            "is_sparse": m / max_edges < 0.3 if max_edges > 0 else True,
            "max_degree": max(dict(G.degree()).values()) if n > 0 else 0,
            "connected_components": nx.number_connected_components(G),
        }
