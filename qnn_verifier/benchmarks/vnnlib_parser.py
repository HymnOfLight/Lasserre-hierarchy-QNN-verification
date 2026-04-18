"""
VNNLIB property file parser.

Parses the VNN-LIB format used by VNN-COMP to specify input/output
constraints for neural network verification properties.

VNNLIB format (simplified SMT-LIB2 subset):
  (declare-const X_0 Real)
  ...
  (assert (<= X_0 0.5))
  (assert (>= X_0 -0.5))
  ...
  (assert (or (and (<= Y_0 Y_1)) ...))
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VNNLIBProperty:
    """Parsed verification property from a VNNLIB file."""
    n_inputs: int = 0
    n_outputs: int = 0
    input_lower: Optional[np.ndarray] = None
    input_upper: Optional[np.ndarray] = None
    # Output constraints: list of disjunctive clauses
    # Each clause is a list of (var_idx, op, bound) or (var_i, op, var_j, offset)
    output_constraints: List = field(default_factory=list)
    raw_text: str = ""


def parse_vnnlib(filepath: str) -> VNNLIBProperty:
    """
    Parse a VNNLIB property file.

    Returns a VNNLIBProperty with input bounds and output constraints.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"VNNLIB file not found: {filepath}")

    text = path.read_text()
    prop = VNNLIBProperty(raw_text=text)

    # Extract variable declarations
    input_vars = re.findall(r"\(declare-const\s+(X_\d+)\s+Real\)", text)
    output_vars = re.findall(r"\(declare-const\s+(Y_\d+)\s+Real\)", text)
    prop.n_inputs = len(input_vars)
    prop.n_outputs = len(output_vars)

    # Parse input bounds from assert statements
    input_lower = np.full(prop.n_inputs, -np.inf)
    input_upper = np.full(prop.n_inputs, np.inf)

    # Match: (assert (>= X_i val)) or (assert (<= X_i val))
    for match in re.finditer(
        r"\(assert\s+\((>=|<=)\s+X_(\d+)\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*\)\s*\)",
        text,
    ):
        op, idx_str, val_str = match.groups()
        idx = int(idx_str)
        val = float(val_str)
        if idx < prop.n_inputs:
            if op == ">=":
                input_lower[idx] = max(input_lower[idx], val)
            else:
                input_upper[idx] = min(input_upper[idx], val)

    # Also match flipped: (assert (<= val X_i)) means X_i >= val
    for match in re.finditer(
        r"\(assert\s+\((>=|<=)\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s+X_(\d+)\s*\)\s*\)",
        text,
    ):
        op, val_str, idx_str = match.groups()
        idx = int(idx_str)
        val = float(val_str)
        if idx < prop.n_inputs:
            if op == "<=":  # val <= X_i  means  X_i >= val
                input_lower[idx] = max(input_lower[idx], val)
            else:  # val >= X_i  means  X_i <= val
                input_upper[idx] = min(input_upper[idx], val)

    prop.input_lower = input_lower
    prop.input_upper = input_upper

    # Parse output constraints (simplified: extract the assert/or/and structure)
    output_constraints = []

    # Simple case: (assert (<= Y_i Y_j)) meaning Y_i <= Y_j
    for match in re.finditer(
        r"\(assert\s+\((<=|>=)\s+Y_(\d+)\s+Y_(\d+)\s*\)\s*\)", text
    ):
        op, i_str, j_str = match.groups()
        output_constraints.append({
            "type": "comparison",
            "op": op,
            "left": int(i_str),
            "right": int(j_str),
        })

    # Disjunctive: (assert (or (and ...) (and ...)))
    or_blocks = re.findall(r"\(assert\s+\(or\s+(.*?)\)\s*\)\s*$", text, re.DOTALL | re.MULTILINE)
    for block in or_blocks:
        clauses = []
        and_blocks = re.findall(r"\(and\s+(.*?)\)", block, re.DOTALL)
        for and_block in and_blocks:
            clause = []
            for cmp in re.finditer(
                r"\((<=|>=)\s+Y_(\d+)\s+Y_(\d+)\)", and_block
            ):
                clause.append({
                    "op": cmp.group(1),
                    "left": int(cmp.group(2)),
                    "right": int(cmp.group(3)),
                })
            for cmp in re.finditer(
                r"\((<=|>=)\s+Y_(\d+)\s+([-+]?[\d.eE+-]+)\)", and_block
            ):
                clause.append({
                    "op": cmp.group(1),
                    "left": int(cmp.group(2)),
                    "bound": float(cmp.group(3)),
                })
            if clause:
                clauses.append(clause)
        if clauses:
            output_constraints.append({"type": "disjunction", "clauses": clauses})

    prop.output_constraints = output_constraints
    return prop
