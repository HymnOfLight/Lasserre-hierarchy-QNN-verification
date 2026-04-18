"""
VNNLIB property file parser.

Parses the VNN-LIB format used by VNN-COMP to specify input/output
constraints for neural network verification properties.

Supported constraint patterns:
  - Input bounds:    (assert (<= X_0 0.5))  /  (assert (>= X_0 -0.5))
  - Output bound:    (assert (>= Y_0 3.99))
  - Output compare:  (assert (<= Y_1 Y_0))
  - Disjunction:     (assert (or (and ...) (and ...)))
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
    output_constraints: List = field(default_factory=list)
    raw_text: str = ""

    def describe(self) -> str:
        """Human-readable description of the property."""
        parts = [f"Inputs: {self.n_inputs}, Outputs: {self.n_outputs}"]
        if self.input_lower is not None:
            bounded = np.isfinite(self.input_lower).sum()
            parts.append(f"Input bounds: {bounded}/{self.n_inputs} bounded")
        for c in self.output_constraints:
            if c["type"] == "output_bound":
                parts.append(f"  Y_{c['var']} {c['op']} {c['bound']:.6f}")
            elif c["type"] == "comparison":
                parts.append(f"  Y_{c['left']} {c['op']} Y_{c['right']}")
            elif c["type"] == "disjunction":
                parts.append(f"  OR of {len(c['clauses'])} conjunctive clauses")
        return "\n".join(parts)


_NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"


def parse_vnnlib(filepath: str) -> VNNLIBProperty:
    """Parse a VNNLIB property file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"VNNLIB file not found: {filepath}")

    text = path.read_text()
    # Strip comments
    lines = [l for l in text.split("\n") if not l.strip().startswith(";")]
    clean = "\n".join(lines)

    prop = VNNLIBProperty(raw_text=text)

    input_vars = re.findall(r"\(declare-const\s+(X_\d+)\s+Real\)", text)
    output_vars = re.findall(r"\(declare-const\s+(Y_\d+)\s+Real\)", text)
    prop.n_inputs = len(input_vars)
    prop.n_outputs = len(output_vars)

    input_lower = np.full(prop.n_inputs, -np.inf)
    input_upper = np.full(prop.n_inputs, np.inf)

    # -- Input bounds: (assert (OP X_i val)) --
    for m in re.finditer(
        r"\(assert\s*\(\s*(>=|<=)\s+X_(\d+)\s+" + _NUM + r"\s*\)\s*\)", clean
    ):
        op, idx, val = m.group(1), int(m.group(2)), float(m.group(3))
        if idx < prop.n_inputs:
            if op == ">=":
                input_lower[idx] = max(input_lower[idx], val)
            else:
                input_upper[idx] = min(input_upper[idx], val)

    # Flipped: (assert (OP val X_i))
    for m in re.finditer(
        r"\(assert\s*\(\s*(>=|<=)\s+" + _NUM + r"\s+X_(\d+)\s*\)\s*\)", clean
    ):
        op, val, idx = m.group(1), float(m.group(2)), int(m.group(3))
        if idx < prop.n_inputs:
            if op == "<=":
                input_lower[idx] = max(input_lower[idx], val)
            else:
                input_upper[idx] = min(input_upper[idx], val)

    prop.input_lower = input_lower
    prop.input_upper = input_upper

    # -- Output bound on single variable: (assert (OP Y_i val)) --
    for m in re.finditer(
        r"\(assert\s*\(\s*(>=|<=)\s+Y_(\d+)\s+" + _NUM + r"\s*\)\s*\)", clean
    ):
        prop.output_constraints.append({
            "type": "output_bound",
            "var": int(m.group(2)),
            "op": m.group(1),
            "bound": float(m.group(3)),
        })

    # -- Comparison between two output vars: (assert (OP Y_i Y_j)) --
    for m in re.finditer(
        r"\(assert\s*\(\s*(>=|<=)\s+Y_(\d+)\s+Y_(\d+)\s*\)\s*\)", clean
    ):
        prop.output_constraints.append({
            "type": "comparison",
            "op": m.group(1),
            "left": int(m.group(2)),
            "right": int(m.group(3)),
        })

    # -- Disjunction: (assert (or ...)) --
    # Use a balanced-paren approach to find the full (assert (or ...)) block
    _parse_disjunctions(clean, prop)

    return prop


def _parse_disjunctions(text: str, prop: VNNLIBProperty):
    """Parse (assert (or ...)) blocks using balanced parenthesis matching."""
    idx = 0
    while True:
        pos = text.find("(assert", idx)
        if pos == -1:
            break
        # Check if this is an (assert (or ...))
        inner_start = text.find("(", pos + 7)
        if inner_start == -1:
            break
        # Skip whitespace
        rest = text[inner_start:].lstrip()
        if not rest.startswith("(or"):
            idx = pos + 7
            continue

        # Find matching closing paren for the outer (assert ...)
        depth = 0
        end = pos
        for i in range(pos, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        block = text[pos:end]
        idx = end

        # Extract the content inside (or ...)
        or_start = block.find("(or")
        if or_start == -1:
            continue
        or_content = block[or_start + 3:]
        # Remove trailing ))
        or_content = or_content.rstrip().rstrip(")")

        clauses = []
        for and_m in re.finditer(r"\(and\s+(.*?)\)", or_content, re.DOTALL):
            clause = _parse_clause(and_m.group(1))
            if clause:
                clauses.append(clause)

        # Handle single-atom clauses without (and ...) wrapper
        if not clauses:
            # Each top-level (...) inside or is a single-atom clause
            for atom_m in re.finditer(r"\(\s*(>=|<=)\s+Y_(\d+)\s+" + _NUM + r"\s*\)", or_content):
                clauses.append([{
                    "op": atom_m.group(1),
                    "var": int(atom_m.group(2)),
                    "bound": float(atom_m.group(3)),
                }])
            for atom_m in re.finditer(r"\(\s*(>=|<=)\s+Y_(\d+)\s+Y_(\d+)\s*\)", or_content):
                clauses.append([{
                    "op": atom_m.group(1),
                    "left": int(atom_m.group(2)),
                    "right": int(atom_m.group(3)),
                }])

        if clauses:
            prop.output_constraints.append({
                "type": "disjunction",
                "clauses": clauses,
            })


def _parse_clause(text: str) -> List[Dict]:
    """Parse atomic constraints inside a conjunctive clause."""
    atoms = []
    # Y_i OP Y_j
    for m in re.finditer(r"\(\s*(>=|<=)\s+Y_(\d+)\s+Y_(\d+)\s*\)", text):
        atoms.append({
            "op": m.group(1), "left": int(m.group(2)), "right": int(m.group(3)),
        })
    # Y_i OP val
    for m in re.finditer(
        r"\(\s*(>=|<=)\s+Y_(\d+)\s+" + _NUM + r"\s*\)", text
    ):
        atoms.append({
            "op": m.group(1), "var": int(m.group(2)), "bound": float(m.group(3)),
        })
    return atoms
