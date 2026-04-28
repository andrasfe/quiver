"""Structural diversity metric.

Two circuits are "structurally similar" if they share most gate-sequence
tokens, route 2-qubit gates over the same qubit pairs, and have similar
depth. The metric returns a similarity in [0, 1]; diversity = 1 - similarity.

The three components are:

1. **Gate-sequence edit distance** (Levenshtein on (name, qubits) tokens),
   normalised by the longer sequence. Captures gate-order and gate-type
   differences — the dominant signal for distinguishing ansatz families.
2. **Qubit-connectivity overlap** (Jaccard on the set of unordered qubit
   pairs touched by 2-qubit gates). Captures wiring topology differences,
   e.g. linear chain vs ring vs all-to-all.
3. **Depth difference**, normalised by the larger depth. Captures whether
   one solution is significantly shallower than another.

Weights default to (0.5, 0.3, 0.2): edit distance dominates because mere
parameter tweaks within the same template should *not* register as diverse,
whereas a different gate sequence should.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from quivercirc.circuit import CircuitSpec


@dataclass(frozen=True)
class DiversityWeights:
    edit: float = 0.5
    connectivity: float = 0.3
    depth: float = 0.2

    def normalised(self) -> "DiversityWeights":
        total = self.edit + self.connectivity + self.depth
        if total <= 0:
            raise ValueError("diversity weights must sum to a positive value")
        return DiversityWeights(self.edit / total, self.connectivity / total, self.depth / total)


def _levenshtein(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cost = 0 if ai == bj else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _edit_similarity(a: CircuitSpec, b: CircuitSpec) -> float:
    sa, sb = a.signature(), b.signature()
    n = max(len(sa), len(sb))
    if n == 0:
        return 1.0
    return 1.0 - _levenshtein(sa, sb) / n


def _connectivity_similarity(a: CircuitSpec, b: CircuitSpec) -> float:
    ca, cb = a.connectivity(), b.connectivity()
    if not ca and not cb:
        return 1.0
    union = ca | cb
    if not union:
        return 1.0
    return len(ca & cb) / len(union)


def _depth_similarity(a: CircuitSpec, b: CircuitSpec) -> float:
    da, db = a.depth, b.depth
    if da == 0 and db == 0:
        return 1.0
    return 1.0 - abs(da - db) / max(da, db)


def structural_similarity(
    a: CircuitSpec, b: CircuitSpec, weights: DiversityWeights | None = None
) -> float:
    """Similarity in [0, 1]. 1.0 means structurally identical."""
    w = (weights or DiversityWeights()).normalised()
    return (
        w.edit * _edit_similarity(a, b)
        + w.connectivity * _connectivity_similarity(a, b)
        + w.depth * _depth_similarity(a, b)
    )


def diversity_score(
    candidate: CircuitSpec,
    existing: Iterable[CircuitSpec],
    weights: DiversityWeights | None = None,
) -> float:
    """Distance from the candidate to its *nearest* existing solution.

    Returns 1.0 when `existing` is empty (the first candidate is maximally
    diverse). Otherwise returns 1 - max similarity, so a low score means
    the candidate is too close to something we have already accepted.
    """
    existing = list(existing)
    if not existing:
        return 1.0
    sims = [structural_similarity(candidate, e, weights) for e in existing]
    return 1.0 - max(sims)
