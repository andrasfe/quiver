"""Library distillation — group fragments by structural pattern and
promote dominant patterns to parametric ansatz templates the system
*invented for itself*.

A fragment's *canonical form* is its gate sequence under the relabeling
that maps qubits in order of first appearance to {0, 1, 2, ...}. Two
fragments with the same canonical form differ only in which absolute
qubits they happened to use — structurally they're the same primitive.

Distillation:
  1. Compute every fragment's canonical form.
  2. Group fragments by canonical form.
  3. Sort groups by population (frequency in the library).
  4. Return the canonical exemplar of the top-k groups.

Each exemplar becomes the seed for a `DiscoveredAnsatz` that tiles the
canonical pattern across an arbitrary host qubit count, with one fresh
parameter slot per parametric gate per tile. Effectively, the canonical
pattern becomes a "module" the new ansatz repeats.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from quiver.circuit import GateSpec
from quiver.microstructures import Fragment, MicrostructureLibrary


CanonicalForm = tuple[tuple[str, tuple[int, ...], bool], ...]


def canonicalize(fragment: Fragment) -> CanonicalForm:
    """Normalize a fragment so structurally-identical fragments hash equal,
    regardless of which absolute qubits they used."""
    qubit_remap: dict[int, int] = {}
    canon: list[tuple[str, tuple[int, ...], bool]] = []
    for g in fragment.gates:
        new_qs: list[int] = []
        for q in g.qubits:
            if q not in qubit_remap:
                qubit_remap[q] = len(qubit_remap)
            new_qs.append(qubit_remap[q])
        canon.append((g.name, tuple(new_qs), g.is_parametric))
    return tuple(canon)


@dataclass
class DistilledPattern:
    canonical: CanonicalForm
    exemplar: Fragment
    occurrences: int

    @property
    def width(self) -> int:
        """Number of distinct qubit slots referenced by the canonical form."""
        if not self.canonical:
            return 0
        return 1 + max(q for _, qs, _ in self.canonical for q in qs)

    @property
    def length(self) -> int:
        return len(self.canonical)


def distill_library(
    library: MicrostructureLibrary,
    top_k: int = 5,
    min_occurrences: int = 2,
    min_width: int = 2,
) -> list[DistilledPattern]:
    """Return the top-k most-frequent canonical patterns in the library
    (filtered by minimum occurrence count and minimum qubit width — single
    qubit patterns aren't useful as ansatz modules)."""
    groups: dict[CanonicalForm, list[Fragment]] = defaultdict(list)
    for f in library.fragments:
        groups[canonicalize(f)].append(f)

    counts = Counter({k: len(v) for k, v in groups.items()})
    patterns: list[DistilledPattern] = []
    for canon, occ in counts.most_common():
        if occ < min_occurrences:
            continue
        exemplar = groups[canon][0]
        pat = DistilledPattern(canonical=canon, exemplar=exemplar, occurrences=occ)
        if pat.width < min_width:
            continue
        patterns.append(pat)
        if len(patterns) >= top_k:
            break
    return patterns


def fragment_from_canonical(canon: CanonicalForm) -> Fragment:
    """Rebuild a Fragment with zero parameters, ready to be tiled."""
    import numpy as np
    gates: list[GateSpec] = []
    param_counter = 0
    for name, qs, is_param in canon:
        if is_param:
            gates.append(GateSpec(name, qs, param_counter))
            param_counter += 1
        else:
            gates.append(GateSpec(name, qs, None))
    return Fragment(gates=gates, params=np.zeros(param_counter, dtype=float))
