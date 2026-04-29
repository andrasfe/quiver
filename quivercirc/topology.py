"""Hardware topology: a coupling map for ansatz generation.

The ansatz family classes (`HardwareEfficient`, `BrickWall`, etc.) and
`AdaptiveGrowth` accept an optional :class:`Topology`. When provided,
every 2-qubit gate emitted by the constructor is required to act on a
pair `(i, j)` that lies on an edge of the topology graph.

This avoids the SWAP overhead the IBM hardware paper documented: a
QAOA "ring" cost layer transpiled onto a heavy-hex line incurs SWAPs
to close the ring; with `topology=Topology.line(n)`, the ring closure
is simply not generated, so transpilation is a no-op.

The intended workflow:

    >>> from quivercirc.topology import Topology
    >>> from quivercirc.ansatz import HardwareEfficient, QAOAInspired
    >>> topo = Topology.line(4)
    >>> spec = HardwareEfficient(num_qubits=4, num_layers=2, topology=topo).build()
    >>> # 2q gates only on (0,1), (1,2), (2,3) — no SWAPs needed on a heavy-hex path.

Edges are stored canonically as ``(min, max)`` so look-ups are
direction-agnostic; ansätze can still emit gates in either direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _canon(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


@dataclass(frozen=True)
class Topology:
    """A coupling-map graph on `num_qubits` logical qubits.

    Edges are an undirected, canonicalized set of (i, j) pairs with i<j.
    """

    num_qubits: int
    edges: frozenset[tuple[int, int]]

    @classmethod
    def line(cls, n: int) -> "Topology":
        return cls(n, frozenset(_canon(i, i + 1) for i in range(n - 1)))

    @classmethod
    def ring(cls, n: int) -> "Topology":
        if n < 3:
            return cls.line(n)
        edges = {_canon(i, i + 1) for i in range(n - 1)} | {_canon(0, n - 1)}
        return cls(n, frozenset(edges))

    @classmethod
    def from_edges(cls, n: int, edges: Iterable[tuple[int, int]]) -> "Topology":
        return cls(n, frozenset(_canon(i, j) for (i, j) in edges))

    @classmethod
    def heavy_hex_path(cls, n: int) -> "Topology":
        """A length-n path embedded in a heavy-hex device.

        For n <= ~12 this is just a linear chain (heavy-hex contains
        long degree-2 paths). The factory exists so user code can be
        explicit about its hardware target without picking a specific
        physical-qubit layout.
        """
        return cls.line(n)

    @classmethod
    def heavy_hex_y(cls, n: int) -> "Topology":
        """A T/Y-shaped subgraph of heavy-hex with one degree-3 hub.

        For n=4 the layout is::

            0 - 1 - 2
                |
                3

        For larger n, qubit 3 starts a second branch off the central
        hub. The shape is illustrative — the practical reason to use
        it is exposing ansätze to a non-line topology where some 2-
        qubit pairs no longer commute through a single SWAP-free path.
        """
        if n < 4:
            return cls.line(n)
        edges = [(0, 1), (1, 2), (1, 3)]
        nxt = 4
        # extend each branch alternately to reach n qubits
        tails = [2, 3]
        while nxt < n:
            t = tails[(nxt - 4) % len(tails)]
            edges.append((t, nxt))
            tails[(nxt - 4) % len(tails)] = nxt
            nxt += 1
        return cls.from_edges(n, edges)

    def has_edge(self, i: int, j: int) -> bool:
        return _canon(i, j) in self.edges

    def edge_list(self) -> list[tuple[int, int]]:
        """Return canonical edges in deterministic ascending order."""
        return sorted(self.edges)

    def neighbors(self, q: int) -> tuple[int, ...]:
        out = []
        for (a, b) in self.edges:
            if a == q:
                out.append(b)
            elif b == q:
                out.append(a)
        return tuple(sorted(out))


__all__ = ["Topology"]
