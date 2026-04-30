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

    `physical_layout`, when set, records the physical-qubit assignment
    for each logical qubit (logical index ``i`` lives on physical qubit
    ``physical_layout[i]``). Constructed by :meth:`from_backend`; pass
    to ``qiskit.transpile(..., initial_layout=topo.physical_layout)`` to
    pin the embedding instead of letting the transpiler choose.
    """

    num_qubits: int
    edges: frozenset[tuple[int, int]]
    physical_layout: tuple[int, ...] | None = None

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

    @classmethod
    def from_backend(
        cls, backend, n: int, layout: str = "path", start_qubit: int | None = None,
    ) -> "Topology":
        """Pull the device coupling map and extract a connected n-qubit
        subgraph; return a Topology with edges relabeled 0..n-1 and a
        physical_layout recording the chosen physical qubits.

        Parameters
        ----------
        backend : a qiskit backend with a ``coupling_map`` attribute (or
            something that quacks like one). Both ``CouplingMap`` and a
            list-of-pairs are accepted.
        n : number of logical qubits to embed.
        layout : ``"path"`` greedy-walks a length-n path (preferring
            low-degree neighbours so we don't waste degree-3 hubs);
            ``"compact"`` BFS-expands from ``start_qubit`` until n
            qubits are collected.
        start_qubit : starting physical qubit index. Defaults to the
            lowest-index qubit in the coupling graph.
        """
        try:
            edge_iter = backend.coupling_map.get_edges()
        except AttributeError:
            edge_iter = list(backend.coupling_map)

        adj: dict[int, set[int]] = {}
        for (a, b) in edge_iter:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        if not adj:
            raise ValueError("backend has empty coupling map")

        if start_qubit is None:
            start_qubit = min(adj.keys())

        if layout == "path":
            # DFS with backtracking — guaranteed to find a length-n
            # path if one exists. Greedy "prefer low-degree" without
            # backtracking can dead-end on heavy-hex side branches.
            def find_path(start):
                path = [start]
                visited = {start}

                def dfs():
                    if len(path) == n:
                        return True
                    # Prefer low-degree neighbours (don't burn hubs early)
                    cands = sorted(adj[path[-1]] - visited,
                                   key=lambda q: (len(adj[q]), q))
                    for nb in cands:
                        path.append(nb)
                        visited.add(nb)
                        if dfs():
                            return True
                        path.pop()
                        visited.discard(nb)
                    return False

                return path if dfs() else None

            physical = find_path(start_qubit)
            if physical is None:
                # try every other starting qubit, low-degree first
                for s in sorted(adj.keys(),
                                key=lambda q: (len(adj[q]), q)):
                    if s == start_qubit:
                        continue
                    physical = find_path(s)
                    if physical is not None:
                        break
            if physical is None:
                raise ValueError(f"could not extract length-{n} path "
                                 f"from coupling map")
        elif layout == "compact":
            physical = [start_qubit]
            frontier = list(adj[start_qubit])
            seen = {start_qubit}
            while len(physical) < n and frontier:
                q = frontier.pop(0)
                if q in seen:
                    continue
                seen.add(q)
                physical.append(q)
                for nb in sorted(adj[q]):
                    if nb not in seen and nb not in frontier:
                        frontier.append(nb)
            if len(physical) < n:
                raise ValueError(f"could not collect {n} qubits via BFS")
        else:
            raise ValueError(f"unknown layout '{layout}'")

        phys_to_logical = {p: i for i, p in enumerate(physical)}
        edges = set()
        for (a, b) in edge_iter:
            if a in phys_to_logical and b in phys_to_logical:
                edges.add(_canon(phys_to_logical[a], phys_to_logical[b]))
        return cls(
            num_qubits=n,
            edges=frozenset(edges),
            physical_layout=tuple(physical),
        )

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
