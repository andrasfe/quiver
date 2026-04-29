"""Brick-wall ansatz with full RZ-RY-RZ (Euler) single-qubit rotations and
alternating-offset CNOT layers.

When a :class:`Topology` is supplied, the brick-wall is built from a
greedy 2-colouring of the topology's edge list: even layers use one
matching, odd layers the other. The pattern degenerates to the original
(0,1)(2,3)... / (1,2)(3,4)... bricks when the topology is a line.
"""

from __future__ import annotations

from dataclasses import dataclass

from quivercirc.circuit import CircuitSpec, GateSpec
from quivercirc.topology import Topology


def _two_colour_matching(edges: list[tuple[int, int]]) -> tuple[list, list]:
    """Greedy 2-colouring into two matchings (sets of edges with no
    shared endpoint). Sufficient for any bipartite-like coupling
    graph; on heavy-hex paths it produces the canonical brick pattern."""
    used_a, used_b = set(), set()
    a, b = [], []
    for (i, j) in edges:
        if i not in used_a and j not in used_a:
            a.append((i, j)); used_a.update([i, j])
        elif i not in used_b and j not in used_b:
            b.append((i, j)); used_b.update([i, j])
        else:
            # neither colour is open — start a new one in the larger
            # bucket; harmless on small graphs.
            (a if len(a) <= len(b) else b).append((i, j))
    return a, b


@dataclass
class BrickWall:
    num_qubits: int
    num_layers: int = 4
    family: str = "brickwall"
    topology: Topology | None = None

    @property
    def num_params(self) -> int:
        return 3 * self.num_qubits * self.num_layers

    def _layer_pairs(self) -> tuple[list, list]:
        if self.topology is None:
            even = [(q, q + 1) for q in range(0, self.num_qubits - 1, 2)]
            odd = [(q, q + 1) for q in range(1, self.num_qubits - 1, 2)]
            return even, odd
        return _two_colour_matching(self.topology.edge_list())

    def build(self) -> CircuitSpec:
        spec = CircuitSpec(num_qubits=self.num_qubits)
        idx = 0
        even, odd = self._layer_pairs()
        for layer in range(self.num_layers):
            for q in range(self.num_qubits):
                spec.add(GateSpec("rz", (q,), idx)); idx += 1
                spec.add(GateSpec("ry", (q,), idx)); idx += 1
                spec.add(GateSpec("rz", (q,), idx)); idx += 1
            pairs = even if (layer % 2 == 0) else odd
            for (a, b) in pairs:
                spec.add(GateSpec("cnot", (a, b)))
        spec.num_params = idx
        return spec
