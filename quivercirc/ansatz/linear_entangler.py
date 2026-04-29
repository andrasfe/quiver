"""Linear-entangler ansatz: a layer of Ry rotations followed by a chain
of CNOTs and another layer of Ry rotations, repeated `num_layers` times.

When a :class:`Topology` is supplied, the CNOT chain is replaced by an
iteration over the topology's edge list (still emitted in canonical
order so the result is deterministic). On a non-line topology this no
longer forms a single chain; instead each layer applies the topology's
edges as a fixed set of disjoint CNOTs (where possible) or in sequence
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

from quivercirc.circuit import CircuitSpec, GateSpec
from quivercirc.topology import Topology


@dataclass
class LinearEntangler:
    num_qubits: int
    num_layers: int = 2
    family: str = "linear_entangler"
    topology: Topology | None = None

    @property
    def num_params(self) -> int:
        return self.num_qubits * (self.num_layers + 1)

    def _chain_pairs(self) -> list[tuple[int, int]]:
        if self.topology is None:
            return [(q, q + 1) for q in range(self.num_qubits - 1)]
        return self.topology.edge_list()

    def build(self) -> CircuitSpec:
        spec = CircuitSpec(num_qubits=self.num_qubits)
        idx = 0
        for q in range(self.num_qubits):
            spec.add(GateSpec("ry", (q,), idx))
            idx += 1
        pairs = self._chain_pairs()
        for _ in range(self.num_layers):
            for (a, b) in pairs:
                spec.add(GateSpec("cnot", (a, b)))
            for q in range(self.num_qubits):
                spec.add(GateSpec("ry", (q,), idx))
                idx += 1
        spec.num_params = idx
        return spec
