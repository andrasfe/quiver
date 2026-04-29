"""QAOA-inspired ansatz: alternating cost (RZZ) and mixer (RX) layers.

When a :class:`Topology` is supplied, RZZ is placed only on edges of the
topology. The `ring` flag is honoured only if the (n-1, 0) wrap-around
is itself an edge of the topology — on a heavy-hex path this means the
ring closure silently disappears, eliminating the SWAP cascade the IBM
hardware paper documented for textbook ring-QAOA.
"""

from __future__ import annotations

from dataclasses import dataclass

from quivercirc.circuit import CircuitSpec, GateSpec
from quivercirc.topology import Topology


@dataclass
class QAOAInspired:
    num_qubits: int
    num_layers: int = 2
    ring: bool = True
    family: str = "qaoa"
    topology: Topology | None = None

    @property
    def num_params(self) -> int:
        return 2 * self.num_layers

    def _cost_pairs(self) -> list[tuple[int, int]]:
        if self.topology is None:
            pairs = [(q, q + 1) for q in range(self.num_qubits - 1)]
            if self.ring and self.num_qubits > 2:
                pairs.append((self.num_qubits - 1, 0))
            return pairs
        return self.topology.edge_list()

    def build(self) -> CircuitSpec:
        spec = CircuitSpec(num_qubits=self.num_qubits)
        for q in range(self.num_qubits):
            spec.add(GateSpec("h", (q,)))
        idx = 0
        pairs = self._cost_pairs()
        for _ in range(self.num_layers):
            gamma = idx
            for (a, b) in pairs:
                spec.add(GateSpec("rzz", (a, b), gamma))
            idx += 1
            beta = idx
            for q in range(self.num_qubits):
                spec.add(GateSpec("rx", (q,), beta))
            idx += 1
        spec.num_params = idx
        return spec
