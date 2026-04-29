"""All-to-all entangler: rotations on every qubit, followed by a CNOT
between every pair (i, j), i < j, repeated num_layers times.

When a :class:`Topology` is supplied, only pairs that are edges of the
topology are kept — i.e. "all-to-all" becomes "all topology edges".
For a line topology this collapses to a hardware-efficient ladder; the
class still exists as a separate family because it preserves a single
rotation block per layer (one-shot full-graph entangling).
"""

from __future__ import annotations

from dataclasses import dataclass

from quivercirc.circuit import CircuitSpec, GateSpec
from quivercirc.topology import Topology


@dataclass
class AllToAll:
    num_qubits: int
    num_layers: int = 2
    rotation_axes: tuple[str, ...] = ("ry", "rz")
    family: str = "all_to_all"
    topology: Topology | None = None

    @property
    def num_params(self) -> int:
        return self.num_qubits * len(self.rotation_axes) * self.num_layers

    def _pairs(self) -> list[tuple[int, int]]:
        if self.topology is None:
            return [(i, j) for i in range(self.num_qubits)
                    for j in range(i + 1, self.num_qubits)]
        return self.topology.edge_list()

    def build(self) -> CircuitSpec:
        spec = CircuitSpec(num_qubits=self.num_qubits)
        idx = 0
        pairs = self._pairs()
        for _ in range(self.num_layers):
            for axis in self.rotation_axes:
                for q in range(self.num_qubits):
                    spec.add(GateSpec(axis, (q,), idx)); idx += 1
            for (a, b) in pairs:
                spec.add(GateSpec("cnot", (a, b)))
        spec.num_params = idx
        return spec
