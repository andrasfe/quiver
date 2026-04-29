"""Hardware-efficient ansatz: alternating single-qubit rotations and
nearest-neighbour CNOT entanglers, repeated `num_layers` times.

When a :class:`Topology` is supplied, the entangling layer iterates the
topology's edge list instead of the implicit (q, q+1) chain. This makes
the ansatz a no-op for the transpiler on devices whose coupling map
matches the chosen topology.
"""

from __future__ import annotations

from dataclasses import dataclass

from quivercirc.circuit import CircuitSpec, GateSpec
from quivercirc.topology import Topology


@dataclass
class HardwareEfficient:
    num_qubits: int
    num_layers: int = 2
    rotation_axes: tuple[str, ...] = ("ry", "rz")
    entangler: str = "cnot"
    family: str = "hardware_efficient"
    topology: Topology | None = None

    @property
    def num_params(self) -> int:
        return self.num_qubits * len(self.rotation_axes) * self.num_layers

    def _entangler_pairs(self) -> list[tuple[int, int]]:
        if self.topology is None:
            return [(q, q + 1) for q in range(self.num_qubits - 1)]
        return self.topology.edge_list()

    def build(self) -> CircuitSpec:
        spec = CircuitSpec(num_qubits=self.num_qubits)
        idx = 0
        pairs = self._entangler_pairs()
        for _ in range(self.num_layers):
            for axis in self.rotation_axes:
                for q in range(self.num_qubits):
                    spec.add(GateSpec(axis, (q,), idx))
                    idx += 1
            for (a, b) in pairs:
                spec.add(GateSpec(self.entangler, (a, b)))
        spec.num_params = idx
        return spec
