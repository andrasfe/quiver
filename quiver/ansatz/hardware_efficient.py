"""Hardware-efficient ansatz: alternating single-qubit rotations and
nearest-neighbour CNOT entanglers, repeated `num_layers` times."""

from __future__ import annotations

from dataclasses import dataclass

from quiver.circuit import CircuitSpec, GateSpec


@dataclass
class HardwareEfficient:
    num_qubits: int
    num_layers: int = 2
    rotation_axes: tuple[str, ...] = ("ry", "rz")
    entangler: str = "cnot"
    family: str = "hardware_efficient"

    @property
    def num_params(self) -> int:
        return self.num_qubits * len(self.rotation_axes) * self.num_layers

    def build(self) -> CircuitSpec:
        spec = CircuitSpec(num_qubits=self.num_qubits)
        idx = 0
        for _ in range(self.num_layers):
            for axis in self.rotation_axes:
                for q in range(self.num_qubits):
                    spec.add(GateSpec(axis, (q,), idx))
                    idx += 1
            for q in range(self.num_qubits - 1):
                spec.add(GateSpec(self.entangler, (q, q + 1)))
        spec.num_params = idx
        return spec
