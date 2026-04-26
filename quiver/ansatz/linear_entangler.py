"""Linear-entangler ansatz: a single layer of Ry rotations followed by a
chain of CNOTs and a closing layer of Ry rotations. Repeats with
`num_layers`. Distinct from the hardware-efficient template in that it
uses only one rotation axis per block, producing a different gate signature.
"""

from __future__ import annotations

from dataclasses import dataclass

from quiver.circuit import CircuitSpec, GateSpec


@dataclass
class LinearEntangler:
    num_qubits: int
    num_layers: int = 2
    family: str = "linear_entangler"

    @property
    def num_params(self) -> int:
        return self.num_qubits * (self.num_layers + 1)

    def build(self) -> CircuitSpec:
        spec = CircuitSpec(num_qubits=self.num_qubits)
        idx = 0
        for q in range(self.num_qubits):
            spec.add(GateSpec("ry", (q,), idx))
            idx += 1
        for _ in range(self.num_layers):
            for q in range(self.num_qubits - 1):
                spec.add(GateSpec("cnot", (q, q + 1)))
            for q in range(self.num_qubits):
                spec.add(GateSpec("ry", (q,), idx))
                idx += 1
        spec.num_params = idx
        return spec
