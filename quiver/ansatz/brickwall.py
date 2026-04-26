"""Brick-wall ansatz with full RZ-RY-RZ (Euler) single-qubit rotations and
alternating-offset CNOT layers.

Each layer consists of:

  1. RZ(θ_a) RY(θ_b) RZ(θ_c)  on every qubit  (3 params/qubit, full SU(2) coverage)
  2. CNOTs on disjoint pairs at offset (layer % 2):
       even layers: (0,1) (2,3) (4,5) ...
       odd  layers: (1,2) (3,4) (5,6) ...

The brick-wall pattern spreads entanglement faster than a linear chain
(every qubit sees the full graph in O(n) layers vs O(n) at *each* depth
for the linear chain). Combined with Euler rotations this gets close to
the expressivity of a 2-design at modest depth — useful for unstructured
targets where simpler templates can't reach the verifier threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

from quiver.circuit import CircuitSpec, GateSpec


@dataclass
class BrickWall:
    num_qubits: int
    num_layers: int = 4
    family: str = "brickwall"

    @property
    def num_params(self) -> int:
        return 3 * self.num_qubits * self.num_layers

    def build(self) -> CircuitSpec:
        spec = CircuitSpec(num_qubits=self.num_qubits)
        idx = 0
        for layer in range(self.num_layers):
            for q in range(self.num_qubits):
                spec.add(GateSpec("rz", (q,), idx)); idx += 1
                spec.add(GateSpec("ry", (q,), idx)); idx += 1
                spec.add(GateSpec("rz", (q,), idx)); idx += 1
            offset = layer % 2
            for q in range(offset, self.num_qubits - 1, 2):
                spec.add(GateSpec("cnot", (q, q + 1)))
        spec.num_params = idx
        return spec
