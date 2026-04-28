"""All-to-all entangler: rotations on every qubit, followed by a CNOT
between every pair (i, j), i < j, repeated num_layers times.

This is structurally maximal for connectivity — the diversity metric
will see this as a different topology family from any chain or
brick-wall. The trade-off is gate count: n*(n-1)/2 CNOTs per layer
versus n-1 for linear chains. Use for small-to-medium n where the
extra entanglement pays for the depth.
"""

from __future__ import annotations

from dataclasses import dataclass

from quivercirc.circuit import CircuitSpec, GateSpec


@dataclass
class AllToAll:
    num_qubits: int
    num_layers: int = 2
    rotation_axes: tuple[str, ...] = ("ry", "rz")
    family: str = "all_to_all"

    @property
    def num_params(self) -> int:
        return self.num_qubits * len(self.rotation_axes) * self.num_layers

    def build(self) -> CircuitSpec:
        spec = CircuitSpec(num_qubits=self.num_qubits)
        idx = 0
        for _ in range(self.num_layers):
            for axis in self.rotation_axes:
                for q in range(self.num_qubits):
                    spec.add(GateSpec(axis, (q,), idx)); idx += 1
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    spec.add(GateSpec("cnot", (i, j)))
        spec.num_params = idx
        return spec
