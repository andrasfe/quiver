"""QAOA-inspired ansatz: alternating cost (RZZ on a ring) and mixer (RX)
layers. Each layer uses two parameters (gamma, beta), giving 2 * p params.
"""

from __future__ import annotations

from dataclasses import dataclass

from quivercirc.circuit import CircuitSpec, GateSpec


@dataclass
class QAOAInspired:
    num_qubits: int
    num_layers: int = 2
    ring: bool = True
    family: str = "qaoa"

    @property
    def num_params(self) -> int:
        return 2 * self.num_layers

    def build(self) -> CircuitSpec:
        spec = CircuitSpec(num_qubits=self.num_qubits)
        for q in range(self.num_qubits):
            spec.add(GateSpec("h", (q,)))
        idx = 0
        for _ in range(self.num_layers):
            gamma = idx
            for q in range(self.num_qubits - 1):
                spec.add(GateSpec("rzz", (q, q + 1), gamma))
            if self.ring and self.num_qubits > 2:
                spec.add(GateSpec("rzz", (self.num_qubits - 1, 0), gamma))
            idx += 1
            beta = idx
            for q in range(self.num_qubits):
                spec.add(GateSpec("rx", (q,), beta))
            idx += 1
        spec.num_params = idx
        return spec
