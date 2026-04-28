"""Strongly-entangling layers (PennyLane-style): per-layer Euler rotations
followed by a ring of CNOT(q, (q + stride) mod n) where the stride
*increases* with layer depth. This injects non-local correlations
directly into early layers, in contrast to chain-only ansätze that need
many layers to mix distant qubits.

stride = 1 + (layer mod (n-1)), so each layer chooses a different jump
distance and consecutive layers use different connectivity patterns.
"""

from __future__ import annotations

from dataclasses import dataclass

from quivercirc.circuit import CircuitSpec, GateSpec


@dataclass
class StronglyEntangling:
    num_qubits: int
    num_layers: int = 3
    family: str = "strongly_entangling"

    @property
    def num_params(self) -> int:
        return 3 * self.num_qubits * self.num_layers

    def build(self) -> CircuitSpec:
        spec = CircuitSpec(num_qubits=self.num_qubits)
        idx = 0
        denom = max(1, self.num_qubits - 1)
        for layer in range(self.num_layers):
            for q in range(self.num_qubits):
                spec.add(GateSpec("rx", (q,), idx)); idx += 1
                spec.add(GateSpec("ry", (q,), idx)); idx += 1
                spec.add(GateSpec("rz", (q,), idx)); idx += 1
            stride = 1 + (layer % denom)
            for q in range(self.num_qubits):
                target = (q + stride) % self.num_qubits
                if target != q:
                    spec.add(GateSpec("cnot", (q, target)))
        spec.num_params = idx
        return spec
