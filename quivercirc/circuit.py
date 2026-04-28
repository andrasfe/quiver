"""Backend-agnostic circuit specification.

A CircuitSpec is a list of GateSpec entries that name a gate, the qubits it
acts on, and (for parameterized gates) an index into a flat parameter vector.
Backends translate a CircuitSpec into their native circuit object before
running it; the diversity metric reads CircuitSpecs directly so that
structural comparison does not depend on which backend produced the circuit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


SINGLE_QUBIT_PARAMETRIC = frozenset({"rx", "ry", "rz", "u3", "p"})
SINGLE_QUBIT_FIXED = frozenset({"h", "x", "y", "z", "s", "t", "sdg", "tdg"})
TWO_QUBIT_FIXED = frozenset({"cnot", "cx", "cz", "swap", "iswap", "sqrt_iswap"})
TWO_QUBIT_PARAMETRIC = frozenset({"rzz", "rxx", "ryy"})
TWO_QUBIT = TWO_QUBIT_FIXED | TWO_QUBIT_PARAMETRIC


@dataclass(frozen=True)
class GateSpec:
    name: str
    qubits: tuple[int, ...]
    param_idx: int | None = None

    @property
    def is_parametric(self) -> bool:
        return self.param_idx is not None

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) == 2


@dataclass
class CircuitSpec:
    num_qubits: int
    gates: list[GateSpec] = field(default_factory=list)
    num_params: int = 0

    def add(self, gate: GateSpec) -> None:
        for q in gate.qubits:
            if not 0 <= q < self.num_qubits:
                raise ValueError(f"qubit {q} out of range [0, {self.num_qubits})")
        self.gates.append(gate)

    def extend(self, gates: Iterable[GateSpec]) -> None:
        for g in gates:
            self.add(g)

    @property
    def depth(self) -> int:
        """Layered depth: greedy packing of gates into parallel layers."""
        if not self.gates:
            return 0
        layer_of: dict[int, int] = {q: -1 for q in range(self.num_qubits)}
        max_layer = 0
        for g in self.gates:
            ready = max(layer_of[q] for q in g.qubits) + 1
            for q in g.qubits:
                layer_of[q] = ready
            if ready > max_layer:
                max_layer = ready
        return max_layer + 1

    @property
    def gate_count(self) -> int:
        return len(self.gates)

    @property
    def two_qubit_count(self) -> int:
        return sum(1 for g in self.gates if g.is_two_qubit)

    def connectivity(self) -> set[tuple[int, int]]:
        """Set of unordered qubit pairs that interact via a 2-qubit gate."""
        pairs: set[tuple[int, int]] = set()
        for g in self.gates:
            if g.is_two_qubit:
                a, b = g.qubits
                pairs.add((a, b) if a < b else (b, a))
        return pairs

    def signature(self) -> list[tuple[str, tuple[int, ...]]]:
        """Token sequence used by the gate-edit-distance diversity component."""
        return [(g.name, g.qubits) for g in self.gates]
