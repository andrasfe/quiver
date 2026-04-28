"""Ansatz protocol.

An ansatz is a function from layer count to a CircuitSpec. The `family`
attribute tags ansätze that share a structural family so the registry can
record which family produced each accepted solution.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from quivercirc.circuit import CircuitSpec


@runtime_checkable
class Ansatz(Protocol):
    family: str
    num_qubits: int
    num_params: int

    def build(self) -> CircuitSpec: ...
