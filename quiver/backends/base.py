"""Backend protocol: simulate a CircuitSpec at given parameters and return
the final statevector."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from quiver.circuit import CircuitSpec


@runtime_checkable
class Backend(Protocol):
    num_qubits: int

    def statevector(self, spec: CircuitSpec, params: np.ndarray) -> np.ndarray: ...
