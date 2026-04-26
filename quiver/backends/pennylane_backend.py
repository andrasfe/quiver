"""PennyLane backend (optional dependency)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quiver.circuit import CircuitSpec


@dataclass
class PennyLaneBackend:
    num_qubits: int
    device_name: str = "default.qubit"

    def __post_init__(self) -> None:
        try:
            import pennylane as qml  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PennyLaneBackend requires the 'pennylane' extra: pip install quiver[pennylane]"
            ) from exc
        import pennylane as qml

        self._qml = qml
        self._device = qml.device(self.device_name, wires=self.num_qubits)

    def statevector(self, spec: CircuitSpec, params: np.ndarray) -> np.ndarray:
        qml = self._qml

        @qml.qnode(self._device, interface="numpy")
        def circuit(p):
            for gate in spec.gates:
                self._apply(gate, p)
            return qml.state()

        return np.asarray(circuit(params))

    def _apply(self, gate, params):
        qml = self._qml
        n = gate.name
        qs = list(gate.qubits)
        theta = params[gate.param_idx] if gate.is_parametric else None
        if n == "h":
            qml.Hadamard(wires=qs[0])
        elif n == "x":
            qml.PauliX(wires=qs[0])
        elif n == "y":
            qml.PauliY(wires=qs[0])
        elif n == "z":
            qml.PauliZ(wires=qs[0])
        elif n == "s":
            qml.S(wires=qs[0])
        elif n == "t":
            qml.T(wires=qs[0])
        elif n == "rx":
            qml.RX(theta, wires=qs[0])
        elif n == "ry":
            qml.RY(theta, wires=qs[0])
        elif n == "rz":
            qml.RZ(theta, wires=qs[0])
        elif n in ("cnot", "cx"):
            qml.CNOT(wires=qs)
        elif n == "cz":
            qml.CZ(wires=qs)
        elif n == "swap":
            qml.SWAP(wires=qs)
        elif n == "rzz":
            qml.IsingZZ(theta, wires=qs)
        elif n == "rxx":
            qml.IsingXX(theta, wires=qs)
        elif n == "ryy":
            qml.IsingYY(theta, wires=qs)
        else:
            raise ValueError(f"unknown gate '{n}'")
