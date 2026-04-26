"""Qiskit backend (optional dependency)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quiver.circuit import CircuitSpec


@dataclass
class QiskitBackend:
    num_qubits: int

    def __post_init__(self) -> None:
        try:
            from qiskit import QuantumCircuit  # noqa: F401
            from qiskit.quantum_info import Statevector  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "QiskitBackend requires the 'qiskit' extra: pip install quiver[qiskit]"
            ) from exc

    def statevector(self, spec: CircuitSpec, params: np.ndarray) -> np.ndarray:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        qc = QuantumCircuit(self.num_qubits)
        for g in spec.gates:
            theta = float(params[g.param_idx]) if g.is_parametric else None
            qs = list(g.qubits)
            n = g.name
            if n == "h":
                qc.h(qs[0])
            elif n == "x":
                qc.x(qs[0])
            elif n == "y":
                qc.y(qs[0])
            elif n == "z":
                qc.z(qs[0])
            elif n == "s":
                qc.s(qs[0])
            elif n == "t":
                qc.t(qs[0])
            elif n == "rx":
                qc.rx(theta, qs[0])
            elif n == "ry":
                qc.ry(theta, qs[0])
            elif n == "rz":
                qc.rz(theta, qs[0])
            elif n in ("cnot", "cx"):
                qc.cx(qs[0], qs[1])
            elif n == "cz":
                qc.cz(qs[0], qs[1])
            elif n == "swap":
                qc.swap(qs[0], qs[1])
            elif n == "rzz":
                qc.rzz(theta, qs[0], qs[1])
            elif n == "rxx":
                qc.rxx(theta, qs[0], qs[1])
            elif n == "ryy":
                qc.ryy(theta, qs[0], qs[1])
            else:
                raise ValueError(f"unknown gate '{n}'")
        sv = Statevector.from_instruction(qc)
        return np.asarray(sv.data)
