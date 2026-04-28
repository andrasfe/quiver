"""Qiskit backend (optional dependency).

Also exposes `to_qiskit_circuit(spec, params)` as a module-level helper
so consumers (notebooks, hardware-submission scripts) can build a
`QuantumCircuit` from a Quiver `CircuitSpec` without instantiating the
full backend or running a simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quiver.circuit import CircuitSpec


def to_qiskit_circuit(spec: CircuitSpec, params: np.ndarray):
    """Translate a CircuitSpec + bound parameters into a qiskit
    QuantumCircuit. Supports every gate Quiver's NumpyBackend supports
    except `iswap` and `sqrt_iswap`, which are inserted as their unitary
    matrices (the qiskit transpiler will decompose them onto the target
    backend's native gate set).
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise ImportError(
            "to_qiskit_circuit requires qiskit: pip install qiskit"
        ) from exc

    qc = QuantumCircuit(spec.num_qubits)
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
        elif n == "sdg":
            qc.sdg(qs[0])
        elif n == "tdg":
            qc.tdg(qs[0])
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
        elif n == "cy":
            qc.cy(qs[0], qs[1])
        elif n == "swap":
            qc.swap(qs[0], qs[1])
        elif n == "rzz":
            qc.rzz(theta, qs[0], qs[1])
        elif n == "rxx":
            qc.rxx(theta, qs[0], qs[1])
        elif n == "ryy":
            qc.ryy(theta, qs[0], qs[1])
        elif n in ("iswap", "sqrt_iswap"):
            from quiver.backends.numpy_backend import _iswap, _sqrt_iswap
            from qiskit.quantum_info import Operator
            mat = _iswap() if n == "iswap" else _sqrt_iswap()
            qc.unitary(Operator(mat), qs, label=n)
        else:
            raise ValueError(f"unknown gate '{n}'")
    return qc


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
        from qiskit.quantum_info import Statevector
        qc = to_qiskit_circuit(spec, params)
        sv = Statevector.from_instruction(qc)
        return np.asarray(sv.data)
