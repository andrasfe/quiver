"""Reference statevector simulator implemented in pure numpy.

Used by Quiver as the default backend so the package is testable without
heavy quantum dependencies. Sized for tutorial / small-circuit use
(num_qubits up to ~12); above that, prefer the PennyLane or Qiskit backends.

Convention: amplitudes are stored as a length-2**N complex array where the
binary representation of the index gives the qubit values, with qubit 0 as
the least-significant bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quiver.circuit import CircuitSpec, GateSpec


_SQRT2 = 1.0 / np.sqrt(2.0)


def _h() -> np.ndarray:
    return np.array([[_SQRT2, _SQRT2], [_SQRT2, -_SQRT2]], dtype=complex)


def _x() -> np.ndarray:
    return np.array([[0, 1], [1, 0]], dtype=complex)


def _y() -> np.ndarray:
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def _z() -> np.ndarray:
    return np.array([[1, 0], [0, -1]], dtype=complex)


def _s() -> np.ndarray:
    return np.array([[1, 0], [0, 1j]], dtype=complex)


def _t() -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def _rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


_FIXED_1Q = {
    "h": _h(),
    "x": _x(),
    "y": _y(),
    "z": _z(),
    "s": _s(),
    "t": _t(),
    "sdg": _s().conj().T,
    "tdg": _t().conj().T,
}


def _apply_1q(state: np.ndarray, mat: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
    state = state.reshape([2] * num_qubits)
    # qubit 0 is least significant -> last axis
    axis = num_qubits - 1 - qubit
    state = np.tensordot(mat, state, axes=([1], [axis]))
    # tensordot puts the new axis at position 0; move it back
    state = np.moveaxis(state, 0, axis)
    return state.reshape(2**num_qubits)


def _apply_2q(state: np.ndarray, mat: np.ndarray, qubits: tuple[int, int], num_qubits: int) -> np.ndarray:
    q1, q2 = qubits
    state = state.reshape([2] * num_qubits)
    a1 = num_qubits - 1 - q1
    a2 = num_qubits - 1 - q2
    mat4 = mat.reshape(2, 2, 2, 2)  # [out_q1, out_q2, in_q1, in_q2]
    state = np.tensordot(mat4, state, axes=([2, 3], [a1, a2]))
    # output axes 0,1 correspond to q1, q2 respectively; move them to a1, a2.
    # After the contraction the remaining axes have shifted; move carefully.
    state = np.moveaxis(state, [0, 1], [a1, a2])
    return state.reshape(2**num_qubits)


def _cnot() -> np.ndarray:
    m = np.eye(4, dtype=complex)
    m[2:, 2:] = [[0, 1], [1, 0]]
    return m


def _cz() -> np.ndarray:
    m = np.eye(4, dtype=complex)
    m[3, 3] = -1
    return m


def _swap() -> np.ndarray:
    m = np.zeros((4, 4), dtype=complex)
    m[0, 0] = m[3, 3] = 1
    m[1, 2] = m[2, 1] = 1
    return m


def _rzz(theta: float) -> np.ndarray:
    # exp(-i theta/2 Z⊗Z): diag(e^-i, e^+i, e^+i, e^-i) * theta/2
    d = np.array([
        np.exp(-1j * theta / 2),
        np.exp(1j * theta / 2),
        np.exp(1j * theta / 2),
        np.exp(-1j * theta / 2),
    ], dtype=complex)
    return np.diag(d)


def _rxx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [[c, 0, 0, -1j * s], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [-1j * s, 0, 0, c]],
        dtype=complex,
    )


def _ryy(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [[c, 0, 0, 1j * s], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [1j * s, 0, 0, c]],
        dtype=complex,
    )


def _iswap() -> np.ndarray:
    m = np.eye(4, dtype=complex)
    m[1, 1] = m[2, 2] = 0.0
    m[1, 2] = m[2, 1] = 1j
    return m


def _sqrt_iswap() -> np.ndarray:
    m = np.eye(4, dtype=complex)
    inv2 = 1.0 / np.sqrt(2.0)
    m[1, 1] = m[2, 2] = inv2
    m[1, 2] = m[2, 1] = 1j * inv2
    return m


_FIXED_2Q = {
    "cnot": _cnot(),
    "cx": _cnot(),
    "cz": _cz(),
    "swap": _swap(),
    "iswap": _iswap(),
    "sqrt_iswap": _sqrt_iswap(),
}


def _gate_matrix(gate: GateSpec, params: np.ndarray) -> np.ndarray:
    name = gate.name
    if name in _FIXED_1Q:
        return _FIXED_1Q[name]
    if name in _FIXED_2Q:
        return _FIXED_2Q[name]
    theta = float(params[gate.param_idx]) if gate.is_parametric else 0.0
    if name == "rx":
        return _rx(theta)
    if name == "ry":
        return _ry(theta)
    if name == "rz":
        return _rz(theta)
    if name == "rzz":
        return _rzz(theta)
    if name == "rxx":
        return _rxx(theta)
    if name == "ryy":
        return _ryy(theta)
    raise ValueError(f"unknown gate '{name}'")


@dataclass
class NumpyBackend:
    num_qubits: int

    def statevector(self, spec: CircuitSpec, params: np.ndarray) -> np.ndarray:
        if spec.num_qubits != self.num_qubits:
            raise ValueError("circuit width does not match backend width")
        state = np.zeros(2**self.num_qubits, dtype=complex)
        state[0] = 1.0
        for gate in spec.gates:
            mat = _gate_matrix(gate, params)
            if gate.is_two_qubit:
                state = _apply_2q(state, mat, gate.qubits, self.num_qubits)
            else:
                state = _apply_1q(state, mat, gate.qubits[0], self.num_qubits)
        return state
