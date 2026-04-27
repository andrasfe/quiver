"""Density-matrix simulator with depolarizing noise.

This is the simplest physically-meaningful noise model for benchmarking:
after every two-qubit gate (where most NISQ error budget actually goes),
each qubit involved is hit by a single-qubit depolarising channel with
strength p:

    N_p(ρ) = (1 − p) ρ + (p/3) (X ρ X + Y ρ Y + Z ρ Z)

The noisy expectation value <H>_noisy = Tr(H ρ) replaces the noiseless
<ψ|H|ψ>, and the parameter-shift rule for gradients still works
identically — at every shifted point we simulate the full noisy
density-matrix evolution.

The Hilbert-space dimension d = 2^n forces ρ to be d × d, so memory
scales as 4^n. Practical for n ≲ 8; we use this to benchmark n=4
portfolios which is plenty to see whether DS-VQE's diversity gives
implicit noise mitigation through the subspace step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quiver.backends.numpy_backend import (
    _FIXED_1Q,
    _FIXED_2Q,
    _apply_1q,
    _apply_2q,
    _gate_matrix,
)
from quiver.circuit import CircuitSpec, GateSpec


_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I = np.eye(2, dtype=complex)


def _lift_single_qubit(U_local: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Lift a 2×2 single-qubit operator to the full 2^n × 2^n Hilbert
    space. qubit-0-is-LSB convention (matches NumpyBackend)."""
    ops = [_I] * n
    ops[qubit] = U_local
    out = ops[n - 1]
    for k in range(n - 2, -1, -1):
        out = np.kron(out, ops[k])
    return out


def _apply_unitary_to_state(spec_or_gate, params: np.ndarray,
                             state: np.ndarray, n: int) -> np.ndarray:
    """Helper: evolve a state through a single gate."""
    g = spec_or_gate
    mat = _gate_matrix(g, params)
    if g.is_two_qubit:
        return _apply_2q(state, mat, g.qubits, n)
    return _apply_1q(state, mat, g.qubits[0], n)


@dataclass
class NoiseModel:
    """Per-2q-gate depolarising-noise model."""
    p_2q_per_qubit: float = 0.01    # depolarising strength applied to each
                                     # qubit involved in a 2q gate
    p_1q_per_qubit: float = 0.0     # optional 1q gate depolarisation


class NoisyDensityBackend:
    """Density-matrix simulator with the noise model above.

    Honours the same gate set as NumpyBackend (RX/RY/RZ/RXX/RYY/RZZ,
    H, X, Y, Z, S, T, CNOT, CZ, SWAP, iSWAP, sqrt-iSWAP). Returns
    noisy expectation values and the final density matrix; states
    are no longer pure.
    """

    def __init__(self, num_qubits: int, noise: NoiseModel | None = None):
        self.num_qubits = num_qubits
        self.dim = 2**num_qubits
        self.noise = noise or NoiseModel()
        # Cache the lifted 1q Pauli operators per qubit.
        self._X_q = [_lift_single_qubit(_X, q, num_qubits) for q in range(num_qubits)]
        self._Y_q = [_lift_single_qubit(_Y, q, num_qubits) for q in range(num_qubits)]
        self._Z_q = [_lift_single_qubit(_Z, q, num_qubits) for q in range(num_qubits)]

    def _build_full_unitary(self, gate: GateSpec, params: np.ndarray) -> np.ndarray:
        """Construct the 2^n × 2^n unitary for a gate by applying it to
        each computational-basis state. O(2^n) state-vector ops; fine
        for n ≤ 8."""
        U_full = np.zeros((self.dim, self.dim), dtype=complex)
        for c in range(self.dim):
            e = np.zeros(self.dim, dtype=complex)
            e[c] = 1.0
            U_full[:, c] = _apply_unitary_to_state(gate, params, e, self.num_qubits)
        return U_full

    def _apply_depolarising_single(self, rho: np.ndarray, qubit: int,
                                   p: float) -> np.ndarray:
        """ρ → (1-p)ρ + (p/3)(X_q ρ X_q + Y_q ρ Y_q + Z_q ρ Z_q)."""
        X = self._X_q[qubit]
        Y = self._Y_q[qubit]
        Z = self._Z_q[qubit]
        return (1.0 - p) * rho + (p / 3.0) * (X @ rho @ X + Y @ rho @ Y + Z @ rho @ Z)

    def density_matrix(self, spec: CircuitSpec, params: np.ndarray) -> np.ndarray:
        """Evolve from |0...0><0...0| through the (noisy) circuit."""
        if spec.num_qubits != self.num_qubits:
            raise ValueError("circuit width does not match backend width")
        rho = np.zeros((self.dim, self.dim), dtype=complex)
        rho[0, 0] = 1.0
        for gate in spec.gates:
            U = self._build_full_unitary(gate, params)
            rho = U @ rho @ U.conj().T
            # Noise after each gate: depolarise every qubit it touched.
            if gate.is_two_qubit and self.noise.p_2q_per_qubit > 0.0:
                for q in gate.qubits:
                    rho = self._apply_depolarising_single(
                        rho, q, self.noise.p_2q_per_qubit
                    )
            elif (not gate.is_two_qubit) and self.noise.p_1q_per_qubit > 0.0:
                rho = self._apply_depolarising_single(
                    rho, gate.qubits[0], self.noise.p_1q_per_qubit
                )
        return rho

    def expectation(self, H: np.ndarray, spec: CircuitSpec,
                    params: np.ndarray) -> float:
        """Tr(H ρ) — the noisy expectation value."""
        rho = self.density_matrix(spec, params)
        return float(np.real(np.trace(H @ rho)))

    # Match the Backend protocol so the rest of Quiver can use this too.
    def statevector(self, spec: CircuitSpec, params: np.ndarray) -> np.ndarray:
        """Return the dominant eigenvector of the (mixed) density matrix
        — the most-probable pure state. Useful as a "what's the circuit
        actually trying to prepare" signal even under noise."""
        rho = self.density_matrix(spec, params)
        eigvals, eigvecs = np.linalg.eigh(rho)
        # Largest eigenvalue first
        return np.asarray(eigvecs[:, -1])
