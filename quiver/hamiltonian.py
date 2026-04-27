"""Hamiltonian / VQE-style targets.

Helpers for building common spin-Hamiltonian matrices and the
state-based loss + verifier pair Quiver needs to optimize them. The
matrices follow Quiver's qubit-0-is-LSB convention (so a state index
i has bit q of i giving qubit q's value), which means tensor products
must be built kron(q_{n-1}, kron(q_{n-2}, ..., kron(q_1, q_0))).

Common Hamiltonians:

  heisenberg(n, periodic=False)
      H = sum_<i,j in E> (X_i X_j + Y_i Y_j + Z_i Z_j)
      where E is the chain {(0,1),(1,2),...,(n-2,n-1)} and additionally
      (n-1, 0) when periodic. Antiferromagnetic ground state.

  tfim(n, J=1.0, h=1.0, periodic=False)
      Transverse-field Ising:
      H = -J sum_<i,j in E> Z_i Z_j  -  h sum_i X_i

  ising_zz(n, h=0.0, periodic=False)
      Pure ZZ-only Ising with optional uniform field.

For each builder we also expose a `vqe_setup` factory that returns the
(target, verifier, state_loss) triple Quiver expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# Pauli matrices.
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron_chain(ops_per_qubit: list[np.ndarray]) -> np.ndarray:
    """ops_per_qubit[q] is the single-qubit operator on qubit q. Returns
    the full 2^n × 2^n operator with qubit 0 = LSB."""
    n = len(ops_per_qubit)
    M = ops_per_qubit[n - 1]
    for q in range(n - 2, -1, -1):
        M = np.kron(M, ops_per_qubit[q])
    return M


def _two_site(n: int, op: np.ndarray, i: int, j: int) -> np.ndarray:
    ops = [_I] * n
    ops[i] = op
    ops[j] = op
    return _kron_chain(ops)


def _single_site(n: int, op: np.ndarray, i: int) -> np.ndarray:
    ops = [_I] * n
    ops[i] = op
    return _kron_chain(ops)


def heisenberg(n: int, periodic: bool = False) -> np.ndarray:
    """Antiferromagnetic Heisenberg chain (J=1)."""
    edges = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:
        edges.append((n - 1, 0))
    H = np.zeros((2**n, 2**n), dtype=complex)
    for i, j in edges:
        H += _two_site(n, _X, i, j)
        H += _two_site(n, _Y, i, j)
        H += _two_site(n, _Z, i, j)
    return H


def tfim(n: int, J: float = 1.0, h: float = 1.0, periodic: bool = False) -> np.ndarray:
    edges = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:
        edges.append((n - 1, 0))
    H = np.zeros((2**n, 2**n), dtype=complex)
    for i, j in edges:
        H -= J * _two_site(n, _Z, i, j)
    for i in range(n):
        H -= h * _single_site(n, _X, i)
    return H


def ising_zz(n: int, h: float = 0.0, periodic: bool = False) -> np.ndarray:
    edges = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:
        edges.append((n - 1, 0))
    H = np.zeros((2**n, 2**n), dtype=complex)
    for i, j in edges:
        H += _two_site(n, _Z, i, j)
    if h != 0.0:
        for i in range(n):
            H += h * _single_site(n, _Z, i)
    return H


def ground_state_energy(H: np.ndarray) -> float:
    """Smallest eigenvalue of H, computed by exact diagonalisation. Use only
    for benchmarking — it is O(8^n) memory and time."""
    eigs = np.linalg.eigvalsh(H)
    return float(eigs[0])


def expectation(H: np.ndarray, state: np.ndarray) -> float:
    """Real part of <state|H|state>. Hermitian H makes this exactly real
    up to numerical noise."""
    return float(np.real(np.vdot(state, H @ state)))


@dataclass
class VQESetup:
    H: np.ndarray
    ground_energy: float
    threshold_above_ground: float
    state_loss: Callable[[np.ndarray], float]
    verifier: Callable[[np.ndarray], "tuple[bool, float]"]


def vqe_setup(H: np.ndarray, threshold_above_ground: float = 0.5) -> VQESetup:
    """Build the (state_loss, verifier) pair for a given Hamiltonian.

    The verifier passes when <state|H|state> is within
    `threshold_above_ground` of the exact ground-state energy. The
    state_loss is just the expectation value, so the optimizer drives
    it down toward the ground.
    """
    e0 = ground_state_energy(H)
    accept_below = e0 + threshold_above_ground

    def state_loss(state: np.ndarray) -> float:
        return expectation(H, state)

    def verifier(state: np.ndarray) -> tuple[bool, float]:
        e = expectation(H, state)
        return e <= accept_below, e

    return VQESetup(
        H=H,
        ground_energy=e0,
        threshold_above_ground=threshold_above_ground,
        state_loss=state_loss,
        verifier=verifier,
    )
