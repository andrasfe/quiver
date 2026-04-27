"""Diversity-driven subspace VQE (DS-VQE).

Given a registry of K verified diverse circuits with state vectors
|ψ_k⟩, k = 1..K, we solve the generalized eigenvalue problem in the
non-orthogonal basis they span:

    M c = E S c     where    M_ij = ⟨ψ_i|H|ψ_j⟩,    S_ij = ⟨ψ_i|ψ_j⟩

The smallest E is the lowest energy reachable by any linear combination
∑ c_k |ψ_k⟩ of the registry states. It is *guaranteed* (Ritz variational
principle) to be ≤ the minimum of the individual ⟨ψ_k|H|ψ_k⟩ values —
strictly less when the registry is structurally diverse and the ⟨ψ_i|ψ_j⟩
overlaps are not 1.

This is a one-shot post-processing step: compute K^2 expectation values
plus one K-dimensional generalized eigenvalue solve. Cheap relative to
the K VQE runs that produced the registry. The novelty here is that the
basis is diversity-optimised by construction — the registry's structural
diversity translates into partially-orthogonal error directions in
state space, which is exactly the regime where subspace expansion
helps most.

Subspace expansion / non-orthogonal VQE is known in chemistry; the
contribution is *what basis to expand in*. We use the registry as
the basis and report the resulting energy improvement over the
single-circuit baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass
class SubspaceResult:
    energies: np.ndarray            # full sorted spectrum in the subspace
    ground_energy: float
    ground_coeffs: np.ndarray        # complex coefficients c_k
    ground_state: np.ndarray          # the combined state vector ∑ c_k |ψ_k⟩
    individual_energies: np.ndarray  # ⟨ψ_k|H|ψ_k⟩ for each registry entry
    overlap_matrix: np.ndarray
    hamiltonian_matrix: np.ndarray
    rank: int                         # effective rank of S (linearly-indep states)


def subspace_diagonalize(
    states: list[np.ndarray],
    H: np.ndarray,
    overlap_threshold: float = 1e-9,
) -> SubspaceResult:
    """Solve the generalized eigenvalue problem in the non-orthogonal basis.

    states: list of normalised statevectors |ψ_k⟩ (length 2^n).
    H: 2^n × 2^n Hermitian Hamiltonian matrix.
    overlap_threshold: drop near-linearly-dependent components by
        truncating the overlap matrix's small eigenvalues. Stabilises
        the solve when registry entries are nearly identical.
    """
    K = len(states)
    if K == 0:
        raise ValueError("need at least one state")

    state_matrix = np.column_stack([s / (np.linalg.norm(s) + 1e-15) for s in states])
    # M = ψ† H ψ, S = ψ† ψ
    HM = H @ state_matrix
    M = state_matrix.conj().T @ HM
    S = state_matrix.conj().T @ state_matrix
    # Make Hermitian to numerical precision
    M = 0.5 * (M + M.conj().T)
    S = 0.5 * (S + S.conj().T)

    individual_energies = np.array([float(np.real(M[i, i])) for i in range(K)])

    # Stabilise: diagonalise S, drop tiny eigenvalues, project into the
    # well-conditioned subspace, solve the resulting *standard* EVP.
    s_eigvals, s_eigvecs = scipy.linalg.eigh(S)
    keep = s_eigvals > overlap_threshold * s_eigvals.max()
    rank = int(keep.sum())
    if rank == 0:
        raise RuntimeError("overlap matrix is degenerate — registry is rank-zero")
    Q = s_eigvecs[:, keep] / np.sqrt(s_eigvals[keep])
    # M' = Q† M Q is the equivalent standard-EVP matrix
    M_prime = Q.conj().T @ M @ Q
    M_prime = 0.5 * (M_prime + M_prime.conj().T)
    eigvals, eigvecs = scipy.linalg.eigh(M_prime)

    # Map ground eigenvector back to the original basis.
    ground_coeffs_orig = Q @ eigvecs[:, 0]
    ground_energy = float(np.real(eigvals[0]))

    # Build the actual quantum state |Φ⟩ = ∑_k c_k |ψ_k⟩, then renormalise.
    ground_state = state_matrix @ ground_coeffs_orig
    ground_state = ground_state / (np.linalg.norm(ground_state) + 1e-15)

    return SubspaceResult(
        energies=np.real(eigvals),
        ground_energy=ground_energy,
        ground_coeffs=ground_coeffs_orig,
        ground_state=ground_state,
        individual_energies=individual_energies,
        overlap_matrix=S,
        hamiltonian_matrix=M,
        rank=rank,
    )
