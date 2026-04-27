import numpy as np

from quiver.hamiltonian import ground_state_energy, heisenberg
from quiver.subspace import subspace_diagonalize


def test_subspace_with_single_state_returns_individual_energy():
    H = heisenberg(2)
    state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    result = subspace_diagonalize([state], H)
    expected = float(np.real(np.vdot(state, H @ state)))
    np.testing.assert_allclose(result.ground_energy, expected, atol=1e-10)


def test_subspace_with_two_orthogonal_states_returns_min_eigenvalue_of_2x2():
    """If the two states are exactly the lowest-energy eigenvectors of H,
    subspace EVP should recover the ground energy."""
    H = heisenberg(2)
    eigvals, eigvecs = np.linalg.eigh(H)
    e0, e1 = eigvals[0], eigvals[1]
    psi0, psi1 = eigvecs[:, 0], eigvecs[:, 1]
    result = subspace_diagonalize([psi0, psi1], H)
    np.testing.assert_allclose(result.ground_energy, e0, atol=1e-10)


def test_subspace_lower_than_individual_when_states_partially_orthogonal():
    """The whole point: the subspace ground energy is <= every individual
    energy, with strict inequality when the basis spans the ground state
    better than any single member."""
    H = heisenberg(4)
    eigvals, eigvecs = np.linalg.eigh(H)
    e0 = float(eigvals[0])
    # Find an eigenvector strictly above the ground (skipping degeneracy).
    higher_idx = next(i for i, e in enumerate(eigvals) if e > e0 + 1e-8)

    # Two states, each a non-trivial mix of ground and a strictly-higher
    # eigenstate. Individual energies are (above-ground) but their span
    # contains the ground exactly.
    psi_a = 0.6 * eigvecs[:, 0] + 0.8 * eigvecs[:, higher_idx]
    psi_b = 0.8 * eigvecs[:, 0] - 0.6 * eigvecs[:, higher_idx]

    result = subspace_diagonalize([psi_a, psi_b], H)
    individual_min = min(result.individual_energies)
    assert result.ground_energy < individual_min - 1e-6
    np.testing.assert_allclose(result.ground_energy, e0, atol=1e-9)


def test_subspace_handles_near_duplicates():
    """Two near-identical states should not crash the solver; the
    overlap-threshold step drops the redundant direction."""
    H = heisenberg(2)
    state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    nearly_same = state + 1e-10 * np.array([0, 1, 0, 0], dtype=complex)
    nearly_same /= np.linalg.norm(nearly_same)
    result = subspace_diagonalize([state, nearly_same], H)
    assert result.rank == 1
    expected = float(np.real(np.vdot(state, H @ state)))
    np.testing.assert_allclose(result.ground_energy, expected, atol=1e-8)
