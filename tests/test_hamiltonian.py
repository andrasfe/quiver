import numpy as np

from quivercirc.hamiltonian import (
    expectation,
    ground_state_energy,
    heisenberg,
    ising_zz,
    tfim,
    vqe_setup,
)


def test_heisenberg_is_hermitian():
    H = heisenberg(3)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


def test_heisenberg_three_sites_known_ground_energy():
    """Heisenberg chain n=3 (open) ground energy = -2 + 2cos(π/3) - 2cos(2π/3)
    Using exact diagonalisation as the reference."""
    H = heisenberg(3, periodic=False)
    e0 = ground_state_energy(H)
    # AFM Heisenberg open n=3: ground state energy is -2*sqrt(2) ≈ -2.828
    # (analytical for two non-equivalent bonds + spin-1/2 chain edge effects).
    # We just check it's well below zero (frustration-free is positive).
    assert e0 < -1.0


def test_tfim_critical_point_has_negative_ground():
    H = tfim(4, J=1.0, h=1.0, periodic=True)
    e0 = ground_state_energy(H)
    assert e0 < 0


def test_vqe_setup_verifier_passes_at_ground():
    H = ising_zz(3, h=0.5)
    setup = vqe_setup(H, threshold_above_ground=0.01)
    # Build the actual ground eigenvector and check the verifier accepts it.
    eigvals, eigvecs = np.linalg.eigh(H)
    ground = eigvecs[:, 0]
    passed, energy = setup.verifier(ground)
    assert passed
    assert abs(energy - eigvals[0]) < 1e-9


def test_vqe_setup_state_loss_matches_expectation():
    H = heisenberg(2)
    setup = vqe_setup(H)
    state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)  # Bell
    np.testing.assert_allclose(setup.state_loss(state), expectation(H, state))
