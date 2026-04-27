import numpy as np

from quiver.ansatz import HardwareEfficient
from quiver.backends import NumpyBackend
from quiver.circuit import CircuitSpec, GateSpec
from quiver.hamiltonian import expectation, heisenberg
from quiver.noise import NoisyDensityBackend, NoiseModel


def test_zero_noise_density_matrix_matches_pure_state():
    """With p=0 the density matrix should equal |ψ⟩⟨ψ| for the same
    circuit on a pure-state backend."""
    spec = HardwareEfficient(num_qubits=3, num_layers=2).build()
    rng = np.random.default_rng(0)
    params = rng.uniform(-np.pi, np.pi, spec.num_params)

    pure_backend = NumpyBackend(num_qubits=3)
    psi = pure_backend.statevector(spec, params)
    rho_pure = np.outer(psi, psi.conj())

    noisy = NoisyDensityBackend(num_qubits=3, noise=NoiseModel(p_2q_per_qubit=0.0))
    rho_noisy = noisy.density_matrix(spec, params)

    np.testing.assert_allclose(rho_noisy, rho_pure, atol=1e-9)


def test_zero_noise_expectation_matches_pure_state():
    H = heisenberg(3)
    spec = HardwareEfficient(num_qubits=3, num_layers=2).build()
    rng = np.random.default_rng(1)
    params = rng.uniform(-np.pi, np.pi, spec.num_params)

    pure_backend = NumpyBackend(num_qubits=3)
    pure_E = expectation(H, pure_backend.statevector(spec, params))

    noisy = NoisyDensityBackend(num_qubits=3, noise=NoiseModel(p_2q_per_qubit=0.0))
    noisy_E = noisy.expectation(H, spec, params)

    np.testing.assert_allclose(noisy_E, pure_E, atol=1e-9)


def test_strong_noise_drives_density_toward_maximally_mixed():
    """With heavy depolarising noise the density matrix should approach
    I/d (maximally mixed). Test on a deep 2q-heavy circuit."""
    spec = CircuitSpec(num_qubits=2)
    for _ in range(20):
        spec.add(GateSpec("cnot", (0, 1)))
    spec.num_params = 0
    backend = NoisyDensityBackend(
        num_qubits=2, noise=NoiseModel(p_2q_per_qubit=0.5)
    )
    rho = backend.density_matrix(spec, np.zeros(0))
    expected = np.eye(4, dtype=complex) / 4.0
    # Off-diagonal should decay; diagonals should be ~uniform.
    np.testing.assert_allclose(np.diag(rho), [0.25] * 4, atol=0.05)


def test_noisy_expectation_is_biased_toward_zero_for_traceless_H():
    """For a traceless Hamiltonian, the maximally mixed state has
    expectation = 0. So heavy noise drives <H> toward 0."""
    H = heisenberg(2)
    H_traceless = H - (np.trace(H) / 4) * np.eye(4)

    spec = HardwareEfficient(num_qubits=2, num_layers=2).build()
    rng = np.random.default_rng(0)
    params = rng.uniform(-np.pi, np.pi, spec.num_params)

    quiet = NoisyDensityBackend(num_qubits=2, noise=NoiseModel(p_2q_per_qubit=0.0))
    loud = NoisyDensityBackend(num_qubits=2, noise=NoiseModel(p_2q_per_qubit=0.4))

    quiet_E = quiet.expectation(H_traceless, spec, params)
    loud_E = loud.expectation(H_traceless, spec, params)

    # Loud noise drags |E| toward zero
    assert abs(loud_E) < abs(quiet_E) + 1e-9
