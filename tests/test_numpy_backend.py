import numpy as np
import pytest

from quiver.ansatz import HardwareEfficient
from quiver.backends import NumpyBackend
from quiver.circuit import CircuitSpec, GateSpec


def test_hadamard_creates_uniform_superposition():
    backend = NumpyBackend(num_qubits=2)
    spec = CircuitSpec(num_qubits=2)
    spec.add(GateSpec("h", (0,)))
    spec.add(GateSpec("h", (1,)))
    state = backend.statevector(spec, np.array([]))
    expected = np.full(4, 0.5, dtype=complex)
    np.testing.assert_allclose(state, expected, atol=1e-10)


def test_bell_state_via_h_cnot():
    backend = NumpyBackend(num_qubits=2)
    spec = CircuitSpec(num_qubits=2)
    spec.add(GateSpec("h", (0,)))
    spec.add(GateSpec("cnot", (0, 1)))
    state = backend.statevector(spec, np.array([]))
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    np.testing.assert_allclose(state, expected, atol=1e-10)


def test_rx_pi_flips_zero_to_minus_i_one():
    backend = NumpyBackend(num_qubits=1)
    spec = CircuitSpec(num_qubits=1)
    spec.add(GateSpec("rx", (0,), 0))
    spec.num_params = 1
    state = backend.statevector(spec, np.array([np.pi]))
    np.testing.assert_allclose(state, np.array([0, -1j], dtype=complex), atol=1e-10)


def test_hardware_efficient_runs_at_random_params():
    ansatz = HardwareEfficient(num_qubits=3, num_layers=2)
    spec = ansatz.build()
    backend = NumpyBackend(num_qubits=3)
    rng = np.random.default_rng(0)
    state = backend.statevector(spec, rng.uniform(-np.pi, np.pi, spec.num_params))
    np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-10)


def test_width_mismatch_raises():
    backend = NumpyBackend(num_qubits=2)
    spec = CircuitSpec(num_qubits=3)
    with pytest.raises(ValueError):
        backend.statevector(spec, np.array([]))
