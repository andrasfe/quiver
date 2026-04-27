import numpy as np

from quiver.ansatz import HardwareEfficient
from quiver.backends import NumpyBackend
from quiver.config import OptimizerConfig
from quiver.optimizer import optimize, parameter_shift_gradient
from quiver.verification import fidelity_objective


def _bell_objective(spec, backend, target):
    loss = fidelity_objective(target)
    def obj(p):
        return float(loss(backend.statevector(spec, p)))
    return obj


def test_parameter_shift_matches_finite_diff_for_bell():
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    backend = NumpyBackend(num_qubits=2)
    spec = HardwareEfficient(num_qubits=2, num_layers=1).build()
    obj = _bell_objective(spec, backend, bell)

    rng = np.random.default_rng(0)
    params = rng.uniform(-np.pi, np.pi, spec.num_params)
    grad_shift = parameter_shift_gradient(obj, params)

    # Compare to centred finite difference (at small h).
    h = 1e-4
    grad_fd = np.zeros_like(grad_shift)
    for i in range(len(params)):
        e = np.zeros_like(params)
        e[i] = h
        grad_fd[i] = (obj(params + e) - obj(params - e)) / (2 * h)

    # Parameter shift is the EXACT gradient, finite diff is an approximation.
    # They should agree to at least 3-4 decimal places.
    np.testing.assert_allclose(grad_shift, grad_fd, atol=1e-3)


def test_lbfgs_solves_bell_state_quickly():
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    backend = NumpyBackend(num_qubits=2)
    spec = HardwareEfficient(num_qubits=2, num_layers=2).build()
    obj = _bell_objective(spec, backend, bell)

    cfg = OptimizerConfig(method="L-BFGS-B", basin_hops=2, max_iter=100,
                          tolerance=1e-8)
    rng = np.random.default_rng(1)
    result = optimize(obj, spec.num_params, cfg, rng)
    assert result.objective < 1e-3  # nearly converged to fidelity = 1
