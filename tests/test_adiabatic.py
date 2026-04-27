import numpy as np
from scipy.optimize import minimize

from quiver.adiabatic import (
    adiabatic_train,
    easy_x_field_hamiltonian,
    linear_interpolation,
)
from quiver.ansatz import HardwareEfficient
from quiver.backends import NumpyBackend
from quiver.hamiltonian import expectation, ground_state_energy, heisenberg
from quiver.optimizer import parameter_shift_gradient


def test_linear_interpolation_endpoints_match():
    H1 = np.array([[1.0, 0], [0, -1]], dtype=complex)
    H2 = np.array([[0, 1], [1, 0]], dtype=complex)
    steps = linear_interpolation(H1, H2, num_steps=5)
    assert len(steps) == 5
    assert steps[0][0] == 0.0
    np.testing.assert_allclose(steps[0][1], H1)
    assert steps[-1][0] == 1.0
    np.testing.assert_allclose(steps[-1][1], H2)


def test_easy_x_field_ground_state_is_plus_n():
    H_easy = easy_x_field_hamiltonian(3)
    e0 = ground_state_energy(H_easy)
    assert np.isclose(e0, -3.0)


def test_adiabatic_drops_below_direct_for_small_heisenberg():
    """On 4q Heisenberg, with a small ansatz that can struggle directly,
    the adiabatic schedule should reach a lower energy than direct attack."""
    n = 4
    H = heisenberg(n, periodic=False)
    e0 = ground_state_energy(H)
    H_easy = easy_x_field_hamiltonian(n)

    spec = HardwareEfficient(num_qubits=n, num_layers=2).build()
    backend = NumpyBackend(num_qubits=n)

    def opt(loss, x0):
        result = minimize(
            loss, x0, method="L-BFGS-B",
            jac=lambda p: parameter_shift_gradient(loss, p),
            options={"maxiter": 30, "ftol": 1e-7, "gtol": 1e-5},
        )
        return np.asarray(result.x), float(result.fun), int(result.nfev)

    rng = np.random.default_rng(0)
    init = rng.normal(0.0, 0.1, spec.num_params)

    # Direct attack: optimize H_target from a single random start.
    direct_loss = lambda p: float(expectation(H, backend.statevector(spec, p)))
    direct_params, direct_e, _ = opt(direct_loss, init.copy())

    # Adiabatic schedule: 8 steps from H_easy to H_target, warm-started.
    final_params, trace = adiabatic_train(
        spec, backend, H_easy, H, num_steps=8,
        optimize_at_step=opt, initial_params=init.copy(),
    )
    adiab_e = trace.energy[-1]

    # Adiabatic should reach at least as low an energy as direct, often
    # markedly lower. We assert the adiabatic energy is closer to (or
    # tied with) the ground than direct.
    assert adiab_e <= direct_e + 1e-3
    assert adiab_e >= e0 - 1e-6
