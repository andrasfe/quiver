import numpy as np

from quivercirc.adaptive import AdaptiveGrowth, _candidate_pool
from quivercirc.backends import NumpyBackend
from quivercirc.circuit import CircuitSpec
from quivercirc.verification import fidelity_objective


def test_candidate_pool_covers_full_gate_set():
    pool = _candidate_pool(num_qubits=3)
    names = {n for n, _ in pool}
    # parametric and fixed, single and two-qubit
    assert {"rx", "ry", "rz"}.issubset(names)
    assert {"cnot", "cz", "iswap", "rzz"}.issubset(names)


def test_adaptive_growth_finds_bell_state():
    """ADAPT must discover *some* circuit that produces a Bell state."""
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    backend = NumpyBackend(num_qubits=2)
    loss_fn = fidelity_objective(bell)

    def make_objective(spec: CircuitSpec):
        def obj(params):
            return loss_fn(backend.statevector(spec, params))
        return obj

    grower = AdaptiveGrowth(
        num_qubits=2, max_gates=20, candidates_per_step=12,
        inner_max_iter=40, plateau_patience=4, epsilon_random=0.1,
        target_loss=1e-3,
    )
    rng = np.random.default_rng(0)
    spec, params = grower.grow(make_objective, rng)
    state = backend.statevector(spec, params)
    fidelity = abs(np.vdot(bell, state)) ** 2
    assert fidelity > 0.95
    assert spec.gate_count > 0


def test_adaptive_growth_produces_distinct_circuits_across_seeds():
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    backend = NumpyBackend(num_qubits=2)
    loss_fn = fidelity_objective(bell)

    def make_objective(spec):
        def obj(params):
            return loss_fn(backend.statevector(spec, params))
        return obj

    grower = AdaptiveGrowth(
        num_qubits=2, max_gates=15, candidates_per_step=10,
        inner_max_iter=30, plateau_patience=3, epsilon_random=0.3,
        target_loss=1e-3,
    )
    spec_a, _ = grower.grow(make_objective, np.random.default_rng(0))
    spec_b, _ = grower.grow(make_objective, np.random.default_rng(7))
    # Different seeds should produce different gate sequences most of the time.
    assert spec_a.signature() != spec_b.signature() or spec_a.gate_count != spec_b.gate_count
