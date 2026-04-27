import numpy as np

from quiver.ansatz import HardwareEfficient, QAOAInspired
from quiver.mutation import (
    Mutator,
    mutate_delete,
    mutate_insert,
    mutate_retarget,
    mutate_retype,
    mutate_swap,
)


def _spec_from_he():
    spec = HardwareEfficient(num_qubits=4, num_layers=2).build()
    return spec, np.zeros(spec.num_params)


def test_insert_increases_gate_count_and_returns_compatible_params():
    spec, params = _spec_from_he()
    rng = np.random.default_rng(0)
    new_spec, new_params = mutate_insert(spec, params, rng)
    assert new_spec.gate_count == spec.gate_count + 1
    assert len(new_params) == new_spec.num_params


def test_insert_parametric_appends_zero_param():
    spec, params = _spec_from_he()
    params = np.linspace(0.1, 0.9, spec.num_params)
    rng = np.random.default_rng(2)
    grew = False
    for _ in range(20):
        new_spec, new_params = mutate_insert(spec, params, rng)
        if new_spec.num_params == spec.num_params + 1:
            grew = True
            assert new_params[-1] == 0.0
            np.testing.assert_allclose(new_params[:-1], params)
            break
    assert grew


def test_delete_compacts_params_and_indices():
    spec, _ = _spec_from_he()
    parent_params = np.linspace(0.1, 0.9, spec.num_params)
    rng = np.random.default_rng(7)
    new_spec, new_params = mutate_delete(spec, parent_params, rng)
    assert len(new_params) == new_spec.num_params
    if new_spec.num_params < spec.num_params:
        for g in new_spec.gates:
            if g.is_parametric:
                assert 0 <= g.param_idx < new_spec.num_params


def test_swap_preserves_param_array():
    spec, _ = _spec_from_he()
    params = np.linspace(0.1, 0.9, spec.num_params)
    rng = np.random.default_rng(1)
    new_spec, new_params = mutate_swap(spec, params, rng)
    assert new_spec.gate_count == spec.gate_count
    np.testing.assert_allclose(new_params, params)


def test_retarget_preserves_qubit_distinctness_and_params():
    spec, _ = _spec_from_he()
    params = np.linspace(0.1, 0.9, spec.num_params)
    rng = np.random.default_rng(3)
    for _ in range(10):
        new_spec, new_params = mutate_retarget(spec, params, rng)
        for g in new_spec.gates:
            assert len(set(g.qubits)) == len(g.qubits)
        np.testing.assert_allclose(new_params, params)


def test_retype_preserves_arity_and_parametricity():
    spec, _ = _spec_from_he()
    params = np.linspace(0.1, 0.9, spec.num_params)
    rng = np.random.default_rng(4)
    for _ in range(20):
        new_spec, _ = mutate_retype(spec, params, rng)
        for orig, mut in zip(spec.gates, new_spec.gates):
            assert len(orig.qubits) == len(mut.qubits)
            assert orig.is_parametric == mut.is_parametric


def test_mutator_chain_threads_params():
    spec, _ = _spec_from_he()
    params = np.linspace(0.1, 0.9, spec.num_params)
    rng = np.random.default_rng(5)
    mutator = Mutator(chain_min=2, chain_max=4)
    new_spec, new_params = mutator.chain(spec, params, rng)
    assert len(new_params) == new_spec.num_params


def test_chain_output_remains_valid_specification():
    spec, _ = _spec_from_he()
    params = np.zeros(spec.num_params)
    rng = np.random.default_rng(11)
    mutator = Mutator(chain_min=3, chain_max=6)
    new_spec, new_params = mutator.chain(spec, params, rng)
    assert len(new_params) == new_spec.num_params
    for g in new_spec.gates:
        for q in g.qubits:
            assert 0 <= q < new_spec.num_qubits
        if g.is_parametric:
            assert 0 <= g.param_idx < new_spec.num_params


def test_mutator_with_library_can_weld_fragments():
    from quiver.microstructures import MicrostructureLibrary
    spec, _ = _spec_from_he()
    params = np.linspace(0.1, 0.9, spec.num_params)
    lib = MicrostructureLibrary(fragments_per_solution=4, min_length=3, max_length=4)
    lib.add_solution(spec, params, np.random.default_rng(0))
    assert lib.fragments

    rng = np.random.default_rng(99)
    mutator = Mutator(
        operations=(),  # disable point edits — only weld available
        weights=(),
        chain_min=1, chain_max=1,
        microstructure_library=lib,
    )
    new_spec, new_params = mutator.step(spec, params, rng)
    # Weld appends; the new spec should be at least as long.
    assert new_spec.gate_count >= spec.gate_count
    assert len(new_params) == new_spec.num_params
