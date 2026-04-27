import numpy as np

from quiver.ansatz import HardwareEfficient
from quiver.circuit import CircuitSpec, GateSpec
from quiver.microstructures import MicrostructureLibrary, weld


def test_add_solution_extracts_fragments():
    spec = HardwareEfficient(num_qubits=4, num_layers=2).build()
    params = np.linspace(0.1, 0.9, spec.num_params)
    lib = MicrostructureLibrary(fragments_per_solution=3,
                                min_length=2, max_length=5)
    rng = np.random.default_rng(0)
    n = lib.add_solution(spec, params, rng)
    assert n == 3
    assert len(lib.fragments) == 3
    for f in lib.fragments:
        assert 2 <= f.length <= 5
        # Fragment param indices remapped to [0, num_params)
        for g in f.gates:
            if g.is_parametric:
                assert 0 <= g.param_idx < f.num_params


def test_sample_returns_compatible_fragment():
    spec = HardwareEfficient(num_qubits=3, num_layers=1).build()
    params = np.zeros(spec.num_params)
    lib = MicrostructureLibrary()
    rng = np.random.default_rng(1)
    lib.add_solution(spec, params, rng)
    f = lib.sample(host_num_qubits=4, rng=rng)
    assert f is not None
    assert max(f.qubits_used()) < 4


def test_sample_returns_none_when_qubits_too_many():
    """All extracted fragments touch qubit 4, so a host of width 3 cannot weld them."""
    spec = CircuitSpec(num_qubits=5)
    spec.add(GateSpec("ry", (4,), 0))
    spec.add(GateSpec("cnot", (3, 4)))
    spec.add(GateSpec("ry", (4,), 1))
    spec.num_params = 2
    params = np.array([0.1, 0.2])
    lib = MicrostructureLibrary(fragments_per_solution=4, min_length=2, max_length=3)
    lib.add_solution(spec, params, np.random.default_rng(0))
    assert all(4 in f.qubits_used() for f in lib.fragments)
    f = lib.sample(host_num_qubits=3, rng=np.random.default_rng(0))
    assert f is None


def test_weld_appends_fragment_correctly():
    host = CircuitSpec(num_qubits=4)
    host.add(GateSpec("ry", (0,), 0))
    host.add(GateSpec("ry", (1,), 1))
    host.num_params = 2
    host_params = np.array([0.5, 0.7])

    spec = HardwareEfficient(num_qubits=3, num_layers=1).build()
    spec_params = np.linspace(0.1, 0.9, spec.num_params)
    lib = MicrostructureLibrary(fragments_per_solution=1, min_length=3, max_length=3)
    rng = np.random.default_rng(2)
    lib.add_solution(spec, spec_params, rng)
    fragment = lib.fragments[0]

    new_spec, new_params = weld(host, host_params, fragment)
    assert new_spec.gate_count == host.gate_count + fragment.length
    assert new_spec.num_params == host.num_params + fragment.num_params
    np.testing.assert_allclose(new_params[: host.num_params], host_params)
    np.testing.assert_allclose(new_params[host.num_params :], fragment.params)
    # Fragment's gates have their param_idx shifted by host.num_params
    appended = new_spec.gates[host.gate_count :]
    for orig, app in zip(fragment.gates, appended):
        if orig.is_parametric:
            assert app.param_idx == orig.param_idx + host.num_params
        else:
            assert app.param_idx is None
