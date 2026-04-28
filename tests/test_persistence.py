import json
from pathlib import Path

import numpy as np
import pytest

from quivercirc.ansatz import HardwareEfficient, QAOAInspired
from quivercirc.circuit import CircuitSpec, GateSpec
from quivercirc.core import Solution
from quivercirc.persistence import (
    load_circuit,
    load_registry,
    save_circuit,
    save_registry,
    spec_from_dict,
    spec_to_dict,
)


def test_spec_round_trip_preserves_gates_and_param_indices():
    original = HardwareEfficient(num_qubits=4, num_layers=2).build()
    d = spec_to_dict(original)
    rebuilt = spec_from_dict(d)
    assert rebuilt.num_qubits == original.num_qubits
    assert rebuilt.num_params == original.num_params
    assert rebuilt.gate_count == original.gate_count
    for g_orig, g_new in zip(original.gates, rebuilt.gates):
        assert g_orig.name == g_new.name
        assert g_orig.qubits == g_new.qubits
        assert g_orig.param_idx == g_new.param_idx


def test_spec_round_trip_handles_non_parametric_gates():
    spec = CircuitSpec(num_qubits=3)
    spec.add(GateSpec("h", (0,)))
    spec.add(GateSpec("cnot", (0, 1)))
    spec.add(GateSpec("ry", (2,), 0))
    spec.num_params = 1
    rebuilt = spec_from_dict(spec_to_dict(spec))
    assert rebuilt.gates[0].param_idx is None
    assert rebuilt.gates[1].param_idx is None
    assert rebuilt.gates[2].param_idx == 0


def test_save_load_circuit(tmp_path: Path):
    spec = QAOAInspired(num_qubits=3, num_layers=2, ring=True).build()
    params = np.linspace(0.1, 0.9, spec.num_params)
    path = tmp_path / "circuit.json"
    save_circuit(spec, params, path, metadata={"target": "test"})
    spec2, params2, meta = load_circuit(path)
    assert spec2.num_qubits == spec.num_qubits
    assert spec2.num_params == spec.num_params
    np.testing.assert_allclose(params2, params)
    assert meta == {"target": "test"}


def _fake_solution(family: str, num_qubits: int = 4, layers: int = 2) -> Solution:
    spec = HardwareEfficient(num_qubits=num_qubits, num_layers=layers).build()
    params = np.linspace(0.0, 1.0, spec.num_params)
    return Solution(
        family=family,
        params=params.tolist(),
        fidelity=0.97,
        objective=0.03,
        gate_count=spec.gate_count,
        depth=spec.depth,
        two_qubit_count=spec.two_qubit_count,
        diversity=0.42,
        spec=spec,
    )


def test_save_load_registry(tmp_path: Path):
    sols = [_fake_solution("hardware_efficient"),
            _fake_solution("hardware_efficient", layers=3),
            _fake_solution("brickwall")]
    path = tmp_path / "registry.json"
    save_registry(sols, path, metadata={"target_kind": "test"})

    rehydrated, meta = load_registry(path)
    assert len(rehydrated) == len(sols)
    assert meta == {"target_kind": "test"}
    for sol, loaded in zip(sols, rehydrated):
        assert loaded.family == sol.family
        assert loaded.gate_count == sol.gate_count
        assert loaded.depth == sol.depth
        np.testing.assert_allclose(loaded.params, sol.params)
        assert loaded.spec.num_qubits == sol.spec.num_qubits
        assert loaded.spec.gate_count == sol.spec.gate_count


def test_loaded_registry_is_human_readable_json(tmp_path: Path):
    sols = [_fake_solution("hardware_efficient")]
    path = tmp_path / "registry.json"
    save_registry(sols, path)
    raw = json.loads(path.read_text())
    assert "schema_version" in raw
    assert "solutions" in raw
    assert raw["solutions"][0]["spec"]["gates"][0]["name"] in {
        "ry", "rz", "rx", "h", "cnot",
    }
