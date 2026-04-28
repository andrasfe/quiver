import pytest

from quivercirc.circuit import CircuitSpec, GateSpec


def test_depth_packs_parallel_single_qubit_gates():
    spec = CircuitSpec(num_qubits=3)
    spec.add(GateSpec("h", (0,)))
    spec.add(GateSpec("h", (1,)))
    spec.add(GateSpec("h", (2,)))
    assert spec.depth == 1


def test_depth_serializes_overlapping_two_qubit_gates():
    spec = CircuitSpec(num_qubits=3)
    spec.add(GateSpec("cnot", (0, 1)))
    spec.add(GateSpec("cnot", (1, 2)))
    assert spec.depth == 2


def test_connectivity_collects_unordered_pairs():
    spec = CircuitSpec(num_qubits=4)
    spec.add(GateSpec("cnot", (2, 0)))
    spec.add(GateSpec("cnot", (0, 2)))  # duplicate ordering, same pair
    spec.add(GateSpec("cnot", (1, 3)))
    assert spec.connectivity() == {(0, 2), (1, 3)}


def test_out_of_range_qubit_raises():
    spec = CircuitSpec(num_qubits=2)
    with pytest.raises(ValueError):
        spec.add(GateSpec("h", (5,)))
