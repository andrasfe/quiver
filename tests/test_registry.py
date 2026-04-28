import numpy as np

from quivercirc.ansatz import HardwareEfficient, QAOAInspired
from quivercirc.registry import SolutionRegistry


def test_first_entry_is_always_accepted():
    reg = SolutionRegistry(diversity_threshold=0.5)
    spec = HardwareEfficient(num_qubits=3, num_layers=2).build()
    entry = reg.try_add(spec, np.zeros(spec.num_params), "he", 0.99, 0.01)
    assert entry is not None
    assert len(reg) == 1


def test_duplicate_spec_is_rejected():
    reg = SolutionRegistry(diversity_threshold=0.5)
    spec = HardwareEfficient(num_qubits=3, num_layers=2).build()
    reg.try_add(spec, np.zeros(spec.num_params), "he", 0.99, 0.01)
    spec2 = HardwareEfficient(num_qubits=3, num_layers=2).build()
    rejected = reg.try_add(spec2, np.ones(spec2.num_params), "he", 0.995, 0.005)
    assert rejected is None
    assert len(reg) == 1


def test_diverse_family_is_accepted():
    reg = SolutionRegistry(diversity_threshold=0.3)
    a = HardwareEfficient(num_qubits=3, num_layers=2).build()
    b = QAOAInspired(num_qubits=3, num_layers=2).build()
    reg.try_add(a, np.zeros(a.num_params), "he", 0.99, 0.01)
    accepted = reg.try_add(b, np.zeros(b.num_params), "qaoa", 0.99, 0.01)
    assert accepted is not None
    assert len(reg) == 2


def test_prefer_compact_replaces_with_smaller_circuit():
    """A 1-layer HE incumbent should be displaced by a 2-layer HE candidate
    only if the candidate is more compact AND structurally similar."""
    reg = SolutionRegistry(diversity_threshold=0.5, prefer_compact=True)
    big = HardwareEfficient(num_qubits=3, num_layers=3).build()
    reg.try_add(big, np.zeros(big.num_params), "he", 0.99, 0.01)
    assert len(reg) == 1
    small = HardwareEfficient(num_qubits=3, num_layers=1).build()
    accepted = reg.try_add(small, np.zeros(small.num_params), "he", 0.99, 0.01)
    # Without prefer_compact this would be rejected (similar to incumbent);
    # with it, the smaller spec replaces the big one.
    assert accepted is not None
    assert len(reg) == 1
    assert reg.entries[0].gate_count == small.gate_count


def test_prefer_compact_does_not_replace_when_candidate_is_larger():
    reg = SolutionRegistry(diversity_threshold=0.5, prefer_compact=True)
    small = HardwareEfficient(num_qubits=3, num_layers=1).build()
    reg.try_add(small, np.zeros(small.num_params), "he", 0.99, 0.01)
    big = HardwareEfficient(num_qubits=3, num_layers=3).build()
    rejected = reg.try_add(big, np.zeros(big.num_params), "he", 0.99, 0.01)
    assert rejected is None
    assert reg.entries[0].gate_count == small.gate_count
