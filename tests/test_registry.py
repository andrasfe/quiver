import numpy as np

from quiver.ansatz import HardwareEfficient, QAOAInspired
from quiver.registry import SolutionRegistry


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
