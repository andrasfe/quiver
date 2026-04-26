from quiver.ansatz import HardwareEfficient, LinearEntangler, QAOAInspired
from quiver.diversity import diversity_score, structural_similarity


def test_identical_specs_have_similarity_one():
    a = HardwareEfficient(num_qubits=3, num_layers=2).build()
    b = HardwareEfficient(num_qubits=3, num_layers=2).build()
    assert structural_similarity(a, b) == 1.0


def test_different_families_are_diverse():
    a = HardwareEfficient(num_qubits=4, num_layers=2).build()
    b = QAOAInspired(num_qubits=4, num_layers=2).build()
    assert structural_similarity(a, b) < 0.5


def test_diversity_score_against_empty_is_one():
    a = LinearEntangler(num_qubits=3, num_layers=2).build()
    assert diversity_score(a, []) == 1.0


def test_param_only_change_does_not_inflate_diversity():
    # Same spec built twice — even with different parameters at runtime,
    # the *structural* signature is identical, so diversity should be 0.
    a = HardwareEfficient(num_qubits=3, num_layers=2).build()
    b = HardwareEfficient(num_qubits=3, num_layers=2).build()
    assert diversity_score(b, [a]) == 0.0
