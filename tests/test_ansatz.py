from quiver.ansatz import HardwareEfficient, LinearEntangler, QAOAInspired


def test_hardware_efficient_param_count_matches_spec():
    a = HardwareEfficient(num_qubits=4, num_layers=3)
    spec = a.build()
    assert spec.num_params == a.num_params
    assert spec.num_qubits == 4
    # 3 layers * (2 axes * 4 qubits + 3 entanglers) = 33 gates
    assert spec.gate_count == 3 * (2 * 4 + 3)


def test_linear_entangler_uses_only_ry_and_cnot():
    a = LinearEntangler(num_qubits=3, num_layers=2)
    spec = a.build()
    names = {g.name for g in spec.gates}
    assert names == {"ry", "cnot"}


def test_qaoa_param_count_is_two_per_layer():
    a = QAOAInspired(num_qubits=4, num_layers=3, ring=True)
    spec = a.build()
    assert spec.num_params == 6
