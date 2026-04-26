from quiver.ansatz import (
    AllToAll,
    BrickWall,
    HardwareEfficient,
    LinearEntangler,
    QAOAInspired,
    StronglyEntangling,
)


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


def test_brickwall_alternates_offset():
    a = BrickWall(num_qubits=4, num_layers=4)
    spec = a.build()
    assert spec.num_params == 3 * 4 * 4
    pairs_per_layer = []
    cur = []
    seen_rotations = 0
    for g in spec.gates:
        if g.name == "cnot":
            cur.append(g.qubits)
        elif g.is_parametric:
            if cur:
                pairs_per_layer.append(cur)
                cur = []
            seen_rotations += 1
    if cur:
        pairs_per_layer.append(cur)
    # Even layers: pairs starting at 0 -> (0,1),(2,3); odd: (1,2)
    assert pairs_per_layer[0] == [(0, 1), (2, 3)]
    assert pairs_per_layer[1] == [(1, 2)]


def test_all_to_all_has_complete_graph_connectivity():
    a = AllToAll(num_qubits=4, num_layers=1)
    spec = a.build()
    assert spec.connectivity() == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}


def test_strongly_entangling_uses_varying_strides():
    a = StronglyEntangling(num_qubits=5, num_layers=4)
    spec = a.build()
    assert spec.num_params == 3 * 5 * 4
    # First layer stride=1 → ring (0,1)(1,2)(2,3)(3,4)(4,0); second stride=2; etc.
    pairs = spec.connectivity()
    assert (0, 2) in pairs and (1, 3) in pairs   # stride-2 layer present
    assert (0, 1) in pairs and (3, 4) in pairs   # stride-1 layer present
