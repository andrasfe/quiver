import numpy as np

from quivercirc.ansatz.discovered import DiscoveredAnsatz
from quivercirc.circuit import GateSpec
from quivercirc.distillation import canonicalize, distill_library, fragment_from_canonical
from quivercirc.microstructures import Fragment, MicrostructureLibrary


def _frag_at(qubits_list, name="ry", parametric=True):
    """Helper: build a fragment with parametric gate `name` on a list of qubits."""
    gates = []
    for i, q in enumerate(qubits_list):
        if parametric:
            gates.append(GateSpec(name, (q,), i))
        else:
            gates.append(GateSpec(name, (q,), None))
    params = np.zeros(len(qubits_list)) if parametric else np.zeros(0)
    return Fragment(gates=gates, params=params)


def test_canonicalize_relabels_qubits_in_order_of_appearance():
    f1 = Fragment(
        gates=[GateSpec("ry", (5,), 0), GateSpec("cnot", (5, 7)), GateSpec("rz", (7,), 1)],
        params=np.array([0.0, 0.0]),
    )
    f2 = Fragment(
        gates=[GateSpec("ry", (1,), 0), GateSpec("cnot", (1, 3)), GateSpec("rz", (3,), 1)],
        params=np.array([0.0, 0.0]),
    )
    assert canonicalize(f1) == canonicalize(f2)


def test_distillation_picks_most_frequent_pattern():
    lib = MicrostructureLibrary()
    common = Fragment(
        gates=[GateSpec("ry", (0,), 0), GateSpec("cnot", (0, 1))],
        params=np.array([0.5]),
    )
    rare = Fragment(
        gates=[GateSpec("rxx", (2, 3), 0), GateSpec("rz", (3,), 1)],
        params=np.array([0.1, 0.2]),
    )
    for _ in range(5):
        lib.fragments.append(common)
    lib.fragments.append(rare)

    patterns = distill_library(lib, top_k=2, min_occurrences=1, min_width=2)
    assert len(patterns) == 2
    assert patterns[0].occurrences == 5
    assert patterns[1].occurrences == 1


def test_discovered_ansatz_tiles_canonical_pattern():
    lib = MicrostructureLibrary()
    f = Fragment(
        gates=[GateSpec("rxx", (0, 1), 0), GateSpec("rz", (1,), 1)],
        params=np.array([0.0, 0.0]),
    )
    lib.fragments.append(f)
    pat = distill_library(lib, top_k=1, min_occurrences=1, min_width=2)[0]
    ansatz = DiscoveredAnsatz.from_pattern(pat, num_qubits=4, num_layers=1, stride=1)
    spec = ansatz.build()
    # Canonical width 2, host width 4 → 3 tiles at offsets 0, 1, 2.
    # Each tile has 2 gates (rxx, rz) and 2 params.
    assert spec.gate_count == 6
    assert spec.num_params == 6
    assert spec.connectivity() == {(0, 1), (1, 2), (2, 3)}


def test_discovered_ansatz_with_layers():
    lib = MicrostructureLibrary()
    f = Fragment(
        gates=[GateSpec("ryy", (0, 1), 0)],
        params=np.array([0.0]),
    )
    lib.fragments.append(f)
    pat = distill_library(lib, top_k=1, min_occurrences=1, min_width=2)[0]
    ansatz = DiscoveredAnsatz.from_pattern(pat, num_qubits=3, num_layers=2, stride=1)
    spec = ansatz.build()
    # canonical_width=2, host=3 → 2 tiles per layer × 2 layers = 4 instances.
    assert spec.gate_count == 4
    assert spec.num_params == 4


def test_fragment_from_canonical_round_trip():
    f = Fragment(
        gates=[GateSpec("rxx", (3, 5), 0), GateSpec("h", (5,), None)],
        params=np.array([0.5]),
    )
    canon = canonicalize(f)
    rebuilt = fragment_from_canonical(canon)
    assert canonicalize(rebuilt) == canon
    assert rebuilt.num_params == 1
