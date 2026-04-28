"""Generate a portable diverse-circuit set for hardware experimentation.

Runs Quiver on the canonical 4-qubit square-graph MaxCut problem and
saves the resulting registry to results/ibm_ready_circuits.json. The
saved JSON contains every circuit's spec, optimised parameters, and
metadata — enough for a downstream consumer (e.g. a Jupyter notebook
submitting to IBM Quantum) to convert each one to a Qiskit circuit
and run it on real hardware.

We pick the 4-qubit square MaxCut because:
  • 4 qubits fits any near-term IBM device with room to spare
  • the optimal cut (alternating bitstrings 0101 / 1010) is well-known
  • multiple ansatz families converge to it, so we get genuine
    structural diversity in the saved set

The generated circuits use only gates Qiskit supports natively (or
that we wrap as a unitary box), so the notebook can submit them
without extra translation work.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quiver import (
    Quiver,
    QuiverConfig,
    save_registry,
)
from quiver.ansatz import (
    BrickWall,
    HardwareEfficient,
    LinearEntangler,
    QAOAInspired,
    StronglyEntangling,
)
from quiver.config import (
    BudgetConfig,
    DiversityConfig,
    ExplorationConfig,
    OptimizerConfig,
)


RESULTS = Path(__file__).resolve().parent.parent / "results"


# ---------- target: 4-qubit square MaxCut -------------------------------

NUM_QUBITS = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
ALT_INDICES = (5, 10)   # binary 0101 and 1010
PROB_THRESHOLD = 0.7


def cycle_alt_target() -> np.ndarray:
    target = np.zeros(2**NUM_QUBITS, dtype=complex)
    target[ALT_INDICES[0]] = 1.0 / np.sqrt(2)
    target[ALT_INDICES[1]] = 1.0 / np.sqrt(2)
    return target


def alternating_prob(state: np.ndarray) -> float:
    return float(abs(state[ALT_INDICES[0]]) ** 2 + abs(state[ALT_INDICES[1]]) ** 2)


def expected_cut_value(state: np.ndarray) -> float:
    probs = np.abs(state) ** 2
    val = 0.0
    for idx in range(probs.size):
        bits = [(idx >> q) & 1 for q in range(NUM_QUBITS)]
        for i, j in EDGES:
            if bits[i] != bits[j]:
                val += float(probs[idx])
    return val


def verifier(state: np.ndarray) -> tuple[bool, float]:
    p = alternating_prob(state)
    return p > PROB_THRESHOLD, p


# ---------- ansatz library ---------------------------------------------


def build_library() -> list:
    return [
        QAOAInspired(num_qubits=NUM_QUBITS, num_layers=1, ring=True),
        QAOAInspired(num_qubits=NUM_QUBITS, num_layers=2, ring=True),
        HardwareEfficient(num_qubits=NUM_QUBITS, num_layers=1),
        HardwareEfficient(num_qubits=NUM_QUBITS, num_layers=2),
        HardwareEfficient(num_qubits=NUM_QUBITS, num_layers=2,
                          rotation_axes=("rx", "ry")),
        LinearEntangler(num_qubits=NUM_QUBITS, num_layers=1),
        LinearEntangler(num_qubits=NUM_QUBITS, num_layers=2),
        BrickWall(num_qubits=NUM_QUBITS, num_layers=2),
        StronglyEntangling(num_qubits=NUM_QUBITS, num_layers=2),
    ]


def main() -> None:
    target = cycle_alt_target()
    config = QuiverConfig(
        exploration=ExplorationConfig(
            num_solutions=8, time_budget_seconds=120, seed=42,
        ),
        optimizer=OptimizerConfig(basin_hops=10, max_iter=200, step_size=1.2),
        diversity=DiversityConfig(threshold=0.15),
        budget=BudgetConfig(max_gates=120, max_depth=40),
    )

    quiver = Quiver(target=target, verifier=verifier, config=config)
    solutions = quiver.explore(build_library())

    print(f"\ngenerated {len(solutions)} verified diverse circuits", flush=True)
    for i, s in enumerate(solutions, 1):
        print(
            f"  [{i}] {s.family:<22} P_alt={s.fidelity:.4f}  "
            f"depth={s.depth:>2}  gates={s.gate_count:>3}  "
            f"2q={s.two_qubit_count:>2}  params={len(s.params):>2}  "
            f"diversity={s.diversity:.3f}",
            flush=True,
        )

    out = RESULTS / "ibm_ready_circuits.json"
    metadata = {
        "target_kind": "maxcut_cycle",
        "target_description": (
            "4-qubit square (4-cycle) MaxCut. Optimal cut value 4 "
            "achieved by alternating bitstrings 0101 (idx 10) and 1010 "
            "(idx 5). Verifier accepts if P(alternating) > 0.7."
        ),
        "num_qubits": NUM_QUBITS,
        "edges": EDGES,
        "optimal_bitstrings_index": list(ALT_INDICES),
        "verifier_threshold_P_alternating": PROB_THRESHOLD,
        "exact_optimal_cut_value": 4,
        "qubit_convention": "qubit 0 is LSB; integer index of |q3 q2 q1 q0> "
                             "is q0 + 2*q1 + 4*q2 + 8*q3",
    }
    save_registry(solutions, out, metadata=metadata)

    print(f"\nwrote {out}", flush=True)
    print(f"  {len(solutions)} circuits, {sum(len(s.params) for s in solutions)} "
          f"total parameters", flush=True)


if __name__ == "__main__":
    main()
