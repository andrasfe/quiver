"""Generate a portable diverse-circuit set for hardware experimentation.

Runs Quiver's full diverse-generation pipeline on the canonical 4-qubit
square-graph MaxCut problem and saves the resulting registry to
results/ibm_ready_circuits.json.

Pipeline features enabled:
  • canonical ansatz templates (HE / QAOA / LinEnt / BrickWall / SE)
  • mutation rounds with microstructure library (genetic recombination
    across previously-verified circuits)
  • adaptive (ADAPT-style) gate-by-gate growth with anti-template reward
    pushing structures away from canonical shapes
  • compactness preference in the registry (similar candidates that are
    smaller / shallower replace their incumbents)

The result is many more circuits than the canonical-template-only run
produces — and crucially, structurally varied at much smaller depths
(useful for noisy hardware where every CNOT is expensive).
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
    AdaptiveConfig,
    BudgetConfig,
    DiversityConfig,
    ExplorationConfig,
    MutationConfig,
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


def main(num_solutions: int = 24, time_budget_s: int = 300) -> None:
    target = cycle_alt_target()
    config = QuiverConfig(
        exploration=ExplorationConfig(
            num_solutions=num_solutions,
            time_budget_seconds=time_budget_s,
            seed=42,
        ),
        optimizer=OptimizerConfig(basin_hops=10, max_iter=200, step_size=1.2),
        diversity=DiversityConfig(
            threshold=0.12,           # slightly looser to admit more variants
            prefer_compact=True,       # replace similar entries with smaller ones
        ),
        budget=BudgetConfig(max_gates=120, max_depth=40),
        mutation=MutationConfig(
            enabled=True, frequency=2,
            chain_min=4, chain_max=12,
            use_microstructures=True, weld_weight=2.0,
        ),
        adaptive=AdaptiveConfig(
            enabled=True, frequency=4,
            max_gates=40,
            candidates_per_step=14,
            inner_max_iter=30,
            plateau_patience=4,
            epsilon_random=0.25,
            target_loss=0.05,
            microstructures_enabled=True,
            microstructures_per_solution=4,
            microstructure_min_length=2,
            microstructure_max_length=5,
            fragment_candidate_fraction=0.4,
            anti_template_weight=0.3,
        ),
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
