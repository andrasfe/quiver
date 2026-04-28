"""Curriculum bootstrap — does easy-to-hard learning crack a wall that
flat-out attacking the hard problem cannot?

We start with an empty MicrostructureLibrary and an empty discovered-
ansätze list. We run a sequence of progressively harder problems:

  Bell-2 → GHZ-3 → GHZ-4 → Heisenberg-4 → Heisenberg-6 → Heisenberg-8

After each problem we **redistill** the library and rebuild the
discovered-ansatz list from the updated patterns. The next problem
sees a richer library and a richer ansatz set. By the time we reach
Heisenberg-8q (the wall the canonical-only baseline cannot cross),
the system has been trained on five smaller problems.

For comparison, we also run Heisenberg-8q with a *fresh* empty library
and just the canonical ansätze — that's the no-curriculum baseline.

The hypothesis: the curriculum cracks Heisenberg-8q in less time and
with more diverse / lower-energy solutions than the no-curriculum run,
because of what the system learned on the earlier problems.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from quivercirc import Quiver, QuiverConfig
from quivercirc.ansatz import (
    BrickWall,
    DiscoveredAnsatz,
    HardwareEfficient,
    LinearEntangler,
    QAOAInspired,
    StronglyEntangling,
)
from quivercirc.backends import NumpyBackend
from quivercirc.config import (
    AdaptiveConfig,
    BudgetConfig,
    DiversityConfig,
    ExplorationConfig,
    MutationConfig,
    OptimizerConfig,
)
from quivercirc.distillation import distill_library, save_patterns
from quivercirc.diversity import structural_similarity
from quivercirc.hamiltonian import heisenberg, vqe_setup
from quivercirc.microstructures import MicrostructureLibrary


RESULTS = Path(__file__).resolve().parent.parent / "results"


# ---------- problem builders --------------------------------------------


def state_prep_problem(name: str, n: int, target: np.ndarray, threshold: float = 0.95):
    def verify(state):
        f = float(abs(np.vdot(target, state)) ** 2)
        return f > threshold, f
    return {
        "name": name, "n": n, "kind": "state_prep",
        "target": target, "verifier": verify,
        "state_loss": None, "backend": None,
        "coupled_pairs": [],
    }


def bell_2():
    t = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return state_prep_problem("Bell-2", 2, t)


def ghz_n(n: int):
    t = np.zeros(2**n, dtype=complex)
    t[0] = t[-1] = 1 / np.sqrt(2)
    return state_prep_problem(f"GHZ-{n}", n, t)


def heisenberg_problem(n: int, threshold_above_ground: float = 1.0, name: str | None = None):
    H = heisenberg(n, periodic=False)
    setup = vqe_setup(H, threshold_above_ground=threshold_above_ground)
    return {
        "name": name or f"Heisenberg-{n}",
        "n": n, "kind": "heisenberg",
        "target": {"hamiltonian_size": H.shape},
        "verifier": setup.verifier,
        "state_loss": setup.state_loss,
        "backend": NumpyBackend(num_qubits=n),
        "coupled_pairs": [[i, i + 1] for i in range(n - 1)],
        "ground_energy": setup.ground_energy,
        "threshold_above_ground": threshold_above_ground,
    }


# ---------- ansatz libraries --------------------------------------------


def canonical_ansätze(n: int) -> list:
    return [
        QAOAInspired(num_qubits=n, num_layers=2, ring=True),
        QAOAInspired(num_qubits=n, num_layers=3, ring=True),
        HardwareEfficient(num_qubits=n, num_layers=2),
        HardwareEfficient(num_qubits=n, num_layers=3),
        LinearEntangler(num_qubits=n, num_layers=2),
        BrickWall(num_qubits=n, num_layers=3),
        BrickWall(num_qubits=n, num_layers=5),
        StronglyEntangling(num_qubits=n, num_layers=3),
    ]


def discovered_ansätze(library: MicrostructureLibrary, n: int,
                       top_k: int = 6, layers=(1, 2, 3)) -> list:
    patterns = distill_library(library, top_k=top_k, min_occurrences=2, min_width=2)
    out = []
    for i, pat in enumerate(patterns):
        for L in layers:
            if pat.width <= n:
                out.append(DiscoveredAnsatz.from_pattern(
                    pat, num_qubits=n, num_layers=L, stride=1,
                    tag=f"discovered_{i}_L{L}",
                ))
    return out, patterns


# ---------- runner ------------------------------------------------------


def run_problem(problem: dict, ansatz_lib: list,
                shared_lib: MicrostructureLibrary,
                hamiltonian_aware: bool, time_budget: int,
                num_target: int = 4):
    n = problem["n"]
    cfg = QuiverConfig(
        exploration=ExplorationConfig(
            num_solutions=num_target, time_budget_seconds=time_budget, seed=42 + n
        ),
        optimizer=OptimizerConfig(basin_hops=8, max_iter=400, step_size=1.5),
        diversity=DiversityConfig(threshold=0.18, prefer_compact=True),
        budget=BudgetConfig(max_gates=400, max_depth=200),
        mutation=MutationConfig(
            enabled=True, frequency=2,
            chain_min=8, chain_max=20,
            use_microstructures=True,
            weld_weight=2.0,
        ),
        adaptive=AdaptiveConfig(
            enabled=True, frequency=2,
            max_gates=50,
            candidates_per_step=18,
            inner_max_iter=35,
            plateau_patience=5,
            epsilon_random=0.25,
            target_loss=-1e18 if problem["kind"] == "heisenberg" else 0.05,
            microstructures_enabled=True,
            microstructures_per_solution=4,
            microstructure_min_length=2,
            microstructure_max_length=5,
            fragment_candidate_fraction=0.4,
            anti_template_weight=0.3,
            coupled_pairs=problem["coupled_pairs"] if hamiltonian_aware else [],
            coupling_bonus=0.05 if hamiltonian_aware else 0.0,
        ),
    )

    q = Quiver(
        target=problem["target"],
        verifier=problem["verifier"],
        backend=problem["backend"],
        state_loss=problem["state_loss"],
        config=cfg,
        microstructure_library=shared_lib,
    )

    started = time.time()
    sols = q.explore(ansatz_lib)
    elapsed = time.time() - started

    rows = []
    for s in sols:
        rows.append({
            "family": s.family,
            "score": float(s.fidelity),
            "gate_count": s.gate_count,
            "depth": s.depth,
            "two_qubit_count": s.two_qubit_count,
            "num_params": len(s.params),
        })
    return {
        "problem": problem["name"],
        "n": n,
        "elapsed_s": elapsed,
        "found": len(sols),
        "best_score": min((r["score"] for r in rows), default=None) if problem["kind"] == "heisenberg" else max((r["score"] for r in rows), default=None),
        "solutions": rows,
    }


# ---------- main: curriculum + control ----------------------------------


def main():
    # --- the curriculum: easy → hard, library carries forward -----------
    print("\n############### CURRICULUM ###############", flush=True)
    curriculum = [
        (bell_2(),                  60),
        (ghz_n(3),                 90),
        (ghz_n(4),                120),
        (heisenberg_problem(4, threshold_above_ground=0.4), 180),
        (heisenberg_problem(6, threshold_above_ground=0.6), 240),
        (heisenberg_problem(8, threshold_above_ground=1.0), 360),
    ]

    shared_lib = MicrostructureLibrary(
        fragments_per_solution=4, min_length=2, max_length=5
    )
    pattern_history: list[dict] = []
    curriculum_results: list[dict] = []

    for problem, budget in curriculum:
        n = problem["n"]
        # Build ansatz library from canonical + whatever the system has discovered.
        canon = canonical_ansätze(n)
        disc, patterns = discovered_ansätze(shared_lib, n, top_k=6, layers=(1, 2, 3))
        ansatz_lib = canon + disc

        print(
            f"\n--- {problem['name']} (n={n}) — canon={len(canon)} disc={len(disc)} "
            f"frags={len(shared_lib.fragments)} ---",
            flush=True,
        )
        if problem["kind"] == "heisenberg":
            print(
                f"    ground={problem['ground_energy']:.4f} "
                f"accept-below={problem['ground_energy'] + problem['threshold_above_ground']:.4f}",
                flush=True,
            )
        result = run_problem(problem, ansatz_lib, shared_lib,
                             hamiltonian_aware=(problem["kind"] == "heisenberg"),
                             time_budget=budget)
        curriculum_results.append(result)

        # Snapshot library state.
        pattern_snapshot = [
            {
                "occurrences": p.occurrences, "width": p.width, "length": p.length,
                "gate_signature": " · ".join(name for name, _, _ in p.canonical),
            }
            for p in patterns
        ]
        pattern_history.append({
            "after_problem": problem["name"],
            "library_size": len(shared_lib.fragments),
            "patterns_distilled": len(patterns),
            "top_patterns": pattern_snapshot[:5],
        })

        flag = (
            f"best_score={result['best_score']:.4f}"
            if result["best_score"] is not None else "no solutions"
        )
        print(
            f"    found {result['found']}/4 in {result['elapsed_s']:.1f}s; "
            f"{flag}; library {len(shared_lib.fragments)} frags, "
            f"{len(patterns)} distilled patterns",
            flush=True,
        )

    # Save the final discovered patterns as a portable artifact.
    final_patterns = distill_library(shared_lib, top_k=10, min_occurrences=2, min_width=2)
    save_patterns(final_patterns, RESULTS / "final_discovered_patterns.json")
    shared_lib.save_json(RESULTS / "curriculum_library.json")

    # --- control: same Heisenberg-8q, no curriculum, fresh library ------
    print("\n############### CONTROL (no curriculum) ###############", flush=True)
    control_problem = heisenberg_problem(8, threshold_above_ground=1.0)
    control_lib = MicrostructureLibrary(
        fragments_per_solution=4, min_length=2, max_length=5
    )
    control_result = run_problem(
        control_problem,
        canonical_ansätze(8),  # no discovered ansätze
        control_lib,
        hamiltonian_aware=True,
        time_budget=360,
    )
    print(
        f"\n--- CONTROL Heisenberg-8 (no curriculum) ---\n"
        f"    ground={control_problem['ground_energy']:.4f} "
        f"accept-below={control_problem['ground_energy'] + control_problem['threshold_above_ground']:.4f}\n"
        f"    found {control_result['found']}/4 in {control_result['elapsed_s']:.1f}s; "
        f"best={control_result['best_score']}",
        flush=True,
    )

    # --- compare ---
    final_curriculum = curriculum_results[-1]
    print("\n############### CURRICULUM vs CONTROL ###############", flush=True)
    print(f"  Heisenberg-8q ground energy:  {control_problem['ground_energy']:.4f}", flush=True)
    print(f"  accept threshold (ground+1):  {control_problem['ground_energy'] + 1.0:.4f}", flush=True)
    print(f"  curriculum  : {final_curriculum['found']}/4 found, best={final_curriculum['best_score']}", flush=True)
    print(f"  control     : {control_result['found']}/4 found, best={control_result['best_score']}", flush=True)

    summary = {
        "study": "Curriculum bootstrap — easy-to-hard self-improvement",
        "curriculum": [r for r in curriculum_results],
        "pattern_history": pattern_history,
        "control_heisenberg_8q": control_result,
        "ground_energy": control_problem["ground_energy"],
        "accept_threshold": control_problem["ground_energy"] + 1.0,
        "final_pattern_count": len(final_patterns),
        "final_library_size": len(shared_lib.fragments),
    }
    out = RESULTS / "curriculum_bootstrap.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}", flush=True)
    print(f"wrote {RESULTS / 'curriculum_library.json'}", flush=True)
    print(f"wrote {RESULTS / 'final_discovered_patterns.json'}", flush=True)


if __name__ == "__main__":
    main()
