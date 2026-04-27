"""Grand invention study — all four extensions in one session.

Combines:
  1. The 96-fragment library learned in the previous multi-target run
     (loaded from results/shared_microstructure_library.json) — the
     system starts with a vocabulary, not from scratch.
  2. Mutation × library: mutation rounds can also weld fragments,
     not just permute gates. Genetic recombination across the library.
  3. Hamiltonian targets via state_loss: 6q and 8q Heisenberg and TFIM
     ground-state preparation. The optimizer minimises <state|H|state>
     directly. Verifier passes when the state is within
     `threshold_above_ground` of the exact ground-state energy.
  4. Compactness preference in the registry: similar candidates that
     are smaller / shallower replace their incumbents. The registry
     converges toward Pareto-optimal solutions.

Five targets:
  - Heisenberg-6q (open chain)
  - TFIM-6q at the critical point (J=h=1)
  - Heisenberg-8q (open chain)
  - TFIM-8q (h=2 — easy regime)
  - 8q cycle MaxCut (state-prep target — round-tripping the previous
    benchmark with the new library)

The library accumulates across all five, so by the last problem the
fragment vocabulary is rich.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from quiver import Quiver, QuiverConfig
from quiver.ansatz import (
    BrickWall,
    HardwareEfficient,
    LinearEntangler,
    QAOAInspired,
    StronglyEntangling,
)
from quiver.backends import NumpyBackend
from quiver.config import (
    AdaptiveConfig,
    BudgetConfig,
    DiversityConfig,
    ExplorationConfig,
    MutationConfig,
    OptimizerConfig,
)
from quiver.diversity import structural_similarity
from quiver.hamiltonian import heisenberg, tfim, vqe_setup
from quiver.microstructures import MicrostructureLibrary


RESULTS = Path(__file__).resolve().parent.parent / "results"


# ---------- target builders ---------------------------------------------


def maxcut_cycle_target(n: int, threshold: float = 0.7):
    a = sum((q & 1) << q for q in range(n))
    b = sum(((q + 1) & 1) << q for q in range(n))
    target = np.zeros(2**n, dtype=complex)
    target[a] = target[b] = 1 / np.sqrt(2)

    def verify(state):
        p = float(abs(state[a]) ** 2 + abs(state[b]) ** 2)
        return p > threshold, p

    return {"kind": "maxcut_cycle", "n": n, "target": target, "verifier": verify,
            "state_loss": None, "backend": None}


def heisenberg_target(n: int, threshold_above_ground: float = 0.5):
    H = heisenberg(n, periodic=False)
    setup = vqe_setup(H, threshold_above_ground=threshold_above_ground)
    return {
        "kind": "heisenberg", "n": n,
        "target": {"hamiltonian_size": H.shape, "ground_energy": setup.ground_energy,
                   "threshold_above_ground": threshold_above_ground},
        "verifier": setup.verifier,
        "state_loss": setup.state_loss,
        "backend": NumpyBackend(num_qubits=n),
    }


def tfim_target(n: int, J: float = 1.0, h: float = 1.0,
                threshold_above_ground: float = 0.5):
    H = tfim(n, J=J, h=h, periodic=False)
    setup = vqe_setup(H, threshold_above_ground=threshold_above_ground)
    return {
        "kind": "tfim", "n": n, "J": J, "h": h,
        "target": {"hamiltonian_size": H.shape, "ground_energy": setup.ground_energy,
                   "threshold_above_ground": threshold_above_ground},
        "verifier": setup.verifier,
        "state_loss": setup.state_loss,
        "backend": NumpyBackend(num_qubits=n),
    }


# ---------- library and config ------------------------------------------


def build_library(n: int):
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


def make_config(n: int, time_budget: int) -> QuiverConfig:
    return QuiverConfig(
        exploration=ExplorationConfig(
            num_solutions=4, time_budget_seconds=time_budget, seed=42 + n
        ),
        optimizer=OptimizerConfig(basin_hops=8, max_iter=300, step_size=1.5),
        diversity=DiversityConfig(threshold=0.18, prefer_compact=True),
        budget=BudgetConfig(max_gates=400, max_depth=200),
        mutation=MutationConfig(
            enabled=True, frequency=2,
            chain_min=8, chain_max=20,
            use_microstructures=True,
            weld_weight=2.0,
        ),
        adaptive=AdaptiveConfig(
            enabled=True, frequency=3,
            max_gates=40,
            candidates_per_step=18,
            inner_max_iter=35,
            plateau_patience=5,
            epsilon_random=0.25,
            target_loss=0.10,           # not always used (state_loss can be negative)
            microstructures_enabled=True,
            microstructures_per_solution=4,
            microstructure_min_length=2,
            microstructure_max_length=5,
            fragment_candidate_fraction=0.4,
            anti_template_weight=0.3,
        ),
    )


# ---------- runner ------------------------------------------------------


def run_problem(name: str, problem: dict, time_budget: int,
                shared_lib: MicrostructureLibrary):
    n = problem["n"]
    cfg = make_config(n, time_budget)
    if problem["kind"] in ("heisenberg", "tfim"):
        # Hamiltonian targets — adaptive's `target_loss` (used to early-exit
        # at "good enough fidelity") is not meaningful when the loss is an
        # energy that can be very negative. Disable that early-exit.
        cfg.adaptive.target_loss = -1e18

    q = Quiver(
        target=problem["target"],
        verifier=problem["verifier"],
        backend=problem["backend"],
        state_loss=problem["state_loss"],
        config=cfg,
        microstructure_library=shared_lib,
    )

    fragments_before = len(shared_lib.fragments)
    started = time.time()
    sols = q.explore(build_library(n))
    elapsed = time.time() - started
    fragments_after = len(shared_lib.fragments)

    template_specs = [a.build() for a in build_library(n)]
    rows = []
    for s in sols:
        sims = [structural_similarity(s.spec, t) for t in template_specs]
        novelty = 1.0 - max(sims)
        gate_types = Counter(g.name for g in s.spec.gates)
        rows.append({
            "family": s.family,
            "verifier_score": float(s.fidelity),
            "gate_count": s.gate_count,
            "depth": s.depth,
            "two_qubit_count": s.two_qubit_count,
            "num_params": len(s.params),
            "novelty": float(novelty),
            "gate_histogram": dict(gate_types),
        })

    print(f"\n=== {name} (n={n}, kind={problem['kind']}) ===", flush=True)
    if problem["kind"] in ("heisenberg", "tfim"):
        gs = problem["target"]["ground_energy"]
        thr = gs + problem["target"]["threshold_above_ground"]
        print(f"  ground energy={gs:.4f}, accept-below={thr:.4f}", flush=True)
    print(
        f"  found {len(sols)}/{cfg.exploration.num_solutions} in {elapsed:.1f}s; "
        f"library {fragments_before} → {fragments_after} fragments",
        flush=True,
    )
    for i, r in enumerate(rows, 1):
        flag = "EXTREME" if r["novelty"] > 0.5 else (
            "moderate" if r["novelty"] > 0.25 else "templated"
        )
        score_label = "energy" if problem["kind"] in ("heisenberg", "tfim") else "score"
        print(
            f"  [{i}] {r['family']:<10} {score_label}={r['verifier_score']:+.4f} "
            f"depth={r['depth']:>3} gates={r['gate_count']:>3} "
            f"params={r['num_params']:>3} novelty={r['novelty']:.3f} ({flag})",
            flush=True,
        )
        print(f"       gates: {r['gate_histogram']}", flush=True)

    return {
        "name": name,
        "kind": problem["kind"],
        "num_qubits": n,
        "elapsed_s": elapsed,
        "fragments_added": fragments_after - fragments_before,
        "found": len(sols),
        "extreme_novel": sum(1 for r in rows if r["novelty"] > 0.5),
        "moderate_novel": sum(1 for r in rows if r["novelty"] > 0.25),
        "novelty_mean": float(np.mean([r["novelty"] for r in rows])) if rows else 0.0,
        "ground_energy": problem["target"].get("ground_energy") if isinstance(problem["target"], dict) else None,
        "solutions": rows,
    }


def main():
    # Load the previously-learned 96-fragment library.
    lib_path = RESULTS / "shared_microstructure_library.json"
    if lib_path.exists():
        shared_lib = MicrostructureLibrary.load_json(lib_path)
        print(f"loaded library with {len(shared_lib.fragments)} fragments from "
              f"{lib_path}", flush=True)
    else:
        shared_lib = MicrostructureLibrary(
            fragments_per_solution=4, min_length=2, max_length=5
        )
        print("starting with empty library", flush=True)

    problems = [
        ("heisenberg_6q",     heisenberg_target(6, threshold_above_ground=0.6),  240),
        ("tfim_6q_critical",  tfim_target(6, J=1.0, h=1.0, threshold_above_ground=0.5), 240),
        ("maxcut_cycle_8q",   maxcut_cycle_target(8, threshold=0.7),             180),
        ("tfim_8q_easy",      tfim_target(8, J=1.0, h=2.0, threshold_above_ground=0.6), 300),
        ("heisenberg_8q",     heisenberg_target(8, threshold_above_ground=1.0),  360),
    ]

    results = []
    for name, problem, budget in problems:
        results.append(run_problem(name, problem, budget, shared_lib))

    # Aggregate analysis
    global_hist: Counter = Counter()
    for r in results:
        for sol in r["solutions"]:
            for n, c in sol["gate_histogram"].items():
                global_hist[n] += c

    bigram_hist: Counter = Counter()
    for f in shared_lib.fragments:
        names = [g.name for g in f.gates]
        for a, b in zip(names, names[1:]):
            bigram_hist[(a, b)] += 1

    total_found = sum(r["found"] for r in results)
    total_extreme = sum(r["extreme_novel"] for r in results)
    total_moderate = sum(r["moderate_novel"] for r in results)
    avg_novelty = float(np.mean([r["novelty_mean"] for r in results if r["found"]])) if results else 0.0

    print("\n=== grand invention summary ===", flush=True)
    print(f"  problems run: {len(results)}", flush=True)
    print(f"  total verified: {total_found}", flush=True)
    print(f"  extreme-novel: {total_extreme}", flush=True)
    print(f"  moderate-novel: {total_moderate}", flush=True)
    print(f"  mean per-target novelty: {avg_novelty:.3f}", flush=True)
    print(f"  library size at end: {len(shared_lib.fragments)} fragments", flush=True)
    print(f"  global gate usage: {dict(global_hist.most_common(10))}", flush=True)
    print(f"  top bigram patterns:", flush=True)
    for (a, b), c in bigram_hist.most_common(10):
        print(f"    [{a}, {b}] × {c}", flush=True)

    summary = {
        "study": "Grand invention — Hamiltonian targets + mut×lib + compactness + persistent library",
        "problems_run": len(results),
        "total_verified": total_found,
        "extreme_novel": total_extreme,
        "moderate_novel": total_moderate,
        "mean_per_target_novelty": avg_novelty,
        "library_size_end": len(shared_lib.fragments),
        "global_gate_usage": dict(global_hist),
        "top_bigrams": [
            {"a": a, "b": b, "count": c} for (a, b), c in bigram_hist.most_common(20)
        ],
        "per_problem": results,
    }
    out = RESULTS / "grand_invention.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    shared_lib.save_json(RESULTS / "grand_invention_library.json")
    print(f"\nwrote {out}", flush=True)
    print(f"wrote {RESULTS / 'grand_invention_library.json'}", flush=True)


if __name__ == "__main__":
    main()
