"""Meta-invention — the system writes its own ansatz library.

We load the 160-fragment library accumulated over the previous studies,
distill its dominant canonical patterns, and promote them to first-class
parametric ansatz templates that tile across any qubit count. We then
re-attack the previous Heisenberg-8q wall using *only the system's own
discovered templates* plus mutation, adaptive growth, and Hamiltonian-
aware scoring.

Three runs:
  A. baseline             — canonical templates only, no library, no
                            adaptive — the original setup
  B. discovered-only      — only the system's own templates, plus
                            adaptive + library + mut×lib + compactness
  C. discovered + canon   — both libraries union, all features on,
                            Hamiltonian-aware scoring with chain
                            couplings provided

If C cracks the wall (any verified solution at n=8 Heisenberg ground
state) where A and B do not, that is the meta-learning result: the
system invented templates and used them to reach a regime its
hand-written templates couldn't.
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
    DiscoveredAnsatz,
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
from quiver.distillation import distill_library, fragment_from_canonical
from quiver.diversity import structural_similarity
from quiver.hamiltonian import heisenberg, vqe_setup
from quiver.microstructures import MicrostructureLibrary


RESULTS = Path(__file__).resolve().parent.parent / "results"


# ---------- target -------------------------------------------------------


def heisenberg_problem(n: int, threshold_above_ground: float = 1.0):
    H = heisenberg(n, periodic=False)
    setup = vqe_setup(H, threshold_above_ground=threshold_above_ground)
    chain_pairs = [[i, i + 1] for i in range(n - 1)]
    return {
        "n": n,
        "H": H,
        "ground_energy": setup.ground_energy,
        "threshold_above_ground": threshold_above_ground,
        "verifier": setup.verifier,
        "state_loss": setup.state_loss,
        "backend": NumpyBackend(num_qubits=n),
        "coupled_pairs": chain_pairs,
    }


# ---------- ansatz libraries --------------------------------------------


def canonical_library(n: int):
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


def discovered_library(library_path: Path, n: int, top_k: int = 6,
                       layers_choices: tuple[int, ...] = (1, 2, 3)
                       ) -> tuple[list, list]:
    """Distill the persisted MicrostructureLibrary and return a list of
    DiscoveredAnsatz templates plus a description of the patterns kept."""
    lib = MicrostructureLibrary.load_json(library_path)
    patterns = distill_library(lib, top_k=top_k, min_occurrences=2, min_width=2)
    ansätze = []
    descriptions = []
    for i, pat in enumerate(patterns):
        for L in layers_choices:
            ansätze.append(
                DiscoveredAnsatz.from_pattern(
                    pat, num_qubits=n, num_layers=L, stride=1,
                    tag=f"discovered_{i}_L{L}",
                )
            )
        gate_signature = " · ".join(name for name, _, _ in pat.canonical)
        descriptions.append({
            "pattern_id": i,
            "occurrences": pat.occurrences,
            "width": pat.width,
            "length": pat.length,
            "gate_signature": gate_signature,
        })
    return ansätze, descriptions


# ---------- runner ------------------------------------------------------


def run(label: str, problem: dict, ansatz_lib: list,
        shared_lib: MicrostructureLibrary | None,
        adaptive_on: bool, mutation_on: bool, hamiltonian_aware: bool,
        time_budget: int):
    n = problem["n"]
    cfg = QuiverConfig(
        exploration=ExplorationConfig(
            num_solutions=4, time_budget_seconds=time_budget, seed=42 + n
        ),
        optimizer=OptimizerConfig(basin_hops=8, max_iter=400, step_size=1.5),
        diversity=DiversityConfig(threshold=0.18, prefer_compact=True),
        budget=BudgetConfig(max_gates=400, max_depth=200),
        mutation=MutationConfig(
            enabled=mutation_on, frequency=2,
            chain_min=8, chain_max=20,
            use_microstructures=mutation_on and shared_lib is not None,
            weld_weight=2.0,
        ),
        adaptive=AdaptiveConfig(
            enabled=adaptive_on, frequency=2,
            max_gates=50,
            candidates_per_step=18,
            inner_max_iter=35,
            plateau_patience=5,
            epsilon_random=0.25,
            target_loss=-1e18,                  # state-loss can be very negative
            microstructures_enabled=shared_lib is not None,
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
        target={"hamiltonian_size": problem["H"].shape},
        verifier=problem["verifier"],
        backend=problem["backend"],
        state_loss=problem["state_loss"],
        config=cfg,
        microstructure_library=shared_lib,
    )

    started = time.time()
    sols = q.explore(ansatz_lib)
    elapsed = time.time() - started

    canon = canonical_library(n)
    canon_specs = [a.build() for a in canon]
    rows = []
    for s in sols:
        sims = [structural_similarity(s.spec, t) for t in canon_specs]
        novelty = 1.0 - max(sims)
        rows.append({
            "family": s.family,
            "energy": float(s.fidelity),
            "gate_count": s.gate_count,
            "depth": s.depth,
            "two_qubit_count": s.two_qubit_count,
            "num_params": len(s.params),
            "novelty_vs_canonical": float(novelty),
            "gate_histogram": dict(Counter(g.name for g in s.spec.gates)),
        })

    print(f"\n--- {label} (n={n}) ---", flush=True)
    print(
        f"  ground={problem['ground_energy']:.4f}, "
        f"accept-below={problem['ground_energy'] + problem['threshold_above_ground']:.4f}",
        flush=True,
    )
    print(f"  found {len(sols)}/4 in {elapsed:.1f}s; "
          f"library {len(shared_lib.fragments) if shared_lib else 0} fragments", flush=True)
    for i, r in enumerate(rows, 1):
        flag = "EXTREME" if r["novelty_vs_canonical"] > 0.5 else (
            "moderate" if r["novelty_vs_canonical"] > 0.25 else "templated"
        )
        print(
            f"  [{i}] {r['family']:<22} energy={r['energy']:+.4f} "
            f"depth={r['depth']:>3} gates={r['gate_count']:>3} "
            f"params={r['num_params']:>3} novelty={r['novelty_vs_canonical']:.3f} ({flag})",
            flush=True,
        )

    return {
        "label": label,
        "elapsed_s": elapsed,
        "found": len(sols),
        "extreme_novel": sum(1 for r in rows if r["novelty_vs_canonical"] > 0.5),
        "best_energy": min((r["energy"] for r in rows), default=None),
        "ground_energy": problem["ground_energy"],
        "solutions": rows,
    }


def main():
    n = 8
    problem = heisenberg_problem(n, threshold_above_ground=1.0)

    library_path = RESULTS / "grand_invention_library.json"
    if not library_path.exists():
        library_path = RESULTS / "shared_microstructure_library.json"
    print(f"loading distilled patterns from {library_path}", flush=True)

    discovered_ansätze, pattern_descriptions = discovered_library(
        library_path, n=n, top_k=6, layers_choices=(1, 2, 3)
    )
    canon = canonical_library(n)

    print(f"\ndistilled {len(pattern_descriptions)} canonical patterns:", flush=True)
    for d in pattern_descriptions:
        print(
            f"  pattern[{d['pattern_id']}] occurs {d['occurrences']:>2}× "
            f"width={d['width']} length={d['length']} :: {d['gate_signature']}",
            flush=True,
        )
    print(f"\ndiscovered ansatz library: {len(discovered_ansätze)} templates", flush=True)

    # Each run reloads the persistent library so all three see the same starting point.
    def fresh_lib():
        return MicrostructureLibrary.load_json(library_path)

    a = run(
        "A_canonical_only", problem, canon,
        shared_lib=None, adaptive_on=False, mutation_on=False, hamiltonian_aware=False,
        time_budget=240,
    )
    b = run(
        "B_discovered_only", problem, discovered_ansätze,
        shared_lib=fresh_lib(), adaptive_on=True, mutation_on=True,
        hamiltonian_aware=False,
        time_budget=300,
    )
    c = run(
        "C_discovered_plus_canon_with_Haware", problem,
        discovered_ansätze + canon,
        shared_lib=fresh_lib(), adaptive_on=True, mutation_on=True,
        hamiltonian_aware=True,
        time_budget=420,
    )

    print("\n=== meta-invention summary ===", flush=True)
    for run_result in (a, b, c):
        print(
            f"  {run_result['label']:<40} found={run_result['found']}  "
            f"best E={run_result['best_energy']}",
            flush=True,
        )

    summary = {
        "study": "Meta-invention — system uses its own distilled ansatz templates",
        "target": f"Heisenberg-{n}q ground state preparation",
        "ground_energy": problem["ground_energy"],
        "accept_threshold": problem["ground_energy"] + problem["threshold_above_ground"],
        "distilled_patterns": pattern_descriptions,
        "runs": [a, b, c],
    }
    out = RESULTS / "meta_invention.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
