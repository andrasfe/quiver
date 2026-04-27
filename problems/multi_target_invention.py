"""Multi-target invention session — a single MicrostructureLibrary
travels across six different state-preparation problems, accumulating
fragments as it goes. Adaptive growth uses fragments learned earlier
as candidate macro-blocks on later problems, so the system effectively
develops its own gate vocabulary.

Targets (size, character):
  Bell-2          — 2q,  reference baseline
  GHZ-3           — 3q,  globally-correlated
  GHZ-4           — 4q,  same shape, larger
  W-3             — 3q,  symmetric single-excitation, hard
  W-4             — 4q,  same shape, larger
  BellPairPair-4  — 4q,  two entangled pairs

For each target we run pure adaptive growth with anti-template reward
λ=0.5 and the shared library. We then save the library to disk and
analyse:
  - which gate types dominate the inventions
  - which fragments transferred from one problem's solution into another
    problem's circuit (cross-target learning)
  - the most novel circuit per target
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from quiver import Quiver, QuiverConfig
from quiver.ansatz import HardwareEfficient, LinearEntangler, QAOAInspired
from quiver.config import (
    AdaptiveConfig,
    BudgetConfig,
    DiversityConfig,
    ExplorationConfig,
    MutationConfig,
    OptimizerConfig,
)
from quiver.diversity import structural_similarity
from quiver.microstructures import MicrostructureLibrary


# ---------- targets ------------------------------------------------------


def bell_2():
    t = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    def verify(state):
        f = float(abs(np.vdot(t, state)) ** 2)
        return f > 0.95, f
    return t, verify, "Bell-2"


def ghz(n: int):
    t = np.zeros(2**n, dtype=complex)
    t[0] = t[-1] = 1 / np.sqrt(2)
    def verify(state):
        f = float(abs(np.vdot(t, state)) ** 2)
        return f > 0.95, f
    return t, verify, f"GHZ-{n}"


def w_state(n: int):
    """W_n = (|10..0> + |01..0> + ... + |0..01>) / sqrt(n)."""
    t = np.zeros(2**n, dtype=complex)
    for q in range(n):
        t[1 << q] = 1.0
    t /= np.linalg.norm(t)
    def verify(state):
        f = float(abs(np.vdot(t, state)) ** 2)
        return f > 0.90, f
    return t, verify, f"W-{n}"


def bell_pair_pair_4():
    t = np.zeros(16, dtype=complex)
    t[3] = t[12] = 1 / np.sqrt(2)
    def verify(state):
        f = float(abs(np.vdot(t, state)) ** 2)
        return f > 0.95, f
    return t, verify, "BellPairPair-4"


# ---------- ansatz library and config ------------------------------------


def build_lib(n: int):
    return [
        QAOAInspired(num_qubits=n, num_layers=1, ring=True),
        QAOAInspired(num_qubits=n, num_layers=2, ring=True),
        HardwareEfficient(num_qubits=n, num_layers=1),
        HardwareEfficient(num_qubits=n, num_layers=2),
        HardwareEfficient(num_qubits=n, num_layers=3),
        LinearEntangler(num_qubits=n, num_layers=1),
        LinearEntangler(num_qubits=n, num_layers=2),
    ]


def make_config(n: int, time_budget: int):
    return QuiverConfig(
        exploration=ExplorationConfig(
            num_solutions=4, time_budget_seconds=time_budget, seed=42 + n
        ),
        optimizer=OptimizerConfig(basin_hops=8, max_iter=200, step_size=1.5),
        diversity=DiversityConfig(threshold=0.18),
        budget=BudgetConfig(max_gates=300, max_depth=120),
        mutation=MutationConfig(enabled=False),
        adaptive=AdaptiveConfig(
            enabled=True,
            frequency=1,                        # every round is adaptive
            max_gates=25,
            candidates_per_step=20,
            inner_max_iter=40,
            plateau_patience=6,                 # generous — W states need it
            epsilon_random=0.25,
            target_loss=0.04,
            microstructures_enabled=True,
            microstructures_per_solution=4,
            microstructure_min_length=2,
            microstructure_max_length=5,
            fragment_candidate_fraction=0.4,
            anti_template_weight=0.5,
        ),
    )


# ---------- run + analysis -----------------------------------------------


def run_target(target_fn, time_budget: int, shared_lib: MicrostructureLibrary):
    target, verify, name = target_fn
    n = int(np.log2(target.size))
    cfg = make_config(n, time_budget)
    q = Quiver(
        target=target,
        verifier=verify,
        config=cfg,
        microstructure_library=shared_lib,
    )

    fragments_before = len(shared_lib.fragments)
    started = time.time()
    sols = q.explore(build_lib(n))
    elapsed = time.time() - started
    fragments_after = len(shared_lib.fragments)

    template_specs = [a.build() for a in build_lib(n)]
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
            "signature": [
                {"name": g.name, "qubits": list(g.qubits), "param_idx": g.param_idx}
                for g in s.spec.gates
            ],
        })

    print(f"\n=== {name} (n={n}) ===", flush=True)
    print(
        f"  found {len(sols)}/{cfg.exploration.num_solutions} in {elapsed:.1f}s; "
        f"library {fragments_before} → {fragments_after} fragments",
        flush=True,
    )
    for i, r in enumerate(rows, 1):
        flag = "EXTREME" if r["novelty"] > 0.5 else (
            "moderate" if r["novelty"] > 0.25 else "templated"
        )
        print(
            f"  [{i}] {r['family']:<10} score={r['verifier_score']:.4f} "
            f"depth={r['depth']:>2} gates={r['gate_count']:>2} "
            f"params={r['num_params']:>2} novelty={r['novelty']:.3f} ({flag})",
            flush=True,
        )
        print(f"       gates: {r['gate_histogram']}", flush=True)

    return {
        "target": name,
        "num_qubits": n,
        "elapsed_s": elapsed,
        "fragments_added": fragments_after - fragments_before,
        "found": len(sols),
        "extreme_novel": sum(1 for r in rows if r["novelty"] > 0.5),
        "moderate_novel": sum(1 for r in rows if r["novelty"] > 0.25),
        "novelty_mean": float(np.mean([r["novelty"] for r in rows])) if rows else 0.0,
        "solutions": rows,
    }


def main():
    shared_lib = MicrostructureLibrary(
        fragments_per_solution=4, min_length=2, max_length=5
    )

    targets = [
        (bell_2(),            45),
        (bell_pair_pair_4(), 120),
        (ghz(3),              90),
        (ghz(4),             180),
        (w_state(3),         120),
        (w_state(4),         240),
    ]

    results = []
    for t, budget in targets:
        results.append(run_target(t, budget, shared_lib))

    # --- analysis -------------------------------------------------------

    # Aggregate gate-type usage across all invented circuits.
    global_hist: Counter = Counter()
    for r in results:
        for sol in r["solutions"]:
            for name, count in sol["gate_histogram"].items():
                global_hist[name] += count

    # Fragment "vocabulary": top gate-type bigrams in fragments.
    bigram_hist: Counter = Counter()
    for f in shared_lib.fragments:
        names = [g.name for g in f.gates]
        for a, b in zip(names, names[1:]):
            bigram_hist[(a, b)] += 1

    print("\n=== invention summary ===", flush=True)
    total_found = sum(r["found"] for r in results)
    total_extreme = sum(r["extreme_novel"] for r in results)
    total_moderate = sum(r["moderate_novel"] for r in results)
    print(f"  total verified circuits: {total_found}", flush=True)
    print(f"  extreme-novel: {total_extreme}", flush=True)
    print(f"  moderate-novel (incl. extreme): {total_moderate}", flush=True)
    print(f"  shared library size: {len(shared_lib.fragments)} fragments", flush=True)
    print(f"  global gate usage: {dict(global_hist.most_common(10))}", flush=True)
    print(f"  top bigram patterns:", flush=True)
    for (a, b), c in bigram_hist.most_common(8):
        print(f"    [{a}, {b}] × {c}", flush=True)

    # Save library and results
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    shared_lib.save_json(out_dir / "shared_microstructure_library.json")

    summary = {
        "study": "Multi-target invention with persistent microstructure library",
        "targets_run": len(targets),
        "total_verified": total_found,
        "extreme_novel": total_extreme,
        "moderate_novel": total_moderate,
        "library_size": len(shared_lib.fragments),
        "global_gate_usage": dict(global_hist),
        "top_bigrams": [
            {"a": a, "b": b, "count": c} for (a, b), c in bigram_hist.most_common(20)
        ],
        "per_target": results,
    }
    (out_dir / "multi_target_invention.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"\nwrote {out_dir / 'multi_target_invention.json'}", flush=True)
    print(f"wrote {out_dir / 'shared_microstructure_library.json'}", flush=True)


if __name__ == "__main__":
    main()
