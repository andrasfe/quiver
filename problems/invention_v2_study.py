"""Invention v2 — does the microstructure library + anti-template reward
push adaptive growth deeper into novel territory?

Three runs on the 4q Bell-pair-pair target:

  A. plain ADAPT          — gate-by-gate growth, no extras
  B. ADAPT + microstructures — fragments learned from each verified
                               solution become candidate macro-blocks
  C. ADAPT + microstructures + anti-template reward — also penalises
                               candidates that resemble templates

Reports per-run novelty distribution and gate histograms so you can see
the shapes the system is inventing.
"""

from __future__ import annotations

import json
import time
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


N = 4
NUM_TARGET = 6


def bell_pair_pair_target():
    t = np.zeros(2**N, dtype=complex)
    t[3] = t[12] = 1 / np.sqrt(2)

    def verifier(state):
        f = float(abs(np.vdot(t, state)) ** 2)
        return f > 0.95, f

    return t, verifier


def build_library():
    return [
        QAOAInspired(num_qubits=N, num_layers=1, ring=True),
        QAOAInspired(num_qubits=N, num_layers=2, ring=True),
        HardwareEfficient(num_qubits=N, num_layers=1),
        HardwareEfficient(num_qubits=N, num_layers=2),
        HardwareEfficient(num_qubits=N, num_layers=3),
        LinearEntangler(num_qubits=N, num_layers=1),
        LinearEntangler(num_qubits=N, num_layers=2),
    ]


def base_config(time_budget=180):
    return QuiverConfig(
        exploration=ExplorationConfig(
            num_solutions=NUM_TARGET, time_budget_seconds=time_budget, seed=42
        ),
        optimizer=OptimizerConfig(basin_hops=8, max_iter=200, step_size=1.5),
        diversity=DiversityConfig(threshold=0.15),
        budget=BudgetConfig(max_gates=400, max_depth=200),
    )


def run(label, microstructures, anti_template_weight, time_budget=180):
    target, verifier = bell_pair_pair_target()
    cfg = base_config(time_budget)
    cfg.mutation = MutationConfig(enabled=False)  # focus on adaptive only
    cfg.adaptive = AdaptiveConfig(
        enabled=True,
        frequency=1,                      # adaptive every round
        max_gates=20,
        candidates_per_step=20,
        inner_max_iter=40,
        plateau_patience=5,
        epsilon_random=0.25,
        target_loss=0.02,
        microstructures_enabled=microstructures,
        microstructures_per_solution=4,
        microstructure_min_length=2,
        microstructure_max_length=5,
        fragment_candidate_fraction=0.4,
        anti_template_weight=anti_template_weight,
    )
    q = Quiver(target=target, verifier=verifier, config=cfg)

    started = time.time()
    sols = q.explore(build_library())
    elapsed = time.time() - started

    template_specs = [a.build() for a in build_library()]
    rows = []
    for s in sols:
        sims = [structural_similarity(s.spec, t) for t in template_specs]
        novelty = 1.0 - max(sims)
        hist: dict[str, int] = {}
        for g in s.spec.gates:
            hist[g.name] = hist.get(g.name, 0) + 1
        rows.append({
            "family": s.family,
            "verifier_score": float(s.fidelity),
            "gate_count": s.gate_count,
            "depth": s.depth,
            "two_qubit_count": s.two_qubit_count,
            "num_params": len(s.params),
            "diversity_in_registry": float(s.diversity),
            "max_similarity_to_template": float(max(sims)),
            "novelty": float(novelty),
            "extreme_novel": novelty > 0.5,
            "moderate_novel": novelty > 0.25,
            "gate_histogram": hist,
        })

    print(f"\n=== {label} (microstructures={microstructures}, "
          f"anti_template_weight={anti_template_weight}) ===", flush=True)
    print(f"  found {len(sols)}/{NUM_TARGET} in {elapsed:.1f}s", flush=True)
    for i, r in enumerate(rows, 1):
        flag = "EXTREME" if r["extreme_novel"] else (
            "moderate" if r["moderate_novel"] else "templated"
        )
        print(
            f"  [{i}] {r['family']:<10} score={r['verifier_score']:.4f} "
            f"depth={r['depth']:>2} gates={r['gate_count']:>2} "
            f"novelty={r['novelty']:.3f} ({flag})",
            flush=True,
        )
        print(f"       gates: {r['gate_histogram']}", flush=True)

    return {
        "label": label,
        "microstructures": microstructures,
        "anti_template_weight": anti_template_weight,
        "elapsed_s": elapsed,
        "num_solutions_found": len(sols),
        "extreme_novel_count": sum(1 for r in rows if r["extreme_novel"]),
        "moderate_novel_count": sum(1 for r in rows if r["moderate_novel"]),
        "novelty_mean": float(np.mean([r["novelty"] for r in rows])) if rows else 0.0,
        "solutions": rows,
    }


def main():
    a = run("A_plain_adapt",     microstructures=False, anti_template_weight=0.0)
    b = run("B_microstructures", microstructures=True,  anti_template_weight=0.0)
    c = run("C_micro_plus_anti", microstructures=True,  anti_template_weight=0.5)

    summary = {
        "study": "Invention v2 — microstructure library + anti-template reward",
        "target": "4q Bell-pair-pair: |0011> + |1100>",
        "fidelity_threshold": 0.95,
        "comparison": {
            "A_plain_adapt":     {"found": a["num_solutions_found"], "extreme": a["extreme_novel_count"], "moderate": a["moderate_novel_count"], "novelty_mean": a["novelty_mean"]},
            "B_microstructures": {"found": b["num_solutions_found"], "extreme": b["extreme_novel_count"], "moderate": b["moderate_novel_count"], "novelty_mean": b["novelty_mean"]},
            "C_micro_plus_anti": {"found": c["num_solutions_found"], "extreme": c["extreme_novel_count"], "moderate": c["moderate_novel_count"], "novelty_mean": c["novelty_mean"]},
        },
        "runs": [a, b, c],
    }
    out = Path(__file__).resolve().parent.parent / "results" / "invention_v2_study.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}", flush=True)
    print(
        f"\n>> A plain-adapt:     {a['extreme_novel_count']} extreme / {a['moderate_novel_count']} moderate / mean novelty={a['novelty_mean']:.3f}"
        f"\n>> B +microstructures: {b['extreme_novel_count']} extreme / {b['moderate_novel_count']} moderate / mean novelty={b['novelty_mean']:.3f}"
        f"\n>> C +anti-template:  {c['extreme_novel_count']} extreme / {c['moderate_novel_count']} moderate / mean novelty={c['novelty_mean']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
