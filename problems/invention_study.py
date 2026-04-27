"""Invention study — does ADAPT-style growth produce circuit structures
that no human-designed template approximates?

We solve the 8-qubit 8-cycle MaxCut three ways with identical seeds and
budgets:

  A. templates only       — baseline
  B. templates + mutation — known-good (from previous study)
  C. templates + mutation + adaptive growth — the new pipeline

The novelty of each accepted solution is measured as
  novelty(spec) = 1 - max_t sim(spec, t)
where the max is over every template configuration we use anywhere in
the library. A solution with novelty ≥ 0.5 has *less than half* its
structure shared with any template.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from quiver import Quiver, QuiverConfig
from quiver.ansatz import (
    AllToAll,
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
from quiver.diversity import structural_similarity


N = 8
THRESHOLD = 0.7
NUM_TARGET = 8


def cycle_maxcut(n: int):
    a = sum((q & 1) << q for q in range(n))
    b = sum(((q + 1) & 1) << q for q in range(n))
    target = np.zeros(2**n, dtype=complex)
    target[a] = target[b] = 1 / np.sqrt(2)

    def verifier(state):
        p = float(abs(state[a]) ** 2 + abs(state[b]) ** 2)
        return p > THRESHOLD, p

    return target, verifier


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
        AllToAll(num_qubits=n, num_layers=2),
    ]


def base_config(time_budget=180):
    return QuiverConfig(
        exploration=ExplorationConfig(
            num_solutions=NUM_TARGET, time_budget_seconds=time_budget, seed=42
        ),
        optimizer=OptimizerConfig(basin_hops=8, max_iter=300, step_size=1.5),
        diversity=DiversityConfig(threshold=0.15),
        budget=BudgetConfig(max_gates=600, max_depth=200),
    )


def evaluate(
    label: str,
    mutation: bool,
    adaptive: bool,
    time_budget: int = 180,
    adaptive_dominant: bool = False,
):
    """adaptive_dominant=True: adaptive runs every round (frequency=1),
    so the registry fills with grown circuits, not templated ones."""
    target, verifier = cycle_maxcut(N)
    cfg = base_config(time_budget)
    cfg.mutation = MutationConfig(
        enabled=mutation, frequency=2, chain_min=15, chain_max=30
    )
    cfg.adaptive = AdaptiveConfig(
        enabled=adaptive,
        frequency=1 if adaptive_dominant else 3,
        max_gates=60 if adaptive_dominant else 60,
        # Wider candidate sweeps + more patience are essential for combinatorial
        # problems where individual gate additions barely move the loss.
        candidates_per_step=28 if adaptive_dominant else 14,
        inner_max_iter=35,
        plateau_patience=6 if adaptive_dominant else 3,
        epsilon_random=0.25,
        target_loss=0.10,
    )
    q = Quiver(target=target, verifier=verifier, config=cfg)

    started = time.time()
    sols = q.explore(build_library(N))
    elapsed = time.time() - started

    template_specs = [a.build() for a in build_library(N)]
    rows = []
    for s in sols:
        sims = [structural_similarity(s.spec, t) for t in template_specs]
        max_sim = max(sims)
        novelty = 1.0 - max_sim
        rows.append({
            "family": s.family,
            "verifier_score": float(s.fidelity),
            "gate_count": s.gate_count,
            "depth": s.depth,
            "two_qubit_count": s.two_qubit_count,
            "num_params": len(s.params),
            "diversity_in_registry": float(s.diversity),
            "max_similarity_to_template": float(max_sim),
            "novelty": float(novelty),
            "novel_extreme": novelty > 0.5,
            "novel_moderate": novelty > 0.25,
        })

    print(f"\n--- {label} (mut={mutation}, adapt={adaptive}) ---", flush=True)
    print(f"  found {len(sols)}/{NUM_TARGET} in {elapsed:.1f}s", flush=True)
    for i, r in enumerate(rows, 1):
        flag = (
            "EXTREME-NOVEL" if r["novel_extreme"]
            else ("moderate-novel" if r["novel_moderate"] else "templated")
        )
        print(
            f"  [{i}] {r['family']:<10} score={r['verifier_score']:.4f} "
            f"depth={r['depth']:>3} gates={r['gate_count']:>3} "
            f"2q={r['two_qubit_count']:>3} params={r['num_params']:>3} "
            f"novelty={r['novelty']:.3f} ({flag})",
            flush=True,
        )

    return {
        "label": label,
        "mutation_enabled": mutation,
        "adaptive_enabled": adaptive,
        "elapsed_s": elapsed,
        "num_solutions_found": len(sols),
        "extreme_novel_count": sum(1 for r in rows if r["novel_extreme"]),
        "moderate_novel_count": sum(1 for r in rows if r["novel_moderate"]),
        "solutions": rows,
    }


def gate_distribution(spec_dict_list, registry_specs):
    """Optional: per-family gate counts (for narrative)."""
    out = {}
    for fam in {s["family"] for s in spec_dict_list}:
        members = [s for s in spec_dict_list if s["family"] == fam]
        out[fam] = {
            "count": len(members),
            "avg_gates": np.mean([s["gate_count"] for s in members]),
            "avg_depth": np.mean([s["depth"] for s in members]),
            "avg_novelty": np.mean([s["novelty"] for s in members]),
        }
    return out


def main():
    # Skip A and B (already characterised in the previous study); focus on
    # C (mixed) and D (pure adaptive on combinatorial).
    a = evaluate("A_templates_only",      mutation=False, adaptive=False, time_budget=120)
    b = evaluate("B_templates_plus_mut",  mutation=True,  adaptive=False, time_budget=120)
    c = evaluate("C_templates_mut_adapt", mutation=True,  adaptive=True,  time_budget=240)
    d = evaluate("D_adaptive_dominant",   mutation=False, adaptive=True,
                 adaptive_dominant=True,  time_budget=300)

    summary = {
        "study": "Invention via ADAPT-style growth (8q 8-cycle MaxCut)",
        "novelty_definitions": {
            "moderate": "novelty > 0.25 (less than 75% template-shared)",
            "extreme":  "novelty > 0.50 (less than 50% template-shared)",
        },
        "comparison": {
            "A_templates_only":       {"found": a["num_solutions_found"], "elapsed_s": a["elapsed_s"], "moderate_novel": a["moderate_novel_count"], "extreme_novel": a["extreme_novel_count"]},
            "B_templates_plus_mut":   {"found": b["num_solutions_found"], "elapsed_s": b["elapsed_s"], "moderate_novel": b["moderate_novel_count"], "extreme_novel": b["extreme_novel_count"]},
            "C_templates_mut_adapt":  {"found": c["num_solutions_found"], "elapsed_s": c["elapsed_s"], "moderate_novel": c["moderate_novel_count"], "extreme_novel": c["extreme_novel_count"]},
            "D_adaptive_dominant":    {"found": d["num_solutions_found"], "elapsed_s": d["elapsed_s"], "moderate_novel": d["moderate_novel_count"], "extreme_novel": d["extreme_novel_count"]},
        },
        "runs": [a, b, c, d],
    }
    out = Path(__file__).resolve().parent.parent / "results" / "invention_study.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}", flush=True)
    print(
        f"\n>> A templates_only:      {a['extreme_novel_count']} extreme / {a['moderate_novel_count']} moderate / {a['num_solutions_found']} total"
        f"\n>> B +mutation:           {b['extreme_novel_count']} extreme / {b['moderate_novel_count']} moderate / {b['num_solutions_found']} total"
        f"\n>> C +mut +adaptive:      {c['extreme_novel_count']} extreme / {c['moderate_novel_count']} moderate / {c['num_solutions_found']} total"
        f"\n>> D adaptive_dominant:   {d['extreme_novel_count']} extreme / {d['moderate_novel_count']} moderate / {d['num_solutions_found']} total",
        flush=True,
    )


if __name__ == "__main__":
    main()
