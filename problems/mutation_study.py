"""Mutation study: do template+mutation rounds discover circuits that no
template alone produces?

Method:
  Run the same problem (12-qubit 6-cycle MaxCut) twice with identical
  seeds and budgets. The only difference: mutation rounds are off in run
  A and on in run B. We compare:

    - the registry's family distribution (B should contain "mutated")
    - the structural-similarity matrix vs known templates
    - the depth/gate/two-qubit profiles
    - the per-solution diversity scores

A mutated solution is "genuinely novel" if its structural similarity to
*every* template in the library is below 0.5 — i.e. it is closer to
something not in the library than to anything in it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from quivercirc import Quiver, QuiverConfig
from quivercirc.ansatz import (
    AllToAll,
    BrickWall,
    HardwareEfficient,
    LinearEntangler,
    QAOAInspired,
    StronglyEntangling,
)
from quivercirc.config import (
    BudgetConfig,
    DiversityConfig,
    ExplorationConfig,
    MutationConfig,
    OptimizerConfig,
)
from quivercirc.diversity import structural_similarity


N = 8                          # 8 qubits keeps each run fast on numpy backend
THRESHOLD = 0.7                # P(0101...|1010...) > 0.7
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


def base_config():
    return QuiverConfig(
        exploration=ExplorationConfig(
            num_solutions=NUM_TARGET, time_budget_seconds=120, seed=42
        ),
        optimizer=OptimizerConfig(basin_hops=8, max_iter=300, step_size=1.5),
        diversity=DiversityConfig(threshold=0.15),
        budget=BudgetConfig(max_gates=600, max_depth=200),
    )


def run_once(label: str, mutation_on: bool):
    target, verifier = cycle_maxcut(N)
    cfg = base_config()
    # Very aggressive mutation: chains of 15-30 operators rewrite roughly
    # half the gates of a typical 50-100-gate parent, so children should
    # land outside the structural neighbourhood of any existing template.
    # frequency=2 means every other round is a mutation round.
    cfg.mutation = MutationConfig(
        enabled=mutation_on, frequency=2, chain_min=15, chain_max=30
    )
    q = Quiver(target=target, verifier=verifier, config=cfg)
    started = time.time()
    sols = q.explore(build_library(N))
    elapsed = time.time() - started

    template_specs = [a.build() for a in build_library(N)]

    rows = []
    for s in sols:
        sims_to_templates = [
            structural_similarity(s.spec, t) for t in template_specs
        ]
        rows.append({
            "family": s.family,
            "verifier_score": float(s.fidelity),
            "gate_count": s.gate_count,
            "depth": s.depth,
            "two_qubit_count": s.two_qubit_count,
            "num_params": len(s.params),
            "diversity_in_registry": float(s.diversity),
            "max_similarity_to_any_template": float(max(sims_to_templates)),
            "min_similarity_to_any_template": float(min(sims_to_templates)),
            "novel_vs_templates": bool(max(sims_to_templates) < 0.5),
        })

    print(f"\n--- {label}: mutation={'ON' if mutation_on else 'OFF'} ---", flush=True)
    print(f"  found {len(sols)}/{NUM_TARGET} in {elapsed:.1f}s", flush=True)
    for i, r in enumerate(rows, 1):
        nov = "NOVEL" if r["novel_vs_templates"] else "templated"
        print(
            f"  [{i}] {r['family']:<22} score={r['verifier_score']:.4f} "
            f"depth={r['depth']:>3} gates={r['gate_count']:>4} "
            f"2q={r['two_qubit_count']:>3} params={r['num_params']:>3} "
            f"max_sim_to_template={r['max_similarity_to_any_template']:.3f} "
            f"({nov})",
            flush=True,
        )

    return {
        "label": label,
        "mutation_enabled": mutation_on,
        "elapsed_s": elapsed,
        "num_solutions_found": len(sols),
        "solutions": rows,
    }


def main():
    a = run_once("A_baseline_templates_only", mutation_on=False)
    b = run_once("B_with_mutation_rounds",   mutation_on=True)

    novel_count_a = sum(1 for r in a["solutions"] if r["novel_vs_templates"])
    novel_count_b = sum(1 for r in b["solutions"] if r["novel_vs_templates"])
    families_a = sorted({r["family"] for r in a["solutions"]})
    families_b = sorted({r["family"] for r in b["solutions"]})

    summary = {
        "study": "Mutation vs templates-only on 8q 8-cycle MaxCut",
        "novelty_rule": "max similarity to any template < 0.5",
        "comparison": {
            "templates_only": {
                "found": a["num_solutions_found"],
                "elapsed_s": a["elapsed_s"],
                "families": families_a,
                "novel_count": novel_count_a,
            },
            "with_mutation": {
                "found": b["num_solutions_found"],
                "elapsed_s": b["elapsed_s"],
                "families": families_b,
                "novel_count": novel_count_b,
            },
        },
        "runs": [a, b],
    }
    out = Path(__file__).resolve().parent.parent / "results" / "mutation_study.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}", flush=True)
    print(
        f"\n>> templates-only: {novel_count_a} novel / {a['num_solutions_found']} found"
        f"\n>> with-mutation:  {novel_count_b} novel / {b['num_solutions_found']} found",
        flush=True,
    )


if __name__ == "__main__":
    main()
