"""Scaling study v3 — push past the wall with optimizer fixes + higher n.

Three changes from v2:
  1. Optimizer now auto-scales `max_iter` to at least `num_params + 50`,
     so high-dim ansätze actually get enough COBYLA evals to converge.
  2. Cascade does NOT halt on a partial failure — we log it and keep
     pushing other stages, since one expressivity wall (Haar-random) does
     not predict the others (structured targets).
  3. Adds a *signal-detection* Haar stage at threshold 0.5 to distinguish
     "the optimizer can't move at all" from "the optimizer makes progress
     but the threshold is too high".

Stages:
  A. 10q Haar @ 0.5 — does the optimizer fix produce *any* signal?
  B. 10q Haar @ 0.85 — retry the v2 wall with the deepened library.
  C. 12q cycle MaxCut — deepening works at higher n on a structured target.
  D. 12q GHZ — same, on a different structured target.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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
    BudgetConfig,
    DiversityConfig,
    ExplorationConfig,
    OptimizerConfig,
)


def cycle_maxcut(n: int, threshold: float = 0.7):
    a = sum((q & 1) << q for q in range(n))
    b = sum(((q + 1) & 1) << q for q in range(n))
    target = np.zeros(2**n, dtype=complex)
    target[a] = target[b] = 1 / np.sqrt(2)

    def verifier(state):
        p = float(abs(state[a]) ** 2 + abs(state[b]) ** 2)
        return p > threshold, p

    return target, verifier, {"kind": "cycle_maxcut", "n": n, "threshold": threshold}


def ghz(n: int, threshold: float = 0.95):
    target = np.zeros(2**n, dtype=complex)
    target[0] = target[-1] = 1 / np.sqrt(2)

    def verifier(state):
        f = float(abs(np.vdot(target, state)) ** 2)
        return f > threshold, f

    return target, verifier, {"kind": "ghz", "n": n, "threshold": threshold}


def haar_random(n: int, seed: int = 13, threshold: float = 0.85):
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    psi = psi.astype(complex) / np.linalg.norm(psi)

    def verifier(state):
        f = float(abs(np.vdot(psi, state)) ** 2)
        return f > threshold, f

    return psi, verifier, {"kind": "haar_random", "n": n, "seed": seed,
                           "threshold": threshold}


def build_library_deep(n: int) -> list:
    return [
        QAOAInspired(num_qubits=n, num_layers=2, ring=True),
        QAOAInspired(num_qubits=n, num_layers=4, ring=True),
        HardwareEfficient(num_qubits=n, num_layers=3),
        HardwareEfficient(num_qubits=n, num_layers=5),
        HardwareEfficient(num_qubits=n, num_layers=4, rotation_axes=("rx", "ry")),
        LinearEntangler(num_qubits=n, num_layers=3),
        BrickWall(num_qubits=n, num_layers=4),
        BrickWall(num_qubits=n, num_layers=8),
        StronglyEntangling(num_qubits=n, num_layers=3),
        StronglyEntangling(num_qubits=n, num_layers=5),
        AllToAll(num_qubits=n, num_layers=2),
    ]


@dataclass
class StageOutcome:
    name: str
    problem_meta: dict
    num_qubits: int
    num_solutions_target: int
    num_solutions_found: int
    elapsed_s: float
    diversity_min: float
    diversity_max: float
    families_used: list = field(default_factory=list)
    capacity_hit: bool = False
    detail: list = field(default_factory=list)


def run_stage(name, n, target, verifier, meta, *,
              num_target=6, time_budget=180, basin_hops=10,
              max_iter=400, threshold=0.15):
    cfg = QuiverConfig(
        exploration=ExplorationConfig(
            num_solutions=num_target, time_budget_seconds=time_budget, seed=42
        ),
        optimizer=OptimizerConfig(
            basin_hops=basin_hops, max_iter=max_iter, step_size=1.5
        ),
        diversity=DiversityConfig(threshold=threshold),
        budget=BudgetConfig(max_gates=4000, max_depth=400),
    )
    q = Quiver(target=target, verifier=verifier, config=cfg)
    started = time.time()
    sols = q.explore(build_library_deep(n))
    elapsed = time.time() - started

    detail = [
        {
            "family": s.family,
            "verifier_score": float(s.fidelity),
            "gate_count": s.gate_count,
            "depth": s.depth,
            "two_qubit_count": s.two_qubit_count,
            "num_params": len(s.params),
            "diversity": float(s.diversity),
        }
        for s in sols
    ]
    div = [s.diversity for s in sols] or [0.0]
    capacity_hit = (
        len(sols) < max(2, num_target // 3) or elapsed > time_budget * 1.5
    )
    return StageOutcome(
        name=name, problem_meta=meta, num_qubits=n,
        num_solutions_target=num_target, num_solutions_found=len(sols),
        elapsed_s=elapsed, diversity_min=float(min(div)),
        diversity_max=float(max(div)),
        families_used=sorted({s.family for s in sols}),
        capacity_hit=capacity_hit, detail=detail,
    )


def main():
    stages = [
        ("10q_haar_thresh_0.50",
         10, haar_random(10, seed=13, threshold=0.50),
         dict(time_budget=180, basin_hops=8, max_iter=600)),
        ("10q_haar_thresh_0.85",
         10, haar_random(10, seed=13, threshold=0.85),
         dict(time_budget=180, basin_hops=8, max_iter=600)),
        ("12q_cycle_maxcut",
         12, cycle_maxcut(12),
         dict(time_budget=180, basin_hops=8, max_iter=400)),
        ("12q_ghz",
         12, ghz(12, threshold=0.95),
         dict(time_budget=180, basin_hops=8, max_iter=400)),
    ]

    outcomes: list[StageOutcome] = []
    for name, n, (target, verifier, meta), kwargs in stages:
        print(f"\n=== {name} (n={n}, kind={meta['kind']}, "
              f"threshold={meta['threshold']}) ===", flush=True)
        out = run_stage(name, n, target, verifier, meta, **kwargs)
        outcomes.append(out)
        marker = "WALL" if out.capacity_hit else "ok"
        print(
            f"  [{marker}] found {out.num_solutions_found}/{out.num_solutions_target} "
            f"in {out.elapsed_s:.1f}s | div {out.diversity_min:.3f}-{out.diversity_max:.3f} "
            f"| families={out.families_used}",
            flush=True,
        )
        for i, d in enumerate(out.detail, 1):
            print(
                f"    [{i}] {d['family']:<22} score={d['verifier_score']:.4f} "
                f"depth={d['depth']:>3} gates={d['gate_count']:>4} "
                f"2q={d['two_qubit_count']:>3} params={d['num_params']:>3} "
                f"div={d['diversity']:.3f}",
                flush=True,
            )

    summary = {
        "study": "Quiver scaling v3 — optimizer auto-scaling + push to n=12",
        "changes_from_v2": [
            "optimizer max_iter scales with num_params (>= num_params + 50)",
            "cascade no longer halts on partial failure",
            "added Haar-at-low-threshold stage to detect optimizer signal",
        ],
        "stages": [
            {
                "name": o.name,
                "problem": o.problem_meta,
                "num_qubits": o.num_qubits,
                "num_solutions_target": o.num_solutions_target,
                "num_solutions_found": o.num_solutions_found,
                "elapsed_s": o.elapsed_s,
                "diversity_min": o.diversity_min,
                "diversity_max": o.diversity_max,
                "families_used": o.families_used,
                "capacity_hit": o.capacity_hit,
                "detail": o.detail,
            }
            for o in outcomes
        ],
    }
    out = Path(__file__).resolve().parent.parent / "results" / "scaling_study_v3.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
