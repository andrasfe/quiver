"""Capacity / scaling study: gradually escalate problem complexity until
Quiver hits a wall.

Two axes are escalated together:
  - qubit count (4 -> 6 -> 8 -> 10 -> 12)
  - target structure (alternating MaxCut -> GHZ -> Haar-random)

A "capacity hit" is declared for any stage where Quiver finds fewer than
max(2, target/3) verified diverse solutions OR overruns its time budget by
more than 50%. The first such stage stops the cascade so we can locate the
boundary cleanly.

Why we expect this to bottom out
--------------------------------
Hardware-efficient ansätze with d layers carry roughly 2nd parameters,
which scale linearly in qubit count. The Hilbert-space dimension is 2^n —
exponential. For a structured target (cycle MaxCut, GHZ) the relevant
manifold is low-dimensional, so a few layers suffice. For a Haar-random
target, the full 2^(n+1) - 2 real degrees of freedom must be matched, and
linear-parameter ansätze cannot reach the verifier threshold no matter
how long basin-hopping runs. That is the wall we expect to find.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from quivercirc import Quiver, QuiverConfig
from quivercirc.ansatz import HardwareEfficient, LinearEntangler, QAOAInspired
from quivercirc.config import (
    BudgetConfig,
    DiversityConfig,
    ExplorationConfig,
    OptimizerConfig,
)


# ---------- target / verifier builders -----------------------------------


def cycle_maxcut(n: int, threshold: float = 0.7) -> tuple[np.ndarray, Callable, dict]:
    """n-cycle MaxCut — even n only (odd cycles aren't bipartite)."""
    if n % 2:
        raise ValueError("cycle MaxCut requires even n")
    bits_a = sum((q & 1) << q for q in range(n))      # 0101... pattern
    bits_b = sum(((q + 1) & 1) << q for q in range(n))  # 1010... pattern
    target = np.zeros(2**n, dtype=complex)
    target[bits_a] = 1 / np.sqrt(2)
    target[bits_b] = 1 / np.sqrt(2)

    def verifier(state: np.ndarray) -> tuple[bool, float]:
        p = float(abs(state[bits_a]) ** 2 + abs(state[bits_b]) ** 2)
        return p > threshold, p

    return target, verifier, {"kind": "cycle_maxcut", "n": n, "threshold": threshold,
                              "alt_indices": [bits_a, bits_b]}


def ghz(n: int, threshold: float = 0.95) -> tuple[np.ndarray, Callable, dict]:
    target = np.zeros(2**n, dtype=complex)
    target[0] = 1 / np.sqrt(2)
    target[-1] = 1 / np.sqrt(2)

    def verifier(state: np.ndarray) -> tuple[bool, float]:
        f = float(abs(np.vdot(target, state)) ** 2)
        return f > threshold, f

    return target, verifier, {"kind": "ghz", "n": n, "threshold": threshold}


def haar_random(n: int, seed: int = 13, threshold: float = 0.85
                ) -> tuple[np.ndarray, Callable, dict]:
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    psi = psi.astype(complex) / np.linalg.norm(psi)

    def verifier(state: np.ndarray) -> tuple[bool, float]:
        f = float(abs(np.vdot(psi, state)) ** 2)
        return f > threshold, f

    return psi, verifier, {"kind": "haar_random", "n": n, "seed": seed,
                           "threshold": threshold}


# ---------- ansatz library scaled to qubit count -------------------------


def build_library(n: int) -> list:
    return [
        QAOAInspired(num_qubits=n, num_layers=1, ring=True),
        QAOAInspired(num_qubits=n, num_layers=2, ring=True),
        QAOAInspired(num_qubits=n, num_layers=3, ring=True),
        HardwareEfficient(num_qubits=n, num_layers=1),
        HardwareEfficient(num_qubits=n, num_layers=2),
        HardwareEfficient(num_qubits=n, num_layers=3),
        HardwareEfficient(num_qubits=n, num_layers=4),
        HardwareEfficient(num_qubits=n, num_layers=2, rotation_axes=("rx", "ry")),
        HardwareEfficient(num_qubits=n, num_layers=3, rotation_axes=("rx", "ry")),
        LinearEntangler(num_qubits=n, num_layers=1),
        LinearEntangler(num_qubits=n, num_layers=2),
        LinearEntangler(num_qubits=n, num_layers=3),
    ]


# ---------- stage runner -------------------------------------------------


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


def run_stage(
    name: str,
    n: int,
    target: np.ndarray,
    verifier: Callable,
    meta: dict,
    num_target: int = 6,
    time_budget: float = 60.0,
    basin_hops: int = 10,
) -> StageOutcome:
    config = QuiverConfig(
        exploration=ExplorationConfig(
            num_solutions=num_target, time_budget_seconds=time_budget, seed=42
        ),
        optimizer=OptimizerConfig(basin_hops=basin_hops, max_iter=200, step_size=1.2),
        diversity=DiversityConfig(threshold=0.15),
        budget=BudgetConfig(max_gates=400, max_depth=120),
    )
    q = Quiver(target=target, verifier=verifier, config=config)

    started = time.time()
    sols = q.explore(build_library(n))
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
        len(sols) < max(2, num_target // 3)
        or elapsed > time_budget * 1.5
    )

    return StageOutcome(
        name=name,
        problem_meta=meta,
        num_qubits=n,
        num_solutions_target=num_target,
        num_solutions_found=len(sols),
        elapsed_s=elapsed,
        diversity_min=float(min(div)),
        diversity_max=float(max(div)),
        families_used=sorted({s.family for s in sols}),
        capacity_hit=capacity_hit,
        detail=detail,
    )


# ---------- main ---------------------------------------------------------


def main() -> None:
    stages = [
        ("6q_cycle_maxcut", 6, cycle_maxcut(6)),
        ("8q_cycle_maxcut", 8, cycle_maxcut(8)),
        ("8q_ghz",           8, ghz(8, threshold=0.95)),
        ("10q_ghz",         10, ghz(10, threshold=0.95)),
        ("10q_haar_random", 10, haar_random(10, seed=13, threshold=0.85)),
        ("12q_ghz",         12, ghz(12, threshold=0.90)),
    ]

    outcomes: list[StageOutcome] = []
    for name, n, (target, verifier, meta) in stages:
        print(f"\n=== {name} (n={n}, kind={meta['kind']}) ===", flush=True)
        out = run_stage(name, n, target, verifier, meta)
        outcomes.append(out)
        print(
            f"  found {out.num_solutions_found}/{out.num_solutions_target} "
            f"in {out.elapsed_s:.1f}s | diversity {out.diversity_min:.3f}-{out.diversity_max:.3f} "
            f"| families={out.families_used} | capacity_hit={out.capacity_hit}",
            flush=True,
        )
        for i, d in enumerate(out.detail, 1):
            print(
                f"    [{i}] {d['family']:<20} score={d['verifier_score']:.4f} "
                f"depth={d['depth']:>3} gates={d['gate_count']:>3} 2q={d['two_qubit_count']:>3} "
                f"params={d['num_params']:>3} div={d['diversity']:.3f}",
                flush=True,
            )
        if out.capacity_hit:
            print(f"  ↳ capacity limit at n={n} ({meta['kind']}); halting cascade",
                  flush=True)
            break

    summary = {
        "study": "Quiver scaling capacity — incremental complexity",
        "stop_rule": "found < max(2, target//3) OR elapsed > 1.5*budget",
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
    out_path = Path(__file__).resolve().parent.parent / "results" / "scaling_study.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
