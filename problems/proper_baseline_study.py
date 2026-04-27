"""Head-to-head with a proper baseline: L-BFGS-B + parameter-shift
gradients. Multiple seeds for error bars. Heisenberg-8q ground state.

Two conditions, identical except for one knob:

  A. canonical-only      — eight hand-designed templates, fresh empty
                           library, adaptive growth + mutation enabled,
                           anti-template + Hamiltonian-aware on
  B. canonical+discovered — same plus DiscoveredAnsätze loaded from
                           final_discovered_patterns.json (the artifact
                           the curriculum distilled). Library loaded
                           from curriculum_library.json so adaptive can
                           draw on prior fragments too.

Both runs use the same optimizer, same time budget, same seed schedule.
The single isolated effect is whether a previously-distilled artifact
(discovered patterns + fragment library) helps when the optimizer is
no longer the bottleneck.
"""

from __future__ import annotations

import json
import time
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
from quiver.distillation import load_patterns
from quiver.hamiltonian import heisenberg, vqe_setup
from quiver.microstructures import MicrostructureLibrary


RESULTS = Path(__file__).resolve().parent.parent / "results"


def heisenberg_8q():
    H = heisenberg(8, periodic=False)
    setup = vqe_setup(H, threshold_above_ground=1.0)
    return {
        "n": 8, "H": H,
        "ground_energy": setup.ground_energy,
        "verifier": setup.verifier,
        "state_loss": setup.state_loss,
        "backend": NumpyBackend(num_qubits=8),
        "coupled_pairs": [[i, i + 1] for i in range(7)],
    }


def canonical_ansätze(n=8):
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


def discovered_ansätze(n=8):
    patterns = load_patterns(RESULTS / "final_discovered_patterns.json")
    out = []
    for i, pat in enumerate(patterns):
        for L in (1, 2, 3):
            if pat.width <= n:
                out.append(DiscoveredAnsatz.from_pattern(
                    pat, num_qubits=n, num_layers=L, stride=1,
                    tag=f"discovered_{i}_L{L}",
                ))
    return out


def make_config(seed: int, time_budget: int) -> QuiverConfig:
    return QuiverConfig(
        exploration=ExplorationConfig(
            num_solutions=4, time_budget_seconds=time_budget, seed=seed,
        ),
        # Proper baseline: L-BFGS-B with parameter-shift gradients.
        optimizer=OptimizerConfig(
            method="L-BFGS-B", basin_hops=4, max_iter=80,
            step_size=1.0, tolerance=1e-5,
        ),
        diversity=DiversityConfig(threshold=0.18, prefer_compact=True),
        budget=BudgetConfig(max_gates=400, max_depth=200),
        mutation=MutationConfig(
            enabled=True, frequency=2,
            chain_min=8, chain_max=20,
            use_microstructures=True, weld_weight=2.0,
        ),
        adaptive=AdaptiveConfig(
            enabled=True, frequency=2,
            max_gates=50, candidates_per_step=14,
            inner_max_iter=25, plateau_patience=5,
            epsilon_random=0.25, target_loss=-1e18,
            microstructures_enabled=True,
            microstructures_per_solution=4,
            microstructure_min_length=2, microstructure_max_length=5,
            fragment_candidate_fraction=0.4,
            anti_template_weight=0.3,
            coupled_pairs=[[i, i + 1] for i in range(7)],
            coupling_bonus=0.05,
        ),
    )


def run_one(label: str, seed: int, condition: str, time_budget: int):
    problem = heisenberg_8q()
    cfg = make_config(seed=seed, time_budget=time_budget)

    if condition == "canonical_only":
        ansatz_lib = canonical_ansätze(8)
        lib = MicrostructureLibrary(
            fragments_per_solution=4, min_length=2, max_length=5
        )
    elif condition == "canonical_plus_discovered":
        ansatz_lib = canonical_ansätze(8) + discovered_ansätze(8)
        lib = MicrostructureLibrary.load_json(RESULTS / "curriculum_library.json")
    else:
        raise ValueError(condition)

    q = Quiver(
        target={"hamiltonian_size": problem["H"].shape},
        verifier=problem["verifier"],
        backend=problem["backend"],
        state_loss=problem["state_loss"],
        config=cfg,
        microstructure_library=lib,
    )

    started = time.time()
    sols = q.explore(ansatz_lib)
    elapsed = time.time() - started

    energies = [float(s.fidelity) for s in sols]
    best = min(energies) if energies else None
    return {
        "label": label,
        "seed": seed,
        "condition": condition,
        "elapsed_s": elapsed,
        "found": len(sols),
        "best_energy": best,
        "all_energies": energies,
        "ground": problem["ground_energy"],
        "accept_below": problem["ground_energy"] + 1.0,
    }


def main():
    seeds = [42, 13, 7]
    time_budget = 360
    results = []

    for cond in ["canonical_only", "canonical_plus_discovered"]:
        for seed in seeds:
            print(f"\n--- {cond} seed={seed} budget={time_budget}s ---", flush=True)
            r = run_one(f"{cond}/seed{seed}", seed, cond, time_budget)
            results.append(r)
            best = f"{r['best_energy']:+.4f}" if r["best_energy"] is not None else "—"
            print(
                f"    found {r['found']}/4 in {r['elapsed_s']:.1f}s; "
                f"best E = {best}; accept-below {r['accept_below']:+.4f}",
                flush=True,
            )

    # Aggregate per condition
    summary = {}
    for cond in ["canonical_only", "canonical_plus_discovered"]:
        these = [r for r in results if r["condition"] == cond]
        verifying = [r for r in these if r["found"] > 0]
        bests = [r["best_energy"] for r in verifying if r["best_energy"] is not None]
        summary[cond] = {
            "seeds_run": len(these),
            "seeds_verifying": len(verifying),
            "best_energy_mean": float(np.mean(bests)) if bests else None,
            "best_energy_std": float(np.std(bests)) if len(bests) >= 2 else None,
            "best_energy_min": float(min(bests)) if bests else None,
            "all_results": these,
        }

    print("\n=== summary (Heisenberg-8q, L-BFGS-B + parameter-shift) ===", flush=True)
    print(f"  ground energy = {results[0]['ground']:.4f}", flush=True)
    print(f"  accept-below  = {results[0]['accept_below']:.4f}\n", flush=True)
    for cond in ["canonical_only", "canonical_plus_discovered"]:
        s = summary[cond]
        if s["best_energy_mean"] is None:
            print(f"  {cond:<28} : verified 0/{s['seeds_run']} seeds", flush=True)
        else:
            std_str = f"±{s['best_energy_std']:.4f}" if s["best_energy_std"] else ""
            print(
                f"  {cond:<28} : verified {s['seeds_verifying']}/{s['seeds_run']} "
                f"seeds; E = {s['best_energy_mean']:+.4f} {std_str} "
                f"(min {s['best_energy_min']:+.4f})",
                flush=True,
            )

    out = RESULTS / "proper_baseline_study.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
