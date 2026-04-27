"""Diversity-driven subspace VQE vs single-circuit VQE.

The claim under test: a registry of K diverse circuits, each trained
to a mediocre energy, beats K independent runs of single-circuit VQE
when post-processed via the generalized eigenvalue problem in the
subspace they span. Matched compute (K circuit trainings each side).

Method A — single-circuit VQE multi-restart:
  Train HE-d on H_target K times with K different seeds. Report the
  *best* individual energy across all K runs.

Method B — diversity-driven subspace VQE (DS-VQE):
  Train K *structurally different* circuits via Quiver (templates with
  different families and depths), each at modest budget. Solve the
  generalized eigenvalue problem in the registry-spanned subspace.
  Report the resulting ground-state energy.

Hypothesis: Method B reaches a lower energy than Method A on
Heisenberg-6q and Heisenberg-8q, because the diversity in the
registry produces partially-orthogonal error directions that the
subspace combination cancels.

We also report the "subspace lift": E_individual_best - E_subspace
on the same registry, which isolates the gain attributable to the
subspace step alone.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from quiver.ansatz import (
    BrickWall,
    HardwareEfficient,
    LinearEntangler,
    QAOAInspired,
    StronglyEntangling,
)
from quiver.backends import NumpyBackend
from quiver.hamiltonian import expectation, ground_state_energy, heisenberg
from quiver.optimizer import parameter_shift_gradient
from quiver.subspace import subspace_diagonalize


RESULTS = Path(__file__).resolve().parent.parent / "results"


def lbfgs_minimize(loss, x0, max_iter=120, ftol=1e-8, gtol=1e-6):
    result = minimize(
        loss, x0, method="L-BFGS-B",
        jac=lambda p: parameter_shift_gradient(loss, p),
        options={"maxiter": max_iter, "ftol": ftol, "gtol": gtol},
    )
    return np.asarray(result.x), float(result.fun), int(result.nfev)


def train_one(spec, backend, H, seed, max_iter=120):
    rng = np.random.default_rng(seed)
    init = rng.normal(0.0, 0.1, spec.num_params)
    loss = lambda p: float(expectation(H, backend.statevector(spec, p)))
    started = time.time()
    params, energy, nfev = lbfgs_minimize(loss, init, max_iter=max_iter)
    return {
        "params": params, "energy": energy,
        "elapsed_s": time.time() - started, "nfev": nfev,
        "state": backend.statevector(spec, params),
    }


def diverse_circuit_set(n: int) -> list:
    """Eight structurally diverse ansätze. Same n, different shapes."""
    return [
        HardwareEfficient(num_qubits=n, num_layers=2),
        HardwareEfficient(num_qubits=n, num_layers=3),
        HardwareEfficient(num_qubits=n, num_layers=4, rotation_axes=("rx", "ry")),
        LinearEntangler(num_qubits=n, num_layers=3),
        BrickWall(num_qubits=n, num_layers=3),
        BrickWall(num_qubits=n, num_layers=5),
        StronglyEntangling(num_qubits=n, num_layers=3),
        QAOAInspired(num_qubits=n, num_layers=3, ring=True),
    ]


def method_a_multirestart(n: int, num_layers: int, seeds: list[int],
                          max_iter: int = 120):
    """Method A: K seeds of one fixed ansatz (HE-d), report best."""
    H = heisenberg(n, periodic=False)
    backend = NumpyBackend(num_qubits=n)
    spec = HardwareEfficient(num_qubits=n, num_layers=num_layers).build()
    runs = [train_one(spec, backend, H, s, max_iter) for s in seeds]
    energies = [r["energy"] for r in runs]
    return {
        "energies": energies, "best": float(min(energies)),
        "total_nfev": sum(r["nfev"] for r in runs),
        "total_time_s": sum(r["elapsed_s"] for r in runs),
    }


def method_b_subspace(n: int, seeds: list[int], max_iter: int = 120):
    """Method B: K diverse circuits, then subspace EVP."""
    H = heisenberg(n, periodic=False)
    backend = NumpyBackend(num_qubits=n)
    ansätze = diverse_circuit_set(n)
    K = len(ansätze)
    # We want the same K = len(seeds). Trim or pad.
    K_use = min(K, len(seeds))
    ansätze = ansätze[:K_use]

    runs = []
    for ansatz, seed in zip(ansätze, seeds[:K_use]):
        spec = ansatz.build()
        runs.append(train_one(spec, backend, H, seed, max_iter))

    states = [r["state"] for r in runs]
    individual_energies = [r["energy"] for r in runs]

    sub = subspace_diagonalize(states, H)

    return {
        "individual_energies": individual_energies,
        "best_individual": float(min(individual_energies)),
        "subspace_energy": sub.ground_energy,
        "subspace_rank": sub.rank,
        "lift_vs_individual_min": float(min(individual_energies)) - sub.ground_energy,
        "total_nfev": sum(r["nfev"] for r in runs),
        "total_time_s": sum(r["elapsed_s"] for r in runs),
    }


def compare(n: int, num_layers_for_a: int, all_seeds: list[int], max_iter=120):
    H = heisenberg(n, periodic=False)
    e0 = ground_state_energy(H)

    a = method_a_multirestart(n, num_layers_for_a, all_seeds, max_iter=max_iter)
    b = method_b_subspace(n, all_seeds, max_iter=max_iter)

    print(f"\n--- Heisenberg-{n}q (ground = {e0:.4f}) ---", flush=True)
    print(
        f"  Method A (HE-{num_layers_for_a}L × {len(all_seeds)} seeds): "
        f"best E = {a['best']:+.4f}  gap = {a['best']-e0:.4f}  nfev = {a['total_nfev']}",
        flush=True,
    )
    print(
        f"  Method B (8 diverse circuits + subspace EVP): "
        f"individual best = {b['best_individual']:+.4f}  gap = {b['best_individual']-e0:.4f}",
        flush=True,
    )
    print(
        f"             subspace E       = {b['subspace_energy']:+.4f}  "
        f"gap = {b['subspace_energy']-e0:.4f}  rank = {b['subspace_rank']}/{len(all_seeds)}",
        flush=True,
    )
    print(
        f"             subspace lift    = {b['lift_vs_individual_min']:.4f}  "
        f"(individual_min - subspace)",
        flush=True,
    )
    print(
        f"  ΔE (A − B subspace) = {a['best'] - b['subspace_energy']:+.4f}  "
        f"(positive = subspace wins)",
        flush=True,
    )

    return {
        "n": n, "ground_energy": e0,
        "method_a_best": a["best"],
        "method_b_individual_best": b["best_individual"],
        "method_b_subspace": b["subspace_energy"],
        "subspace_lift": b["lift_vs_individual_min"],
        "delta_a_minus_b_subspace": a["best"] - b["subspace_energy"],
        "method_a": a, "method_b": b,
    }


def main():
    seeds = [42, 13, 7, 19, 31, 23, 1, 100]   # 8 seeds → 8 diverse circuits
    studies = [
        compare(n=4, num_layers_for_a=3, all_seeds=seeds, max_iter=120),
        compare(n=6, num_layers_for_a=3, all_seeds=seeds, max_iter=120),
        compare(n=8, num_layers_for_a=3, all_seeds=seeds, max_iter=150),
    ]

    print("\n=== summary ===", flush=True)
    print(
        f"{'problem':<18} {'A best':>14} {'B best ind.':>14} "
        f"{'B subspace':>14} {'A − B subsp.':>14} {'subsp lift':>12}",
        flush=True,
    )
    for s in studies:
        print(
            f"Heisenberg-{s['n']}q       "
            f"{s['method_a_best']:>+12.4f}  {s['method_b_individual_best']:>+12.4f}  "
            f"{s['method_b_subspace']:>+12.4f}  {s['delta_a_minus_b_subspace']:>+12.4f}  "
            f"{s['subspace_lift']:>+10.4f}",
            flush=True,
        )

    out = RESULTS / "subspace_vs_single.json"
    out.write_text(json.dumps(studies, indent=2, default=str))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
