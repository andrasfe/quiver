"""Noisy Heisenberg benchmark — the regime where DS-VQE should win.

Unlike portfolio QUBOs (diagonal in computational basis), the AFM
Heisenberg ground state is an entangled superposition. Depolarising
noise destroys the off-diagonal coherences the ideal state needs;
training under noise pushes parameters toward states that look low
under noise but aren't actually close to the ideal ground.

If DS-VQE's subspace expansion provides implicit noise mitigation by
combining diverse-circuit ideal states (each with different noise-
induced parameter biases), the combined output should reach a lower
energy than the best individual ansatz under matched noisy training.

Method A — HE-3L × 8 seeds, trained under noise.
Method B — 8 diverse circuits, trained under noise, subspace EVP on
           ideal-state evaluation of trained parameters.
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
from quiver.noise import NoiseModel, NoisyDensityBackend
from quiver.optimizer import parameter_shift_gradient
from quiver.subspace import subspace_diagonalize


RESULTS = Path(__file__).resolve().parent.parent / "results"


def lbfgs(loss, x0, max_iter=80, ftol=1e-7, gtol=1e-5):
    res = minimize(
        loss, x0, method="L-BFGS-B",
        jac=lambda p: parameter_shift_gradient(loss, p),
        options={"maxiter": max_iter, "ftol": ftol, "gtol": gtol},
    )
    return np.asarray(res.x), float(res.fun), int(res.nfev)


def train_noisy(spec, noisy_backend, ideal_backend, H, seed, max_iter=80):
    rng = np.random.default_rng(seed)
    init = rng.normal(0.0, 0.1, spec.num_params)
    noisy_loss = lambda p: noisy_backend.expectation(H, spec, p)
    started = time.time()
    params, noisy_E, nfev = lbfgs(noisy_loss, init, max_iter=max_iter)
    elapsed = time.time() - started
    ideal_state = ideal_backend.statevector(spec, params)
    ideal_E = float(expectation(H, ideal_state))
    return {
        "params": params, "noisy_energy": noisy_E,
        "ideal_energy": ideal_E, "ideal_state": ideal_state,
        "elapsed_s": elapsed, "nfev": nfev,
    }


def he_baseline(n, depth, seeds, H, noise_p):
    ideal = NumpyBackend(num_qubits=n)
    noisy = NoisyDensityBackend(
        num_qubits=n, noise=NoiseModel(p_2q_per_qubit=noise_p),
    )
    spec = HardwareEfficient(num_qubits=n, num_layers=depth).build()
    runs = [train_noisy(spec, noisy, ideal, H, s) for s in seeds]
    return {
        "best_ideal": float(min(r["ideal_energy"] for r in runs)),
        "all_ideal_energies": [r["ideal_energy"] for r in runs],
        "all_noisy_energies": [r["noisy_energy"] for r in runs],
        "params_per_circuit": spec.num_params,
    }


def dsvqe(n, seeds, H, noise_p):
    ideal = NumpyBackend(num_qubits=n)
    noisy = NoisyDensityBackend(
        num_qubits=n, noise=NoiseModel(p_2q_per_qubit=noise_p),
    )
    ansätze = [
        HardwareEfficient(num_qubits=n, num_layers=2),
        HardwareEfficient(num_qubits=n, num_layers=3),
        HardwareEfficient(num_qubits=n, num_layers=2, rotation_axes=("rx", "ry")),
        LinearEntangler(num_qubits=n, num_layers=2),
        BrickWall(num_qubits=n, num_layers=2),
        BrickWall(num_qubits=n, num_layers=3),
        StronglyEntangling(num_qubits=n, num_layers=2),
        QAOAInspired(num_qubits=n, num_layers=2, ring=True),
    ][: len(seeds)]
    runs = []
    for ansatz, seed in zip(ansätze, seeds):
        spec = ansatz.build()
        runs.append(train_noisy(spec, noisy, ideal, H, seed))
    states = [r["ideal_state"] for r in runs]
    individual_ideal = [r["ideal_energy"] for r in runs]
    sub = subspace_diagonalize(states, H)
    return {
        "individual_ideal_energies": individual_ideal,
        "best_individual_ideal": float(min(individual_ideal)),
        "subspace_energy": sub.ground_energy,
        "subspace_rank": sub.rank,
    }


def main():
    n = 4
    H = heisenberg(n, periodic=False)
    e0 = ground_state_energy(H)
    seeds = [42, 13, 7, 19, 31, 23, 1, 100]

    print(f"# Noisy Heisenberg-{n}q benchmark", flush=True)
    print(f"#   exact ground E₀ = {e0:.6f}", flush=True)
    print(f"#   8 seeds, L-BFGS-B + parameter-shift, density-matrix sim", flush=True)
    print(f"#   noise on 2q gates only", flush=True)

    studies = []
    for p in (0.0, 0.005, 0.01, 0.02, 0.05):
        a = he_baseline(n, depth=3, seeds=seeds, H=H, noise_p=p)
        b = dsvqe(n, seeds=seeds, H=H, noise_p=p)
        a_gap = a["best_ideal"] - e0
        b_indiv_gap = b["best_individual_ideal"] - e0
        b_sub_gap = b["subspace_energy"] - e0
        studies.append({
            "noise_p": p,
            "a_best_ideal": a["best_ideal"],
            "b_best_individual": b["best_individual_ideal"],
            "b_subspace": b["subspace_energy"],
            "a_gap": a_gap,
            "b_individual_gap": b_indiv_gap,
            "b_subspace_gap": b_sub_gap,
            "a_full": a,
            "b_full": {**{k: v for k, v in b.items() if k != "individual_ideal_energies"},
                       "individual_ideal_energies": b["individual_ideal_energies"]},
        })
        print(
            f"\n  p_2q = {p:.4f}", flush=True,
        )
        print(
            f"    A: HE-3L × 8 seeds      ideal best E = {a['best_ideal']:+.6f}  "
            f"gap = {a_gap:+.6f}", flush=True,
        )
        print(
            f"    B: DS-VQE indiv. best     ideal     E = {b['best_individual_ideal']:+.6f}  "
            f"gap = {b_indiv_gap:+.6f}", flush=True,
        )
        print(
            f"       DS-VQE subspace EVP        E = {b['subspace_energy']:+.6f}  "
            f"gap = {b_sub_gap:+.6f}", flush=True,
        )

    print("\n" + "=" * 78, flush=True)
    print("GAP TABLE (smaller is better; H_2q acting on entangled ground state)",
          flush=True)
    print("=" * 78, flush=True)
    print(f"{'p_2q':>10}   {'A: HE-3L gap':>14}   {'B: indiv.':>12}   "
          f"{'B: subsp.':>12}   {'subsp/A':>9}", flush=True)
    print("-" * 78, flush=True)
    for s in studies:
        ratio = s["b_subspace_gap"] / s["a_gap"] if abs(s["a_gap"]) > 1e-9 else float("nan")
        print(
            f"{s['noise_p']:>10.4f}   {s['a_gap']:>14.6f}   "
            f"{s['b_individual_gap']:>12.6f}   {s['b_subspace_gap']:>12.6f}   "
            f"{ratio:>9.3f}", flush=True,
        )
    print("=" * 78, flush=True)
    print("(subsp/A < 1 means DS-VQE subspace is closer to ground than HE-3L)",
          flush=True)

    out = RESULTS / "noisy_heisenberg_benchmark.json"
    out.write_text(json.dumps(studies, indent=2, default=str))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
