"""Noisy portfolio benchmark — DS-VQE vs Kandala-style HE-ansatz under
realistic depolarising noise.

This is the regime where Hodson et al. (2019), Barkoutsos et al. (2020),
and Mugel et al. (2022) report 0.80–0.95 approximation ratios. Noiseless
gradient-based VQE solves portfolio QUBOs trivially (see
problems/portfolio_dsvqe_benchmark.py); under noise, the optimizer
biases toward parameters that look better in the noisy expectation but
correspond to a different state than the ideal optimum.

Hypothesis under test: DS-VQE's diversity acts as implicit noise
mitigation. Each diverse circuit's noise pattern partially cancels
under the subspace generalized eigenvalue solve when the basis states
are structurally distinct.

Method A — Kandala-style HE-3L × 8 seeds, trained under noise.
Method B — DS-VQE with 8 diverse circuits, trained under noise, with
subspace EVP applied to the *ideal* (noiseless) statevectors at the
trained parameters. The training adapts to noise; the post-processing
subspace step is performed without noise (matching the standard
"variational state preparation, classical post-processing" workflow).

We report:
  • noisy training energy (what the optimizer minimises)
  • ideal energy at the trained parameters (the "intended" state)
  • subspace energy (DS-VQE's combined output)
  • approximation ratio against the cardinality-feasible classical
    optimum

Compute is matched: 8 trainings each side, same noise level, same seeds.
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
from quiver.finance import (
    PortfolioBenchmark,
    build_portfolio,
)
from quiver.hamiltonian import expectation
from quiver.noise import NoiseModel, NoisyDensityBackend
from quiver.optimizer import parameter_shift_gradient
from quiver.subspace import subspace_diagonalize


RESULTS = Path(__file__).resolve().parent.parent / "results"


def stylized_4asset(seed: int = 42) -> PortfolioBenchmark:
    """4-asset stylised portfolio. Small enough that noisy density
    matrix simulation is fast (16×16 ρ)."""
    rng = np.random.default_rng(seed)
    n = 4
    mu = 0.05 + 0.10 * rng.random(n)
    L = 0.05 * rng.standard_normal((n, n))
    sigma = L @ L.T + 0.01 * np.eye(n)
    return build_portfolio(
        mu=mu, sigma=sigma, K=2, risk_aversion=0.5, cardinality_penalty=2.0,
        citation="4-asset stylised portfolio for noisy benchmark.",
    )


def lbfgs(loss, x0, max_iter=80, ftol=1e-7, gtol=1e-5):
    res = minimize(
        loss, x0, method="L-BFGS-B",
        jac=lambda p: parameter_shift_gradient(loss, p),
        options={"maxiter": max_iter, "ftol": ftol, "gtol": gtol},
    )
    return np.asarray(res.x), float(res.fun), int(res.nfev)


def train_noisy(spec, noisy_backend, ideal_backend, H, seed, max_iter=80):
    """Train under noise; record both noisy and ideal energies at the
    final parameters."""
    rng = np.random.default_rng(seed)
    init = rng.normal(0.0, 0.1, spec.num_params)
    noisy_loss = lambda p: noisy_backend.expectation(H, spec, p)

    started = time.time()
    params, noisy_E, nfev = lbfgs(noisy_loss, init, max_iter=max_iter)
    elapsed = time.time() - started

    # Ideal evaluation: same params, no noise.
    ideal_state = ideal_backend.statevector(spec, params)
    ideal_E = float(expectation(H, ideal_state))

    return {
        "params": params,
        "noisy_energy": noisy_E,
        "ideal_energy": ideal_E,
        "ideal_state": ideal_state,
        "elapsed_s": elapsed,
        "nfev": nfev,
    }


def baseline_he_noisy(bench: PortfolioBenchmark, depth: int, seeds: list[int],
                     noise_p: float):
    n = bench.mu.size
    ideal = NumpyBackend(num_qubits=n)
    noisy = NoisyDensityBackend(num_qubits=n, noise=NoiseModel(p_2q_per_qubit=noise_p))
    spec = HardwareEfficient(num_qubits=n, num_layers=depth).build()
    runs = [train_noisy(spec, noisy, ideal, bench.H, s) for s in seeds]
    ideal_energies = [r["ideal_energy"] for r in runs]
    noisy_energies = [r["noisy_energy"] for r in runs]
    return {
        "method": f"HE-{depth}L × {len(seeds)} seeds (noisy training)",
        "best_ideal_energy": float(min(ideal_energies)),
        "best_noisy_energy": float(min(noisy_energies)),
        "all_ideal_energies": ideal_energies,
        "all_noisy_energies": noisy_energies,
        "params_per_circuit": spec.num_params,
        "total_nfev": sum(r["nfev"] for r in runs),
        "total_time_s": sum(r["elapsed_s"] for r in runs),
    }


def dsvqe_noisy(bench: PortfolioBenchmark, seeds: list[int], noise_p: float):
    n = bench.mu.size
    ideal = NumpyBackend(num_qubits=n)
    noisy = NoisyDensityBackend(num_qubits=n, noise=NoiseModel(p_2q_per_qubit=noise_p))
    ansätze = [
        HardwareEfficient(num_qubits=n, num_layers=1),
        HardwareEfficient(num_qubits=n, num_layers=2),
        HardwareEfficient(num_qubits=n, num_layers=2, rotation_axes=("rx", "ry")),
        LinearEntangler(num_qubits=n, num_layers=2),
        BrickWall(num_qubits=n, num_layers=2),
        BrickWall(num_qubits=n, num_layers=3),
        StronglyEntangling(num_qubits=n, num_layers=2),
        QAOAInspired(num_qubits=n, num_layers=2, ring=False),
    ][: len(seeds)]

    runs = []
    for ansatz, seed in zip(ansätze, seeds):
        spec = ansatz.build()
        runs.append(train_noisy(spec, noisy, ideal, bench.H, seed))
    states = [r["ideal_state"] for r in runs]
    individual_ideal = [r["ideal_energy"] for r in runs]
    sub = subspace_diagonalize(states, bench.H)
    return {
        "method": f"DS-VQE: {len(runs)} diverse + subspace EVP (trained noisy)",
        "individual_ideal_energies": individual_ideal,
        "best_individual_ideal": float(min(individual_ideal)),
        "subspace_energy": sub.ground_energy,
        "subspace_rank": sub.rank,
        "total_nfev": sum(r["nfev"] for r in runs),
        "total_time_s": sum(r["elapsed_s"] for r in runs),
    }


def feasibility_filter(bench: PortfolioBenchmark) -> tuple[float, float]:
    n = bench.mu.size
    feasible_costs = []
    for idx in range(2**n):
        x = np.array([(idx >> q) & 1 for q in range(n)], dtype=float)
        if int(x.sum()) != bench.cardinality_K:
            continue
        cost = float(-bench.mu @ x + bench.risk_aversion * x @ bench.sigma @ x)
        feasible_costs.append(cost)
    return float(min(feasible_costs)), float(max(feasible_costs))


def approx_ratio(method_energy: float, bench: PortfolioBenchmark,
                 e_min: float, e_max: float) -> float:
    cost = method_energy + bench.constant_offset
    return (e_max - cost) / (e_max - e_min)


def compare_at_noise(bench, seeds, noise_p, baseline_depth=3):
    print(f"\n  ─── noise level p_2q = {noise_p:.3f} ───", flush=True)
    e_min, e_max = feasibility_filter(bench)

    a = baseline_he_noisy(bench, baseline_depth, seeds, noise_p)
    b = dsvqe_noisy(bench, seeds, noise_p)

    a_ratio = approx_ratio(a["best_ideal_energy"], bench, e_min, e_max)
    b_indiv_ratio = approx_ratio(b["best_individual_ideal"], bench, e_min, e_max)
    b_sub_ratio = approx_ratio(b["subspace_energy"], bench, e_min, e_max)

    print(
        f"     A: HE-{baseline_depth}L × {len(seeds)} seeds  "
        f"best ideal E = {a['best_ideal_energy']:+.6f}  "
        f"approx ratio = {a_ratio:.4f}",
        flush=True,
    )
    print(
        f"        ↳ best NOISY E (what optimizer saw) = "
        f"{a['best_noisy_energy']:+.6f}", flush=True,
    )
    print(
        f"     B: DS-VQE individual best ideal       = "
        f"{b['best_individual_ideal']:+.6f}  approx ratio = {b_indiv_ratio:.4f}",
        flush=True,
    )
    print(
        f"        subspace EVP energy                = "
        f"{b['subspace_energy']:+.6f}  approx ratio = {b_sub_ratio:.4f}",
        flush=True,
    )

    return {
        "noise_p": noise_p,
        "method_a": a, "method_a_ratio": a_ratio,
        "method_b": {**{k: v for k, v in b.items() if k != "individual_ideal_energies"},
                     "individual_ideal_energies": b["individual_ideal_energies"]},
        "method_b_individual_ratio": b_indiv_ratio,
        "method_b_subspace_ratio": b_sub_ratio,
    }


def main():
    bench = stylized_4asset()
    seeds = [42, 13, 7, 19, 31, 23, 1, 100]

    e_min, e_max = feasibility_filter(bench)
    print(f"# Noisy portfolio benchmark: 4 assets, K=2", flush=True)
    print(f"#   exact feasible optimum E_min = {e_min:+.6f}", flush=True)
    print(f"#   worst feasible          E_max = {e_max:+.6f}", flush=True)
    print(f"#   8 seeds, L-BFGS-B + parameter-shift", flush=True)
    print(f"#   density-matrix simulation, depolarising noise on 2q gates",
          flush=True)
    print(f"#   approximation ratio: 1.0 = optimal, 0.0 = worst feasible",
          flush=True)

    studies = []
    for p in (0.005, 0.01, 0.02, 0.05):
        print(f"\n=== noise level p_2q = {p:.3f} ===", flush=True)
        studies.append(compare_at_noise(bench, seeds, noise_p=p))

    # Summary
    print("\n" + "=" * 78, flush=True)
    print("APPROXIMATION-RATIO TABLE (higher is better)", flush=True)
    print("=" * 78, flush=True)
    print(f"{'p_2q':>8}   {'A: HE-3L':>14}   {'B: DS-VQE indiv.':>18}   "
          f"{'B: DS-VQE subsp.':>18}", flush=True)
    print("-" * 78, flush=True)
    for s in studies:
        print(
            f"{s['noise_p']:>8.4f}   {s['method_a_ratio']:>14.4f}   "
            f"{s['method_b_individual_ratio']:>18.4f}   "
            f"{s['method_b_subspace_ratio']:>18.4f}",
            flush=True,
        )
    print("=" * 78, flush=True)

    out = RESULTS / "noisy_portfolio_benchmark.json"
    out.write_text(json.dumps(studies, indent=2, default=str))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
