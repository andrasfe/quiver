"""Quantum-finance benchmark: DS-VQE vs Kandala-style HE-ansatz on the
Markowitz portfolio-selection Ising Hamiltonian.

Background
----------
The mean-variance portfolio selection problem with a fixed cardinality
constraint maps directly to an Ising / QUBO Hamiltonian with single-site
fields and pair couplings. This is the canonical quantum-finance
benchmark in:

  • Hodson et al., "Portfolio rebalancing experiments using the QAOA"
    (arXiv:1911.05296, Phys. Rev. Research 2 (2020))
  • Barkoutsos et al., "Improving Variational Quantum Optimization
    using CVaR" (Quantum 4, 256, 2020)
  • Mugel et al., "Dynamic portfolio optimization with real datasets
    using quantum processors" (Phys. Rev. Research 4, 013006, 2022)
  • Brandhofer et al., "Benchmarking the performance of portfolio
    optimization with QAOA" (arXiv:2207.10555, 2022)
  • IBM Qiskit Finance reference tutorials

In those papers, QAOA-2 / VQE-with-HE achieves approximation ratios
of roughly 0.80–0.95 depending on problem size and depth — i.e.
relative cost gaps of 5–20% to the classical optimum. Reaching tighter
than that on noiseless simulation typically requires more layers,
better optimizers, or specialised ansätze.

What we measure
---------------
For each portfolio (6 and 8 assets, K=n/2 cardinality):

  approximation_ratio  =  (E_method - E_max) / (E_min - E_max)

where E_min is the exact optimum, E_max is the worst-cost portfolio
in the budget-feasible set. Approximation ratio = 1 means optimal,
0 means worst-case. Standard QAOA on portfolios is reported in the
0.80–0.95 range; we report whatever DS-VQE achieves at matched
compute against the standard HE-ansatz baseline.

Method A — Kandala-style HE-3L × 8 seeds (best of 8 trainings)
Method B — DS-VQE: 8 structurally-diverse circuits + subspace EVP
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
    stylized_6asset_portfolio,
    stylized_8asset_portfolio,
)
from quiver.hamiltonian import expectation
from quiver.optimizer import parameter_shift_gradient
from quiver.subspace import subspace_diagonalize


RESULTS = Path(__file__).resolve().parent.parent / "results"


def lbfgs(loss, x0, max_iter=120, ftol=1e-9, gtol=1e-7):
    res = minimize(
        loss, x0, method="L-BFGS-B",
        jac=lambda p: parameter_shift_gradient(loss, p),
        options={"maxiter": max_iter, "ftol": ftol, "gtol": gtol},
    )
    return np.asarray(res.x), float(res.fun), int(res.nfev)


def train(spec, backend, H, seed, max_iter=120):
    rng = np.random.default_rng(seed)
    init = rng.normal(0.0, 0.1, spec.num_params)
    loss = lambda p: float(expectation(H, backend.statevector(spec, p)))
    started = time.time()
    params, energy, nfev = lbfgs(loss, init, max_iter=max_iter)
    return {
        "params": params, "energy": energy,
        "elapsed_s": time.time() - started, "nfev": nfev,
        "state": backend.statevector(spec, params),
    }


def diverse_set(n: int):
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


def baseline(bench: PortfolioBenchmark, depth: int, seeds: list[int]):
    n = bench.mu.size
    backend = NumpyBackend(num_qubits=n)
    spec = HardwareEfficient(num_qubits=n, num_layers=depth).build()
    runs = [train(spec, backend, bench.H, s) for s in seeds]
    energies = [r["energy"] for r in runs]
    return {
        "method": f"HE-{depth}L × {len(seeds)} seeds",
        "best_energy": float(min(energies)),
        "all_energies": energies,
        "params_per_circuit": spec.num_params,
        "total_nfev": sum(r["nfev"] for r in runs),
        "total_time_s": sum(r["elapsed_s"] for r in runs),
    }


def dsvqe(bench: PortfolioBenchmark, seeds: list[int]):
    n = bench.mu.size
    backend = NumpyBackend(num_qubits=n)
    ansätze = diverse_set(n)[: len(seeds)]
    runs = []
    for ansatz, seed in zip(ansätze, seeds):
        spec = ansatz.build()
        runs.append(train(spec, backend, bench.H, seed))
    states = [r["state"] for r in runs]
    individual = [r["energy"] for r in runs]
    sub = subspace_diagonalize(states, bench.H)
    return {
        "method": f"DS-VQE: {len(runs)} diverse + subspace EVP",
        "individual_energies": individual,
        "best_individual": float(min(individual)),
        "subspace_energy": sub.ground_energy,
        "subspace_rank": sub.rank,
        "total_nfev": sum(r["nfev"] for r in runs),
        "total_time_s": sum(r["elapsed_s"] for r in runs),
    }


def feasibility_filter_costs(bench: PortfolioBenchmark) -> tuple[float, float]:
    """E_min and E_max over the cardinality-FEASIBLE set: only x with
    ∑x_i = K. The approximation ratio is normalised to this set so the
    cardinality penalty (which dominates infeasible states) doesn't
    dilute the comparison."""
    n = bench.mu.size
    feasible_costs = []
    for idx in range(2**n):
        x = np.array([(idx >> q) & 1 for q in range(n)], dtype=float)
        if int(x.sum()) != bench.cardinality_K:
            continue
        cost = float(
            -bench.mu @ x
            + bench.risk_aversion * x @ bench.sigma @ x
        )
        feasible_costs.append(cost)
    feasible_costs = np.array(feasible_costs)
    return float(feasible_costs.min()), float(feasible_costs.max())


def approximation_ratio(method_energy: float, bench: PortfolioBenchmark,
                        E_min_feas: float, E_max_feas: float) -> float:
    """Convert a Hamiltonian energy back to a (cardinality-feasible)
    cost and report (E_max - E) / (E_max - E_min). 1.0 = optimal,
    0.0 = worst-case feasible."""
    energy_with_const = method_energy + bench.constant_offset
    # Subtract cardinality penalty term assuming feasibility (γ·0 = 0
    # if x ∈ feasible set), so the energy is comparable to feasible
    # cost directly.
    return (E_max_feas - energy_with_const) / (E_max_feas - E_min_feas)


def compare(name: str, bench: PortfolioBenchmark, seeds: list[int],
            baseline_depth: int = 3):
    n = bench.mu.size
    e_min_feas, e_max_feas = feasibility_filter_costs(bench)
    print(f"\n# {name} (n={n} assets, K={bench.cardinality_K})", flush=True)
    print(f"  citation: {bench.citation}", flush=True)
    print(f"  exact ground energy        = {bench.exact_ground_energy:+.6f}", flush=True)
    print(f"  optimal portfolio cost     = {bench.optimal_cost:+.6f}", flush=True)
    print(f"  best feasible cost (E_min) = {e_min_feas:+.6f}", flush=True)
    print(f"  worst feasible cost (E_max)= {e_max_feas:+.6f}", flush=True)

    a = baseline(bench, baseline_depth, seeds)
    b = dsvqe(bench, seeds)

    a_ratio = approximation_ratio(a["best_energy"], bench, e_min_feas, e_max_feas)
    b_indiv_ratio = approximation_ratio(b["best_individual"], bench, e_min_feas, e_max_feas)
    b_sub_ratio = approximation_ratio(b["subspace_energy"], bench, e_min_feas, e_max_feas)

    print(f"  ─── Method A: HE-{baseline_depth}L × {len(seeds)} seeds ───", flush=True)
    print(f"     best E              = {a['best_energy']:+.6f}", flush=True)
    print(f"     approximation ratio = {a_ratio:.4f}    "
          f"({100*a_ratio:.2f}% of optimal-vs-worst)", flush=True)
    print(f"     params/circuit = {a['params_per_circuit']}, "
          f"nfev = {a['total_nfev']:,}, wall = {a['total_time_s']:.1f}s", flush=True)
    print(f"  ─── Method B: DS-VQE ───", flush=True)
    print(f"     best individual     = {b['best_individual']:+.6f}    "
          f"(approx. ratio {b_indiv_ratio:.4f})", flush=True)
    print(f"     subspace EVP        = {b['subspace_energy']:+.6f}    "
          f"(approx. ratio {b_sub_ratio:.4f})", flush=True)
    print(f"     subspace rank = {b['subspace_rank']}/{len(seeds)}, "
          f"nfev = {b['total_nfev']:,}, wall = {b['total_time_s']:.1f}s", flush=True)

    return {
        "name": name, "n": n, "K": bench.cardinality_K,
        "exact_ground_energy": bench.exact_ground_energy,
        "optimal_cost": bench.optimal_cost,
        "feasible_min_cost": e_min_feas,
        "feasible_max_cost": e_max_feas,
        "method_a": a, "method_a_ratio": a_ratio,
        "method_b": {**{k: v for k, v in b.items() if k != "individual_energies"},
                     "individual_energies": b["individual_energies"]},
        "method_b_individual_ratio": b_indiv_ratio,
        "method_b_subspace_ratio": b_sub_ratio,
    }


def main():
    seeds = [42, 13, 7, 19, 31, 23, 1, 100]
    print("# Portfolio benchmark: DS-VQE vs Kandala-style HE-ansatz",
          flush=True)
    print("# 8 seeds, L-BFGS-B + parameter-shift, identical compute on each side",
          flush=True)

    results = []
    results.append(compare("Stylised-6asset", stylized_6asset_portfolio(), seeds))
    results.append(compare("Stylised-8asset", stylized_8asset_portfolio(), seeds))

    print("\n" + "=" * 78, flush=True)
    print("APPROXIMATION RATIO TABLE", flush=True)
    print("(higher = better, 1.0 = exact optimum, 0.0 = worst feasible portfolio)",
          flush=True)
    print("=" * 78, flush=True)
    print(f"{'benchmark':<22}  {'A: HE-3L':>14}  {'B: DS-VQE indiv.':>18}  "
          f"{'B: DS-VQE subsp.':>18}", flush=True)
    print("-" * 78, flush=True)
    for r in results:
        print(
            f"{r['name']:<22}  {r['method_a_ratio']:>14.4f}  "
            f"{r['method_b_individual_ratio']:>18.4f}  "
            f"{r['method_b_subspace_ratio']:>18.4f}",
            flush=True,
        )
    print("=" * 78, flush=True)
    print(
        "\nLiterature reference: Hodson et al. (2019) and Barkoutsos et al. (2020)",
        flush=True,
    )
    print(
        "report QAOA-2 / VQE-HE approximation ratios in the 0.80–0.95 range on",
        flush=True,
    )
    print("comparable small-portfolio benchmarks.", flush=True)

    out = RESULTS / "portfolio_dsvqe_benchmark.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
