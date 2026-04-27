"""H₂ molecule benchmark — head-to-head against published baselines.

The H₂ molecule (STO-3G basis, 2-qubit Bravyi-Kitaev reduced form) at
the equilibrium bond distance R = 0.7414 Å is THE canonical small-
molecule VQE benchmark used in:

  • O'Malley et al., PRX 6, 031007 (2016) — first-principles VQE
  • Kandala et al., Nature 549, 242 (2017) — hardware-efficient ansatz
                                              experimental landmark
  • Hempel et al., PRX 8, 031022 (2018) — trapped-ion implementation
  • dozens of follow-ups

The published bar:
  • Exact ground (full CI / FCI):   E₀ = -1.857275 Ha (electronic)
  • Chemical accuracy threshold:    1.6 milli-Hartree (1.6 × 10⁻³ Ha)
  • Anything within chemical accuracy is considered a "solved" molecule

The Kandala et al. 2017 paper demonstrated that hardware-efficient
ansatz at low depth can reach chemical accuracy on H₂ in noiseless
simulation; their landmark result was reaching it on real hardware.

Our claim under test on this benchmark:
  Diversity-driven subspace VQE (DS-VQE) reaches chemical accuracy
  with a small ensemble of structurally diverse 2-qubit circuits,
  with the same total compute as 8 multi-restart runs of the standard
  hardware-efficient ansatz.

For 2 qubits, the comparison is mostly a sanity check (Hilbert space
dim 4 leaves little room for ansatz diversity to matter), so we also
report the previously-established result on Heisenberg-8q where the
diversity advantage is clearer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from quiver.ansatz import (
    AllToAll,
    BrickWall,
    HardwareEfficient,
    LinearEntangler,
    QAOAInspired,
    StronglyEntangling,
)
from quiver.backends import NumpyBackend
from quiver.hamiltonian import expectation
from quiver.molecules import h2_2qubit_bk
from quiver.optimizer import parameter_shift_gradient
from quiver.subspace import subspace_diagonalize


RESULTS = Path(__file__).resolve().parent.parent / "results"


def lbfgs(loss, x0, max_iter=200, ftol=1e-10, gtol=1e-8):
    res = minimize(
        loss, x0, method="L-BFGS-B",
        jac=lambda p: parameter_shift_gradient(loss, p),
        options={"maxiter": max_iter, "ftol": ftol, "gtol": gtol},
    )
    return np.asarray(res.x), float(res.fun), int(res.nfev)


def train_one_circuit(spec, backend, H, seed, max_iter=200):
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


def diverse_h2_circuits(n: int = 2):
    """Eight structurally diverse circuits compatible with n=2."""
    return [
        HardwareEfficient(num_qubits=n, num_layers=1),
        HardwareEfficient(num_qubits=n, num_layers=2),
        HardwareEfficient(num_qubits=n, num_layers=2, rotation_axes=("rx", "ry")),
        LinearEntangler(num_qubits=n, num_layers=2),
        BrickWall(num_qubits=n, num_layers=2),
        BrickWall(num_qubits=n, num_layers=4),
        StronglyEntangling(num_qubits=n, num_layers=2),
        QAOAInspired(num_qubits=n, num_layers=3, ring=False),
    ]


def method_a_he_multirestart(H, num_layers: int, seeds: list[int]):
    """Method A — the Kandala-style baseline: hardware-efficient ansatz
    at fixed depth, multi-restart with K different seeds, report best."""
    backend = NumpyBackend(num_qubits=2)
    spec = HardwareEfficient(num_qubits=2, num_layers=num_layers).build()
    runs = [train_one_circuit(spec, backend, H, s) for s in seeds]
    energies = [r["energy"] for r in runs]
    return {
        "method": f"HE-{num_layers}L × {len(seeds)} seeds",
        "best_energy": float(min(energies)),
        "energies": energies,
        "total_nfev": sum(r["nfev"] for r in runs),
        "total_time_s": sum(r["elapsed_s"] for r in runs),
        "num_params_per_circuit": spec.num_params,
    }


def method_b_dsvqe(H, seeds: list[int]):
    """Method B — DS-VQE: structurally diverse circuits + subspace EVP."""
    backend = NumpyBackend(num_qubits=2)
    ansätze = diverse_h2_circuits(n=2)[: len(seeds)]
    runs = []
    for ansatz, seed in zip(ansätze, seeds):
        spec = ansatz.build()
        runs.append(train_one_circuit(spec, backend, H, seed))

    states = [r["state"] for r in runs]
    individual_energies = [r["energy"] for r in runs]
    sub = subspace_diagonalize(states, H)
    return {
        "method": f"DS-VQE: {len(runs)} diverse circuits + subspace EVP",
        "individual_energies": individual_energies,
        "best_individual": float(min(individual_energies)),
        "subspace_energy": sub.ground_energy,
        "subspace_rank": sub.rank,
        "total_nfev": sum(r["nfev"] for r in runs),
        "total_time_s": sum(r["elapsed_s"] for r in runs),
    }


def main():
    bench = h2_2qubit_bk()
    H = bench["H"]
    e0 = bench["ground_energy"]
    chem_acc = bench["chemical_accuracy_Ha"]

    print(f"\n# H₂ (STO-3G, 2-qubit BK) at R = {bench['R_angstroms']} Å", flush=True)
    print(f"  citation: {bench['citation']}", flush=True)
    print(f"  exact electronic E₀ = {e0:.6f} Ha", flush=True)
    print(f"  total ground (incl. nuclear repulsion) = {bench['total_ground']:.6f} Ha", flush=True)
    print(f"  chemical-accuracy threshold = {chem_acc:.4f} Ha\n", flush=True)

    seeds = [42, 13, 7, 19, 31, 23, 1, 100]   # 8 seeds → 8 diverse circuits

    # ---- Method A baselines: HE at depths 1, 2, 3 ----
    print("# Method A — hardware-efficient ansatz, multi-restart (Kandala-style)", flush=True)
    a_results = {}
    for depth in (1, 2, 3):
        r = method_a_he_multirestart(H, num_layers=depth, seeds=seeds)
        gap = r["best_energy"] - e0
        passes = gap < chem_acc
        a_results[f"HE-{depth}L"] = r
        print(
            f"  HE-{depth}L (params={r['num_params_per_circuit']}) × 8 seeds → "
            f"best E = {r['best_energy']:+.6f} Ha   gap = {gap:.2e} Ha   "
            f"chemical_accuracy {'PASS' if passes else 'FAIL'}",
            flush=True,
        )

    # ---- Method B: DS-VQE ----
    print("\n# Method B — DS-VQE (diversity-driven subspace VQE)", flush=True)
    b = method_b_dsvqe(H, seeds=seeds)
    gap_individual = b["best_individual"] - e0
    gap_subspace = b["subspace_energy"] - e0
    print(
        f"  individual best     E = {b['best_individual']:+.6f} Ha   "
        f"gap = {gap_individual:.2e} Ha", flush=True,
    )
    print(
        f"  subspace EVP        E = {b['subspace_energy']:+.6f} Ha   "
        f"gap = {gap_subspace:.2e} Ha   "
        f"chemical_accuracy {'PASS' if gap_subspace < chem_acc else 'FAIL'}",
        flush=True,
    )
    print(f"  subspace rank = {b['subspace_rank']}/{len(seeds)}", flush=True)

    # ---- Summary ----
    print("\n# Summary", flush=True)
    print(f"  exact ground:                 {e0:+.6f} Ha", flush=True)
    print(f"  chemical-accuracy threshold:  {chem_acc:.6f} Ha", flush=True)
    for name, r in a_results.items():
        gap = r["best_energy"] - e0
        print(
            f"  {name + ' multi-restart':<28}  best = {r['best_energy']:+.6f} Ha  "
            f"gap = {gap:.2e} Ha  "
            f"({'PASS' if gap < chem_acc else 'FAIL'})",
            flush=True,
        )
    print(
        f"  {'DS-VQE subspace':<28}  best = {b['subspace_energy']:+.6f} Ha  "
        f"gap = {gap_subspace:.2e} Ha  "
        f"({'PASS' if gap_subspace < chem_acc else 'FAIL'})",
        flush=True,
    )

    # Persist
    out = {
        "benchmark": "H2 STO-3G 2-qubit BK at R=0.7414 A",
        "citation": bench["citation"],
        "exact_electronic_ground": e0,
        "chemical_accuracy_Ha": chem_acc,
        "method_a": {k: {sk: sv for sk, sv in r.items() if sk != "energies"}
                     for k, r in a_results.items()},
        "method_a_energies": {k: r["energies"] for k, r in a_results.items()},
        "method_b": {sk: sv for sk, sv in b.items() if sk != "individual_energies"},
        "method_b_individual_energies": b["individual_energies"],
    }
    fp = RESULTS / "h2_published_benchmark.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {fp}", flush=True)


if __name__ == "__main__":
    main()
