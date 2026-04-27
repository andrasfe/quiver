"""Adiabatic schedule vs direct VQE — does annealing close the gap to
chemical-accuracy thresholds where direct gradient-based VQE plateaus?

Both methods use the SAME circuit (HardwareEfficient with d layers) and
the SAME final optimizer (L-BFGS-B + parameter-shift). The only
difference is whether the loss function is the target Hamiltonian
directly, or an adiabatic schedule

    H(s) = (1 − s) · H_easy + s · H_target

with H_easy = −∑_i X_i (whose ground state is |+⟩^n, prepared by H on
every qubit). For the schedule we optimise at each of `num_steps` values
of s, warm-starting from the previous step's solution.

Three problems × multiple seeds. We report best energy reached, and
the *gap to ground* in absolute and relative terms. The publishable
claim form: for tight thresholds (gap < 0.1) on n=6, n=8 Heisenberg,
adiabatic significantly outperforms direct.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from quiver.adiabatic import adiabatic_train, easy_x_field_hamiltonian
from quiver.ansatz import HardwareEfficient
from quiver.backends import NumpyBackend
from quiver.hamiltonian import expectation, ground_state_energy, heisenberg
from quiver.optimizer import parameter_shift_gradient


RESULTS = Path(__file__).resolve().parent.parent / "results"


def lbfgs_minimize(loss, x0, max_iter=80, ftol=1e-8, gtol=1e-6):
    result = minimize(
        loss, x0, method="L-BFGS-B",
        jac=lambda p: parameter_shift_gradient(loss, p),
        options={"maxiter": max_iter, "ftol": ftol, "gtol": gtol},
    )
    return np.asarray(result.x), float(result.fun), int(result.nfev)


def run_direct(spec, backend, H_target, init_params, max_iter=120):
    loss = lambda p: float(expectation(H_target, backend.statevector(spec, p)))
    started = time.time()
    params, energy, nfev = lbfgs_minimize(loss, init_params, max_iter=max_iter)
    return {
        "energy": energy, "params": params.tolist(),
        "elapsed_s": time.time() - started, "nfev": nfev,
    }


def run_adiabatic(spec, backend, H_target, H_easy, init_params, num_steps=10,
                  per_step_max_iter=60):
    started = time.time()
    final_params, trace = adiabatic_train(
        spec, backend, H_easy, H_target,
        num_steps=num_steps,
        optimize_at_step=lambda loss, x0: lbfgs_minimize(
            loss, x0, max_iter=per_step_max_iter, ftol=1e-9, gtol=1e-7,
        ),
        initial_params=init_params,
    )
    return {
        "energy": trace.energy[-1],
        "params": final_params.tolist(),
        "elapsed_s": time.time() - started,
        "nfev": trace.nfev_total,
        "trace_s": trace.s,
        "trace_energy": trace.energy,
    }


def compare(n: int, num_layers: int, seeds: list[int], adiabatic_steps: int = 10):
    H = heisenberg(n, periodic=False)
    H_easy = easy_x_field_hamiltonian(n)
    e0 = ground_state_energy(H)
    spec = HardwareEfficient(num_qubits=n, num_layers=num_layers).build()
    backend = NumpyBackend(num_qubits=n)

    rows: list[dict] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        init = rng.normal(0.0, 0.1, spec.num_params)

        direct = run_direct(spec, backend, H, init.copy())
        adiab = run_adiabatic(spec, backend, H, H_easy, init.copy(),
                              num_steps=adiabatic_steps)
        rows.append({
            "seed": seed,
            "direct": direct,
            "adiabatic": adiab,
            "ground": e0,
            "direct_gap": direct["energy"] - e0,
            "adiab_gap": adiab["energy"] - e0,
        })
        print(
            f"  n={n} L={num_layers} seed={seed:>3} | "
            f"direct E={direct['energy']:+.4f} (gap {direct['energy']-e0:.4f}) | "
            f"adiab E={adiab['energy']:+.4f} (gap {adiab['energy']-e0:.4f}) | "
            f"adiab/direct nfev = {adiab['nfev']}/{direct['nfev']}",
            flush=True,
        )

    direct_gaps = [r["direct_gap"] for r in rows]
    adiab_gaps = [r["adiab_gap"] for r in rows]
    return {
        "problem": f"Heisenberg-{n}q",
        "ansatz": f"HE-{num_layers}L",
        "ground_energy": e0,
        "spec_num_params": spec.num_params,
        "direct_gap_mean": float(np.mean(direct_gaps)),
        "direct_gap_std": float(np.std(direct_gaps)),
        "adiab_gap_mean": float(np.mean(adiab_gaps)),
        "adiab_gap_std": float(np.std(adiab_gaps)),
        "improvement_mean": float(np.mean(direct_gaps) - np.mean(adiab_gaps)),
        "rows": rows,
    }


def main():
    seeds = [42, 13, 7, 19, 31]   # 5 seeds for tighter error bars
    adiabatic_steps = 12

    studies = [
        compare(n=4, num_layers=2, seeds=seeds, adiabatic_steps=adiabatic_steps),
        compare(n=6, num_layers=3, seeds=seeds, adiabatic_steps=adiabatic_steps),
        compare(n=8, num_layers=3, seeds=seeds, adiabatic_steps=adiabatic_steps),
    ]

    print("\n=== summary ===", flush=True)
    print(f"{'problem':<18} {'ansatz':<8} {'direct gap':>20} {'adiab gap':>20} {'Δ':>10}",
          flush=True)
    for st in studies:
        d = st["direct_gap_mean"]; ds = st["direct_gap_std"]
        a = st["adiab_gap_mean"]; asd = st["adiab_gap_std"]
        print(
            f"{st['problem']:<18} {st['ansatz']:<8} "
            f"{d:>10.4f} ± {ds:>5.4f}     "
            f"{a:>10.4f} ± {asd:>5.4f}     "
            f"{d - a:>+8.4f}",
            flush=True,
        )

    out = RESULTS / "adiabatic_vs_direct.json"
    out.write_text(json.dumps({"studies": studies, "seeds": seeds,
                               "adiabatic_steps": adiabatic_steps},
                              indent=2, default=str))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
