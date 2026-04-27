"""Pure-ADAPT invention demo on small state-preparation problems.

For a *state preparation* target (smooth loss landscape with respect to
gate insertions), ADAPT-style growth invents circuits whose structure has
no template lineage. This demo runs adaptive growth multiple times with
different seeds on small problems and reports the resulting structures.

Two problems:
  1. 4-qubit GHZ state    — shallowest known is depth 4 (H + 3 CNOTs)
  2. 4-qubit Bell-pair pair — |00⟩|11⟩ + |11⟩|00⟩, requires entanglement

For each, we check whether ADAPT discovers a working circuit and how its
structure compares to known templates and to each other across seeds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from quiver.adaptive import AdaptiveGrowth
from quiver.ansatz import (
    HardwareEfficient,
    LinearEntangler,
    QAOAInspired,
)
from quiver.backends import NumpyBackend
from quiver.diversity import structural_similarity
from quiver.verification import fidelity_objective


def ghz_target(n: int) -> np.ndarray:
    t = np.zeros(2**n, dtype=complex)
    t[0] = t[-1] = 1 / np.sqrt(2)
    return t


def bell_pair_pair_target() -> np.ndarray:
    """|00⟩|11⟩ + |11⟩|00⟩ on 4 qubits, normalized."""
    t = np.zeros(16, dtype=complex)
    # |q3 q2 q1 q0⟩ ordering: |0011⟩ = bits q0=1,q1=1,q2=0,q3=0 → idx 3
    # |1100⟩ = q0=0,q1=0,q2=1,q3=1 → idx 12
    t[3] = t[12] = 1 / np.sqrt(2)
    return t


def run_adaptive(target: np.ndarray, label: str, seeds: list[int]) -> dict:
    n = int(np.log2(target.size))
    backend = NumpyBackend(num_qubits=n)
    loss_fn = fidelity_objective(target)

    def make_objective(spec):
        def obj(p):
            return loss_fn(backend.statevector(spec, p))
        return obj

    grower = AdaptiveGrowth(
        num_qubits=n,
        max_gates=30,
        candidates_per_step=20,
        inner_max_iter=40,
        plateau_patience=5,
        epsilon_random=0.30,
        target_loss=0.005,
    )

    template_specs = [
        HardwareEfficient(num_qubits=n, num_layers=1).build(),
        HardwareEfficient(num_qubits=n, num_layers=2).build(),
        HardwareEfficient(num_qubits=n, num_layers=3).build(),
        LinearEntangler(num_qubits=n, num_layers=1).build(),
        LinearEntangler(num_qubits=n, num_layers=2).build(),
        QAOAInspired(num_qubits=n, num_layers=1, ring=True).build(),
        QAOAInspired(num_qubits=n, num_layers=2, ring=True).build(),
    ]

    runs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        started = time.time()
        spec, params = grower.grow(make_objective, rng)
        elapsed = time.time() - started
        state = backend.statevector(spec, params)
        fidelity = float(abs(np.vdot(target, state)) ** 2)
        sims_to_templates = [structural_similarity(spec, t) for t in template_specs]
        max_sim = max(sims_to_templates)
        novelty = 1.0 - max_sim

        # Gate-name histogram — a fingerprint of the circuit's "shape" no
        # template-name conveys.
        hist: dict[str, int] = {}
        for g in spec.gates:
            hist[g.name] = hist.get(g.name, 0) + 1

        runs.append({
            "seed": seed,
            "fidelity": fidelity,
            "elapsed_s": elapsed,
            "gate_count": spec.gate_count,
            "depth": spec.depth,
            "two_qubit_count": spec.two_qubit_count,
            "num_params": len(params),
            "max_similarity_to_template": float(max_sim),
            "novelty": float(novelty),
            "gate_histogram": hist,
            "signature": [
                {"name": g.name, "qubits": list(g.qubits), "param_idx": g.param_idx}
                for g in spec.gates
            ],
        })

    print(f"\n=== {label} (n={n}) ===", flush=True)
    print(f"  ran {len(seeds)} ADAPT growths", flush=True)
    for r in runs:
        flag = "EXTREME-NOVEL" if r["novelty"] > 0.5 else (
            "moderate-novel" if r["novelty"] > 0.25 else "templated"
        )
        print(
            f"  seed={r['seed']:>3} fidelity={r['fidelity']:.4f} "
            f"depth={r['depth']:>2} gates={r['gate_count']:>2} "
            f"2q={r['two_qubit_count']:>2} params={r['num_params']:>2} "
            f"novelty={r['novelty']:.3f} ({flag})",
            flush=True,
        )
        print(f"      gate histogram: {r['gate_histogram']}", flush=True)

    # Cross-seed structural comparison: are the inventions diverse?
    cross_div: list[float] = []
    for i, ri in enumerate(runs):
        for j, rj in enumerate(runs):
            if i >= j:
                continue
            spec_i = _spec_from_signature(ri["signature"], n)
            spec_j = _spec_from_signature(rj["signature"], n)
            cross_div.append(1.0 - structural_similarity(spec_i, spec_j))
    if cross_div:
        print(
            f"  cross-seed novelty (mean diversity between runs): "
            f"{np.mean(cross_div):.3f} (range {min(cross_div):.3f}-{max(cross_div):.3f})",
            flush=True,
        )

    return {
        "label": label,
        "num_qubits": n,
        "runs": runs,
        "cross_seed_diversity_mean": float(np.mean(cross_div)) if cross_div else 0.0,
    }


def _spec_from_signature(sig, num_qubits):
    from quiver.circuit import CircuitSpec, GateSpec
    spec = CircuitSpec(num_qubits=num_qubits)
    max_p = -1
    for s in sig:
        spec.gates.append(GateSpec(s["name"], tuple(s["qubits"]), s["param_idx"]))
        if s["param_idx"] is not None and s["param_idx"] > max_p:
            max_p = s["param_idx"]
    spec.num_params = max_p + 1
    return spec


def main():
    seeds = [3, 7, 13, 21, 42]
    out = {
        "study": "Pure-ADAPT invention on small state-preparation problems",
        "results": [
            run_adaptive(ghz_target(4),         "GHZ-4",          seeds),
            run_adaptive(bell_pair_pair_target(), "BellPairPair-4", seeds),
        ],
    }
    p = Path(__file__).resolve().parent.parent / "results" / "adaptive_invention_demo.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
