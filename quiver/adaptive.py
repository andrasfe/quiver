"""Adaptive (ADAPT-VQE-style) circuit growth.

The template+mutation pipeline starts from human-designed shapes and
varies them. This module starts from an empty circuit and **grows it
gate-by-gate**: at each step, we evaluate a random subsample of the
full candidate-gate pool, briefly optimize the parameters of each
extended circuit, and commit the gate that most reduces the loss. The
result is a circuit whose structure is dictated by the problem, not by
any human's intuition about ansatz design.

Two design choices that matter:

  1. **Random subsampling** of the candidate pool per step. Evaluating
     all ~500 candidates would dominate runtime; sampling 12-24 keeps
     each step under a second while still exploring broadly.

  2. **ε-greedy** acceptance: with probability `epsilon_random` we
     accept a *random* candidate instead of the best one. This breaks
     symmetry across runs (different RNG seeds produce structurally
     different growths) and avoids getting stuck on a single greedy
     trajectory the moment the surface flattens.

The grown circuit is returned as a (CircuitSpec, params) pair so the
calling Quiver instance can run it through the same verifier and
diversity gate as everything else in the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from quiver.circuit import CircuitSpec, GateSpec


_PARAM_1Q = ("rx", "ry", "rz")
_FIXED_1Q = ("h", "x", "y", "z", "s", "t")
_PARAM_2Q = ("rxx", "ryy", "rzz")
_FIXED_2Q = ("cnot", "cz", "swap", "iswap", "sqrt_iswap")


def _is_parametric(name: str) -> bool:
    return name in _PARAM_1Q or name in _PARAM_2Q


def _candidate_pool(num_qubits: int) -> list[tuple[str, tuple[int, ...]]]:
    pool: list[tuple[str, tuple[int, ...]]] = []
    for q in range(num_qubits):
        for name in _PARAM_1Q + _FIXED_1Q:
            pool.append((name, (q,)))
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i == j:
                continue
            for name in _PARAM_2Q + _FIXED_2Q:
                pool.append((name, (i, j)))
    return pool


def _append(
    spec: CircuitSpec, params: np.ndarray, name: str, qubits: tuple[int, ...]
) -> tuple[CircuitSpec, np.ndarray]:
    new_spec = CircuitSpec(num_qubits=spec.num_qubits)
    new_spec.gates = list(spec.gates)
    new_spec.num_params = spec.num_params
    if _is_parametric(name):
        new_spec.gates.append(GateSpec(name, qubits, spec.num_params))
        new_spec.num_params += 1
        new_params = np.concatenate([params, [0.0]])
    else:
        new_spec.gates.append(GateSpec(name, qubits, None))
        new_params = params.copy()
    return new_spec, new_params


@dataclass
class AdaptiveGrowth:
    """Grow a circuit by greedy gate-by-gate selection.

    Parameters
    ----------
    num_qubits : int
    max_gates : int
        Hard cap on the grown circuit length.
    candidates_per_step : int
        How many random candidates from the pool to evaluate at each
        step. Larger = better gate choices, slower per step.
    inner_max_iter : int
        COBYLA iteration budget when scoring a candidate. Keep small —
        we only need to know if a candidate has *potential*, not converge.
    plateau_patience : int
        Stop after this many consecutive steps that fail to reduce loss.
    epsilon_random : float
        Probability of picking a random valid candidate instead of the
        best, to break greedy symmetry across runs.
    target_loss : float
        Stop early when the loss drops below this — verification will
        almost certainly pass.
    family : str
        Family tag attached to grown circuits in the registry.
    """

    num_qubits: int
    max_gates: int = 80
    candidates_per_step: int = 16
    inner_max_iter: int = 40
    plateau_patience: int = 3
    epsilon_random: float = 0.15
    target_loss: float = 1e-3
    family: str = "adaptive"

    def grow(
        self,
        make_objective: Callable[[CircuitSpec], Callable[[np.ndarray], float]],
        rng: np.random.Generator,
    ) -> tuple[CircuitSpec, np.ndarray]:
        spec = CircuitSpec(num_qubits=self.num_qubits)
        params = np.zeros(0)
        # Loss of an empty circuit (just |0...0>): may be very high if
        # target is far from the all-zero state.
        try:
            best_loss = float(make_objective(spec)(params))
        except Exception:
            best_loss = 1.0

        pool = _candidate_pool(self.num_qubits)
        plateau = 0

        while spec.gate_count < self.max_gates and plateau < self.plateau_patience:
            n_sample = min(self.candidates_per_step, len(pool))
            sampled_idx = rng.choice(len(pool), size=n_sample, replace=False)
            candidates = [pool[i] for i in sampled_idx]

            best_cand: tuple[str, tuple[int, ...]] | None = None
            best_cand_loss = best_loss
            best_cand_params = params

            scored: list[tuple[float, tuple[str, tuple[int, ...]], np.ndarray]] = []
            for name, qubits in candidates:
                trial_spec, trial_params = _append(spec, params, name, qubits)
                obj = make_objective(trial_spec)

                if trial_spec.num_params == 0:
                    val = float(obj(trial_params))
                    scored.append((val, (name, qubits), trial_params))
                    if val < best_cand_loss:
                        best_cand_loss = val
                        best_cand = (name, qubits)
                        best_cand_params = trial_params
                    continue

                try:
                    result = minimize(
                        obj,
                        trial_params,
                        method="COBYLA",
                        options={
                            "maxiter": max(self.inner_max_iter, trial_spec.num_params + 5),
                            "rhobeg": 0.3,
                            "catol": 1e-4,
                        },
                    )
                    val = float(result.fun)
                    final_p = np.asarray(result.x)
                except Exception:
                    continue

                scored.append((val, (name, qubits), final_p))
                if val < best_cand_loss:
                    best_cand_loss = val
                    best_cand = (name, qubits)
                    best_cand_params = final_p

            if best_cand is None:
                plateau += 1
                continue

            # ε-greedy: occasionally take a *non-best* improving candidate.
            if scored and rng.random() < self.epsilon_random:
                improving = [
                    s for s in scored if s[0] < best_loss - 1e-6 and s[1] != best_cand
                ]
                if improving:
                    pick = improving[int(rng.integers(0, len(improving)))]
                    best_cand_loss, best_cand, best_cand_params = pick

            spec, _ = _append(spec, params, *best_cand)
            params = best_cand_params
            best_loss = best_cand_loss
            plateau = 0

            if best_loss < self.target_loss:
                break

        return spec, params
