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
from quiver.diversity import structural_similarity
from quiver.microstructures import MicrostructureLibrary, weld


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
    # Continual learning: when present, the library is mined for fragment
    # candidates in addition to single-gate candidates. Quiver populates
    # this from verified registry entries during exploration.
    microstructure_library: MicrostructureLibrary | None = None
    fragment_candidate_fraction: float = 0.4
    # Anti-template active reward: candidates are scored by
    #   loss(after_brief_optimisation) - anti_template_weight * novelty
    # so circuits that drift from canonical shapes win ties. Pass the
    # template specs you want to push *away* from.
    anti_template_specs: tuple[CircuitSpec, ...] = ()
    anti_template_weight: float = 0.0

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
            single_count, fragment_count = self._budget_split()
            single_idx = rng.choice(len(pool), size=min(single_count, len(pool)),
                                    replace=False)
            single_candidates = [pool[i] for i in single_idx]

            # scored entries: (combined_score, raw_loss, applier, final_params)
            # applier(spec, params) -> (new_spec, new_params)
            scored: list = []

            for name, qubits in single_candidates:
                applier = _single_gate_applier(name, qubits)
                trial_spec, trial_params = applier(spec, params)
                self._score_candidate(
                    trial_spec, trial_params, applier, make_objective, scored,
                )

            if self.microstructure_library is not None and fragment_count > 0:
                for _ in range(fragment_count):
                    frag = self.microstructure_library.sample(self.num_qubits, rng)
                    if frag is None or spec.gate_count + frag.length > self.max_gates:
                        continue
                    applier = _fragment_applier(frag)
                    trial_spec, trial_params = applier(spec, params)
                    self._score_candidate(
                        trial_spec, trial_params, applier, make_objective, scored,
                    )

            if not scored:
                plateau += 1
                continue

            # Best = lowest combined score (loss minus novelty bonus).
            scored.sort(key=lambda x: x[0])
            best_combined, best_loss_after, best_applier, best_final_params = scored[0]

            # ε-greedy among non-worse candidates.
            if rng.random() < self.epsilon_random:
                non_worse = [s for s in scored if s[1] <= best_loss + 1e-6]
                if len(non_worse) > 1:
                    pick = non_worse[int(rng.integers(0, len(non_worse)))]
                    best_combined, best_loss_after, best_applier, best_final_params = pick

            # Reject only if the candidate makes loss strictly *worse*.
            # Committing tie-steps lets the optimizer build up multi-gate
            # combinations whose individual additions were no-ops but whose
            # combined effect crosses the basin (e.g. RY then CNOT for Bell).
            if best_loss_after > best_loss + 1e-6:
                plateau += 1
                continue

            spec, _ = best_applier(spec, params)
            params = best_final_params
            improved = best_loss_after < best_loss - 1e-6
            best_loss = best_loss_after
            plateau = 0 if improved else plateau + 1

            if best_loss < self.target_loss:
                break

        return spec, params

    # ---------- helpers --------------------------------------------------

    def _budget_split(self) -> tuple[int, int]:
        if self.microstructure_library is None or not self.microstructure_library.fragments:
            return self.candidates_per_step, 0
        frag = int(round(self.candidates_per_step * self.fragment_candidate_fraction))
        return self.candidates_per_step - frag, frag

    def _novelty(self, spec: CircuitSpec) -> float:
        if not self.anti_template_specs:
            return 0.0
        sims = [structural_similarity(spec, t) for t in self.anti_template_specs]
        return 1.0 - max(sims)

    def _score_candidate(
        self, trial_spec, trial_params, applier, make_objective, scored
    ) -> None:
        obj = make_objective(trial_spec)
        if trial_spec.num_params == 0:
            try:
                raw = float(obj(trial_params))
            except Exception:
                return
            combined = raw - self.anti_template_weight * self._novelty(trial_spec)
            scored.append((combined, raw, applier, trial_params))
            return
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
            raw = float(result.fun)
            final = np.asarray(result.x)
        except Exception:
            return
        combined = raw - self.anti_template_weight * self._novelty(trial_spec)
        scored.append((combined, raw, applier, final))


def _single_gate_applier(name: str, qubits: tuple[int, ...]):
    def apply(spec: CircuitSpec, params: np.ndarray) -> tuple[CircuitSpec, np.ndarray]:
        return _append(spec, params, name, qubits)
    return apply


def _fragment_applier(fragment):
    def apply(spec: CircuitSpec, params: np.ndarray) -> tuple[CircuitSpec, np.ndarray]:
        return weld(spec, params, fragment)
    return apply
