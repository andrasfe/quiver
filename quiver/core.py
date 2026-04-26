"""Quiver: orchestrates ansatz exploration, optimization, verification, and
diversity-filtered registry insertion."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from quiver.ansatz.base import Ansatz
from quiver.backends.base import Backend
from quiver.backends.numpy_backend import NumpyBackend
from quiver.circuit import CircuitSpec
from quiver.config import QuiverConfig
from quiver.diversity import DiversityWeights
from quiver.mutation import Mutator
from quiver.optimizer import optimize
from quiver.registry import RegistryEntry, SolutionRegistry
from quiver.verification import Verifier, fidelity_objective, fidelity_verifier


Objective = Callable[[np.ndarray], float]


@dataclass
class Solution:
    family: str
    params: list[float]
    fidelity: float
    objective: float
    gate_count: int
    depth: int
    two_qubit_count: int
    diversity: float
    spec: CircuitSpec

    @classmethod
    def from_entry(cls, entry: RegistryEntry) -> "Solution":
        return cls(
            family=entry.family,
            params=entry.params.tolist(),
            fidelity=entry.fidelity,
            objective=entry.objective,
            gate_count=entry.gate_count,
            depth=entry.depth,
            two_qubit_count=entry.two_qubit_count,
            diversity=entry.diversity,
            spec=entry.spec,
        )

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "params": list(self.params),
            "fidelity": self.fidelity,
            "objective": self.objective,
            "gate_count": self.gate_count,
            "depth": self.depth,
            "two_qubit_count": self.two_qubit_count,
            "diversity": self.diversity,
        }


def _within_budget(spec: CircuitSpec, config: QuiverConfig) -> bool:
    b = config.budget
    if b.max_gates is not None and spec.gate_count > b.max_gates:
        return False
    if b.max_depth is not None and spec.depth > b.max_depth:
        return False
    return True


@dataclass
class Quiver:
    """Find multiple structurally diverse solutions to a target outcome.

    Parameters
    ----------
    target : np.ndarray | dict
        For circuit problems, a target statevector. For metrics-driven
        problems, a dict consumed by a custom verifier.
    verifier : Verifier | None
        Callable returning (passed, score). Defaults to a fidelity verifier
        when `target` is a statevector.
    backend : Backend | None
        Statevector backend. Defaults to NumpyBackend sized to the target.
    objective : Objective | None
        Loss in parameter space. Defaults to (1 - fidelity to target).
    config : QuiverConfig
    """

    target: np.ndarray | dict
    verifier: Verifier | None = None
    backend: Backend | None = None
    objective: Callable[[np.ndarray], float] | None = None
    config: QuiverConfig = field(default_factory=QuiverConfig)

    def __post_init__(self) -> None:
        if isinstance(self.target, np.ndarray):
            num_qubits = int(np.log2(self.target.size))
            if 2**num_qubits != self.target.size:
                raise ValueError("target statevector length must be a power of 2")
            if self.backend is None:
                self.backend = NumpyBackend(num_qubits=num_qubits)
            if self.verifier is None:
                self.verifier = fidelity_verifier(
                    self.target, threshold=self._fidelity_threshold()
                )
        else:
            if self.backend is None:
                raise ValueError("backend is required when target is not a statevector")
            if self.verifier is None:
                raise ValueError("verifier is required when target is not a statevector")

    def _fidelity_threshold(self) -> float:
        # Threshold lives implicitly in the verifier; default 0.99 for circuits.
        return 0.99

    def _build_objective(self, spec: CircuitSpec) -> Objective:
        if self.objective is not None:
            user_obj = self.objective
            return lambda p: float(user_obj(p))

        if not isinstance(self.target, np.ndarray):
            raise ValueError("objective is required for non-statevector targets")

        loss_fn = fidelity_objective(self.target)
        backend = self.backend
        assert backend is not None

        def obj(params: np.ndarray) -> float:
            state = backend.statevector(spec, params)
            return loss_fn(state)

        return obj

    def explore(
        self,
        ansatz_library: Iterable[Ansatz],
        num_solutions: int | None = None,
        time_budget: float | None = None,
    ) -> list[Solution]:
        """Run exploration until num_solutions are found or time runs out."""
        cfg = self.config
        target_n = num_solutions if num_solutions is not None else cfg.exploration.num_solutions
        budget_s = time_budget if time_budget is not None else cfg.exploration.time_budget_seconds

        weights = DiversityWeights(
            edit=cfg.diversity.edit_weight,
            connectivity=cfg.diversity.connectivity_weight,
            depth=cfg.diversity.depth_weight,
        )
        registry = SolutionRegistry(
            diversity_threshold=cfg.diversity.threshold, weights=weights
        )

        rng = np.random.default_rng(cfg.exploration.seed)
        ansatz_list = list(ansatz_library)
        if not ansatz_list:
            raise ValueError("ansatz_library is empty")

        mutator = Mutator(
            chain_min=cfg.mutation.chain_min,
            chain_max=cfg.mutation.chain_max,
        )

        deadline = time.monotonic() + budget_s
        rounds_done = 0
        ansatz_idx = 0
        rejected = 0

        while len(registry) < target_n and time.monotonic() < deadline:
            is_mutation_round = (
                cfg.mutation.enabled
                and len(registry) > 0
                and rounds_done % cfg.mutation.frequency == cfg.mutation.frequency - 1
            )

            warm_start: np.ndarray | None = None
            if is_mutation_round:
                parent_idx = int(rng.integers(0, len(registry)))
                parent = registry.entries[parent_idx]
                spec, warm_start = mutator.chain(parent.spec, parent.params, rng)
                family_tag = "mutated"
            else:
                ansatz = ansatz_list[ansatz_idx % len(ansatz_list)]
                ansatz_idx += 1
                spec = ansatz.build()
                family_tag = ansatz.family

            rounds_done += 1

            if not _within_budget(spec, cfg):
                continue

            objective = self._build_objective(spec)
            result = optimize(
                objective, spec.num_params, cfg.optimizer, rng,
                warm_start=warm_start,
            )

            if not isinstance(self.target, np.ndarray):
                # Caller-provided objective + verifier: feed params through verifier.
                passed, score = self.verifier(result.params)  # type: ignore[arg-type]
            else:
                state = self.backend.statevector(spec, result.params)  # type: ignore[union-attr]
                passed, score = self.verifier(state)  # type: ignore[misc]

            if not passed:
                rejected += 1
                continue

            entry = registry.try_add(
                spec=spec,
                params=result.params,
                family=family_tag,
                fidelity=score,
                objective=result.objective,
            )
            if entry is None:
                rejected += 1

        return [Solution.from_entry(e) for e in registry]
