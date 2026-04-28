"""Registry of verified, structurally-diverse solutions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from quivercirc.circuit import CircuitSpec
from quivercirc.diversity import DiversityWeights, diversity_score, structural_similarity


@dataclass
class RegistryEntry:
    spec: CircuitSpec
    params: np.ndarray
    family: str
    fidelity: float
    objective: float
    gate_count: int
    depth: int
    two_qubit_count: int
    diversity: float  # diversity vs prior entries at insertion time

    def summary(self) -> dict:
        return {
            "family": self.family,
            "fidelity": float(self.fidelity),
            "objective": float(self.objective),
            "gate_count": self.gate_count,
            "depth": self.depth,
            "two_qubit_count": self.two_qubit_count,
            "diversity": float(self.diversity),
            "params": self.params.tolist(),
        }


@dataclass
class SolutionRegistry:
    diversity_threshold: float = 0.25
    weights: DiversityWeights = field(default_factory=DiversityWeights)
    entries: list[RegistryEntry] = field(default_factory=list)
    # When True, a candidate that fails the diversity gate but is *more
    # compact* (fewer gates AND not deeper) than its closest existing
    # entry replaces that entry instead of being rejected. This drives
    # the registry toward Pareto-optimal solutions over time without
    # sacrificing structural variety.
    prefer_compact: bool = False

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[RegistryEntry]:
        return iter(self.entries)

    def specs(self) -> list[CircuitSpec]:
        return [e.spec for e in self.entries]

    def _make_entry(
        self,
        spec: CircuitSpec,
        params: np.ndarray,
        family: str,
        fidelity: float,
        objective: float,
        diversity: float,
    ) -> RegistryEntry:
        return RegistryEntry(
            spec=spec,
            params=np.array(params, copy=True),
            family=family,
            fidelity=float(fidelity),
            objective=float(objective),
            gate_count=spec.gate_count,
            depth=spec.depth,
            two_qubit_count=spec.two_qubit_count,
            diversity=float(diversity),
        )

    def try_add(
        self,
        spec: CircuitSpec,
        params: np.ndarray,
        family: str,
        fidelity: float,
        objective: float,
    ) -> RegistryEntry | None:
        score = diversity_score(spec, self.specs(), self.weights)

        if not self.entries or score >= self.diversity_threshold:
            entry = self._make_entry(spec, params, family, fidelity, objective, score)
            self.entries.append(entry)
            return entry

        if self.prefer_compact:
            # Find the most-similar existing entry; replace if candidate
            # dominates it on (gates, depth) and at least one is strict.
            sims = [
                structural_similarity(spec, e.spec, self.weights)
                for e in self.entries
            ]
            idx = max(range(len(self.entries)), key=lambda i: sims[i])
            incumbent = self.entries[idx]
            strictly_smaller = (
                spec.gate_count <= incumbent.gate_count
                and spec.depth <= incumbent.depth
                and (
                    spec.gate_count < incumbent.gate_count
                    or spec.depth < incumbent.depth
                )
            )
            if strictly_smaller:
                entry = self._make_entry(
                    spec, params, family, fidelity, objective, score
                )
                self.entries[idx] = entry
                return entry

        return None
