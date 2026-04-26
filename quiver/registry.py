"""Registry of verified, structurally-diverse solutions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from quiver.circuit import CircuitSpec
from quiver.diversity import DiversityWeights, diversity_score


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

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[RegistryEntry]:
        return iter(self.entries)

    def specs(self) -> list[CircuitSpec]:
        return [e.spec for e in self.entries]

    def try_add(
        self,
        spec: CircuitSpec,
        params: np.ndarray,
        family: str,
        fidelity: float,
        objective: float,
    ) -> RegistryEntry | None:
        score = diversity_score(spec, self.specs(), self.weights)
        if self.entries and score < self.diversity_threshold:
            return None
        entry = RegistryEntry(
            spec=spec,
            params=np.array(params, copy=True),
            family=family,
            fidelity=float(fidelity),
            objective=float(objective),
            gate_count=spec.gate_count,
            depth=spec.depth,
            two_qubit_count=spec.two_qubit_count,
            diversity=float(score),
        )
        self.entries.append(entry)
        return entry
