"""TOML config loader.

Example::

    [exploration]
    num_solutions = 10
    time_budget_seconds = 300
    seed = 42

    [optimizer]
    method = "COBYLA"
    max_iter = 200
    basin_hops = 12
    step_size = 1.0
    tolerance = 1e-4

    [diversity]
    threshold = 0.25
    edit_weight = 0.5
    connectivity_weight = 0.3
    depth_weight = 0.2

    [budget]
    max_gates = 80
    max_depth = 30

    [ansatz]
    families = ["hardware_efficient", "qaoa", "linear_entangler"]
    num_layers = [1, 2, 3]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore


@dataclass
class ExplorationConfig:
    num_solutions: int = 10
    time_budget_seconds: float = 300.0
    seed: int | None = 42


@dataclass
class OptimizerConfig:
    method: str = "COBYLA"
    max_iter: int = 200
    basin_hops: int = 12
    step_size: float = 1.0
    tolerance: float = 1e-4


@dataclass
class DiversityConfig:
    threshold: float = 0.25
    edit_weight: float = 0.5
    connectivity_weight: float = 0.3
    depth_weight: float = 0.2


@dataclass
class BudgetConfig:
    max_gates: int | None = None
    max_depth: int | None = None


@dataclass
class AnsatzConfig:
    families: list[str] = field(
        default_factory=lambda: ["hardware_efficient", "qaoa", "linear_entangler"]
    )
    num_layers: list[int] = field(default_factory=lambda: [1, 2, 3])


@dataclass
class QuiverConfig:
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    ansatz: AnsatzConfig = field(default_factory=AnsatzConfig)


def _section(data: dict, name: str) -> dict:
    return data.get(name, {}) or {}


def load_config(path: str | Path) -> QuiverConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return QuiverConfig(
        exploration=ExplorationConfig(**_section(data, "exploration")),
        optimizer=OptimizerConfig(**_section(data, "optimizer")),
        diversity=DiversityConfig(**_section(data, "diversity")),
        budget=BudgetConfig(**_section(data, "budget")),
        ansatz=AnsatzConfig(**_section(data, "ansatz")),
    )
