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
class MutationConfig:
    """Structural mutation of verified solutions during exploration.

    When enabled, every `frequency`-th round in `Quiver.explore()` skips
    the template library and instead mutates a randomly-chosen registry
    entry. The mutated spec is then optimized and verified like any other
    candidate. Default: disabled (backwards compatible).
    """
    enabled: bool = False
    frequency: int = 3
    chain_min: int = 1
    chain_max: int = 3


@dataclass
class AdaptiveConfig:
    """ADAPT-style growth: build circuits gate-by-gate using the verifier.

    When enabled, every `frequency`-th round runs an `AdaptiveGrowth`
    instead of using a template or mutation. The grown circuits have no
    template ancestry and are problem-shaped — useful for *inventing*
    novel circuit families rather than rediscovering known ones.
    """
    enabled: bool = False
    frequency: int = 4
    max_gates: int = 80
    candidates_per_step: int = 16
    inner_max_iter: int = 40
    plateau_patience: int = 3
    epsilon_random: float = 0.15
    target_loss: float = 1e-3


@dataclass
class QuiverConfig:
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    ansatz: AnsatzConfig = field(default_factory=AnsatzConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)


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
        mutation=MutationConfig(**_section(data, "mutation")),
        adaptive=AdaptiveConfig(**_section(data, "adaptive")),
    )
