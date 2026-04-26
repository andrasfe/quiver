"""Quiver: structurally diverse quantum circuit exploration."""

from quiver.circuit import CircuitSpec, GateSpec
from quiver.config import MutationConfig, QuiverConfig, load_config
from quiver.core import Quiver, Solution
from quiver.diversity import diversity_score, structural_similarity
from quiver.mutation import Mutator
from quiver.registry import SolutionRegistry

__version__ = "0.1.0"

__all__ = [
    "Quiver",
    "Solution",
    "SolutionRegistry",
    "CircuitSpec",
    "GateSpec",
    "Mutator",
    "MutationConfig",
    "QuiverConfig",
    "load_config",
    "diversity_score",
    "structural_similarity",
]
