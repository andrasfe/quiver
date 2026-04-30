"""Quiver: structurally diverse quantum circuit exploration."""

from quivercirc.adaptive import AdaptiveGrowth
from quivercirc.circuit import CircuitSpec, GateSpec
from quivercirc.config import AdaptiveConfig, MutationConfig, QuiverConfig, load_config
from quivercirc.core import Quiver, Solution
from quivercirc.diversity import diversity_score, structural_similarity
from quivercirc.mutation import Mutator
from quivercirc.persistence import (
    LoadedSolution,
    load_circuit,
    load_registry,
    save_circuit,
    save_registry,
    spec_from_dict,
    spec_to_dict,
)
from quivercirc.registry import SolutionRegistry
from quivercirc.subspace import (
    AdaptiveKResult,
    select_k_adaptive,
    subspace_diagonalize,
)
from quivercirc.topology import Topology

__version__ = "0.1.5"

__all__ = [
    "Quiver",
    "Solution",
    "SolutionRegistry",
    "CircuitSpec",
    "GateSpec",
    "AdaptiveGrowth",
    "AdaptiveConfig",
    "Mutator",
    "MutationConfig",
    "QuiverConfig",
    "load_config",
    "diversity_score",
    "structural_similarity",
    # persistence
    "LoadedSolution",
    "save_registry",
    "load_registry",
    "save_circuit",
    "load_circuit",
    "spec_to_dict",
    "spec_from_dict",
    # subspace expansion / adaptive K
    "subspace_diagonalize",
    "select_k_adaptive",
    "AdaptiveKResult",
    # hardware topology
    "Topology",
]
