"""Built-in ansatz templates."""

from quivercirc.ansatz.all_to_all import AllToAll
from quivercirc.ansatz.base import Ansatz
from quivercirc.ansatz.brickwall import BrickWall
from quivercirc.ansatz.discovered import DiscoveredAnsatz
from quivercirc.ansatz.hardware_efficient import HardwareEfficient
from quivercirc.ansatz.linear_entangler import LinearEntangler
from quivercirc.ansatz.qaoa import QAOAInspired
from quivercirc.ansatz.strongly_entangling import StronglyEntangling

__all__ = [
    "Ansatz",
    "AllToAll",
    "BrickWall",
    "DiscoveredAnsatz",
    "HardwareEfficient",
    "LinearEntangler",
    "QAOAInspired",
    "StronglyEntangling",
]
