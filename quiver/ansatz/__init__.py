"""Built-in ansatz templates."""

from quiver.ansatz.all_to_all import AllToAll
from quiver.ansatz.base import Ansatz
from quiver.ansatz.brickwall import BrickWall
from quiver.ansatz.discovered import DiscoveredAnsatz
from quiver.ansatz.hardware_efficient import HardwareEfficient
from quiver.ansatz.linear_entangler import LinearEntangler
from quiver.ansatz.qaoa import QAOAInspired
from quiver.ansatz.strongly_entangling import StronglyEntangling

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
