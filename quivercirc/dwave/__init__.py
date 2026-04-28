"""D-Wave QUBO / CQM exploration: enumerate diverse penalty formulations
of the same constrained problem."""

from quivercirc.dwave.qubo import QUBOExplorer, QUBOFormulation
from quivercirc.dwave.penalties import (
    PenaltyStrategy,
    quadratic_penalty,
    slack_penalty,
    one_hot_penalty,
    log_encoded_penalty,
)

__all__ = [
    "QUBOExplorer",
    "QUBOFormulation",
    "PenaltyStrategy",
    "quadratic_penalty",
    "slack_penalty",
    "one_hot_penalty",
    "log_encoded_penalty",
]
