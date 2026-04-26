import numpy as np

from quiver import Quiver, QuiverConfig
from quiver.ansatz import HardwareEfficient, LinearEntangler, QAOAInspired
from quiver.config import DiversityConfig, ExplorationConfig, OptimizerConfig


def test_finds_bell_state():
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    config = QuiverConfig(
        exploration=ExplorationConfig(num_solutions=2, time_budget_seconds=30, seed=11),
        optimizer=OptimizerConfig(basin_hops=4, max_iter=120),
        diversity=DiversityConfig(threshold=0.1),
    )
    q = Quiver(target=bell, config=config)
    sols = q.explore([
        HardwareEfficient(num_qubits=2, num_layers=2),
        LinearEntangler(num_qubits=2, num_layers=2),
        QAOAInspired(num_qubits=2, num_layers=2),
    ])
    assert len(sols) >= 1
    assert sols[0].fidelity >= 0.99
    families = {s.family for s in sols}
    if len(sols) >= 2:
        assert len(families) >= 2  # structurally diverse families


def test_explore_respects_num_solutions_override():
    plus_plus = np.full(4, 0.5, dtype=complex)
    config = QuiverConfig(
        exploration=ExplorationConfig(num_solutions=10, time_budget_seconds=20, seed=3),
        optimizer=OptimizerConfig(basin_hops=2, max_iter=80),
        diversity=DiversityConfig(threshold=0.05),
    )
    q = Quiver(target=plus_plus, config=config)
    sols = q.explore(
        [HardwareEfficient(num_qubits=2, num_layers=1)],
        num_solutions=1,
    )
    assert len(sols) == 1
