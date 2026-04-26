"""Find multiple structurally diverse circuits that prepare a Bell state."""

from __future__ import annotations

import numpy as np

from quiver import Quiver, QuiverConfig
from quiver.ansatz import HardwareEfficient, LinearEntangler, QAOAInspired
from quiver.config import (
    DiversityConfig,
    ExplorationConfig,
    OptimizerConfig,
)


def main() -> None:
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

    config = QuiverConfig(
        exploration=ExplorationConfig(num_solutions=4, time_budget_seconds=60, seed=7),
        optimizer=OptimizerConfig(basin_hops=6, max_iter=150),
        diversity=DiversityConfig(threshold=0.15),
    )

    q = Quiver(target=bell, config=config)

    library = [
        HardwareEfficient(num_qubits=2, num_layers=1),
        HardwareEfficient(num_qubits=2, num_layers=2),
        LinearEntangler(num_qubits=2, num_layers=1),
        LinearEntangler(num_qubits=2, num_layers=2),
        QAOAInspired(num_qubits=2, num_layers=1),
        QAOAInspired(num_qubits=2, num_layers=2),
    ]

    solutions = q.explore(library)

    print(f"found {len(solutions)} diverse solutions:")
    for i, sol in enumerate(solutions, 1):
        print(
            f"  [{i}] family={sol.family:<20} fidelity={sol.fidelity:.4f} "
            f"depth={sol.depth} gates={sol.gate_count} 2q={sol.two_qubit_count} "
            f"diversity={sol.diversity:.3f}"
        )


if __name__ == "__main__":
    main()
