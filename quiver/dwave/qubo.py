"""Explore multiple QUBO formulations of the same problem.

For portfolio reverse stress testing, the same constrained optimization
is encoded several different ways — quadratic-equality, slack-inequality,
one-hot, log-encoded — and Quiver's job is to enumerate formulations that
all satisfy the same target risk metrics but encode the constraints
differently. Diversity here is measured by the structural
fingerprint of the resulting Q matrix (sparsity pattern, auxiliary-variable
count, penalty family).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np


@dataclass
class QUBOFormulation:
    Q: np.ndarray
    family: str
    aux_vars: int
    metrics: dict
    description: str = ""

    def fingerprint(self) -> tuple:
        # Structural fingerprint: family + sparsity pattern.
        nnz = tuple(sorted(zip(*np.nonzero(self.Q))))
        return (self.family, self.aux_vars, len(nnz))


@dataclass
class QUBOExplorer:
    """Generate diverse QUBO formulations validated against a verifier.

    The verifier receives a candidate `Q` matrix (and optional metadata) and
    returns (passed, metrics_dict). Formulations are kept only when they
    pass verification and have a fingerprint distinct from existing ones.
    """

    base_Q: np.ndarray
    verifier: Callable[[np.ndarray, dict], "tuple[bool, dict]"]
    formulations: list[QUBOFormulation] = field(default_factory=list)

    def explore(
        self,
        strategies: Iterable[Callable[[np.ndarray], "tuple[np.ndarray, int, dict]"]],
        max_results: int = 10,
    ) -> list[QUBOFormulation]:
        seen: set[tuple] = set()
        for strat in strategies:
            if len(self.formulations) >= max_results:
                break
            Q, aux, meta = strat(self.base_Q.copy())
            passed, metrics = self.verifier(Q, meta)
            if not passed:
                continue
            f = QUBOFormulation(
                Q=Q,
                family=meta.get("family", "unknown"),
                aux_vars=aux,
                metrics=metrics,
                description=meta.get("description", ""),
            )
            fp = f.fingerprint()
            if fp in seen:
                continue
            seen.add(fp)
            self.formulations.append(f)
        return list(self.formulations)
