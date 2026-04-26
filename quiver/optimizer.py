"""Basin-hopping wrapper around scipy's COBYLA.

Each call to `optimize` runs `basin_hops` independent COBYLA restarts with
random perturbations of the starting point and returns the best parameters
found. Callers seed the RNG to keep the optimization path deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from quiver.config import OptimizerConfig


@dataclass
class OptimizeResult:
    params: np.ndarray
    objective: float
    nfev: int
    converged: bool


def optimize(
    objective: Callable[[np.ndarray], float],
    num_params: int,
    config: OptimizerConfig,
    rng: np.random.Generator,
) -> OptimizeResult:
    best_params = rng.uniform(-np.pi, np.pi, size=num_params)
    best_val = float(objective(best_params))
    total_nfev = 1

    for hop in range(max(1, config.basin_hops)):
        if hop == 0:
            x0 = best_params.copy()
        else:
            perturb = rng.normal(0.0, config.step_size, size=num_params)
            x0 = best_params + perturb

        result = minimize(
            objective,
            x0,
            method=config.method,
            options={
                "maxiter": config.max_iter,
                "rhobeg": 0.5,
                "catol": config.tolerance,
            },
        )
        total_nfev += int(result.nfev)
        if float(result.fun) < best_val:
            best_val = float(result.fun)
            best_params = np.asarray(result.x)

        if best_val < config.tolerance:
            break

    return OptimizeResult(
        params=best_params,
        objective=best_val,
        nfev=total_nfev,
        converged=best_val < config.tolerance,
    )
