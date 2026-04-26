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


def _init_strategies(num_params: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Diverse seeds: zero (good for variational identity), small Gaussian
    (perturbed identity), and full uniform random."""
    return [
        np.zeros(num_params),
        rng.normal(0.0, 0.1, size=num_params),
        rng.uniform(-np.pi, np.pi, size=num_params),
    ]


def optimize(
    objective: Callable[[np.ndarray], float],
    num_params: int,
    config: OptimizerConfig,
    rng: np.random.Generator,
) -> OptimizeResult:
    seeds = _init_strategies(num_params, rng)
    best_params = seeds[0]
    best_val = float(objective(best_params))
    total_nfev = 1

    # COBYLA needs at least num_params + 2 function evals to be well-defined;
    # raise the cap automatically so high-dim ansätze are not silently starved.
    effective_max_iter = max(config.max_iter, num_params + 50)

    for hop in range(max(1, config.basin_hops)):
        if hop < len(seeds):
            x0 = seeds[hop]
        else:
            # Annealed perturbation around best-so-far: large early,
            # narrowing as basin-hopping proceeds.
            anneal = max(0.25, 1.0 - 0.05 * (hop - len(seeds)))
            scale = config.step_size * anneal
            x0 = best_params + rng.normal(0.0, scale, size=num_params)

        result = minimize(
            objective,
            x0,
            method=config.method,
            options={
                "maxiter": effective_max_iter,
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
