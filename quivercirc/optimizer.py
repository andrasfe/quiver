"""Basin-hopping wrapper around scipy's COBYLA.

Each call to `optimize` runs `basin_hops` independent COBYLA restarts
with diverse starting points (zero / small Gaussian / uniform) and
returns the best parameters found. Callers seed the RNG to keep the
optimization path deterministic. A `warm_start` may be passed to
front-load the seed list (used by mutation rounds when the parent's
verified parameters are a useful neighbourhood).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from quivercirc.config import OptimizerConfig


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
    warm_start: np.ndarray | None = None,
) -> OptimizeResult:
    # Edge case: a fully non-parametric circuit (all gates fixed) — common
    # for mutated specs after deletes — has nothing to optimize.
    if num_params == 0:
        params = np.zeros(0)
        val = float(objective(params))
        return OptimizeResult(
            params=params, objective=val, nfev=1,
            converged=val < config.tolerance,
        )

    if warm_start is not None and len(warm_start) == num_params:
        # Mutation rounds: parent's verified params are usually a good
        # neighbourhood. Front-load the seed list with warm starts.
        warm = np.asarray(warm_start, dtype=float)
        seeds = [
            warm,
            warm + rng.normal(0.0, 0.1, size=num_params),
            warm + rng.normal(0.0, 0.5, size=num_params),
        ]
    else:
        seeds = _init_strategies(num_params, rng)
    best_params = seeds[0]
    best_val = float(objective(best_params))
    total_nfev = 1

    # COBYLA needs at least num_params + 2 function evals to be well-defined;
    # scale up automatically so high-dim ansätze are not silently starved.
    effective_max_iter = max(config.max_iter, num_params + 50)

    for hop in range(max(1, config.basin_hops)):
        if hop < len(seeds):
            x0 = seeds[hop]
        else:
            anneal = max(0.25, 1.0 - 0.05 * (hop - len(seeds)))
            scale = config.step_size * anneal
            x0 = best_params + rng.normal(0.0, scale, size=num_params)

        result = minimize(
            objective, x0,
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
