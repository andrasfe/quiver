"""Built-in verifiers and objective functions.

A verifier is a callable `(state) -> (passed: bool, fidelity: float)`. It
should be deterministic — the registry only accepts a candidate if the
verifier returns True.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Verifier = Callable[[np.ndarray], "tuple[bool, float]"]


def state_fidelity(target: np.ndarray, state: np.ndarray) -> float:
    """|<target|state>|^2 with both vectors normalised."""
    t = target / (np.linalg.norm(target) + 1e-15)
    s = state / (np.linalg.norm(state) + 1e-15)
    return float(np.abs(np.vdot(t, s)) ** 2)


def fidelity_verifier(target: np.ndarray, threshold: float = 0.99) -> Verifier:
    """Verifier that passes when |<target|state>|^2 ≥ threshold."""
    target = np.asarray(target, dtype=complex)

    def verify(state: np.ndarray) -> tuple[bool, float]:
        f = state_fidelity(target, state)
        return f >= threshold, f

    return verify


def fidelity_objective(target: np.ndarray) -> Callable[[np.ndarray], float]:
    """Objective for COBYLA: 1 - fidelity (so minimising drives fidelity → 1)."""
    target = np.asarray(target, dtype=complex)

    def loss(state: np.ndarray) -> float:
        return 1.0 - state_fidelity(target, state)

    return loss
