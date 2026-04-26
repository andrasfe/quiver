"""Penalty-formulation strategies for encoding constraints as QUBO terms.

Each strategy produces additional Q matrix entries (and possibly auxiliary
variables) that penalise constraint violations. The strategies differ
structurally — quadratic, slack, one-hot, and log-encoded — so the
explorer treats them as distinct formulation families when measuring
diversity across CQM solutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class PenaltyStrategy:
    name: str
    family: str
    apply: Callable[..., "tuple[np.ndarray, int]"]


def _ensure_size(Q: np.ndarray, size: int) -> np.ndarray:
    if Q.shape[0] >= size:
        return Q
    new = np.zeros((size, size))
    new[: Q.shape[0], : Q.shape[1]] = Q
    return new


def quadratic_penalty(
    Q: np.ndarray,
    coefficients: Sequence[float],
    target: float,
    weight: float,
) -> tuple[np.ndarray, int]:
    """Encode (sum_i a_i x_i - target)^2 directly into Q. No aux vars."""
    a = np.asarray(coefficients, dtype=float)
    n = Q.shape[0]
    if a.size > n:
        Q = _ensure_size(Q, a.size)
        n = a.size
    Q = Q.copy()
    for i in range(a.size):
        Q[i, i] += weight * (a[i] ** 2 - 2 * a[i] * target)
        for j in range(i + 1, a.size):
            Q[i, j] += 2 * weight * a[i] * a[j]
    return Q, 0


def slack_penalty(
    Q: np.ndarray,
    coefficients: Sequence[float],
    upper_bound: float,
    weight: float,
    slack_bits: int = 4,
) -> tuple[np.ndarray, int]:
    """Encode sum_i a_i x_i + s ≤ upper_bound with binary slack s."""
    a = np.asarray(coefficients, dtype=float)
    n = Q.shape[0]
    start = max(n, a.size)
    Q = _ensure_size(Q, start + slack_bits)
    slack_coeffs = np.array([2**k for k in range(slack_bits)], dtype=float)

    coeffs = np.zeros(Q.shape[0])
    coeffs[: a.size] = a
    coeffs[start : start + slack_bits] = slack_coeffs

    for i in range(Q.shape[0]):
        Q[i, i] += weight * (coeffs[i] ** 2 - 2 * coeffs[i] * upper_bound)
        for j in range(i + 1, Q.shape[0]):
            Q[i, j] += 2 * weight * coeffs[i] * coeffs[j]
    return Q, slack_bits


def one_hot_penalty(
    Q: np.ndarray, group: Sequence[int], weight: float
) -> tuple[np.ndarray, int]:
    """Penalise (sum_{i in group} x_i - 1)^2 — exactly one variable selected."""
    n = max(Q.shape[0], max(group) + 1)
    Q = _ensure_size(Q, n).copy()
    for i in group:
        Q[i, i] += weight * (1 - 2)
        for j in group:
            if j > i:
                Q[i, j] += 2 * weight
    return Q, 0


def log_encoded_penalty(
    Q: np.ndarray,
    coefficients: Sequence[float],
    target: float,
    weight: float,
    num_bits: int = 4,
) -> tuple[np.ndarray, int]:
    """Quadratic-equality penalty using a log-encoded auxiliary integer."""
    a = np.asarray(coefficients, dtype=float)
    n = Q.shape[0]
    start = max(n, a.size)
    Q = _ensure_size(Q, start + num_bits)
    aux_coeffs = -np.array([2**k for k in range(num_bits)], dtype=float)

    coeffs = np.zeros(Q.shape[0])
    coeffs[: a.size] = a
    coeffs[start : start + num_bits] = aux_coeffs

    for i in range(Q.shape[0]):
        Q[i, i] += weight * (coeffs[i] ** 2 - 2 * coeffs[i] * target)
        for j in range(i + 1, Q.shape[0]):
            Q[i, j] += 2 * weight * coeffs[i] * coeffs[j]
    return Q, num_bits
