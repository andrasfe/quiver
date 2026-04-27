"""Variational adiabatic state preparation.

For a target Hamiltonian H_target whose ground state is hard to reach
directly, we interpolate from an easy Hamiltonian H_easy:

    H(s) = (1 − s) · H_easy + s · H_target ,    s ∈ [0, 1]

We sweep s from 0 to 1 in num_steps points. At each s we minimise
⟨ψ(θ)|H(s)|ψ(θ)⟩ over the circuit's parameters, **warm-starting from
the previous step's optimum**. The intuition (and the textbook adiabatic
theorem) is that the ground state of H(s) deforms continuously with s,
so the circuit's parameters need only small updates between steps. The
optimizer never has to climb out of the barren-plateau corner that
direct attack on H_target would leave it in.

This is an annealing schedule applied to *training* — it doesn't change
the circuit at all, only the loss function over time. You still get a
parameterised circuit at the end whose state approximates the target
ground state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from quiver.circuit import CircuitSpec
from quiver.hamiltonian import expectation


def linear_interpolation(
    H_easy: np.ndarray, H_target: np.ndarray, num_steps: int
) -> list[tuple[float, np.ndarray]]:
    """Return a list of (s, H(s)) tuples for s evenly spaced on [0, 1]."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    out: list[tuple[float, np.ndarray]] = []
    for k in range(num_steps):
        s = k / (num_steps - 1)
        out.append((float(s), (1.0 - s) * H_easy + s * H_target))
    return out


@dataclass
class AdiabaticTrace:
    s: list[float]
    energy: list[float]
    params: list[np.ndarray]
    nfev_total: int


def adiabatic_train(
    spec: CircuitSpec,
    backend,
    H_easy: np.ndarray,
    H_target: np.ndarray,
    num_steps: int,
    optimize_at_step: Callable[
        [Callable[[np.ndarray], float], np.ndarray], "tuple[np.ndarray, float, int]"
    ],
    initial_params: np.ndarray | None = None,
) -> tuple[np.ndarray, AdiabaticTrace]:
    """Run the adiabatic schedule over `num_steps` points.

    `optimize_at_step(objective, x0)` is the user-supplied local optimizer.
    It must return (best_params, best_value, nfev). We give it the current
    step's expectation-value loss and the warm-start `x0`, and we use its
    output as the warm-start for the next step.

    Returns the parameters at s = 1 plus a trace of (s, energy, params)
    at every step for diagnostics.
    """
    schedule = linear_interpolation(H_easy, H_target, num_steps)

    if initial_params is None:
        params = np.zeros(spec.num_params)
    else:
        params = np.asarray(initial_params, dtype=float).copy()

    trace_s: list[float] = []
    trace_e: list[float] = []
    trace_p: list[np.ndarray] = []
    total_nfev = 0

    for s, H_s in schedule:
        def loss(p, H=H_s):
            state = backend.statevector(spec, p)
            return float(expectation(H, state))

        params, val, nfev = optimize_at_step(loss, params)
        total_nfev += int(nfev)
        trace_s.append(s)
        trace_e.append(float(val))
        trace_p.append(params.copy())

    trace = AdiabaticTrace(
        s=trace_s, energy=trace_e, params=trace_p, nfev_total=total_nfev
    )
    return params, trace


def easy_x_field_hamiltonian(n: int) -> np.ndarray:
    """Diagonal-in-X-basis easy starting Hamiltonian: H_easy = -∑_i X_i.
    Ground state = |+>^n (uniform superposition), trivial to prepare with
    one Hadamard per qubit. Use as a starting point for problems on n
    qubits whose Hamiltonian is more complicated.
    """
    from quiver.hamiltonian import _kron_chain, _I, _X
    H = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n):
        ops = [_I] * n
        ops[i] = _X
        H -= _kron_chain(ops)
    return H
