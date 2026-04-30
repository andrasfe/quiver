"""Subspace expansion utilities and adaptive K selection.

The DS-VQE construction trains K circuits and solves the generalized
eigenvalue problem ``M c = E S c`` in the span of their final states.
Picking K up-front is wasteful: most quantum runtime budget is spent on
circuits that turn out to be near-collinear with circuits already in the
basis, contributing zero new rank to S.

`select_k_adaptive` does the screening *classically* (noiseless training
on a numpy state-vector simulator) and only commits to circuits whose
trained state grows the effective rank of the overlap matrix beyond a
threshold. The caller then runs only the surviving K on hardware.

Public API:

  - :func:`subspace_diagonalize` — Ritz solve in the span of K states.
  - :func:`select_k_adaptive`   — greedy classical pre-screen of an
    ansatz pool, returning indices/states/params worth running.

Both functions are pure-numpy: no qiskit dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import scipy.linalg
from scipy.optimize import minimize

from quivercirc.backends import NumpyBackend
from quivercirc.circuit import CircuitSpec


def subspace_diagonalize(
    states: Sequence[np.ndarray],
    H: np.ndarray,
    eps: float = 1e-9,
) -> tuple[float, list[float], int]:
    """Solve ``M c = E S c`` in the span of K (non-orthogonal) states.

    Parameters
    ----------
    states : K state vectors of dimension 2**n
    H      : (2**n, 2**n) Hermitian Hamiltonian
    eps    : eigenvalues of S below ``eps * max(eig(S))`` are dropped
             when forming the canonical orthonormal basis (numerical
             rank cutoff for the Cholesky-like inverse-square-root step).

    Returns
    -------
    (E_min, eigvals_of_S_descending, effective_rank)
    """
    norm_states = [s / (np.linalg.norm(s) + 1e-15) for s in states]
    K = len(norm_states)
    M = np.zeros((K, K), dtype=complex)
    S = np.zeros((K, K), dtype=complex)
    for i in range(K):
        for j in range(K):
            M[i, j] = np.vdot(norm_states[i], H @ norm_states[j])
            S[i, j] = np.vdot(norm_states[i], norm_states[j])
    M = 0.5 * (M + M.conj().T)
    S = 0.5 * (S + S.conj().T)
    s_eig, s_vec = scipy.linalg.eigh(S)
    keep = s_eig > eps * s_eig.max()
    rank = int(keep.sum())
    if rank == 0:
        return float("nan"), [float(np.real(x)) for x in s_eig[::-1]], 0
    Q = s_vec[:, keep] / np.sqrt(s_eig[keep])
    Mp = Q.conj().T @ M @ Q
    Mp = 0.5 * (Mp + Mp.conj().T)
    e_eig, _ = scipy.linalg.eigh(Mp)
    return (
        float(np.real(e_eig[0])),
        [float(np.real(x)) for x in s_eig[::-1]],
        rank,
    )


def _parameter_shift_grad(loss, params, shift=np.pi / 2):
    grad = np.zeros(len(params))
    for i in range(len(params)):
        e = np.zeros_like(params)
        e[i] = shift
        grad[i] = 0.5 * (loss(params + e) - loss(params - e))
    return grad


def _default_train(
    spec: CircuitSpec,
    H,
    backend,
    seed: int,
    max_iter: int = 80,
    init_scale: float = 0.1,
    H_diag=None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Cheap noiseless training: L-BFGS-B with gradients.

    If `backend` exposes a `make_loss_and_grad` method (e.g.
    :class:`JaxBackend`) it is used directly — JAX autodiff replaces
    the explicit parameter-shift loop and is ~3000x faster at n=20.
    Otherwise falls back to numpy parameter-shift.
    """
    rng = np.random.default_rng(seed)
    init = rng.normal(0.0, init_scale, spec.num_params)

    if hasattr(backend, "make_loss_and_grad"):
        # Backend-native autodiff path (JAX). Returns (val, grad) per call.
        loss_and_grad = backend.make_loss_and_grad(spec, H_diag=H_diag, H=H)

        def loss(p):
            v, _ = loss_and_grad(p)
            return v

        res = minimize(
            loss_and_grad, init, method="L-BFGS-B", jac=True,
            options={"maxiter": max_iter, "ftol": 1e-8, "gtol": 1e-6},
        )
        params = np.asarray(res.x)
        state = backend.statevector(spec, params)
        return params, float(res.fun), state

    def loss(p):
        psi = backend.statevector(spec, p)
        return float(np.real(np.vdot(psi, H @ psi)))

    res = minimize(
        loss, init, method="L-BFGS-B",
        jac=lambda p: _parameter_shift_grad(loss, p),
        options={"maxiter": max_iter, "ftol": 1e-8, "gtol": 1e-6},
    )
    params = np.asarray(res.x)
    state = backend.statevector(spec, params)
    return params, float(res.fun), state


@dataclass
class AdaptiveKResult:
    """Outcome of an adaptive-K screening pass.

    Attributes
    ----------
    selected_indices : positions in the input pool that were accepted.
    states           : trained state vectors for selected circuits.
    params           : trained parameters for selected circuits.
    individual_energies : per-circuit ideal energy at the trained params.
    subspace_energy  : Ritz-eigenvalue estimate from the accepted basis.
    overlap_eigvals  : eigenvalues of S (descending) at termination.
    history          : per-iteration log of (idx, accepted, rank, λ_min, E_sub)
                       — useful for diagnostics and Table 5.
    """
    selected_indices: list[int] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    params: list[np.ndarray] = field(default_factory=list)
    individual_energies: list[float] = field(default_factory=list)
    subspace_energy: float = float("inf")
    overlap_eigvals: list[float] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)


def select_k_adaptive(
    candidates: Sequence[CircuitSpec],
    H: np.ndarray,
    *,
    backend: NumpyBackend | None = None,
    seeds: Sequence[int] | None = None,
    max_k: int | None = None,
    rank_threshold: float = 1e-4,
    energy_tol: float = 1e-3,
    patience: int = 2,
    train_fn: Callable[[CircuitSpec, int], tuple[np.ndarray, float, np.ndarray]] | None = None,
) -> AdaptiveKResult:
    """Pick K adaptively by greedy overlap-rank growth.

    Each candidate is trained noiselessly, then its state is offered to
    a growing basis; we accept only if (a) the smallest non-trivial
    eigenvalue of the new K×K overlap matrix S exceeds
    ``rank_threshold``, *or* (b) the subspace Ritz energy improves by
    more than ``energy_tol``. After ``patience`` consecutive rejections
    *or* on reaching ``max_k`` accepted circuits, screening stops.

    The expensive resource (hardware shot budget) is touched zero times
    — this is a fully classical pre-screen meant to decide which of the
    pool's circuits are worth submitting to a real device.

    Parameters
    ----------
    candidates : pool of pre-built :class:`CircuitSpec` objects (as from
        ``Ansatz.build()``).
    H          : Hamiltonian as a dense matrix.
    backend    : :class:`NumpyBackend`. Default constructs one matching
        ``candidates[0].num_qubits``.
    seeds      : per-candidate RNG seeds for parameter initialisation.
        Default is ``range(len(candidates))`` so screening is reproducible.
    max_k      : cap on accepted circuits. Default ``len(candidates)``.
    rank_threshold : minimum tolerated value of the smallest S-eigenvalue
        after canonical normalisation. Below this the new state is
        considered numerically dependent on the existing basis.
    energy_tol : if rank check fails, accept anyway when the subspace
        energy drops by at least this much.
    patience   : terminate after this many consecutive rejections.
    train_fn   : custom training callback, signature
        ``(spec, seed) -> (params, energy, state)``. If ``None``, uses
        ``_default_train`` with the given backend.

    Returns
    -------
    :class:`AdaptiveKResult` with the accepted basis and full diagnostics.
    """
    if not candidates:
        return AdaptiveKResult()

    if backend is None:
        backend = NumpyBackend(num_qubits=candidates[0].num_qubits)
    if seeds is None:
        seeds = list(range(len(candidates)))
    if len(seeds) < len(candidates):
        raise ValueError("seeds must be at least as long as candidates")
    if max_k is None:
        max_k = len(candidates)
    if train_fn is None:
        def train_fn(spec, seed):
            return _default_train(spec, H, backend, seed)

    result = AdaptiveKResult()
    rejections = 0
    prev_E = float("inf")

    for idx, (spec, seed) in enumerate(zip(candidates, seeds)):
        params, energy, state = train_fn(spec, seed)
        norm = np.linalg.norm(state)
        if norm < 1e-12:
            result.history.append({
                "idx": idx, "accepted": False, "reason": "zero-norm state",
            })
            continue
        state = state / norm

        trial_states = result.states + [state]
        E_sub, s_eig, rank = subspace_diagonalize(trial_states, H)
        # smallest *non-trivial* S-eigenvalue (descending order)
        lam_min = s_eig[len(trial_states) - 1] if s_eig else 0.0

        accept = False
        reason = ""
        if not result.states:
            accept = True
            reason = "first-state"
        elif lam_min >= rank_threshold:
            accept = True
            reason = f"rank-grew (lam_min={lam_min:.2e})"
        elif (prev_E - E_sub) >= energy_tol:
            accept = True
            reason = f"energy-improved ({prev_E - E_sub:.2e})"
        else:
            reason = (
                f"redundant (lam_min={lam_min:.2e} < {rank_threshold:.0e}, "
                f"ΔE={prev_E - E_sub:.2e})"
            )

        result.history.append({
            "idx": idx, "accepted": accept, "reason": reason,
            "ideal_energy": energy, "lam_min": lam_min,
            "rank": rank, "E_sub": E_sub,
        })

        if accept:
            result.selected_indices.append(idx)
            result.states.append(state)
            result.params.append(params)
            result.individual_energies.append(energy)
            result.subspace_energy = E_sub
            result.overlap_eigvals = s_eig
            prev_E = E_sub
            rejections = 0
            if len(result.selected_indices) >= max_k:
                break
        else:
            rejections += 1
            if rejections >= patience:
                break

    return result


__all__ = ["subspace_diagonalize", "select_k_adaptive", "AdaptiveKResult"]
