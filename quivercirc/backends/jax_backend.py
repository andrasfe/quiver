"""JAX-based statevector simulator with JIT compilation.

Designed for the same `Backend` interface as :class:`NumpyBackend` but
much faster on circuits with many parameter-shift gradient evaluations
(typical VQE workload at n>=15). Each :class:`CircuitSpec` is compiled
once into a closed-over JAX function that maps the parameter vector to
the final state vector; subsequent calls run as a fused XLA kernel.

Two notable improvements over the numpy backend:

  1. JIT-fused forward pass — the entire circuit becomes a single XLA
     graph after the first call, so per-gate Python overhead disappears
     and contiguous arithmetic is auto-fused.
  2. Native autodiff — paired with the helper in
     :mod:`quivercirc.subspace`, gradients are computed in one
     reverse-mode pass instead of `2 * num_params` parameter-shift
     forward passes. This is roughly an `O(num_params)` win on top of
     the JIT speedup, dominant for shallow circuits with tens of
     parameters.

Defaults to complex64 (single precision) — accuracy is much better
than the shot-noise floor of any realistic quantum experiment, and
single precision halves memory and compute on top of JIT.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp

from quivercirc.circuit import CircuitSpec, GateSpec


# enable double precision globally — float32 is too imprecise for
# accumulated inner products over 2^n amplitudes at n>=15 (one
# observed failure mode: classical |<psi_i|psi_j>|^2 rounds to zero
# at n=20 in complex64, which collapses the subspace eigenvalue
# problem to "pick the best diagonal"). The 2x slowdown over float32
# is still ~1500x faster than numpy parameter-shift gradient at n=20.
jax.config.update("jax_enable_x64", True)
_DTYPE = jnp.complex128


# ---------- gate primitives (JAX) ------------------------------------------


def _h_jax():
    inv = 1.0 / jnp.sqrt(jnp.array(2.0, dtype=_DTYPE))
    return jnp.array([[inv, inv], [inv, -inv]], dtype=_DTYPE)


def _x_jax():
    return jnp.array([[0, 1], [1, 0]], dtype=_DTYPE)


def _y_jax():
    return jnp.array([[0, -1j], [1j, 0]], dtype=_DTYPE)


def _z_jax():
    return jnp.array([[1, 0], [0, -1]], dtype=_DTYPE)


def _s_jax():
    return jnp.array([[1, 0], [0, 1j]], dtype=_DTYPE)


def _t_jax():
    return jnp.array(
        [[1, 0], [0, jnp.exp(1j * jnp.pi / 4)]], dtype=_DTYPE,
    )


def _rx_jax(theta):
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    zero = jnp.zeros_like(c)
    return jnp.stack([
        jnp.stack([c.astype(_DTYPE), (-1j * s).astype(_DTYPE)]),
        jnp.stack([(-1j * s).astype(_DTYPE), c.astype(_DTYPE)]),
    ])


def _ry_jax(theta):
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.stack([
        jnp.stack([c.astype(_DTYPE), (-s).astype(_DTYPE)]),
        jnp.stack([s.astype(_DTYPE), c.astype(_DTYPE)]),
    ])


def _rz_jax(theta):
    em = jnp.exp(-1j * theta / 2).astype(_DTYPE)
    ep = jnp.exp(1j * theta / 2).astype(_DTYPE)
    zero = jnp.zeros_like(em)
    return jnp.stack([
        jnp.stack([em, zero]),
        jnp.stack([zero, ep]),
    ])


def _cnot_jax():
    return jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=_DTYPE,
    )


def _cz_jax():
    return jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=_DTYPE,
    )


def _swap_jax():
    return jnp.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=_DTYPE,
    )


def _rzz_jax(theta):
    em = jnp.exp(-1j * theta / 2).astype(_DTYPE)
    ep = jnp.exp(1j * theta / 2).astype(_DTYPE)
    return jnp.diag(jnp.stack([em, ep, ep, em]))


def _rxx_jax(theta):
    c = jnp.cos(theta / 2).astype(_DTYPE)
    s = jnp.sin(theta / 2).astype(_DTYPE)
    is_ = -1j * s
    z = jnp.zeros((), dtype=_DTYPE)
    return jnp.array([
        [c,  z,  z,  is_],
        [z,  c,  is_, z ],
        [z,  is_, c,  z ],
        [is_, z,  z,  c ],
    ])


def _ryy_jax(theta):
    c = jnp.cos(theta / 2).astype(_DTYPE)
    s = jnp.sin(theta / 2).astype(_DTYPE)
    is_ = -1j * s
    pis = 1j * s
    z = jnp.zeros((), dtype=_DTYPE)
    return jnp.array([
        [c,  z,  z,  pis],
        [z,  c,  is_, z ],
        [z,  is_, c,  z ],
        [pis, z,  z,  c ],
    ])


_FIXED_1Q_NAMES = {"h", "x", "y", "z", "s", "t", "sdg", "tdg"}
_FIXED_2Q_NAMES = {"cnot", "cx", "cz", "swap"}


def _fixed_1q_matrix(name):
    if name == "h":
        return _h_jax()
    if name == "x":
        return _x_jax()
    if name == "y":
        return _y_jax()
    if name == "z":
        return _z_jax()
    if name == "s":
        return _s_jax()
    if name == "t":
        return _t_jax()
    if name == "sdg":
        return _s_jax().conj().T
    if name == "tdg":
        return _t_jax().conj().T
    raise ValueError(name)


def _fixed_2q_matrix(name):
    if name in ("cnot", "cx"):
        return _cnot_jax()
    if name == "cz":
        return _cz_jax()
    if name == "swap":
        return _swap_jax()
    raise ValueError(name)


def _gate_matrix_jax(gate: GateSpec, params: jnp.ndarray):
    name = gate.name
    if name in _FIXED_1Q_NAMES:
        return _fixed_1q_matrix(name)
    if name in _FIXED_2Q_NAMES:
        return _fixed_2q_matrix(name)
    theta = params[gate.param_idx]
    if name == "rx":
        return _rx_jax(theta)
    if name == "ry":
        return _ry_jax(theta)
    if name == "rz":
        return _rz_jax(theta)
    if name == "rzz":
        return _rzz_jax(theta)
    if name == "rxx":
        return _rxx_jax(theta)
    if name == "ryy":
        return _ryy_jax(theta)
    raise ValueError(f"unknown gate '{name}'")


# ---------- gate application -----------------------------------------------


def _apply_1q_jax(state, mat, qubit, num_qubits):
    state = state.reshape([2] * num_qubits)
    axis = num_qubits - 1 - qubit
    state = jnp.tensordot(mat, state, axes=([1], [axis]))
    state = jnp.moveaxis(state, 0, axis)
    return state.reshape(2 ** num_qubits)


def _apply_2q_jax(state, mat, qubits, num_qubits):
    q1, q2 = qubits
    state = state.reshape([2] * num_qubits)
    a1 = num_qubits - 1 - q1
    a2 = num_qubits - 1 - q2
    mat4 = mat.reshape(2, 2, 2, 2)
    state = jnp.tensordot(mat4, state, axes=([2, 3], [a1, a2]))
    state = jnp.moveaxis(state, [0, 1], [a1, a2])
    return state.reshape(2 ** num_qubits)


# ---------- backend --------------------------------------------------------


def _make_circuit_fn(spec: CircuitSpec, num_qubits: int) -> Callable:
    """Return a JIT-compiled function (params: jnp.ndarray) -> statevector.
    The gate sequence is captured statically so JAX traces the graph
    once at first call and reuses the compiled XLA program afterwards.
    """
    gates = list(spec.gates)
    n = num_qubits

    def circuit_fn(params):
        psi = jnp.zeros(2 ** n, dtype=_DTYPE)
        psi = psi.at[0].set(jnp.array(1.0, dtype=_DTYPE))
        for g in gates:
            mat = _gate_matrix_jax(g, params)
            if g.is_two_qubit:
                psi = _apply_2q_jax(psi, mat, g.qubits, n)
            else:
                psi = _apply_1q_jax(psi, mat, g.qubits[0], n)
        return psi

    return jax.jit(circuit_fn)


@dataclass
class JaxBackend:
    """A drop-in replacement for :class:`NumpyBackend` that uses JAX.

    Caches JIT-compiled circuit functions keyed by id(spec), so repeated
    statevector queries on the same spec at different parameters reuse
    the compiled XLA program. Numpy arrays are accepted and returned at
    the API boundary; JAX arrays only live inside the simulator.
    """

    num_qubits: int

    def __post_init__(self) -> None:
        self._fn_cache: dict[int, Callable] = {}

    def _get_fn(self, spec: CircuitSpec) -> Callable:
        if spec.num_qubits != self.num_qubits:
            raise ValueError("circuit width does not match backend width")
        key = id(spec)
        fn = self._fn_cache.get(key)
        if fn is None:
            fn = _make_circuit_fn(spec, self.num_qubits)
            self._fn_cache[key] = fn
        return fn

    def statevector(self, spec: CircuitSpec, params) -> np.ndarray:
        fn = self._get_fn(spec)
        psi = fn(jnp.asarray(params, dtype=jnp.float64))
        return np.asarray(psi).astype(np.complex128)

    # --- helpers tailored for VQE training (autodiff over <psi|H|psi>) --

    def make_loss_and_grad(
        self, spec: CircuitSpec, H_diag: np.ndarray | None = None,
        H: np.ndarray | None = None,
    ) -> Callable:
        """Return a JIT'd `(params: ndarray) -> (loss, grad)` closure
        for the energy <psi|H|psi>. If ``H_diag`` is given, treats H as
        a diagonal operator (saves O(2^n) memory and matvec). Otherwise
        falls back to the dense ``H`` matrix.
        """
        fn = self._get_fn(spec)
        if H_diag is not None:
            Hd = jnp.asarray(H_diag, dtype=_DTYPE)

            def loss(p):
                psi = fn(p)
                return jnp.real(jnp.sum(jnp.conj(psi) * (Hd * psi)))
        elif H is not None:
            Hm = jnp.asarray(H, dtype=_DTYPE)

            def loss(p):
                psi = fn(p)
                return jnp.real(jnp.vdot(psi, Hm @ psi))
        else:
            raise ValueError("provide H_diag or H")

        loss_and_grad = jax.jit(jax.value_and_grad(loss))

        def closure(params_np):
            params_j = jnp.asarray(params_np, dtype=jnp.float64)
            val, grad = loss_and_grad(params_j)
            return float(val), np.asarray(grad).astype(np.float64)

        return closure

    def make_fidelity_loss_and_grad(
        self, spec: CircuitSpec, target: np.ndarray,
    ) -> Callable:
        """Return a JIT'd `(params) -> (loss, grad)` for the *infidelity*
        ``loss(theta) = 1 - |<target | U(theta) |0>|^2``. Used by the
        warmstart phase of coordinated training: minimising this drives
        the trained state toward the reference ``target`` state.
        """
        fn = self._get_fn(spec)
        target_j = jnp.asarray(target, dtype=_DTYPE)

        def loss(p):
            psi = fn(p)
            ovl = jnp.vdot(target_j, psi)
            return 1.0 - jnp.real(ovl * jnp.conj(ovl))

        loss_and_grad = jax.jit(jax.value_and_grad(loss))

        def closure(params_np):
            params_j = jnp.asarray(params_np, dtype=jnp.float64)
            val, grad = loss_and_grad(params_j)
            return float(val), np.asarray(grad).astype(np.float64)

        return closure


__all__ = ["JaxBackend"]
