"""Structural mutation of CircuitSpecs — the primitive that lets Quiver
discover circuits whose shape is not in any human-designed template.

Five operators, each producing a new (CircuitSpec, params) pair (the
input is never modified). Threading params through mutation lets us
**warm-start** the optimizer from the parent's verified parameters,
which is essential — mutated structures are too random for the optimizer
to find good params from a cold start.

  insert    — drop a fresh random gate at a random position.
              If parametric, append 0.0 to the params array.
  delete    — remove a gate; if parametric, drop its slot from params
              and shift later indices down.
  swap      — swap two adjacent gates. params unchanged.
  retarget  — change one of a gate's qubits. params unchanged.
  retype    — replace gate name within same arity/parametricity class.
              params unchanged.

After mutation the param array's length always matches the new spec's
num_params. The Mutator.chain() helper applies k operators in sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from quivercirc.circuit import CircuitSpec, GateSpec
from quivercirc.microstructures import MicrostructureLibrary, weld


PARAM_1Q = ("rx", "ry", "rz")
FIXED_1Q = ("h", "x", "y", "z", "s", "t")
FIXED_2Q = ("cnot", "cz", "swap", "iswap", "sqrt_iswap")
PARAM_2Q = ("rxx", "ryy", "rzz")


def _clone(spec: CircuitSpec, gates: list[GateSpec], num_params: int) -> CircuitSpec:
    out = CircuitSpec(num_qubits=spec.num_qubits)
    out.gates = list(gates)
    out.num_params = num_params
    return out


def _random_gate_skeleton(num_qubits: int, rng: np.random.Generator
                          ) -> tuple[str, tuple[int, ...]]:
    arity = 1 if (num_qubits < 2 or rng.random() < 0.5) else 2
    if arity == 1:
        pool = PARAM_1Q + FIXED_1Q
        name = str(rng.choice(pool))
        q = int(rng.integers(0, num_qubits))
        return name, (q,)
    pool = FIXED_2Q + PARAM_2Q
    name = str(rng.choice(pool))
    q1 = int(rng.integers(0, num_qubits))
    q2 = int(rng.integers(0, num_qubits))
    while q2 == q1:
        q2 = int(rng.integers(0, num_qubits))
    return name, (q1, q2)


def _is_parametric(name: str) -> bool:
    return name in PARAM_1Q or name in PARAM_2Q


# ---------- mutation operators -------------------------------------------


def mutate_insert(spec: CircuitSpec, params: np.ndarray, rng: np.random.Generator
                  ) -> tuple[CircuitSpec, np.ndarray]:
    name, qubits = _random_gate_skeleton(spec.num_qubits, rng)
    pos = int(rng.integers(0, spec.gate_count + 1))
    new_gates = list(spec.gates)
    if _is_parametric(name):
        gate = GateSpec(name, qubits, spec.num_params)
        new_gates.insert(pos, gate)
        new_spec = _clone(spec, new_gates, spec.num_params + 1)
        new_params = np.concatenate([params, [0.0]])
        return new_spec, new_params
    gate = GateSpec(name, qubits, None)
    new_gates.insert(pos, gate)
    return _clone(spec, new_gates, spec.num_params), params.copy()


def mutate_delete(spec: CircuitSpec, params: np.ndarray, rng: np.random.Generator
                  ) -> tuple[CircuitSpec, np.ndarray]:
    if not spec.gates:
        return spec, params.copy()
    idx = int(rng.integers(0, spec.gate_count))
    removed = spec.gates[idx]
    remaining = [g for i, g in enumerate(spec.gates) if i != idx]
    if not removed.is_parametric:
        return _clone(spec, remaining, spec.num_params), params.copy()
    slot = removed.param_idx
    compacted: list[GateSpec] = []
    for g in remaining:
        if g.is_parametric and g.param_idx > slot:
            compacted.append(GateSpec(g.name, g.qubits, g.param_idx - 1))
        else:
            compacted.append(g)
    new_params = np.concatenate([params[:slot], params[slot + 1 :]])
    return _clone(spec, compacted, spec.num_params - 1), new_params


def mutate_swap(spec: CircuitSpec, params: np.ndarray, rng: np.random.Generator
                ) -> tuple[CircuitSpec, np.ndarray]:
    if spec.gate_count < 2:
        return spec, params.copy()
    i = int(rng.integers(0, spec.gate_count - 1))
    new_gates = list(spec.gates)
    new_gates[i], new_gates[i + 1] = new_gates[i + 1], new_gates[i]
    return _clone(spec, new_gates, spec.num_params), params.copy()


def mutate_retarget(spec: CircuitSpec, params: np.ndarray, rng: np.random.Generator
                    ) -> tuple[CircuitSpec, np.ndarray]:
    if not spec.gates or spec.num_qubits < 2:
        return spec, params.copy()
    idx = int(rng.integers(0, spec.gate_count))
    g = spec.gates[idx]
    qs = list(g.qubits)
    pos = int(rng.integers(0, len(qs)))
    others = [qs[k] for k in range(len(qs)) if k != pos]
    candidates = [q for q in range(spec.num_qubits) if q not in others]
    if not candidates:
        return spec, params.copy()
    qs[pos] = int(rng.choice(candidates))
    new_gates = list(spec.gates)
    new_gates[idx] = GateSpec(g.name, tuple(qs), g.param_idx)
    return _clone(spec, new_gates, spec.num_params), params.copy()


def mutate_retype(spec: CircuitSpec, params: np.ndarray, rng: np.random.Generator
                  ) -> tuple[CircuitSpec, np.ndarray]:
    if not spec.gates:
        return spec, params.copy()
    idx = int(rng.integers(0, spec.gate_count))
    g = spec.gates[idx]
    if g.is_parametric:
        pool = PARAM_1Q if len(g.qubits) == 1 else PARAM_2Q
    else:
        pool = FIXED_1Q if len(g.qubits) == 1 else FIXED_2Q
    alternatives = [n for n in pool if n != g.name]
    if not alternatives:
        return spec, params.copy()
    new_name = str(rng.choice(alternatives))
    new_gates = list(spec.gates)
    new_gates[idx] = GateSpec(new_name, g.qubits, g.param_idx)
    return _clone(spec, new_gates, spec.num_params), params.copy()


# ---------- mutator façade ----------------------------------------------


@dataclass
class Mutator:
    operations: tuple[Callable, ...] = (
        mutate_insert,
        mutate_delete,
        mutate_swap,
        mutate_retarget,
        mutate_retype,
    )
    weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    chain_min: int = 1
    chain_max: int = 3
    # When set, mutation rounds can also weld a learned fragment onto the
    # parent circuit — this is structural recombination across the
    # accumulated library, not just point edits.
    microstructure_library: MicrostructureLibrary | None = None
    weld_weight: float = 1.5

    def _ops_and_weights(self) -> tuple[list[Callable], list[float]]:
        ops = list(self.operations)
        weights = list(self.weights)
        if self.microstructure_library is not None and self.microstructure_library.fragments:
            ops.append(self._weld_op)
            weights.append(self.weld_weight)
        return ops, weights

    def _weld_op(self, spec: CircuitSpec, params: np.ndarray,
                 rng: np.random.Generator) -> tuple[CircuitSpec, np.ndarray]:
        assert self.microstructure_library is not None
        frag = self.microstructure_library.sample(spec.num_qubits, rng)
        if frag is None:
            return spec, params.copy()
        return weld(spec, params, frag)

    def step(self, spec: CircuitSpec, params: np.ndarray,
             rng: np.random.Generator) -> tuple[CircuitSpec, np.ndarray]:
        ops, weights = self._ops_and_weights()
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        op = ops[int(rng.choice(len(ops), p=w))]
        return op(spec, params, rng)

    def chain(self, spec: CircuitSpec, params: np.ndarray,
              rng: np.random.Generator) -> tuple[CircuitSpec, np.ndarray]:
        k = int(rng.integers(self.chain_min, self.chain_max + 1))
        s, p = spec, np.asarray(params, dtype=float)
        for _ in range(k):
            s, p = self.step(s, p, rng)
        return s, p
