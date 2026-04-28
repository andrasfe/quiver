"""Persistence: round-trippable JSON serialization for CircuitSpecs,
parameters, and registries.

The format is human-readable JSON so the serialised circuits can be
loaded by any consumer — Python notebook, another language, hardware-
submission tooling — without the consumer needing to import Quiver.
Each saved registry contains:

  - top-level metadata: target description, problem family, exact
    reference value (if any), generation timestamp, software version
  - a list of solutions, each a dict with:
      * family            : str — ansatz family tag from the registry
      * num_qubits        : int
      * gate_count        : int
      * depth             : int
      * two_qubit_count   : int
      * num_params        : int
      * params            : list[float]
      * verifier_score    : float
      * objective         : float
      * diversity         : float
      * spec              : { gates: [...], num_qubits, num_params }

The `spec.gates` items are dicts of {name, qubits, param_idx} matching
GateSpec exactly — directly reconstructable.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from quivercirc.circuit import CircuitSpec, GateSpec
from quivercirc.core import Solution


# ---------- core (de)serialisation --------------------------------------


def spec_to_dict(spec: CircuitSpec) -> dict:
    return {
        "num_qubits": spec.num_qubits,
        "num_params": spec.num_params,
        "gates": [
            {"name": g.name, "qubits": list(g.qubits), "param_idx": g.param_idx}
            for g in spec.gates
        ],
    }


def spec_from_dict(d: dict) -> CircuitSpec:
    spec = CircuitSpec(num_qubits=int(d["num_qubits"]))
    spec.num_params = int(d["num_params"])
    for g in d["gates"]:
        spec.gates.append(GateSpec(
            name=str(g["name"]),
            qubits=tuple(int(q) for q in g["qubits"]),
            param_idx=None if g["param_idx"] is None else int(g["param_idx"]),
        ))
    return spec


def solution_to_dict(sol: Solution) -> dict:
    return {
        "family": sol.family,
        "num_qubits": sol.spec.num_qubits,
        "gate_count": sol.gate_count,
        "depth": sol.depth,
        "two_qubit_count": sol.two_qubit_count,
        "num_params": len(sol.params),
        "params": [float(p) for p in sol.params],
        "verifier_score": float(sol.fidelity),
        "objective": float(sol.objective),
        "diversity": float(sol.diversity),
        "spec": spec_to_dict(sol.spec),
    }


@dataclass
class LoadedSolution:
    """A solution rehydrated from disk. Carries the spec and params plus
    the metadata that was present at save time. Plays the same role as a
    `quiver.core.Solution` but doesn't require a live registry."""
    family: str
    spec: CircuitSpec
    params: np.ndarray
    verifier_score: float
    objective: float
    gate_count: int
    depth: int
    two_qubit_count: int
    num_params: int
    diversity: float


def loaded_solution_from_dict(d: dict) -> LoadedSolution:
    return LoadedSolution(
        family=str(d["family"]),
        spec=spec_from_dict(d["spec"]),
        params=np.asarray(d["params"], dtype=float),
        verifier_score=float(d["verifier_score"]),
        objective=float(d["objective"]),
        gate_count=int(d["gate_count"]),
        depth=int(d["depth"]),
        two_qubit_count=int(d["two_qubit_count"]),
        num_params=int(d["num_params"]),
        diversity=float(d["diversity"]),
    )


# ---------- registry-level helpers --------------------------------------


def save_registry(
    solutions: Iterable[Solution],
    path: str | Path,
    metadata: dict | None = None,
) -> None:
    """Write a registry of Solutions plus arbitrary metadata to JSON.

    Suggested metadata keys:
        target_description : str — what problem these circuits solve
        target_kind        : str — e.g. "state_prep", "maxcut", "vqe"
        num_qubits         : int — width of every circuit
        exact_reference    : float | None — optimal value if known
    """
    payload = {
        "schema_version": 1,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "metadata": metadata or {},
        "solutions": [solution_to_dict(s) for s in solutions],
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_registry(path: str | Path) -> "tuple[list[LoadedSolution], dict]":
    payload = json.loads(Path(path).read_text())
    sols = [loaded_solution_from_dict(d) for d in payload["solutions"]]
    return sols, dict(payload.get("metadata", {}))


# ---------- single-circuit convenience ----------------------------------


def save_circuit(
    spec: CircuitSpec,
    params: np.ndarray,
    path: str | Path,
    metadata: dict | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "metadata": metadata or {},
        "spec": spec_to_dict(spec),
        "params": [float(p) for p in params],
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_circuit(path: str | Path) -> "tuple[CircuitSpec, np.ndarray, dict]":
    payload = json.loads(Path(path).read_text())
    spec = spec_from_dict(payload["spec"])
    params = np.asarray(payload["params"], dtype=float)
    return spec, params, dict(payload.get("metadata", {}))
