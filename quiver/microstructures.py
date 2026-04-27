"""Microstructure library — continual learning of useful gate fragments.

Every time Quiver accepts a verified circuit, the library indexes random
contiguous sub-sequences of that circuit as reusable fragments. Adaptive
growth can then propose adding either a single gate *or* a learned
fragment in any given step. As the registry grows, the library grows
with it, and the growth process effectively learns its own primitives.

A "fragment" is a list of `GateSpec`s with parameter slots remapped to
[0, k) so a fragment with k parametric gates becomes a self-contained
sub-circuit that can be welded onto any growing spec by:

  1. allocating k new param slots in the host spec
  2. shifting the fragment's param_idx values into those slots
  3. appending the gates verbatim (qubits unchanged)

Fragments only weld in when their qubit set is a subset of the host's
qubits — same-problem assumption keeps the design simple.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quiver.circuit import CircuitSpec, GateSpec


@dataclass
class Fragment:
    gates: list[GateSpec]                # param_idx values are 0..k-1
    params: np.ndarray                    # length k

    @property
    def num_params(self) -> int:
        return len(self.params)

    @property
    def length(self) -> int:
        return len(self.gates)

    def qubits_used(self) -> set[int]:
        out: set[int] = set()
        for g in self.gates:
            out.update(g.qubits)
        return out


@dataclass
class MicrostructureLibrary:
    fragments: list[Fragment] = field(default_factory=list)
    fragments_per_solution: int = 4
    min_length: int = 2
    max_length: int = 6

    def to_dict(self) -> dict:
        return {
            "fragments_per_solution": self.fragments_per_solution,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "fragments": [
                {
                    "gates": [
                        {"name": g.name, "qubits": list(g.qubits),
                         "param_idx": g.param_idx}
                        for g in f.gates
                    ],
                    "params": f.params.tolist(),
                }
                for f in self.fragments
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MicrostructureLibrary":
        lib = cls(
            fragments_per_solution=d.get("fragments_per_solution", 4),
            min_length=d.get("min_length", 2),
            max_length=d.get("max_length", 6),
        )
        for f_dict in d.get("fragments", []):
            gates = [
                GateSpec(g["name"], tuple(g["qubits"]), g["param_idx"])
                for g in f_dict["gates"]
            ]
            params = np.array(f_dict["params"], dtype=float)
            lib.fragments.append(Fragment(gates=gates, params=params))
        return lib

    def save_json(self, path) -> None:
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path) -> "MicrostructureLibrary":
        import json
        from pathlib import Path
        return cls.from_dict(json.loads(Path(path).read_text()))

    def add_solution(
        self,
        spec: CircuitSpec,
        params: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """Index up to `fragments_per_solution` random contiguous windows
        from the verified circuit. Returns the number actually added."""
        if spec.gate_count < self.min_length:
            return 0
        added = 0
        for _ in range(self.fragments_per_solution):
            length = int(rng.integers(
                self.min_length,
                min(self.max_length, spec.gate_count) + 1,
            ))
            start = int(rng.integers(0, spec.gate_count - length + 1))
            window = spec.gates[start : start + length]

            param_idx_in_window = sorted(
                {g.param_idx for g in window if g.is_parametric}
            )
            slot_remap = {old: new for new, old in enumerate(param_idx_in_window)}
            remapped_gates = []
            for g in window:
                if g.is_parametric:
                    remapped_gates.append(
                        GateSpec(g.name, g.qubits, slot_remap[g.param_idx])
                    )
                else:
                    remapped_gates.append(GateSpec(g.name, g.qubits, None))

            fragment_params = np.array(
                [params[idx] for idx in param_idx_in_window], dtype=float
            )
            self.fragments.append(
                Fragment(gates=remapped_gates, params=fragment_params)
            )
            added += 1
        return added

    def sample(
        self, host_num_qubits: int, rng: np.random.Generator
    ) -> Fragment | None:
        """Return a random fragment whose qubits fit inside the host's
        qubit set. None if no such fragment exists."""
        if not self.fragments:
            return None
        viable = [
            f for f in self.fragments if max(f.qubits_used(), default=-1) < host_num_qubits
        ]
        if not viable:
            return None
        return viable[int(rng.integers(0, len(viable)))]

    def sample_with_offset(
        self, host_num_qubits: int, rng: np.random.Generator
    ) -> tuple["Fragment | None", int]:
        """Like sample(), but also returns a random valid `qubit_offset`
        for cross-scale welding. The fragment's footprint plus offset
        is guaranteed to fit inside the host's qubit register."""
        if not self.fragments:
            return None, 0
        f = self.fragments[int(rng.integers(0, len(self.fragments)))]
        qubits = f.qubits_used()
        if not qubits:
            return f, 0
        footprint = max(qubits) + 1
        if footprint > host_num_qubits:
            return None, 0
        max_offset = host_num_qubits - footprint
        if max_offset <= 0:
            return f, 0
        return f, int(rng.integers(0, max_offset + 1))


def weld(
    spec: CircuitSpec,
    params: np.ndarray,
    fragment: Fragment,
    qubit_offset: int = 0,
) -> tuple[CircuitSpec, np.ndarray]:
    """Append a fragment to the host spec, allocating new param slots.

    `qubit_offset` shifts every fragment qubit by a constant amount so a
    fragment learned at qubits {0,1,2} can be welded into a host of
    larger width at any valid starting position. The shift is rejected
    silently if any resulting qubit would be out of range — caller is
    expected to choose offsets within bounds.
    """
    new_spec = CircuitSpec(num_qubits=spec.num_qubits)
    new_spec.gates = list(spec.gates)
    new_spec.num_params = spec.num_params

    base = spec.num_params
    for g in fragment.gates:
        shifted = tuple(q + qubit_offset for q in g.qubits)
        if any(not (0 <= q < spec.num_qubits) for q in shifted):
            # Skip gates that would land outside the host. Conservative —
            # better than a malformed spec.
            continue
        if g.is_parametric:
            new_spec.gates.append(GateSpec(g.name, shifted, base + g.param_idx))
        else:
            new_spec.gates.append(GateSpec(g.name, shifted, None))
    new_spec.num_params = base + fragment.num_params

    new_params = np.concatenate([params, fragment.params])
    return new_spec, new_params
