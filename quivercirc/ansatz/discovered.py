"""DiscoveredAnsatz — parametric ansatz built from a distilled fragment.

The fragment's canonical form (e.g. `[ry(0), rxx(0,1), rz(1)]` on three
abstract qubit slots) is tiled across the host's qubit register at
configurable offsets, with each tile getting its own fresh parameter
slots. The result is a regular-structured circuit whose primitive
*was discovered by the system*, not specified by a human — but which
behaves identically to any hand-written ansatz from the explorer's
point of view.

Tiling strategies:

  - "stride": tile at offsets 0, stride, 2*stride, ... up to host width
    minus the canonical width. Sliding-window-style.
  - "all_starts": every valid start offset in [0, n - canonical_width].
"""

from __future__ import annotations

from dataclasses import dataclass

from quivercirc.circuit import CircuitSpec, GateSpec
from quivercirc.distillation import CanonicalForm, DistilledPattern


@dataclass
class DiscoveredAnsatz:
    canonical: CanonicalForm
    num_qubits: int
    canonical_width: int
    num_layers: int = 1
    stride: int = 1
    family: str = "discovered"

    @classmethod
    def from_pattern(
        cls,
        pattern: DistilledPattern,
        num_qubits: int,
        num_layers: int = 1,
        stride: int = 1,
        tag: str | None = None,
    ) -> "DiscoveredAnsatz":
        return cls(
            canonical=pattern.canonical,
            num_qubits=num_qubits,
            canonical_width=pattern.width,
            num_layers=num_layers,
            stride=stride,
            family=tag or f"discovered_w{pattern.width}_l{pattern.length}",
        )

    def _tile_offsets(self) -> list[int]:
        if self.canonical_width <= 0:
            return []
        max_start = self.num_qubits - self.canonical_width
        if max_start < 0:
            return []
        offsets: list[int] = []
        for off in range(0, max_start + 1, self.stride):
            offsets.append(off)
        if not offsets:
            offsets.append(0)
        return offsets

    @property
    def num_params(self) -> int:
        param_per_tile = sum(1 for _, _, is_param in self.canonical if is_param)
        return param_per_tile * len(self._tile_offsets()) * self.num_layers

    def build(self) -> CircuitSpec:
        spec = CircuitSpec(num_qubits=self.num_qubits)
        idx = 0
        for _ in range(self.num_layers):
            for offset in self._tile_offsets():
                for name, qs, is_param in self.canonical:
                    shifted = tuple(q + offset for q in qs)
                    if any(not (0 <= q < self.num_qubits) for q in shifted):
                        continue
                    if is_param:
                        spec.gates.append(GateSpec(name, shifted, idx))
                        idx += 1
                    else:
                        spec.gates.append(GateSpec(name, shifted, None))
        spec.num_params = idx
        return spec
