# Quiver

> Quantum circuit exploration framework that finds **multiple structurally diverse**
> solutions to problems where the target outcome is known in advance.

Quiver is the complement to a single-solution convergence approach (such as
`loopy`'s gradient-driven descent toward one minimum). Rather than asking
*"what is the best circuit?"*, Quiver asks *"what are the meaningfully
different circuits that all reach the goal?"* — and returns a curated set of
verified, structurally distinct candidates.

A motivating use case is **portfolio reverse stress testing**: generate
multiple valid CQM formulations that encode the same target risk metrics
in structurally different ways, so analysts can reason about formulation
sensitivity rather than trusting a single encoding.

---

## Table of contents

1. [What problem Quiver solves](#what-problem-quiver-solves)
2. [How it works (algorithm)](#how-it-works-algorithm)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quickstart](#quickstart)
6. [Configuration](#configuration)
7. [Ansatz library](#ansatz-library)
8. [Backends](#backends)
9. [Diversity metric — design rationale](#diversity-metric--design-rationale)
10. [Solution registry](#solution-registry)
11. [Verification](#verification)
12. [D-Wave / QUBO mode](#d-wave--qubo-mode)
13. [Reproducibility](#reproducibility)
14. [Performance targets](#performance-targets)
15. [Extending Quiver](#extending-quiver)
16. [Project layout](#project-layout)

---

## What problem Quiver solves

Variational quantum algorithms (VQE, QAOA, state preparation, etc.) and QUBO
formulations of constrained optimization problems both have a property in
common: **the target is known up front**, but the *circuit* or *encoding*
that achieves the target is not unique. Multiple architectures can prepare
the same statevector; multiple penalty schemes can enforce the same
constraint set.

A standard solver returns *one* answer — the first parameter assignment
that crosses the verification bar. In practice teams want more than that:

- **Architectural choice** — a deeper circuit on cheap qubits may be
  preferable to a shallower one that requires all-to-all connectivity.
- **Sensitivity analysis** — does the answer change qualitatively when the
  encoding changes? If two structurally different formulations agree, the
  result is more trustworthy.
- **Hardware-aware compilation** — different ansatz families compile
  differently to different topologies; you want options.
- **Compliance / auditability** — for regulated domains (financial stress
  testing), defending a single formulation is harder than defending a
  *family* of formulations that all pass the same audit.

Quiver's job is to fan out, find a diverse set of solutions that all pass
verification, and hand back a registry of trade-offs.

---

## How it works (algorithm)

The core loop is **basin-hopping over ansatz templates with diversity-filtered
acceptance**:

```
registry := empty
seed RNG(config.seed)
while len(registry) < num_solutions and time < deadline:
    ansatz := next ansatz template from library         # round-robin
    spec := ansatz.build()                              # CircuitSpec
    if spec.gate_count > budget: continue
    objective := lambda params: 1 - fidelity(backend(spec, params), target)

    # COBYLA × basin_hops independent restarts:
    for hop in range(basin_hops):
        x0 := perturb(best_so_far, step_size)            if hop > 0 else random
        result := scipy.optimize.minimize(objective, x0, method="COBYLA", ...)
        if result.fun < best.fun: best := result

    state := backend(spec, best.params)
    passed, fidelity := verifier(state)
    if not passed: continue

    diversity := 1 - max(structural_similarity(spec, e.spec) for e in registry)
    if diversity >= diversity_threshold:
        registry.add(spec, best.params, family=ansatz.family, fidelity, ...)
```

The key design choices:

- **Each restart begins from a different ansatz template**, not just a
  perturbed parameter vector. This is what gives structural diversity rather
  than parameter-only diversity.
- **COBYLA** — gradient-free, constraint-tolerant, and well-behaved on
  noisy quantum-circuit objectives. Wrapped in basin-hopping so we don't
  get stuck in the first local minimum the optimizer finds.
- **Verification is deterministic** — the registry only accepts candidates
  that the verifier returns `(True, score)` for. The fidelity threshold
  (default 0.99) is configurable per-problem.
- **Diversity is structural, not numerical** — see
  [Diversity metric](#diversity-metric--design-rationale).

---

## Architecture

```
                        ┌──────────────────┐
                        │  QuiverConfig    │   TOML / dataclass
                        │  (seed, budget,  │
                        │   thresholds,    │
                        │   weights)       │
                        └────────┬─────────┘
                                 │
                ┌────────────────▼────────────────┐
                │            Quiver               │
                │   ┌────────────────────────┐    │
                │   │   explore(library)     │    │
                │   └──┬──────────┬──────┬───┘    │
                └──────┼──────────┼──────┼────────┘
                       │          │      │
              ┌────────▼──┐  ┌────▼───┐  │
              │ Ansatz    │  │ Back-  │  │
              │ library   │  │ end    │  │
              │ (HE,      │  │ (numpy │  │
              │  QAOA,    │  │ /pl/   │  │
              │  LinEnt)  │  │  qiskit)  │
              └─────┬─────┘  └────┬───┘  │
                    │             │      │
                    │             ▼      │
                    │      ┌──────────┐  │
                    │      │ Statevec │  │
                    │      └────┬─────┘  │
                    │           │        │
                    │           ▼        ▼
                    │     ┌──────────────────┐
                    │     │   Optimizer      │
                    │     │ (COBYLA + basin- │
                    │     │  hopping)        │
                    │     └────────┬─────────┘
                    │              │
                    │              ▼
                    │       ┌────────────┐
                    └──────►│ Verifier   │
                            │ (fidelity) │
                            └────┬───────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │  Diversity     │
                        │  metric        │
                        │  (edit, conn,  │
                        │   depth)       │
                        └────────┬───────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │ SolutionRegistry│
                        │ (verified +    │
                        │  diverse)      │
                        └────────────────┘
```

Backend-agnostic representation is critical: ansätze emit a `CircuitSpec`
(a list of `GateSpec` tokens), backends translate it to their native
circuit object, and the diversity metric reads `CircuitSpec`s directly so
that structural comparison is independent of which backend ran the
simulation.

---

## Installation

From PyPI (the package is named `quiver-circuits` because the bare
`quiver` namespace is taken; `import quiver` still works):

```bash
pip install quiver-circuits                # core (numpy + scipy)
pip install quiver-circuits[pennylane]     # + PennyLane backend
pip install quiver-circuits[qiskit]        # + Qiskit backend
pip install quiver-circuits[ibm]           # + qiskit-ibm-runtime + qiskit-aer
pip install quiver-circuits[dwave]         # + dimod / dwave-system
pip install quiver-circuits[all]           # everything
pip install quiver-circuits[test]          # pytest tooling
```

For development against this repo:

```bash
git clone https://github.com/andrasfe/quiver.git
cd quiver
pip install -e .[test]
```

Python 3.10+ required.

---

## Quickstart

```python
import numpy as np
from quiver import Quiver, QuiverConfig
from quiver.ansatz import HardwareEfficient, LinearEntangler, QAOAInspired
from quiver.config import DiversityConfig, ExplorationConfig, OptimizerConfig

# Target: a Bell state
bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

config = QuiverConfig(
    exploration=ExplorationConfig(num_solutions=4, time_budget_seconds=60, seed=7),
    optimizer=OptimizerConfig(basin_hops=6, max_iter=150),
    diversity=DiversityConfig(threshold=0.15),
)

q = Quiver(target=bell, config=config)

library = [
    HardwareEfficient(num_qubits=2, num_layers=1),
    HardwareEfficient(num_qubits=2, num_layers=2),
    LinearEntangler(num_qubits=2, num_layers=1),
    LinearEntangler(num_qubits=2, num_layers=2),
    QAOAInspired(num_qubits=2, num_layers=1),
    QAOAInspired(num_qubits=2, num_layers=2),
]

solutions = q.explore(library)

for i, s in enumerate(solutions, 1):
    print(f"[{i}] {s.family:<20} fidelity={s.fidelity:.4f} "
          f"depth={s.depth} gates={s.gate_count} 2q={s.two_qubit_count} "
          f"diversity={s.diversity:.3f}")
```

Sample output:

```
[1] hardware_efficient   fidelity=1.0000 depth=3 gates=5 2q=1 diversity=1.000
[2] hardware_efficient   fidelity=1.0000 depth=6 gates=10 2q=2 diversity=0.350
[3] linear_entangler     fidelity=1.0000 depth=3 gates=5 2q=1 diversity=0.300
[4] linear_entangler     fidelity=1.0000 depth=5 gates=8 2q=2 diversity=0.267
```

Each `Solution` carries: `family`, `params`, `fidelity`, `objective`,
`gate_count`, `depth`, `two_qubit_count`, `diversity`, and the underlying
`CircuitSpec`.

---

## Configuration

Configuration is via TOML or programmatically through `QuiverConfig`.

```toml
# examples/quiver.toml

[exploration]
num_solutions       = 10            # target size of the solution set
time_budget_seconds = 300           # hard wall-clock cap
seed                = 42            # RNG seed for reproducibility

[optimizer]
method     = "COBYLA"               # scipy minimize method
max_iter   = 200                    # per-restart iteration cap
basin_hops = 12                     # COBYLA restarts per ansatz attempt
step_size  = 1.0                    # σ of gaussian perturbation between hops
tolerance  = 1e-4                   # early-exit threshold on objective

[diversity]
threshold           = 0.25          # minimum diversity score to accept
edit_weight         = 0.5           # gate-sequence edit-distance weight
connectivity_weight = 0.3           # qubit-pair-set Jaccard weight
depth_weight        = 0.2           # circuit-depth-difference weight

[budget]
max_gates = 80                      # reject ansätze whose spec exceeds this
max_depth = 30

[ansatz]
families   = ["hardware_efficient", "qaoa", "linear_entangler"]
num_layers = [1, 2, 3]
```

Load with:

```python
from quiver import load_config
config = load_config("quiver.toml")
```

---

## Ansatz library

An ansatz is a structural template: it knows the qubit count, the layer
count, and how to emit a `CircuitSpec`. Three are built in.

### HardwareEfficient

Repeated blocks of `[Ry, Rz on every qubit] → [linear chain of CNOTs]`.
The default rotation axes are `("ry", "rz")` and the default entangler is
`cnot`; both can be overridden.

```python
HardwareEfficient(num_qubits=4, num_layers=3, rotation_axes=("ry", "rz"))
# 3 * (2 * 4 + 3) = 33 gates, 24 parameters
```

### LinearEntangler

`Ry(θ)` on all qubits, then a linear chain of CNOTs, then `Ry(θ)` again,
repeated for `num_layers`. Distinct from `HardwareEfficient` because it
uses only one rotation axis per block — the gate signature differs, so
the diversity metric distinguishes them.

```python
LinearEntangler(num_qubits=3, num_layers=2)
# (2+1) * 3 ry + 2 * 2 cnots = 9 + 4 = 13 gates, 9 parameters
```

### QAOAInspired

Hadamard-prep, then alternating `Rzz(γ)` cost layers (on a chain or ring)
and `Rx(β)` mixer layers. Parameters are 2 per layer (γ, β), shared
across all qubits within a layer — much smaller search space than HE.

```python
QAOAInspired(num_qubits=4, num_layers=3, ring=True)
# 4 H + 3 * (4 rzz + 4 rx) = 28 gates, 6 parameters
```

### Custom ansätze

Implement the `Ansatz` protocol:

```python
from quiver.ansatz.base import Ansatz
from quiver.circuit import CircuitSpec, GateSpec

class MyAnsatz:
    family = "my_custom"
    num_qubits = 4
    num_params = 8

    def build(self) -> CircuitSpec:
        spec = CircuitSpec(num_qubits=self.num_qubits)
        # ...add GateSpecs, set spec.num_params...
        return spec
```

---

## Backends

### NumpyBackend (default)

Pure-numpy statevector simulator — no quantum dependencies. Suitable for
circuits up to roughly 12 qubits. Used by all tests and examples.

Implements: `h`, `x`, `y`, `z`, `s`, `t`, `sdg`, `tdg`, `rx`, `ry`, `rz`,
`cnot`/`cx`, `cz`, `swap`, `rzz`, `rxx`, `ryy`.

Convention: amplitudes are stored as a length-2ⁿ complex array where the
binary index gives the qubit values, with **qubit 0 as the
least-significant bit**.

### PennyLaneBackend

Lazy-imports `pennylane`. Wraps each `CircuitSpec` in a `qml.qnode` against
`default.qubit` (or any device name you pass).

```python
from quiver.backends import pennylane_backend
backend = pennylane_backend(num_qubits=6, device_name="default.qubit")
```

### QiskitBackend

Lazy-imports `qiskit`. Builds a `QuantumCircuit` and evaluates with
`Statevector.from_instruction`.

```python
from quiver.backends import qiskit_backend
backend = qiskit_backend(num_qubits=6)
```

All backends conform to the same `Backend` protocol:

```python
class Backend(Protocol):
    num_qubits: int
    def statevector(self, spec: CircuitSpec, params: np.ndarray) -> np.ndarray: ...
```

---

## Diversity metric — design rationale

Two circuits are *structurally similar* if they share most gate-sequence
tokens, route 2-qubit gates over the same qubit pairs, and have similar
depth. The metric returns a similarity in `[0, 1]`; **diversity = 1 −
similarity**.

The three weighted components:

1. **Gate-sequence edit distance** (default weight 0.5).
   Levenshtein on `(gate_name, qubit_tuple)` tokens, normalised by the
   longer sequence. This is the dominant signal: it distinguishes ansatz
   *families* and is invariant to the parameter values, which is exactly
   what we want — mere parameter tweaks must not register as diversity.

2. **Qubit-connectivity overlap** (default weight 0.3).
   Jaccard similarity on the set of unordered qubit pairs touched by 2-qubit
   gates. Captures wiring topology: a linear chain `{(0,1),(1,2)}` and a
   ring `{(0,1),(1,2),(2,0)}` are partially overlapping, while a star and
   a chain disagree more.

3. **Depth difference** (default weight 0.2).
   `1 − |d₁ − d₂| / max(d₁, d₂)`. Captures whether one solution is
   significantly shallower than another — useful for hardware-aware
   selection.

Weights are normalised to sum to 1 internally, so you can pass arbitrary
positive numbers in the config.

The `diversity_score(candidate, existing)` returns the candidate's distance
to its **nearest** existing solution (`1 − max similarity`), so the registry
rejects anything close to *any* prior entry — preventing a pile-up of
near-duplicates around one architectural archetype.

---

## Solution registry

`SolutionRegistry` holds verified, diverse solutions:

```python
@dataclass
class RegistryEntry:
    spec:             CircuitSpec
    params:           np.ndarray
    family:           str           # ansatz family tag
    fidelity:         float         # verifier score
    objective:        float         # final loss value
    gate_count:       int
    depth:            int
    two_qubit_count:  int
    diversity:        float         # vs prior entries at insertion time
```

`registry.try_add(...)` returns the entry if it was accepted, or `None` if
it was rejected as a near-duplicate. Callers don't need to manage diversity
themselves — the registry is the gatekeeper.

---

## Verification

A verifier is a callable returning `(passed: bool, score: float)`. The
built-in `fidelity_verifier(target, threshold=0.99)` computes
|⟨target|state⟩|² and passes when it crosses the threshold.

```python
from quiver.verification import fidelity_verifier
verify = fidelity_verifier(target_state, threshold=0.995)
ok, fid = verify(state)
```

For non-state-preparation problems (e.g., when the target is a metrics
dict rather than a statevector), pass your own `verifier` and `objective`
to `Quiver`. In that case, the verifier receives the **parameter vector**
rather than a statevector, and the objective is whatever loss you choose.

---

## D-Wave / QUBO mode

For QUBO/CQM exploration the same diversity philosophy applies, but the
"structure" being varied is the **penalty formulation**, not a circuit
template.

```python
from quiver.dwave import QUBOExplorer, quadratic_penalty, slack_penalty, one_hot_penalty
import numpy as np

base_Q = np.zeros((10, 10))

def verifier(Q, meta):
    # Solve Q (e.g. with dimod.ExactSolver or D-Wave) and check that the
    # result satisfies the same target risk metrics.
    metrics = solve_and_score(Q)
    return metrics["risk_var"] <= 0.05, metrics

explorer = QUBOExplorer(base_Q=base_Q, verifier=verifier)

strategies = [
    lambda Q: (*quadratic_penalty(Q, [1, 2, 3], target=4, weight=2.0),
               {"family": "quadratic", "description": "equality penalty"}),
    lambda Q: (*slack_penalty(Q, [1, 2, 3], upper_bound=4, weight=2.0),
               {"family": "slack", "description": "inequality with binary slack"}),
    lambda Q: (*one_hot_penalty(Q, group=[0, 1, 2], weight=3.0),
               {"family": "one_hot", "description": "exactly-one constraint"}),
]

formulations = explorer.explore(strategies, max_results=10)
```

Each accepted `QUBOFormulation` records its penalty `family`, auxiliary-variable
count, sparsity fingerprint, and the metrics returned by the verifier — so
downstream analysis can compare formulations on equal footing.

---

## Reproducibility

Every random source in Quiver is seeded from `config.exploration.seed`:

- The `numpy.random.Generator` that initialises COBYLA starting points.
- The same RNG drives basin-hopping perturbations.
- COBYLA itself is deterministic given a starting point.

Two runs with the same seed, config, and ansatz library produce
**identical** solution sets. This is critical for audit and for diffing
runs as the codebase evolves.

---

## Performance targets

The success criterion in the original spec is:

> ten verified, diverse solutions in under five minutes for typical problems

Practically, that means:

- 2–6 qubit state-preparation problems → seconds, not minutes
- 8–10 qubit VQE-style targets → typically under the budget on the
  numpy backend with `basin_hops=8`–`12`
- The hard wall-clock cap (`time_budget_seconds`) is enforced inside
  `Quiver.explore`, so the function never overruns

If you're not hitting the budget, the most effective levers are
`basin_hops` (more = more chances to escape local minima, more time per
ansatz) and the size of the ansatz library (more templates = more
structural variety to draw from).

---

## Extending Quiver

| Want to... | Implement... | Where |
|---|---|---|
| New ansatz family | `Ansatz` protocol (`family`, `num_qubits`, `num_params`, `build()`) | `quiver/ansatz/` |
| New backend | `Backend` protocol (`statevector(spec, params)`) | `quiver/backends/` |
| New diversity component | Add to `quiver/diversity.py` and a weight to `DiversityConfig` | `quiver/diversity.py`, `quiver/config.py` |
| New verifier | Callable `(state \| params) -> (bool, float)` | pass to `Quiver(verifier=...)` |
| New QUBO penalty | `(Q, **kwargs) -> (Q', aux_vars, meta)` | `quiver/dwave/penalties.py` |

The protocols are runtime-checkable (`@runtime_checkable`), so duck typing
works: any class with the right attributes plugs in.

---

## Project layout

```
quiver/
├── pyproject.toml
├── README.md
├── examples/
│   ├── quiver.toml             # reference TOML config
│   └── bell_state.py           # end-to-end demo
├── quiver/
│   ├── __init__.py             # public API
│   ├── core.py                 # Quiver, Solution, .explore()
│   ├── circuit.py              # CircuitSpec, GateSpec
│   ├── config.py               # QuiverConfig dataclasses + load_config()
│   ├── diversity.py            # similarity & diversity metrics
│   ├── optimizer.py            # COBYLA + basin-hopping wrapper
│   ├── registry.py             # SolutionRegistry
│   ├── verification.py         # fidelity_verifier, fidelity_objective
│   ├── ansatz/
│   │   ├── base.py             # Ansatz protocol
│   │   ├── hardware_efficient.py
│   │   ├── linear_entangler.py
│   │   └── qaoa.py
│   ├── backends/
│   │   ├── base.py             # Backend protocol
│   │   ├── numpy_backend.py    # default, pure-numpy simulator
│   │   ├── pennylane_backend.py
│   │   └── qiskit_backend.py
│   └── dwave/
│       ├── qubo.py             # QUBOExplorer, QUBOFormulation
│       └── penalties.py        # quadratic / slack / one-hot / log-encoded
└── tests/                      # 22 tests, all passing
    ├── test_circuit.py
    ├── test_ansatz.py
    ├── test_diversity.py
    ├── test_registry.py
    ├── test_numpy_backend.py
    ├── test_core.py
    └── test_config.py
```

---

## License

MIT.
