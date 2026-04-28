# Changelog

All notable changes to **quivercirc** are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-04-28

Initial public release.

### Core framework

- **Quiver class** with diversity-filtered registry, structural-similarity
  metric (gate-edit distance + qubit connectivity overlap + depth
  difference), and a configurable acceptance gate.
- **Six built-in ansatz templates** — `HardwareEfficient`, `QAOAInspired`,
  `LinearEntangler`, `BrickWall`, `AllToAll`, `StronglyEntangling`, plus
  a `DiscoveredAnsatz` class for runtime-promoted patterns.
- **Backend-agnostic CircuitSpec** — translatable to NumPy
  (default, pure-numpy simulator up to ~12 qubits), PennyLane, or Qiskit.
- **COBYLA + basin-hopping optimizer** with multi-strategy initialisation
  (zero / Gaussian / uniform random) and warm-start support.
- **Mutation operators** — insert / delete / swap / retarget / retype,
  threading parameters through structural changes for warm-start
  fidelity.
- **Microstructure library** — extracts gate fragments from accepted
  registry entries; mutation and adaptive growth can recombine them as
  macro-blocks.
- **Adaptive (ADAPT-style) growth** — gate-by-gate construction with
  ε-greedy candidate selection and anti-template novelty reward.
- **Library distillation** — promotes recurring fragment patterns to
  parametric `DiscoveredAnsatz` templates.

### Persistence

- **`quiver.persistence`** — round-trippable JSON serialization for
  circuit specs, parameters, and registries (`save_registry`,
  `load_registry`, `save_circuit`, `load_circuit`).
- **`quiver.backends.qiskit_backend.to_qiskit_circuit`** — module-level
  helper that converts a Quiver `CircuitSpec` to a Qiskit
  `QuantumCircuit` for hardware submission.

### Tooling

- 60 tests passing (`pytest tests/`).
- GitHub Actions: `ci.yml` (test matrix on Python 3.10/3.11/3.12) and
  `publish.yml` (PyPI Trusted Publishing on tag push).
- TOML configuration loader.

[0.1.0]: https://github.com/andrasfe/quiver/releases/tag/v0.1.0
