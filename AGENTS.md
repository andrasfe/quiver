# AGENTS.md

Notes for AI agents and human contributors working in this repo.
Skim this before making changes.

## What this project is

`quivercirc` (PyPI: `quivercirc`, repo: `andrasfe/quiver`) — a Python
framework that finds **multiple structurally diverse quantum circuits**
that all solve the same problem (typically a state-preparation or VQE
target). The user-facing entry point is the `Quiver` class; the unique
contribution is the **diversity-filtered registry** that maintains K
verified solutions distinguished by gate-sequence edit distance, qubit
connectivity, and circuit depth — not just parameter values.

Distribution name and import name are the same: `pip install quivercirc`,
then `import quivercirc`.

## Current status (2026-04-29)

- Released to PyPI as **`quivercirc==0.1.1`** via Trusted Publishing
  (no API tokens stored; OIDC handshake from GitHub Actions).
- `main` is the only long-lived branch.
- 60 tests passing locally and on CI (Python 3.10 / 3.11 / 3.12).
- Two GitHub Actions workflows: `.github/workflows/ci.yml` (push/PR
  tests + wheel smoke test) and `.github/workflows/publish.yml`
  (tag-triggered Trusted Publishing).
- `git-crypt` encryption protects the IBM hardware notebook and any
  per-run hardware-result JSON. Symmetric key lives at
  `~/.quiver-git-crypt.key` (also at `.git/git-crypt/keys/default`
  inside the working tree). Repository is public; encrypted files
  appear as binary blobs on GitHub.

## Repo layout

```
quivercirc/             # the import package, what ships to PyPI
  __init__.py             public API (Quiver, Solution, ansatz, persistence)
  core.py                 Quiver class + explore() loop
  registry.py             SolutionRegistry, RegistryEntry
  diversity.py            structural-similarity metric (edit + connectivity + depth)
  circuit.py              CircuitSpec, GateSpec — the backend-agnostic IR
  config.py               QuiverConfig + sub-configs (TOML loader)
  optimizer.py            COBYLA + basin-hopping wrapper
  mutation.py             5 mutation operators + Mutator class
  microstructures.py      MicrostructureLibrary + weld()
  adaptive.py             AdaptiveGrowth (ADAPT-VQE-style gate-by-gate)
  distillation.py         canonicalize fragments → DiscoveredAnsatz patterns
  hamiltonian.py          Heisenberg / TFIM builders + vqe_setup helper
  persistence.py          save_registry / load_registry (JSON)
  verification.py         fidelity_verifier, fidelity_objective
  ansatz/                 6 canonical templates + DiscoveredAnsatz
  backends/               NumpyBackend (default), PennyLaneBackend, QiskitBackend
  dwave/                  QUBO / penalty-strategy module
tests/                  60 pytest tests, all passing
problems/               runnable demo / study scripts (not shipped to PyPI)
results/                JSON outputs from problem scripts
notebooks/              ENCRYPTED — see "git-crypt" section below
.github/workflows/      ci.yml, publish.yml
examples/               quiver.toml reference config + bell_state.py
```

Top-level docs: `README.md`, `CHANGELOG.md`, `PUBLISHING.md`,
`LICENSE`, `MANIFEST.in`, `pyproject.toml`.

## Public API surface (stable as of 0.1.1)

The 30+ symbols re-exported from `quivercirc.__init__`. Most-used:

```python
from quivercirc import (
    Quiver, Solution, QuiverConfig,           # core
    save_registry, load_registry,              # persistence
    Mutator, MutationConfig,                   # structural mutation
    AdaptiveGrowth, AdaptiveConfig,             # ADAPT-style growth
    SolutionRegistry, CircuitSpec, GateSpec,    # internals if needed
    diversity_score, structural_similarity,     # metric
)
from quivercirc.ansatz import (
    HardwareEfficient, QAOAInspired, LinearEntangler,
    BrickWall, AllToAll, StronglyEntangling,    # canonical templates
    DiscoveredAnsatz,                            # runtime-derived templates
)
from quivercirc.backends import NumpyBackend
from quivercirc.backends.qiskit_backend import to_qiskit_circuit  # for hardware paths
```

## Conventions worth knowing

- **Qubit-0 is the LSB** throughout. State `|q3 q2 q1 q0⟩` is at
  integer index `q0 + 2*q1 + 4*q2 + 8*q3`. The numpy backend uses
  this; the qiskit backend converts when needed.
- **Backend protocol**: anything with `num_qubits: int` and
  `statevector(spec, params) -> np.ndarray`. NumpyBackend is the
  default and requires no quantum dependencies.
- **Diversity metric weights** default to 0.5 / 0.3 / 0.2 (edit
  distance / connectivity Jaccard / depth-difference). Configurable
  via `DiversityConfig`.
- **Gate set** supported in NumpyBackend: `h, x, y, z, s, t, sdg,
  tdg, rx, ry, rz, cnot/cx, cz, swap, iswap, sqrt_iswap, rzz, rxx,
  ryy`. The qiskit translator covers the same set; `iswap` and
  `sqrt_iswap` are inserted as `qc.unitary` boxes for the transpiler.
- **Verifier signature**: `(state) -> (passed: bool, score: float)`
  for state-based targets, `(params) -> (bool, float)` for params-only
  custom losses. The Quiver class auto-detects which based on
  whether `target` is an `np.ndarray`.

## Dev workflow

```bash
git clone git@github.com:andrasfe/quiver.git && cd quiver
pip install -e .[test]      # editable install + pytest
pytest tests/               # 60 tests, ~0.6s on a laptop
```

Running a problem/demo locally:

```bash
PYTHONPATH=. python problems/maxcut_square_4node.py
PYTHONPATH=. python problems/generate_circuits_for_hardware.py
```

Adding a new feature: write the test first (`tests/test_*.py`),
expose anything public via `quivercirc/__init__.py`'s `__all__`,
update `CHANGELOG.md`'s Unreleased section.

## Release flow

PyPI releases are entirely automated via Trusted Publishing:

1. Bump `version` in `pyproject.toml` *and* `quivercirc/__init__.py`'s
   `__version__` (these must match).
2. Add a `CHANGELOG.md` entry.
3. `git commit && git push`.
4. `git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`.
5. Watch `https://github.com/andrasfe/quiver/actions` — the
   `publish.yml` workflow builds, runs tests, and uploads via OIDC
   (no token, no manual upload).

Do **not** `twine upload` manually — the project is configured for
Trusted Publishing only on the maintainer's PyPI side.

## git-crypt: what's encrypted, where the key lives

`.gitattributes` configures git-crypt to encrypt:

- `notebooks/*.ipynb` — the IBM hardware experimentation notebook(s)
- `results/ibm_run_*.json` — per-run hardware result files
- `notebooks/README.md` is explicitly *not* encrypted so GitHub
  shows a useful explanation in the folder.

These files appear as opaque binary blobs (`\0GITCRYPT\0` header +
AES ciphertext) on github.com. Locally they decrypt transparently
via the smudge/clean filter — `git status`, `git diff`, Jupyter,
nbconvert all work normally once the repo is unlocked.

Symmetric key:
- `.git/git-crypt/keys/default` — active key inside the working tree
- `~/.quiver-git-crypt.key` — backup export (same bytes)
- Both are 148-byte cleartext binaries protected only by `chmod 0600`
  (and FileVault if enabled). **The key file itself is not encrypted at
  rest.** Operational guidance: store the backup in a password
  manager (1Password Documents / Bitwarden Attachments).

To work on a new device:

```bash
brew install git-crypt
git clone git@github.com:andrasfe/quiver.git && cd quiver
# transfer ~/.quiver-git-crypt.key to the new device via password manager
git-crypt unlock /path/to/quiver-git-crypt.key
```

After unlock, the `.ipynb` files are readable; `git pull` and
`git push` operate transparently (re-encrypts on commit).

Everything outside the encrypted patterns is in the clear and
shipped to PyPI users normally.

## Local-only files (gitignored, *not* git-crypt protected)

These exist on the maintainer's machine but are not in the repo:

- `.env` — `IBM_API_KEY`, `IBM_CRN` for hardware submissions
- `idea.md` — DS-VQE design notes from earlier exploration
- `dist/`, `build/`, `*.egg-info/` — build artefacts
- `.pytest_cache/`, `.ipynb_checkpoints/`

If an agent needs the IBM credentials, they're loaded by the
notebook via `python-dotenv` from this `.env`.

## Things not to do

- Don't rename the package. The `quiver` → `quivercirc` rename
  in 0.1.1 was deliberate and final.
- Don't push without bumping the version when `pyproject.toml`
  changes. Trusted Publishing rejects duplicate versions.
- Don't commit the symmetric key, and don't echo the contents of
  `.env` to logs.
- Don't decrypt + re-commit the notebook with results that contain
  actual IBM job IDs / device-internal data unless the notebook
  remains git-crypt encrypted.
- Don't add new top-level dependencies without considering whether
  they belong in an `[project.optional-dependencies]` extra. Core
  install is currently only `numpy` + `scipy` (+ `tomli` on
  Python < 3.11) — keep it that way.

## Things on the wishlist

- **Workflows**: bump `actions/setup-python` to a Node-24-supporting
  version when one releases, before June 2026 deadline.
- **CI matrix**: consider adding Python 3.13 once it's stable in
  setup-python.
- **Docs**: a published Sphinx site (`docs/` → ReadTheDocs / GH
  Pages) — currently README is the only documentation.
- **Hardware-aware ranking**: extend the IBM notebook to transpile
  each registry circuit against the chosen backend's coupling map
  and rank by post-transpile depth/CNOT count before submission.

## Branch / commit style

- One change per commit, descriptive subject + body.
- Co-author trailer is fine for AI-pair-programmed commits but not
  required.
- `main` should always be releasable. CI must be green before merging.
