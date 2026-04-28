# Publishing `quivercirc` to PyPI

The package is published as **`quivercirc`** on PyPI. Distribution
name and import name match. Users install with:

```bash
pip install quivercirc
```

and then `import quivercirc` works as normal.

## Prerequisites

```bash
pip install --upgrade build twine
```

You'll need an account on PyPI (https://pypi.org/account/register/)
and an API token (https://pypi.org/manage/account/token/). Save the
token as an environment variable:

```bash
export TWINE_PASSWORD="pypi-<your-token>"
export TWINE_USERNAME="__token__"
```

## Test on TestPyPI first (recommended)

TestPyPI is the staging registry — same workflow as PyPI but the
account/token are separate (https://test.pypi.org/account/register/).

```bash
# Build (creates dist/quiver_quantum-X.Y.Z.tar.gz and .whl)
rm -rf dist build
python -m build

# Verify the wheel is well-formed
twine check dist/*

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Install from TestPyPI to confirm
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            quivercirc
```

The `--extra-index-url` is needed because TestPyPI doesn't mirror the
real PyPI, so dependencies (`numpy`, `scipy`) need to come from the
production index.

## Real PyPI publish

When TestPyPI works:

```bash
rm -rf dist build
python -m build
twine check dist/*
twine upload dist/*
```

After upload, the package appears at
https://pypi.org/project/quivercirc/ within a minute or two.

## Versioning

Bump `version` in `pyproject.toml` for every upload — PyPI rejects
duplicate versions. Recommended scheme:

- `0.1.0` → first public release (current)
- `0.1.1` → bugfix releases
- `0.2.0` → new features added since `0.1.0`
- `1.0.0` → API stability commitment

Tag the release in git:

```bash
git tag -a v0.1.0 -m "Initial PyPI release"
git push origin v0.1.0
```

## What gets included

The build picks up everything matched by `[tool.setuptools.packages.find]`
in `pyproject.toml` (the whole `quiver/` tree) plus the files listed
in `MANIFEST.in` (`README.md`, `LICENSE`, `pyproject.toml`).
**`tests/`, `problems/`, `results/`, and `notebooks/` are not
shipped** — they're development artefacts.

## Optional extras

The `[project.optional-dependencies]` block defines named installable
extras:

```bash
pip install quivercirc[ibm]        # qiskit + qiskit-ibm-runtime + qiskit-aer
pip install quivercirc[pennylane]  # PennyLane backend
pip install quivercirc[dwave]      # D-Wave penalty exploration
pip install quivercirc[all]        # everything above
pip install quivercirc[test]       # pytest tooling for development
```

## Trouble-shooting

- **"File already exists"** — bump the version number; PyPI doesn't
  allow re-uploading the same version.
- **"Invalid distribution name"** — the package name on PyPI must be
  unique. If `quivercirc` is also taken, pick another (e.g.
  `quiver-search`, `quiver-explorer`, `pyquiver`) and update `name` in
  `pyproject.toml`.
- **Import name and distribution name match** — `pip install quivercirc`
  installs the `quivercirc` import package; no name aliasing.
