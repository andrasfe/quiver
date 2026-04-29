# notebooks/

The `.ipynb` files in this directory are **encrypted at rest** with
[`git-crypt`](https://github.com/AGWA/git-crypt). On github.com they
look like opaque binary blobs and the notebook renderer shows
*"Invalid Notebook — the notebook does not appear to be valid JSON"*.
That is the encryption working — it means people without the key
cannot read these notebooks.

## What's in here

- `ibm_hardware_experiment.ipynb` — submits the diverse-circuit
  registry from `results/ibm_ready_circuits.json` to either a local
  Aer simulator or real IBM Quantum hardware via
  `qiskit-ibm-runtime`. Reads `IBM_API_KEY` and `IBM_CRN` from a
  local `.env` (which is gitignored).

## To unlock the notebooks

You need the symmetric key (kept off-repo by the maintainer):

```bash
brew install git-crypt          # macOS
# apt install git-crypt         # Debian/Ubuntu

git clone git@github.com:andrasfe/quiver.git
cd quiver
git-crypt unlock /path/to/saved/quiver-git-crypt.key
```

After unlock, the `.ipynb` files decrypt transparently for the rest
of the working tree's lifetime. `git status`, `git diff`, Jupyter,
`nbconvert` — all work normally.

## What's outside this folder

The whole `quivercirc` Python package (everything else in the repo)
stays in the clear and ships to PyPI. The diversity-driven circuit
generator that produces the registry these notebooks consume is at
`problems/generate_circuits_for_hardware.py`.
