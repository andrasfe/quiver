# paper/

This folder contains an in-progress academic paper. The actual source
files are **encrypted at rest** with [`git-crypt`](https://github.com/AGWA/git-crypt).
On github.com they appear as binary blobs; locally they decrypt
transparently once the repo is unlocked.

## Files (encrypted)

- `ds_vqe.tex` — main paper source (LaTeX, `article` class)
- `ds_vqe.md`  — companion markdown version for quick reading
- `references.bib` — bibliography (BibTeX)

## Topic

*Diversity-Driven Subspace VQE (DS-VQE): Implicit Noise Mitigation in
Variational Quantum Eigensolvers via Structurally Diverse Circuit
Ensembles.*

## To unlock and read

```bash
brew install git-crypt
git clone git@github.com:andrasfe/quiver.git && cd quiver
git-crypt unlock /path/to/saved/quiver-git-crypt.key
# now paper/*.tex etc. decrypt automatically
```

To compile (after unlock):

```bash
cd paper
pdflatex ds_vqe.tex && bibtex ds_vqe && pdflatex ds_vqe.tex && pdflatex ds_vqe.tex
# produces ds_vqe.pdf
```
