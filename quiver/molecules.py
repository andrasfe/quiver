"""Reference molecular Hamiltonians for VQE benchmarking.

These are the canonical small-molecule Hamiltonians used in published
VQE work — hardcoded so we don't depend on a quantum-chemistry stack
(OpenFermion / PySCF) for benchmarks.

Citations are inline. The Hamiltonians are returned as dense numpy
arrays in the qubit-0-is-LSB convention used by NumpyBackend.
"""

from __future__ import annotations

import numpy as np


# Pauli matrices.
_I = np.eye(2, dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


def _kron2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tensor product. Convention: q0 is LSB → q0 ends up as the right
    factor of the kron product."""
    return np.kron(A, B)


def h2_2qubit_bk(R: float = 0.7414) -> dict:
    """H₂ molecule, STO-3G basis, parity-mapped + qubit-tapered to 2 qubits
    via the Bravyi-Kitaev encoding. Equilibrium internuclear distance.

    Coefficients from O'Malley et al., PRX 6, 031007 (2016), Table I,
    and used identically in Kandala et al., Nature 549, 242 (2017).

    H_q = c_I · I + c_Z0 · Z_0 + c_Z1 · Z_1 + c_ZZ · Z_0 Z_1 + c_XX · X_0 X_1

    With these coefficients the electronic ground state energy is
        E₀ ≈ -1.857275 Hartree
    and the total energy (including nuclear repulsion 0.71999 Ha) is
        E_total ≈ -1.137 Ha — the textbook H₂ minimum.

    Returns a dict with keys:
        H               — 4x4 Hamiltonian matrix (Hartree)
        ground_energy   — exact smallest eigenvalue
        nuclear_repulsion — constant offset (Hartree)
        total_ground    — ground_energy + nuclear_repulsion
        chemical_accuracy_Ha — the 1.6 mH bar
        coefficients    — the c_X factors above
        citation        — the source string
    """
    if abs(R - 0.7414) > 1e-3:
        raise NotImplementedError(
            "Only R = 0.7414 Å is hardcoded. "
            "Other distances need integral re-tabulation."
        )

    II = _kron2(_I, _I)
    IZ = _kron2(_I, _Z)   # Z on q0 (LSB) — matches Quiver's convention
    ZI = _kron2(_Z, _I)   # Z on q1 (MSB)
    ZZ = _kron2(_Z, _Z)
    XX = _kron2(_X, _X)

    coeffs = {
        "I":   -1.0523732,
        "Z_0": +0.39793742,
        "Z_1": -0.39793742,
        "ZZ":  -0.01128010,
        "XX":  +0.18093119,
    }
    H = (coeffs["I"]   * II
         + coeffs["Z_0"] * IZ
         + coeffs["Z_1"] * ZI
         + coeffs["ZZ"]  * ZZ
         + coeffs["XX"]  * XX)

    ground = float(np.linalg.eigvalsh(H)[0])
    nuclear_repulsion = 0.71999  # Hartree, R=0.7414 Å

    return {
        "H": H,
        "ground_energy": ground,
        "nuclear_repulsion": nuclear_repulsion,
        "total_ground": ground + nuclear_repulsion,
        "chemical_accuracy_Ha": 1.6e-3,
        "coefficients": coeffs,
        "citation": "O'Malley et al. PRX 6, 031007 (2016); "
                    "Kandala et al. Nature 549, 242 (2017)",
        "R_angstroms": R,
        "num_qubits": 2,
    }
