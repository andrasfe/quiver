import numpy as np

from quiver.molecules import h2_2qubit_bk


def test_h2_hamiltonian_is_hermitian():
    bench = h2_2qubit_bk()
    H = bench["H"]
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


def test_h2_ground_energy_matches_published_value():
    """O'Malley et al. (2016) and Kandala et al. (2017) report the
    electronic ground-state energy of this Hamiltonian as -1.857275 Ha.
    A correctness check on our coefficients."""
    bench = h2_2qubit_bk()
    expected_electronic = -1.857275
    np.testing.assert_allclose(
        bench["ground_energy"], expected_electronic, atol=1e-4
    )


def test_h2_total_energy_matches_textbook_value():
    """Adding the nuclear repulsion E_nuc = 0.71999 Ha at R=0.7414 Å
    gives the textbook H₂ total energy ≈ -1.137 Ha."""
    bench = h2_2qubit_bk()
    np.testing.assert_allclose(bench["total_ground"], -1.137, atol=2e-3)
