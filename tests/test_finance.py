import numpy as np

from quiver.finance import (
    build_portfolio,
    stylized_6asset_portfolio,
    stylized_8asset_portfolio,
)


def _enumerate_costs(bench):
    """Brute-force every bitstring's classical cost."""
    n = bench.mu.size
    out = []
    for idx in range(2**n):
        x = np.array([(idx >> q) & 1 for q in range(n)], dtype=float)
        cost = float(
            -bench.mu @ x
            + bench.risk_aversion * x @ bench.sigma @ x
            + bench.cardinality_penalty * (x.sum() - bench.cardinality_K) ** 2
        )
        out.append(cost)
    return np.array(out)


def test_hamiltonian_eigenspectrum_matches_classical_costs():
    """The Hamiltonian's diagonal (it has only Z and ZZ terms) should
    produce eigenvalues equal to classical_costs - constant_offset for
    every computational-basis index."""
    bench = stylized_6asset_portfolio()
    classical_costs = _enumerate_costs(bench)
    diag = np.real(np.diag(bench.H))
    np.testing.assert_allclose(diag + bench.constant_offset, classical_costs, atol=1e-9)


def test_optimal_bitstring_matches_ground_eigenvector():
    bench = stylized_8asset_portfolio()
    eigvals, eigvecs = np.linalg.eigh(bench.H)
    ground_eig = eigvecs[:, 0]
    # Ground eigenvector should be a computational-basis state since H
    # is diagonal (Z, ZZ only). Find the dominant index.
    idx = int(np.argmax(np.abs(ground_eig)))
    assert idx == bench.optimal_bitstring


def test_constant_offset_recovers_classical_cost_at_ground():
    bench = stylized_8asset_portfolio()
    np.testing.assert_allclose(
        bench.exact_ground_energy + bench.constant_offset,
        bench.optimal_cost,
        atol=1e-9,
    )


def test_cardinality_constraint_is_satisfied_at_optimum():
    bench = stylized_8asset_portfolio()
    x = np.array([(bench.optimal_bitstring >> q) & 1
                  for q in range(bench.mu.size)])
    assert int(x.sum()) == bench.cardinality_K
