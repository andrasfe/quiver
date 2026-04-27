"""Portfolio-optimization Hamiltonians for quantum finance benchmarks.

The Markowitz mean-variance portfolio-selection problem with a fixed
cardinality constraint reads:

    C(x) = -μᵀ x + λ xᵀ Σ x + γ (∑_i x_i - K)²

where x ∈ {0,1}ⁿ chooses which assets to include, μ is expected
returns, Σ is the covariance matrix, λ is risk aversion, K is the
cardinality budget, and γ is the cardinality-penalty weight. This is
the canonical formulation in:

  • Hodson et al., "Portfolio rebalancing experiments using the QAOA"
    (Phys. Rev. Research / arXiv:1911.05296, 2019/2020)
  • Barkoutsos et al., "Improving Variational Quantum Optimization
    using CVaR" (Quantum 4, 256, 2020)
  • Mugel et al., "Dynamic portfolio optimization with real datasets
    using quantum processors" (Phys. Rev. Research 4, 013006, 2022)
  • IBM Qiskit Finance tutorials

To map the QUBO C(x) onto an Ising Hamiltonian acting on n qubits we
substitute x_i = (1 − Z_i)/2 (so x_i = 0 ↔ Z_i = +1, x_i = 1 ↔ Z_i = −1):

    H = ∑_i h_i Z_i + ∑_{i<j} J_ij Z_i Z_j  +  const

with single-site fields and pair couplings derived in build_portfolio.
The constant offset is omitted since it shifts every eigenvalue equally
and the variational principle is unaffected.

The ground state of H corresponds to the optimal portfolio x*, and we
can recover the binary selection from the bitstring index of the
ground eigenvector (qubit 0 is LSB).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_I = np.eye(2, dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron_chain(ops: list[np.ndarray]) -> np.ndarray:
    """qubit 0 is LSB (rightmost factor)."""
    n = len(ops)
    out = ops[n - 1]
    for k in range(n - 2, -1, -1):
        out = np.kron(out, ops[k])
    return out


def _single_z(n: int, i: int) -> np.ndarray:
    ops = [_I] * n
    ops[i] = _Z
    return _kron_chain(ops)


def _double_z(n: int, i: int, j: int) -> np.ndarray:
    ops = [_I] * n
    ops[i] = _Z
    ops[j] = _Z
    return _kron_chain(ops)


@dataclass
class PortfolioBenchmark:
    H: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    cardinality_K: int
    risk_aversion: float
    cardinality_penalty: float
    optimal_bitstring: int
    optimal_cost: float
    exact_ground_energy: float
    constant_offset: float
    citation: str


def build_portfolio(
    mu: np.ndarray,
    sigma: np.ndarray,
    K: int,
    risk_aversion: float = 0.5,
    cardinality_penalty: float = 1.0,
    citation: str = "",
) -> PortfolioBenchmark:
    """Construct the Ising Hamiltonian for the Markowitz problem and
    enumerate to find the exact optimum (n ≤ 12 is fine).

    The exact optimum gives us an unambiguous yardstick for any VQE-style
    approximation method's approximation ratio.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = mu.size
    if sigma.shape != (n, n):
        raise ValueError("sigma must be n×n")
    sigma = 0.5 * (sigma + sigma.T)  # symmetrise
    A = 0.5 * n - K   # convenient constant

    # Single-site fields
    h = np.zeros(n)
    for i in range(n):
        row_sum = float(sigma[i].sum())
        h[i] = (
            mu[i] / 2.0
            - (risk_aversion / 2.0) * row_sum
            - cardinality_penalty * A
        )

    # Pair couplings
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = (risk_aversion / 2.0) * sigma[i, j] + cardinality_penalty / 2.0

    H = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n):
        H += h[i] * _single_z(n, i)
    for i in range(n):
        for j in range(i + 1, n):
            H += J[i, j] * _double_z(n, i, j)

    # Constant offset (we omit from H but compute for cost-recovery).
    # The risk term contributes (λ/2)·∑Σ_ii + (λ/2)·∑_{i<j}Σ_ij to the
    # constant, which simplifies to (λ/4)·(trace(Σ) + Σ.sum()) using the
    # identity Σ.sum() = trace(Σ) + 2·∑_{i<j}Σ_ij.
    const = (
        -mu.sum() / 2.0
        + (risk_aversion / 4.0) * (np.trace(sigma) + sigma.sum())
        + cardinality_penalty * (A * A + n / 4.0)
    )

    # Enumerate to find exact optimum and its cost.
    optimum_idx = -1
    optimum_cost = float("inf")
    for idx in range(2**n):
        x = np.array([(idx >> q) & 1 for q in range(n)], dtype=float)
        cost = float(
            -mu @ x
            + risk_aversion * x @ sigma @ x
            + cardinality_penalty * (x.sum() - K) ** 2
        )
        if cost < optimum_cost:
            optimum_cost = cost
            optimum_idx = idx

    exact_ground = float(np.linalg.eigvalsh(H)[0])

    return PortfolioBenchmark(
        H=H, mu=mu, sigma=sigma,
        cardinality_K=K,
        risk_aversion=risk_aversion,
        cardinality_penalty=cardinality_penalty,
        optimal_bitstring=optimum_idx,
        optimal_cost=optimum_cost,
        exact_ground_energy=exact_ground,
        constant_offset=const,
        citation=citation,
    )


def stylized_8asset_portfolio(seed: int = 42) -> PortfolioBenchmark:
    """8 stylised assets with realistic-magnitude annualised returns
    (5–15%) and a positive-definite covariance matrix from a random
    Cholesky factor (deterministic via `seed`). Cardinality K=4
    (select half the assets), λ=0.5, γ=2.0 (large enough that
    cardinality is binding).

    Designed to match the size and structure of the small-portfolio
    benchmarks in Hodson et al. 2019 (4-12 asset QAOA experiments)
    and Barkoutsos et al. 2020 (CVaR-VQE, 6-asset and similar)."""
    rng = np.random.default_rng(seed)
    n = 8
    mu = 0.05 + 0.10 * rng.random(n)
    L = 0.05 * rng.standard_normal((n, n))
    sigma = L @ L.T + 0.01 * np.eye(n)
    return build_portfolio(
        mu=mu, sigma=sigma, K=4,
        risk_aversion=0.5, cardinality_penalty=2.0,
        citation="Stylised portfolio benchmark in the form used by "
                 "Hodson et al. (2019) and Barkoutsos et al. (2020).",
    )


def stylized_6asset_portfolio(seed: int = 42) -> PortfolioBenchmark:
    """6-asset variant — matches the Barkoutsos et al. 2020 size."""
    rng = np.random.default_rng(seed)
    n = 6
    mu = 0.05 + 0.10 * rng.random(n)
    L = 0.05 * rng.standard_normal((n, n))
    sigma = L @ L.T + 0.01 * np.eye(n)
    return build_portfolio(
        mu=mu, sigma=sigma, K=3,
        risk_aversion=0.5, cardinality_penalty=2.0,
        citation="Stylised portfolio benchmark in the form used by "
                 "Hodson et al. (2019) and Barkoutsos et al. (2020).",
    )
