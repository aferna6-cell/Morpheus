"""Bregman divergence and KL projection for LMSR markets.

Mathematical background
-----------------------
In a Logarithmic Market Scoring Rule (LMSR) market, the cost function is
based on negative entropy:

    R(μ) = Σ_i μ_i · ln(μ_i)        (negative entropy, convex)

The Bregman divergence induced by R is the KL divergence:

    D_R(μ ‖ θ) = R(μ) - R(θ) - ∇R(θ)·(μ - θ)
               = Σ_i μ_i · ln(μ_i / θ_i)
               = D_KL(μ ‖ θ)

Arbitrage detection reduces to projecting market prices θ onto the
marginal polytope M (the set of valid probability distributions):

    μ* = argmin_{μ ∈ M}  D_KL(μ ‖ θ)

If θ ∈ M, no arbitrage exists (D_KL = 0).
If θ ∉ M, the projection distance D_KL(μ* ‖ θ) equals the maximum
guaranteed profit from arbitrage trading.

References
----------
- arXiv:2508.03474v1, Section 3 (Bregman projection framework)
- Hanson (2003), Logarithmic Market Scoring Rules
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import LinearConstraint, minimize

import structlog

logger = structlog.get_logger(__name__)

# Small epsilon to avoid log(0) and division by zero
_EPS = 1e-15


def kl_divergence(mu: np.ndarray, theta: np.ndarray) -> float:
    """Compute KL divergence D_KL(μ ‖ θ) = Σ_i μ_i · ln(μ_i / θ_i).

    Parameters
    ----------
    mu : array-like
        Target distribution (must be non-negative, should sum to 1).
    theta : array-like
        Reference distribution (market prices, must be positive).

    Returns
    -------
    float
        KL divergence. Returns 0.0 if distributions are identical.
        Returns +inf if theta has zeros where mu is positive.

    Notes
    -----
    Convention: 0 · ln(0/θ) = 0  (by continuity).
    If θ_i = 0 and μ_i > 0, D_KL = +∞.
    """
    mu = np.asarray(mu, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    if mu.shape != theta.shape:
        raise ValueError(f"Shape mismatch: mu={mu.shape}, theta={theta.shape}")

    # Handle zeros in theta: if mu_i > 0 and theta_i = 0 → infinity
    mask = mu > _EPS
    if np.any((theta[mask] <= 0)):
        return float("inf")

    # 0 * log(0/x) = 0 by convention
    result = np.zeros_like(mu)
    result[mask] = mu[mask] * np.log(mu[mask] / np.maximum(theta[mask], _EPS))

    divergence = float(np.sum(result))

    # KL divergence is always non-negative; clamp numerical noise
    return max(0.0, divergence)


def bregman_divergence(mu: np.ndarray, theta: np.ndarray) -> float:
    """Bregman divergence for negative entropy R(x) = Σ x_i · ln(x_i).

    For the negative entropy generator, the Bregman divergence is exactly
    the KL divergence:

        D_R(μ ‖ θ) = R(μ) - R(θ) - ∇R(θ)·(μ - θ)

    where ∇R(θ)_i = ln(θ_i) + 1.

    Expanding:
        D_R = Σ μ_i ln(μ_i) - Σ θ_i ln(θ_i) - Σ (ln(θ_i)+1)(μ_i - θ_i)
            = Σ μ_i ln(μ_i/θ_i) - Σ (μ_i - θ_i)
            = D_KL(μ‖θ)  when Σμ_i = Σθ_i  (both on simplex)

    For generality, we compute the full Bregman form without assuming
    equal mass.

    Parameters
    ----------
    mu, theta : array-like
        Non-negative vectors.

    Returns
    -------
    float
        Bregman divergence D_R(μ ‖ θ).
    """
    mu = np.asarray(mu, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    if mu.shape != theta.shape:
        raise ValueError(f"Shape mismatch: mu={mu.shape}, theta={theta.shape}")

    # R(x) = Σ x_i ln(x_i), with 0·ln(0) = 0
    def neg_entropy(x: np.ndarray) -> float:
        mask = x > _EPS
        vals = np.zeros_like(x)
        vals[mask] = x[mask] * np.log(x[mask])
        return float(np.sum(vals))

    # ∇R(θ)_i = ln(θ_i) + 1
    grad_theta = np.log(np.maximum(theta, _EPS)) + 1.0

    result = neg_entropy(mu) - neg_entropy(theta) - np.dot(grad_theta, mu - theta)
    return max(0.0, float(result))


def project_onto_simplex(prices: np.ndarray) -> np.ndarray:
    """Project a price vector onto the probability simplex.

    Finds μ* = argmin_{μ ∈ Δ} ‖μ - prices‖² where Δ = {μ ≥ 0, Σμ = 1}.

    Uses the efficient O(n log n) algorithm from:
        Duchi et al. (2008), "Efficient Projections onto the ℓ1-Ball"

    Parameters
    ----------
    prices : array-like
        Input price vector (may be outside simplex).

    Returns
    -------
    np.ndarray
        Projected vector on the simplex (sums to 1, all ≥ 0).
    """
    prices = np.asarray(prices, dtype=np.float64).ravel()
    n = len(prices)

    if n == 0:
        return prices

    # Sort in descending order
    sorted_prices = np.sort(prices)[::-1]

    # Find the threshold τ
    cumsum = np.cumsum(sorted_prices)
    rho_candidates = sorted_prices - (cumsum - 1.0) / np.arange(1, n + 1)
    rho = int(np.max(np.where(rho_candidates > 0)[0])) + 1 if np.any(rho_candidates > 0) else 1

    tau = (cumsum[rho - 1] - 1.0) / rho

    # Project
    result = np.maximum(prices - tau, 0.0)

    # Ensure exact sum = 1 (fix floating point)
    result /= result.sum()

    return result


def project_onto_polytope(
    prices: np.ndarray,
    A_eq: Optional[np.ndarray] = None,
    b_eq: Optional[np.ndarray] = None,
    A_ub: Optional[np.ndarray] = None,
    b_ub: Optional[np.ndarray] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Project prices onto a constrained polytope via KL minimization.

    Solves:  μ* = argmin_{μ ∈ M} D_KL(μ ‖ θ)

    where M is defined by:
        A_eq · μ = b_eq     (equality constraints)
        A_ub · μ ≤ b_ub     (inequality constraints)
        bounds[i] ≤ μ_i ≤ bounds[i]  (box constraints)

    Parameters
    ----------
    prices : array-like
        Current market prices θ (reference point for KL projection).
    A_eq, b_eq : optional
        Equality constraint matrix and RHS.
    A_ub, b_ub : optional
        Inequality constraint matrix and RHS (A_ub @ μ ≤ b_ub).
    bounds : optional
        Per-variable (lower, upper) bounds. Defaults to (ε, 1-ε) for each.

    Returns
    -------
    np.ndarray
        The KL-projection μ* onto the polytope.

    Notes
    -----
    Uses scipy.optimize.minimize with SLSQP. The objective is:
        min Σ_i μ_i · ln(μ_i / θ_i)
    which is convex, so SLSQP finds the global minimum.
    """
    prices = np.asarray(prices, dtype=np.float64).ravel()
    n = len(prices)

    # Clamp prices to avoid log(0)
    theta = np.maximum(prices, _EPS)

    # Default bounds: stay in (0, 1) open interval
    if bounds is None:
        bounds = [(_EPS, 1.0 - _EPS)] * n

    # KL divergence objective: min D_KL(μ ‖ θ)
    def objective(mu):
        mu_safe = np.maximum(mu, _EPS)
        return float(np.sum(mu_safe * np.log(mu_safe / theta)))

    # Gradient of KL: ∂D_KL/∂μ_i = ln(μ_i/θ_i) + 1
    def gradient(mu):
        mu_safe = np.maximum(mu, _EPS)
        return np.log(mu_safe / theta) + 1.0

    # Build constraints list for SLSQP
    constraints = []

    if A_eq is not None and b_eq is not None:
        A_eq = np.atleast_2d(A_eq)
        b_eq = np.atleast_1d(b_eq)
        for i in range(len(b_eq)):
            row = A_eq[i]
            rhs = b_eq[i]
            constraints.append({
                "type": "eq",
                "fun": lambda mu, r=row, v=rhs: float(r @ mu - v),
                "jac": lambda mu, r=row: r.astype(np.float64),
            })

    if A_ub is not None and b_ub is not None:
        A_ub = np.atleast_2d(A_ub)
        b_ub = np.atleast_1d(b_ub)
        for i in range(len(b_ub)):
            row = A_ub[i]
            rhs = b_ub[i]
            # SLSQP convention: ineq means fun(x) >= 0
            # We want A_ub @ μ ≤ b_ub  →  b_ub - A_ub @ μ ≥ 0
            constraints.append({
                "type": "ineq",
                "fun": lambda mu, r=row, v=rhs: float(v - r @ mu),
                "jac": lambda mu, r=row: -r.astype(np.float64),
            })

    # Initial guess: project prices onto simplex as starting point
    x0 = project_onto_simplex(theta)

    # Clamp x0 within bounds
    for i, (lo, hi) in enumerate(bounds):
        x0[i] = np.clip(x0[i], lo, hi)

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12, "disp": False},
    )

    if not result.success:
        logger.warning(
            "KL projection did not converge",
            message=result.message,
            fun=result.fun,
        )

    mu_star = np.maximum(result.x, 0.0)
    return mu_star
