"""Marginal polytope construction for prediction market arbitrage.

A marginal polytope M is the set of all valid probability distributions
over market outcomes, accounting for logical dependencies between markets.

For a single market with n outcomes:
    M = {μ ∈ R^n : μ_i ≥ 0, Σ_i μ_i = 1}  (the probability simplex)

For multiple dependent markets, M is smaller than the product of simplices
because logical constraints eliminate impossible outcome combinations.

Example:
    Market A: "Trump wins PA?" (Yes/No)
    Market B: "Republicans win PA by 5+?" (Yes/No)

    Dependency: B=Yes → A=Yes (can't win by 5+ without winning).
    Constraint: p(B=Yes) ≤ p(A=Yes)
    This reduces the valid outcome space.

Arbitrage exists when market prices θ lie OUTSIDE the polytope M.
The guaranteed profit equals the Bregman divergence D_KL(μ* ‖ θ)
where μ* is the KL-projection of θ onto M.

References
----------
- arXiv:2508.03474v1, Section 4 (Polytope construction)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import structlog

from .bregman import kl_divergence, project_onto_polytope, project_onto_simplex

logger = structlog.get_logger(__name__)


@dataclass
class Polytope:
    """Representation of a marginal polytope for arbitrage detection.

    The polytope is defined by:
        A_eq @ μ = b_eq     (equality constraints, e.g. prob sums = 1)
        A_ub @ μ ≤ b_ub     (inequality constraints from dependencies)
        bounds[i] = (lo, hi) per variable

    And optionally by explicit vertex enumeration for small cases.

    Attributes
    ----------
    n_vars : int
        Total dimension (sum of outcomes across all markets).
    A_eq, b_eq : np.ndarray
        Equality constraints.
    A_ub, b_ub : np.ndarray
        Inequality constraints (A_ub @ μ ≤ b_ub).
    bounds : list of (float, float)
        Per-variable bounds.
    vertices : optional np.ndarray
        Explicit vertex list (for small polytopes).
    market_slices : dict
        Maps market_id → (start_idx, end_idx) into the variable vector.
    """

    n_vars: int
    A_eq: np.ndarray
    b_eq: np.ndarray
    A_ub: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    b_ub: np.ndarray = field(default_factory=lambda: np.empty(0))
    bounds: List[Tuple[float, float]] = field(default_factory=list)
    vertices: Optional[np.ndarray] = None
    market_slices: Dict[str, Tuple[int, int]] = field(default_factory=dict)


def build_single_market_polytope(n_outcomes: int) -> Polytope:
    """Build the simplex polytope for a single market.

    Constraints:
        Σ_i μ_i = 1  (probabilities sum to 1)
        μ_i ≥ 0      (non-negative)

    Parameters
    ----------
    n_outcomes : int
        Number of outcomes (typically 2 for Yes/No markets).

    Returns
    -------
    Polytope
        Simplex polytope with equality and bound constraints.
    """
    # One equality: sum = 1
    A_eq = np.ones((1, n_outcomes))
    b_eq = np.array([1.0])

    # No inequality constraints beyond bounds
    A_ub = np.empty((0, n_outcomes))
    b_ub = np.empty(0)

    # Each variable in [0, 1]
    eps = 1e-15
    bounds = [(eps, 1.0 - eps)] * n_outcomes

    # Vertices of the simplex: standard basis vectors
    vertices = np.eye(n_outcomes)

    return Polytope(
        n_vars=n_outcomes,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        vertices=vertices,
        market_slices={"market_0": (0, n_outcomes)},
    )


@dataclass
class DependencyConstraint:
    """A linear constraint from a market dependency.

    Represents: coeff_a · μ[idx_a] + coeff_b · μ[idx_b] ≤ rhs
    (or = rhs for equality constraints).
    """

    indices: List[int]
    coefficients: List[float]
    rhs: float
    constraint_type: str = "ub"  # "ub" for ≤, "eq" for =


def build_multi_market_polytope(
    markets: List[Dict],
    dependencies: Optional[List[DependencyConstraint]] = None,
) -> Polytope:
    """Build the combined marginal polytope for multiple markets.

    Each market contributes a simplex constraint (outcomes sum to 1).
    Dependencies add cross-market inequality constraints.

    Parameters
    ----------
    markets : list of dict
        Each dict has:
            - "id": str, market identifier
            - "n_outcomes": int, number of outcomes
    dependencies : optional list of DependencyConstraint
        Cross-market constraints derived from logical dependencies.

    Returns
    -------
    Polytope
        Combined polytope with all constraints.

    Example
    -------
    Two binary markets (4 variables: [A_yes, A_no, B_yes, B_no]):

        markets = [
            {"id": "market_A", "n_outcomes": 2},
            {"id": "market_B", "n_outcomes": 2},
        ]
        # Dependency: p(B_yes) ≤ p(A_yes)  →  μ[2] - μ[0] ≤ 0
        deps = [DependencyConstraint(indices=[2, 0], coefficients=[1.0, -1.0], rhs=0.0)]
    """
    if dependencies is None:
        dependencies = []

    # Calculate total dimension and market slices
    n_vars = 0
    market_slices: Dict[str, Tuple[int, int]] = {}

    for m in markets:
        start = n_vars
        n_out = m["n_outcomes"]
        market_slices[m["id"]] = (start, start + n_out)
        n_vars += n_out

    # Equality constraints: each market's outcomes sum to 1
    n_markets = len(markets)
    A_eq = np.zeros((n_markets, n_vars))
    b_eq = np.ones(n_markets)

    for i, m in enumerate(markets):
        start, end = market_slices[m["id"]]
        A_eq[i, start:end] = 1.0

    # Inequality constraints from dependencies
    ub_rows = []
    eq_extra_rows = []

    for dep in dependencies:
        row = np.zeros(n_vars)
        for idx, coeff in zip(dep.indices, dep.coefficients):
            row[idx] = coeff
        if dep.constraint_type == "ub":
            ub_rows.append((row, dep.rhs))
        elif dep.constraint_type == "eq":
            eq_extra_rows.append((row, dep.rhs))

    if ub_rows:
        A_ub = np.array([r for r, _ in ub_rows])
        b_ub = np.array([v for _, v in ub_rows])
    else:
        A_ub = np.empty((0, n_vars))
        b_ub = np.empty(0)

    if eq_extra_rows:
        A_eq_extra = np.array([r for r, _ in eq_extra_rows])
        b_eq_extra = np.array([v for _, v in eq_extra_rows])
        A_eq = np.vstack([A_eq, A_eq_extra])
        b_eq = np.concatenate([b_eq, b_eq_extra])

    eps = 1e-15
    bounds = [(eps, 1.0 - eps)] * n_vars

    return Polytope(
        n_vars=n_vars,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        vertices=None,
        market_slices=market_slices,
    )


def enumerate_valid_outcomes(
    markets: List[Dict],
    validity_fn: Optional[callable] = None,
) -> np.ndarray:
    """Enumerate all valid outcome vectors for a set of markets.

    For k markets with n_1, n_2, ..., n_k outcomes, there are
    Π n_i possible outcome combinations. Each combination is an
    indicator vector in R^(Σ n_i).

    Parameters
    ----------
    markets : list of dict
        Each dict has "id" and "n_outcomes".
    validity_fn : optional callable
        Takes a dict mapping market_id → outcome_index, returns bool.
        If provided, only valid combinations are included.
        If None, all combinations are valid (independent markets).

    Returns
    -------
    np.ndarray of shape (n_valid, n_vars)
        Each row is a valid outcome indicator vector.

    Warning
    -------
    Exponential in the number of markets! Only use for small cases
    (≤ ~15-20 total combinations).
    """
    n_vars = sum(m["n_outcomes"] for m in markets)
    outcome_ranges = [range(m["n_outcomes"]) for m in markets]

    # Build market slices
    slices = {}
    offset = 0
    for m in markets:
        slices[m["id"]] = (offset, offset + m["n_outcomes"])
        offset += m["n_outcomes"]

    vertices = []
    for combo in itertools.product(*outcome_ranges):
        # Check validity
        if validity_fn is not None:
            outcome_dict = {m["id"]: combo[i] for i, m in enumerate(markets)}
            if not validity_fn(outcome_dict):
                continue

        # Build indicator vector
        v = np.zeros(n_vars)
        for i, m in enumerate(markets):
            start, _ = slices[m["id"]]
            v[start + combo[i]] = 1.0
        vertices.append(v)

    if not vertices:
        logger.warning("No valid outcomes found!")
        return np.empty((0, n_vars))

    return np.array(vertices)


def detect_arbitrage(
    prices: np.ndarray,
    polytope: Polytope,
) -> Tuple[bool, float, np.ndarray]:
    """Detect if market prices admit arbitrage.

    Projects the price vector θ onto the marginal polytope M via KL
    minimization. The profit metric combines two components:

    1. **Per-market mispricing**: For each market, |Σ prices - 1.0|
       represents direct over/under-pricing on a CLOB.
    2. **Cross-market mispricing**: The L1 distance ‖μ* - θ‖₁ captures
       how far prices deviate from the nearest valid distribution,
       including dependency constraint violations.

    For LMSR markets, D_KL(μ*‖θ) is the exact profit. For CLOB markets
    (like Polymarket), we use the L1/sum-based metric which directly
    maps to tradeable profit.

    Parameters
    ----------
    prices : np.ndarray
        Market price vector θ (concatenated across markets).
    polytope : Polytope
        The marginal polytope defining valid distributions.

    Returns
    -------
    is_arb : bool
        True if arbitrage opportunity exists.
    profit : float
        Guaranteed arbitrage profit estimate.
    mu_star : np.ndarray
        The optimal projection point (nearest valid distribution).
    """
    prices = np.asarray(prices, dtype=np.float64).ravel()

    if len(prices) != polytope.n_vars:
        raise ValueError(
            f"Price vector length {len(prices)} != polytope dimension {polytope.n_vars}"
        )

    # Project onto the polytope (KL minimization)
    mu_star = project_onto_polytope(
        prices,
        A_eq=polytope.A_eq,
        b_eq=polytope.b_eq,
        A_ub=polytope.A_ub if polytope.A_ub.size > 0 else None,
        b_ub=polytope.b_ub if polytope.b_ub.size > 0 else None,
        bounds=polytope.bounds if polytope.bounds else None,
    )

    # Compute profit metric:
    # 1) Per-market sum deviation (CLOB-native metric)
    sum_deviation = 0.0
    for mid, (s, e) in polytope.market_slices.items():
        market_sum = float(np.sum(prices[s:e]))
        sum_deviation += abs(market_sum - 1.0)

    # 2) L1 distance from projection (captures dependency violations too)
    l1_dist = float(np.sum(np.abs(mu_star - prices)))

    # 3) KL divergence (only meaningful when θ is close to normalized)
    # Normalize θ per market for KL calculation
    theta_norm = prices.copy()
    for mid, (s, e) in polytope.market_slices.items():
        seg = theta_norm[s:e]
        seg_sum = seg.sum()
        if seg_sum > 0:
            theta_norm[s:e] = seg / seg_sum
    kl_profit = kl_divergence(mu_star, theta_norm)

    # Use the maximum of all metrics as the profit indicator
    profit = max(sum_deviation, l1_dist / 2.0, kl_profit)

    # Threshold for meaningful arbitrage (accounts for numerical noise)
    arb_threshold = 1e-6
    is_arb = profit > arb_threshold

    if is_arb:
        logger.info(
            "arbitrage_detected",
            profit=profit,
            sum_deviation=sum_deviation,
            l1_distance=l1_dist,
            kl_divergence=kl_profit,
            price_sum_check={
                mid: float(np.sum(prices[s:e]))
                for mid, (s, e) in polytope.market_slices.items()
            },
        )

    return is_arb, profit, mu_star


def calculate_arbitrage_profit(
    prices: np.ndarray,
    polytope: Polytope,
    liquidity_b: float = 1.0,
) -> Dict:
    """Calculate detailed arbitrage profit and trading directions.

    For LMSR markets, the cost of moving from price θ to μ is:
        cost = b · [C(q_new) - C(q_old)]
    where C(q) = b · ln(Σ exp(q_i/b))  is the LMSR cost function
    and q is the share vector.

    The guaranteed profit from buying the arbitrage portfolio is:
        profit = b · D_KL(μ* ‖ θ)

    For CLOB (order book) markets like Polymarket, the profit calculation
    is simpler: buy each outcome at its current price, guaranteed payout
    is $1 for one outcome → profit = 1 - Σ prices (if prices overlap).

    Parameters
    ----------
    prices : np.ndarray
        Current market price vector.
    polytope : Polytope
        Marginal polytope.
    liquidity_b : float
        LMSR liquidity parameter (1.0 for CLOB markets).

    Returns
    -------
    dict with keys:
        - is_arbitrage: bool
        - kl_divergence: float (raw KL distance)
        - profit_per_unit: float (guaranteed profit per $1 risked)
        - trade_directions: dict mapping market_id → {outcome_idx: delta}
          Positive delta = buy, negative = sell.
        - projected_prices: np.ndarray
    """
    prices = np.asarray(prices, dtype=np.float64).ravel()

    is_arb, kl_profit, mu_star = detect_arbitrage(prices, polytope)

    # Trade directions: move from current prices θ to projected μ*
    delta = mu_star - prices
    trade_directions = {}
    for mid, (start, end) in polytope.market_slices.items():
        directions = {}
        for i in range(start, end):
            if abs(delta[i]) > 1e-8:
                directions[i - start] = float(delta[i])
        if directions:
            trade_directions[mid] = directions

    # For CLOB markets: profit = sum of (θ_i - μ*_i) for overpriced outcomes
    # Intuitively: if sum of YES prices > 1, you can sell each YES and guarantee
    # profit of (sum - 1) per unit.
    clob_profits = {}
    for mid, (start, end) in polytope.market_slices.items():
        market_prices = prices[start:end]
        price_sum = float(np.sum(market_prices))
        if price_sum > 1.0 + 1e-6:
            # Overpriced: sell all outcomes → guaranteed profit
            clob_profits[mid] = price_sum - 1.0
        elif price_sum < 1.0 - 1e-6:
            # Underpriced: buy all outcomes → guaranteed profit
            clob_profits[mid] = 1.0 - price_sum

    return {
        "is_arbitrage": is_arb,
        "kl_divergence": kl_profit,
        "profit_lmsr": liquidity_b * kl_profit,
        "profit_clob": clob_profits,
        "trade_directions": trade_directions,
        "projected_prices": mu_star,
        "original_prices": prices,
    }
