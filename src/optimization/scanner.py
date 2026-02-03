"""Arbitrage scanner for Polymarket prediction markets.

Ties together Bregman projection, Frank-Wolfe optimization, polytope
construction, and dependency detection into a unified scanner.

Three levels of arbitrage detection:
    1. Single-market: YES + NO prices ≠ $1.00 within one market
    2. Multi-market: Exhaustive outcome group prices ≠ $1.00
       (e.g., "Trump Electoral Votes" ranges should sum to 1)
    3. Cross-market: Dependent markets with polytope constraints
       (uses Bregman projection / Frank-Wolfe for detection)

References
----------
- arXiv:2508.03474v1
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import structlog

from .bregman import kl_divergence, project_onto_polytope, project_onto_simplex
from .dependency import DependencyDetector, MarketDependency
from .frank_wolfe import FrankWolfe
from .polytope import (
    DependencyConstraint,
    Polytope,
    build_multi_market_polytope,
    build_single_market_polytope,
    calculate_arbitrage_profit,
    detect_arbitrage,
    enumerate_valid_outcomes,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExecutionLeg:
    """One leg of an arbitrage trade."""

    market_id: str
    side: str  # "buy_yes", "buy_no", "sell_yes", "sell_no"
    outcome_index: int
    entry_price: float
    size_usd: float
    token_id: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Plan for executing an arbitrage opportunity.

    Attributes
    ----------
    legs : list of ExecutionLeg
        Individual trade legs (executed in parallel where possible).
    total_cost : float
        Total capital required.
    guaranteed_payout : float
        Minimum payout regardless of outcome.
    guaranteed_profit : float
        guaranteed_payout - total_cost (always positive for real arb).
    parallel_groups : list of list of int
        Groups of leg indices that can execute simultaneously.
    max_slippage_pct : float
        Maximum acceptable slippage before aborting.
    """

    legs: List[ExecutionLeg] = field(default_factory=list)
    total_cost: float = 0.0
    guaranteed_payout: float = 0.0
    guaranteed_profit: float = 0.0
    parallel_groups: List[List[int]] = field(default_factory=list)
    max_slippage_pct: float = 2.0


@dataclass
class ArbOpportunity:
    """A detected arbitrage opportunity.

    Attributes
    ----------
    arb_type : str
        "single_market", "multi_market", or "cross_market".
    market_ids : list of str
        Markets involved.
    sides : dict
        Mapping market_id → "buy_yes" / "buy_no" per market.
    entry_prices : dict
        Mapping market_id → price at detection time.
    guaranteed_profit : float
        Minimum guaranteed profit in dollars per $1 risked.
    profit_pct : float
        Profit as percentage of capital deployed.
    execution_plan : ExecutionPlan
        Detailed execution plan.
    kl_divergence : float
        Raw KL divergence (mathematical profit metric).
    projected_prices : Optional[np.ndarray]
        KL-projected prices onto the valid polytope.
    detected_at : float
        Unix timestamp of detection.
    confidence : float
        Confidence in the opportunity (0-1).
    notes : str
        Human-readable description.
    """

    arb_type: str
    market_ids: List[str]
    sides: Dict[str, str]
    entry_prices: Dict[str, float]
    guaranteed_profit: float
    profit_pct: float
    execution_plan: ExecutionPlan = field(default_factory=ExecutionPlan)
    kl_divergence: float = 0.0
    projected_prices: Optional[np.ndarray] = None
    detected_at: float = field(default_factory=time.time)
    confidence: float = 1.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class ArbScanner:
    """Arbitrage scanner for Polymarket.

    Parameters
    ----------
    min_profit_pct : float
        Minimum profit percentage to report (default 0.5% to cover fees).
    fee_rate : float
        Polymarket fee rate per trade (default ~2% round-trip).
    dependency_detector : optional DependencyDetector
        LLM-based dependency detector for cross-market arb.
    frank_wolfe_iters : int
        Max iterations for Frank-Wolfe solver.
    """

    def __init__(
        self,
        min_profit_pct: float = 0.5,
        fee_rate: float = 0.02,
        dependency_detector: Optional[DependencyDetector] = None,
        frank_wolfe_iters: int = 150,
    ):
        self.min_profit_pct = min_profit_pct
        self.fee_rate = fee_rate
        self.dependency_detector = dependency_detector or DependencyDetector()
        self.fw_solver = FrankWolfe(
            max_iter=frank_wolfe_iters,
            convergence_threshold=1e-7,
        )

    # ------------------------------------------------------------------
    # Level 1: Single-market arbitrage
    # ------------------------------------------------------------------

    async def scan_single_market_arb(
        self,
        markets: List[Dict[str, Any]],
    ) -> List[ArbOpportunity]:
        """Check if YES + NO prices ≠ $1.00 in individual markets.

        For a binary market (Yes/No), the prices should sum to $1.00.
        Any deviation (net of fees) is a risk-free arbitrage:

        If sum > 1.0: Sell both sides → guaranteed profit = sum - 1.0
        If sum < 1.0: Buy both sides → guaranteed profit = 1.0 - sum

        Parameters
        ----------
        markets : list of dict
            Each dict has: "id", "question", "yes_price", "no_price",
            optionally "yes_token_id", "no_token_id".

        Returns
        -------
        list of ArbOpportunity
        """
        opportunities = []

        for m in markets:
            mid = m.get("id", "unknown")
            yes_p = m.get("yes_price", 0.0)
            no_p = m.get("no_price", 0.0)

            if yes_p <= 0 or no_p <= 0:
                continue

            price_sum = yes_p + no_p
            deviation = abs(price_sum - 1.0)

            # Net profit after fees on both legs
            fee_cost = self.fee_rate * 2  # fee on each leg
            net_profit = deviation - fee_cost

            if net_profit <= 0:
                continue

            profit_pct = (net_profit / price_sum) * 100

            if profit_pct < self.min_profit_pct:
                continue

            if price_sum > 1.0:
                # Overpriced: sell YES and sell NO (or equivalently, do nothing
                # and let market makers cover). On Polymarket CLOB, this means
                # buying NO on both sides isn't possible — instead:
                # Strategy: The "overpriced" case is rare on CLOB; usually it's
                # the underpriced case. For CLOB: buy YES at yes_p + buy NO at
                # no_p → guaranteed payout $1.00 → profit = 1.0 - sum if sum < 1.
                # If sum > 1, there's no direct arb on CLOB (can't short).
                continue

            # Underpriced: buy YES + buy NO → payout $1.00
            plan = ExecutionPlan(
                legs=[
                    ExecutionLeg(
                        market_id=mid,
                        side="buy_yes",
                        outcome_index=0,
                        entry_price=yes_p,
                        size_usd=yes_p,
                        token_id=m.get("yes_token_id"),
                    ),
                    ExecutionLeg(
                        market_id=mid,
                        side="buy_no",
                        outcome_index=1,
                        entry_price=no_p,
                        size_usd=no_p,
                        token_id=m.get("no_token_id"),
                    ),
                ],
                total_cost=price_sum,
                guaranteed_payout=1.0,
                guaranteed_profit=net_profit,
                parallel_groups=[[0, 1]],  # both legs can execute simultaneously
            )

            opp = ArbOpportunity(
                arb_type="single_market",
                market_ids=[mid],
                sides={mid: "buy_yes+buy_no"},
                entry_prices={mid: price_sum},
                guaranteed_profit=net_profit,
                profit_pct=profit_pct,
                execution_plan=plan,
                kl_divergence=0.0,
                notes=f"YES({yes_p:.3f})+NO({no_p:.3f})={price_sum:.4f}, "
                      f"profit={net_profit:.4f} ({profit_pct:.2f}%)",
            )
            opportunities.append(opp)

        logger.info(
            "single_market_scan_complete",
            n_markets=len(markets),
            n_opportunities=len(opportunities),
        )
        return opportunities

    # ------------------------------------------------------------------
    # Level 2: Multi-market (event group) arbitrage
    # ------------------------------------------------------------------

    async def scan_multi_market_arb(
        self,
        event_markets: Dict[str, List[Dict[str, Any]]],
    ) -> List[ArbOpportunity]:
        """Check event groups where all outcomes should sum to $1.00.

        Example: "Trump Electoral Votes" might have markets for ranges
        [0-99, 100-199, 200-269, 270-319, 320-399, 400+].
        These are exhaustive and mutually exclusive → prices must sum to 1.

        Parameters
        ----------
        event_markets : dict
            Maps event_group_id → list of markets in that group.
            Each market dict has: "id", "question", "yes_price",
            "outcome_label" (the specific outcome this market covers).

        Returns
        -------
        list of ArbOpportunity
        """
        opportunities = []

        for event_id, markets in event_markets.items():
            if len(markets) < 2:
                continue

            # Sum of YES prices across all outcomes in the event
            yes_prices = []
            market_ids = []
            for m in markets:
                yp = m.get("yes_price", 0.0)
                if yp <= 0:
                    continue
                yes_prices.append(yp)
                market_ids.append(m.get("id", "unknown"))

            if len(yes_prices) < 2:
                continue

            price_sum = sum(yes_prices)
            n_legs = len(yes_prices)

            # Fee cost: one fee per leg
            fee_cost = self.fee_rate * n_legs

            if price_sum < 1.0:
                # Underpriced: buy YES on all outcomes
                net_profit = (1.0 - price_sum) - fee_cost
                if net_profit <= 0:
                    continue

                profit_pct = (net_profit / price_sum) * 100
                if profit_pct < self.min_profit_pct:
                    continue

                legs = []
                for i, (m, yp) in enumerate(zip(markets, yes_prices)):
                    legs.append(ExecutionLeg(
                        market_id=m.get("id", "unknown"),
                        side="buy_yes",
                        outcome_index=0,
                        entry_price=yp,
                        size_usd=yp,
                        token_id=m.get("yes_token_id"),
                    ))

                plan = ExecutionPlan(
                    legs=legs,
                    total_cost=price_sum,
                    guaranteed_payout=1.0,
                    guaranteed_profit=net_profit,
                    parallel_groups=[list(range(n_legs))],
                )

                opp = ArbOpportunity(
                    arb_type="multi_market",
                    market_ids=market_ids,
                    sides={mid: "buy_yes" for mid in market_ids},
                    entry_prices={mid: yp for mid, yp in zip(market_ids, yes_prices)},
                    guaranteed_profit=net_profit,
                    profit_pct=profit_pct,
                    execution_plan=plan,
                    notes=f"Event {event_id}: {n_legs} outcomes sum={price_sum:.4f}, "
                          f"profit={net_profit:.4f} ({profit_pct:.2f}%)",
                )
                opportunities.append(opp)

            elif price_sum > 1.0:
                # Overpriced: buy NO on all outcomes
                # Each NO is priced at (1 - yes_price) approximately
                no_prices = [1.0 - yp for yp in yes_prices]
                no_sum = sum(no_prices)

                # Buying NO on all: cost = no_sum, payout = (n-1) × $1
                # because exactly one outcome will be YES → that NO loses,
                # all others pay $1.
                guaranteed_payout = n_legs - 1
                net_profit = guaranteed_payout - no_sum - fee_cost

                if net_profit <= 0:
                    continue

                profit_pct = (net_profit / no_sum) * 100
                if profit_pct < self.min_profit_pct:
                    continue

                legs = []
                for i, (m, np_val) in enumerate(zip(markets, no_prices)):
                    legs.append(ExecutionLeg(
                        market_id=m.get("id", "unknown"),
                        side="buy_no",
                        outcome_index=1,
                        entry_price=np_val,
                        size_usd=np_val,
                        token_id=m.get("no_token_id"),
                    ))

                plan = ExecutionPlan(
                    legs=legs,
                    total_cost=no_sum,
                    guaranteed_payout=guaranteed_payout,
                    guaranteed_profit=net_profit,
                    parallel_groups=[list(range(n_legs))],
                )

                opp = ArbOpportunity(
                    arb_type="multi_market",
                    market_ids=market_ids,
                    sides={mid: "buy_no" for mid in market_ids},
                    entry_prices={mid: np_val for mid, np_val in zip(market_ids, no_prices)},
                    guaranteed_profit=net_profit,
                    profit_pct=profit_pct,
                    execution_plan=plan,
                    notes=f"Event {event_id}: {n_legs} outcomes YES sum={price_sum:.4f} (overpriced), "
                          f"buy all NOs for {no_sum:.4f}, payout={guaranteed_payout}, "
                          f"profit={net_profit:.4f} ({profit_pct:.2f}%)",
                )
                opportunities.append(opp)

        logger.info(
            "multi_market_scan_complete",
            n_events=len(event_markets),
            n_opportunities=len(opportunities),
        )
        return opportunities

    # ------------------------------------------------------------------
    # Level 3: Cross-market arbitrage (Bregman projection)
    # ------------------------------------------------------------------

    async def scan_cross_market_arb(
        self,
        markets: List[Dict[str, Any]],
        dependencies: Optional[List[MarketDependency]] = None,
    ) -> List[ArbOpportunity]:
        """Detect cross-market arbitrage using Bregman projection.

        This is the core algorithm from the paper. Steps:
        1. Build the marginal polytope from market structure + dependencies
        2. Concatenate all market prices into a single vector θ
        3. Project θ onto the polytope: μ* = argmin D_KL(μ ‖ θ)
        4. If D_KL(μ* ‖ θ) > threshold, arbitrage exists
        5. The trade direction is θ → μ* (buy underpriced, sell overpriced)

        Parameters
        ----------
        markets : list of dict
            Each market has: "id", "question", "outcomes" (list of str),
            "prices" (list of float, one per outcome).
        dependencies : optional list of MarketDependency
            Pre-computed dependencies. If None, uses LLM detector.

        Returns
        -------
        list of ArbOpportunity
        """
        if len(markets) < 2:
            return []

        # Step 1: Detect dependencies if not provided
        if dependencies is None:
            dependencies = await self.dependency_detector.detect_dependencies(markets)

        if not dependencies:
            logger.info("no_dependencies_found, skipping cross-market scan")
            return []

        # Step 2: Build polytope
        market_defs = [
            {"id": m["id"], "n_outcomes": len(m.get("outcomes", ["Yes", "No"]))}
            for m in markets
        ]

        # Convert MarketDependency → DependencyConstraint
        dep_constraints = self._dependencies_to_constraints(
            dependencies, market_defs
        )

        polytope = build_multi_market_polytope(market_defs, dep_constraints)

        # Step 3: Build price vector
        prices = []
        market_id_map = {}
        for m in markets:
            market_id_map[m["id"]] = m
            m_prices = m.get("prices", [])
            if not m_prices:
                # Fallback: construct from yes_price
                yp = m.get("yes_price", 0.5)
                m_prices = [yp, 1.0 - yp]
            prices.extend(m_prices)

        prices = np.array(prices, dtype=np.float64)
        prices = np.clip(prices, 1e-6, 1.0 - 1e-6)  # avoid boundary issues

        # Step 4: Detect arbitrage via projection
        is_arb, kl_profit, mu_star = detect_arbitrage(prices, polytope)

        if not is_arb:
            logger.info("no_cross_market_arbitrage")
            return []

        # Step 5: Calculate practical profit and execution plan
        result = calculate_arbitrage_profit(prices, polytope)
        delta = mu_star - prices

        # Build execution plan
        opportunities = []
        sides = {}
        entry_prices = {}
        legs = []
        leg_idx = 0

        for m in markets:
            mid = m["id"]
            start, end = polytope.market_slices[mid]
            market_delta = delta[start:end]
            market_prices = prices[start:end]

            for i, (d, p) in enumerate(zip(market_delta, market_prices)):
                if abs(d) < 1e-6:
                    continue

                outcome_names = m.get("outcomes", ["Yes", "No"])
                outcome_name = outcome_names[i] if i < len(outcome_names) else f"outcome_{i}"

                if d > 0:
                    # Price should be higher → outcome is underpriced → buy
                    side = f"buy_{outcome_name.lower()}"
                else:
                    # Price should be lower → outcome is overpriced → sell/skip
                    side = f"sell_{outcome_name.lower()}"

                sides[f"{mid}_{outcome_name}"] = side
                entry_prices[f"{mid}_{outcome_name}"] = float(p)

                legs.append(ExecutionLeg(
                    market_id=mid,
                    side=side,
                    outcome_index=i,
                    entry_price=float(p),
                    size_usd=abs(float(d)),
                ))
                leg_idx += 1

        # CLOB profit estimation
        # For independent legs, total profit = sum of per-market mispricing
        clob_profit = sum(result.get("profit_clob", {}).values())
        total_cost = sum(leg.size_usd for leg in legs if "buy" in leg.side)
        net_profit = max(clob_profit - self.fee_rate * len(legs), kl_profit)
        profit_pct = (net_profit / max(total_cost, 0.01)) * 100 if total_cost > 0 else 0.0

        if profit_pct < self.min_profit_pct:
            return []

        plan = ExecutionPlan(
            legs=legs,
            total_cost=total_cost,
            guaranteed_payout=total_cost + net_profit,
            guaranteed_profit=net_profit,
            parallel_groups=[list(range(len(legs)))],
        )

        opp = ArbOpportunity(
            arb_type="cross_market",
            market_ids=[m["id"] for m in markets],
            sides=sides,
            entry_prices=entry_prices,
            guaranteed_profit=net_profit,
            profit_pct=profit_pct,
            execution_plan=plan,
            kl_divergence=kl_profit,
            projected_prices=mu_star,
            confidence=min(d.confidence for d in dependencies) if dependencies else 0.5,
            notes=f"Cross-market arb: KL={kl_profit:.6f}, "
                  f"profit={net_profit:.4f} ({profit_pct:.2f}%), "
                  f"{len(dependencies)} dependencies",
        )
        opportunities.append(opp)

        logger.info(
            "cross_market_arb_found",
            kl_divergence=kl_profit,
            profit=net_profit,
            profit_pct=profit_pct,
            n_legs=len(legs),
        )

        return opportunities

    def _dependencies_to_constraints(
        self,
        dependencies: List[MarketDependency],
        market_defs: List[Dict],
    ) -> List[DependencyConstraint]:
        """Convert MarketDependency objects to DependencyConstraint for polytope.

        Maps market-local outcome indices to global variable indices.
        """
        # Build global index map
        offset = 0
        global_offset: Dict[str, int] = {}
        for m in market_defs:
            global_offset[m["id"]] = offset
            offset += m["n_outcomes"]

        constraints = []

        for dep in dependencies:
            off_a = global_offset.get(dep.market_a_id)
            off_b = global_offset.get(dep.market_b_id)

            if off_a is None or off_b is None:
                continue

            for lc in dep.linear_constraints:
                a_idx = lc["a_outcome_idx"]
                b_idx = lc["b_outcome_idx"]

                global_a = off_a + a_idx
                global_b = off_b + b_idx

                if lc["type"] == "incompatible":
                    # p(A=a) + p(B=b) ≤ 1
                    constraints.append(DependencyConstraint(
                        indices=[global_a, global_b],
                        coefficients=[1.0, 1.0],
                        rhs=1.0,
                        constraint_type="ub",
                    ))
                elif lc["type"] == "implication":
                    # p(A=a) ≤ p(B=b)  →  p(A=a) - p(B=b) ≤ 0
                    constraints.append(DependencyConstraint(
                        indices=[global_a, global_b],
                        coefficients=[1.0, -1.0],
                        rhs=0.0,
                        constraint_type="ub",
                    ))

        return constraints

    def calculate_execution_plan(
        self,
        opportunity: ArbOpportunity,
        max_position_usd: float = 100.0,
    ) -> ExecutionPlan:
        """Refine an execution plan with VWAP-aware sizing.

        Parameters
        ----------
        opportunity : ArbOpportunity
            The detected opportunity.
        max_position_usd : float
            Maximum total position size in USD.

        Returns
        -------
        ExecutionPlan
            Refined plan with proper sizing.
        """
        plan = opportunity.execution_plan

        if not plan.legs:
            return plan

        # Scale all legs proportionally to fit within max_position
        current_total = sum(leg.size_usd for leg in plan.legs)
        if current_total <= 0:
            return plan

        scale = min(1.0, max_position_usd / current_total)

        for leg in plan.legs:
            leg.size_usd *= scale

        plan.total_cost *= scale
        plan.guaranteed_payout *= scale
        plan.guaranteed_profit *= scale

        return plan


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

async def _run_tests():
    """Run basic tests to verify the optimization engine."""
    import sys

    print("=" * 60)
    print("Polymarket Arbitrage Optimization Engine — Self-Test")
    print("=" * 60)

    errors = 0

    # --- Test 1: KL Divergence ---
    print("\n[Test 1] KL Divergence")
    from .bregman import kl_divergence, bregman_divergence

    mu = np.array([0.5, 0.5])
    theta = np.array([0.5, 0.5])
    kl = kl_divergence(mu, theta)
    assert abs(kl) < 1e-10, f"KL(same, same) should be 0, got {kl}"
    print(f"  KL([0.5,0.5] || [0.5,0.5]) = {kl:.10f} ✓")

    mu = np.array([0.7, 0.3])
    theta = np.array([0.5, 0.5])
    kl = kl_divergence(mu, theta)
    expected = 0.7 * math.log(0.7 / 0.5) + 0.3 * math.log(0.3 / 0.5)
    assert abs(kl - expected) < 1e-10, f"KL mismatch: {kl} vs {expected}"
    print(f"  KL([0.7,0.3] || [0.5,0.5]) = {kl:.6f} (expected {expected:.6f}) ✓")

    # Bregman should equal KL on simplex
    breg = bregman_divergence(mu, theta)
    assert abs(breg - kl) < 1e-10, f"Bregman != KL on simplex: {breg} vs {kl}"
    print(f"  Bregman divergence matches KL on simplex ✓")

    # --- Test 2: Simplex Projection ---
    print("\n[Test 2] Simplex Projection")
    from .bregman import project_onto_simplex

    prices = np.array([0.6, 0.6])  # sums to 1.2
    proj = project_onto_simplex(prices)
    assert abs(proj.sum() - 1.0) < 1e-10, f"Projection doesn't sum to 1: {proj.sum()}"
    assert all(p >= 0 for p in proj), f"Projection has negatives: {proj}"
    print(f"  project([0.6, 0.6]) = {proj} (sum={proj.sum():.10f}) ✓")

    prices = np.array([0.3, 0.2])  # sums to 0.5
    proj = project_onto_simplex(prices)
    assert abs(proj.sum() - 1.0) < 1e-10
    print(f"  project([0.3, 0.2]) = {proj} (sum={proj.sum():.10f}) ✓")

    # Already on simplex
    prices = np.array([0.4, 0.6])
    proj = project_onto_simplex(prices)
    assert np.allclose(proj, prices, atol=1e-8)
    print(f"  project([0.4, 0.6]) = {proj} (already on simplex) ✓")

    # --- Test 3: Single Market Polytope ---
    print("\n[Test 3] Single Market Polytope & Arbitrage Detection")

    poly = build_single_market_polytope(2)
    assert poly.n_vars == 2
    assert poly.A_eq.shape == (1, 2)
    print(f"  Built 2-outcome simplex polytope ✓")

    # Fair prices: no arbitrage
    fair_prices = np.array([0.6, 0.4])
    is_arb, profit, mu_star = detect_arbitrage(fair_prices, poly)
    assert not is_arb, f"False positive: fair prices detected as arb"
    print(f"  Fair prices [0.6, 0.4]: arb={is_arb}, profit={profit:.8f} ✓")

    # Mispriced: arbitrage
    bad_prices = np.array([0.6, 0.6])  # sum=1.2
    is_arb, profit, mu_star = detect_arbitrage(bad_prices, poly)
    assert is_arb, f"Missed arbitrage in overpriced market"
    print(f"  Overpriced [0.6, 0.6]: arb={is_arb}, profit={profit:.6f}, μ*={mu_star} ✓")

    # --- Test 4: Multi-market Polytope with Dependencies ---
    print("\n[Test 4] Multi-market Polytope with Dependencies")

    market_defs = [
        {"id": "A", "n_outcomes": 2},  # A: Yes(0)/No(1)
        {"id": "B", "n_outcomes": 2},  # B: Yes(0)/No(1)
    ]

    # Dependency: B=Yes → A=Yes, so p(B_yes) ≤ p(A_yes)
    # Global indices: A_yes=0, A_no=1, B_yes=2, B_no=3
    dep = DependencyConstraint(
        indices=[2, 0],       # B_yes - A_yes ≤ 0
        coefficients=[1.0, -1.0],
        rhs=0.0,
        constraint_type="ub",
    )

    poly = build_multi_market_polytope(market_defs, [dep])
    assert poly.n_vars == 4
    assert poly.A_ub.shape[0] == 1  # one inequality constraint
    print(f"  Built 2-market polytope with 1 dependency ✓")

    # Prices violating dependency: B_yes > A_yes
    # A_yes=0.3, A_no=0.7, B_yes=0.5, B_no=0.5
    # This violates p(B_yes) ≤ p(A_yes) since 0.5 > 0.3
    bad_prices = np.array([0.3, 0.7, 0.5, 0.5])
    is_arb, profit, mu_star = detect_arbitrage(bad_prices, poly)
    assert is_arb, f"Missed cross-market arbitrage"
    assert mu_star[2] <= mu_star[0] + 1e-6, f"Projection violates constraint: B_yes={mu_star[2]} > A_yes={mu_star[0]}"
    print(f"  Dependency violation detected: arb={is_arb}, profit={profit:.6f}")
    print(f"  Original: A_yes={bad_prices[0]}, B_yes={bad_prices[2]}")
    print(f"  Projected: A_yes={mu_star[0]:.4f}, B_yes={mu_star[2]:.4f} ✓")

    # --- Test 5: Frank-Wolfe Solver ---
    print("\n[Test 5] Frank-Wolfe Solver")
    from .frank_wolfe import FrankWolfe

    # Minimize ‖x - target‖² over the simplex
    target = np.array([0.7, 0.2, 0.1])
    vertices = np.eye(3)  # simplex vertices

    def obj(x):
        return float(0.5 * np.sum((x - target) ** 2))

    def grad(x):
        return x - target

    fw = FrankWolfe(max_iter=100, convergence_threshold=1e-8)
    result = fw.solve(obj, grad, vertices=vertices)

    assert result.converged, f"Frank-Wolfe did not converge (gap={result.gap})"
    assert np.allclose(result.x, target, atol=1e-4), f"FW solution {result.x} far from target {target}"
    print(f"  Minimized ‖x-[0.7,0.2,0.1]‖² over simplex: x={result.x}")
    print(f"  Converged in {result.iterations} iters, gap={result.gap:.2e} ✓")

    # --- Test 6: Scanner ---
    print("\n[Test 6] ArbScanner - Single Market")
    scanner = ArbScanner(min_profit_pct=0.1, fee_rate=0.01)

    test_markets = [
        {"id": "m1", "question": "Test?", "yes_price": 0.45, "no_price": 0.45},
        {"id": "m2", "question": "Test2?", "yes_price": 0.50, "no_price": 0.50},
        {"id": "m3", "question": "Test3?", "yes_price": 0.60, "no_price": 0.42},  # no arb (sum > 1)
    ]

    opps = await scanner.scan_single_market_arb(test_markets)
    print(f"  Found {len(opps)} single-market opportunities")
    for opp in opps:
        print(f"    {opp.market_ids[0]}: profit={opp.guaranteed_profit:.4f} "
              f"({opp.profit_pct:.2f}%) — {opp.notes}")

    # --- Test 7: Scanner - Multi Market ---
    print("\n[Test 7] ArbScanner - Multi Market (Event Groups)")

    event_markets = {
        "electoral_votes": [
            {"id": "ev1", "question": "Trump 0-99 EV?", "yes_price": 0.05},
            {"id": "ev2", "question": "Trump 100-199 EV?", "yes_price": 0.10},
            {"id": "ev3", "question": "Trump 200-269 EV?", "yes_price": 0.15},
            {"id": "ev4", "question": "Trump 270-319 EV?", "yes_price": 0.25},
            {"id": "ev5", "question": "Trump 320-399 EV?", "yes_price": 0.20},
            {"id": "ev6", "question": "Trump 400+ EV?", "yes_price": 0.10},
            # Sum = 0.85 < 1.0 → buy all YES for guaranteed profit
        ],
    }

    opps = await scanner.scan_multi_market_arb(event_markets)
    print(f"  Found {len(opps)} multi-market opportunities")
    for opp in opps:
        print(f"    {opp.notes}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(_run_tests())
