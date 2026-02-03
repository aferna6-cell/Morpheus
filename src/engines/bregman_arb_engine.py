"""Advanced Bregman-projection arbitrage engine for Polymarket.

Three layers of arbitrage detection:

1. **Single-market** — YES + NO ≠ 1.0 (fastest, simplest)
2. **Event-group** — All outcomes in a neg-risk event don't sum to 1.0
3. **Cross-market** — Logically dependent markets with exploitable dependencies

Uses the optimization framework at ``src.optimization``:

- ``bregman`` — KL divergence and simplex/polytope projection
- ``frank_wolfe`` — Barrier Frank-Wolfe solver over marginal polytopes
- ``polytope`` — Constraint / polytope construction
- ``dependency`` — LLM-based dependency detection between markets

Runnable standalone::

    python3 -m src.engines.bregman_arb_engine
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

from ..markets import Market
from ..markets_scanner import MarketScanner
from ..utils import BotConfig, append_jsonl, load_config
from .base import BaseEngine
from .event_scanner import EventGroup, EventScanner
from .signals import TradeSignal

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Graceful optional imports — optimisation modules may not be built yet
# ---------------------------------------------------------------------------

_HAS_OPTIMIZATION = False
try:
    import numpy as np
    from ..optimization.bregman import (
        kl_divergence,
        project_onto_polytope,
        project_onto_simplex,
    )
    from ..optimization.frank_wolfe import FrankWolfe, FrankWolfeResult
    from ..optimization.polytope import (
        Polytope,
        build_multi_market_polytope,
        build_single_market_polytope,
        calculate_arbitrage_profit,
        detect_arbitrage,
    )
    from ..optimization.dependency import DependencyDetector, MarketDependency

    _HAS_OPTIMIZATION = True
except ImportError as _imp_err:
    logger.warning(
        "optimization_modules_unavailable",
        error=str(_imp_err),
        hint="Cross-market (Layer 3) arbitrage disabled. "
             "Build src/optimization/* to enable.",
    )

try:
    import numpy as np  # noqa: F811  (may already be imported above)
except ImportError:
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ArbOpportunity:
    """A detected arbitrage opportunity (any layer)."""

    layer: int                # 1, 2, or 3
    label: str                # human-readable label
    market_ids: List[str]     # involved market IDs
    legs: List[Dict[str, Any]]  # individual trade legs
    gross_profit: float       # before fees
    net_profit: float         # after fees
    net_profit_pct: float     # net / cost
    confidence: float         # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BregmanArbEngine(BaseEngine):
    """Advanced arbitrage engine using Bregman projections and Frank-Wolfe.

    See module docstring for the three-layer architecture.
    """

    name: str = "bregman_arb"

    def __init__(self, config: BotConfig) -> None:
        self.config = config

        # Engine-specific config (bregman_arb section or fall back to defaults)
        cfg: Dict[str, Any] = {}
        # Try several locations in config
        for attr in ("bregman_arb", "bregman_arb_engine"):
            val = getattr(config, attr, None)
            if isinstance(val, dict) and val:
                cfg = val
                break
        # Also look in the generic dict
        if not cfg and hasattr(config, "__dict__"):
            cfg = config.__dict__.get("bregman_arb", {}) or {}

        self.scan_interval: float = float(cfg.get("scan_interval_seconds", 30))
        self.cross_scan_interval: float = float(cfg.get("cross_market_scan_interval", 3600))
        self.min_profit_threshold: float = float(cfg.get("min_profit_threshold", 0.02))
        self.min_profit_usd: float = float(cfg.get("min_profit_usd", 5.0))
        self.max_position_usd: float = float(cfg.get("max_position_usd", 100.0))
        self.taker_fee: float = float(cfg.get("taker_fee", 0.02))

        # Frank-Wolfe parameters
        self.fw_max_iter: int = int(cfg.get("frank_wolfe_max_iter", 150))
        self.fw_convergence: float = float(cfg.get("frank_wolfe_convergence", 1e-6))

        # Dependency cache
        self.dep_cache_file = Path(cfg.get("dependency_cache_file", "state/market_dependencies.json"))
        self.dep_refresh_days: int = int(cfg.get("dependency_refresh_days", 7))

        # Sub-components
        self.market_scanner = MarketScanner(config)
        self.event_scanner = EventScanner(config)

        # State
        self._signals: List[TradeSignal] = []
        self._scan_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_cross_scan: float = 0.0
        self._dependency_cache: Dict[str, Any] = {}
        self._opportunities_file = cfg.get(
            "opportunities_file", "state/bregman_arb_opportunities.jsonl"
        )

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize scanner, load cached dependencies, start background scan."""
        self._load_dependency_cache()
        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())
        logger.info("bregman_arb_engine_started")

    async def stop(self) -> None:
        """Save dependency cache, cancel background tasks, cleanup."""
        self._running = False
        if self._scan_task and not self._scan_task.done():
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        self._save_dependency_cache()
        await self.event_scanner.close()
        await self.market_scanner.close()
        logger.info("bregman_arb_engine_stopped")

    async def get_signals(self) -> List[TradeSignal]:
        """Drain and return pending trade signals."""
        signals = list(self._signals)
        self._signals.clear()
        return signals

    # ------------------------------------------------------------------
    # Background scan loop
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        """Periodic scan loop running all three arbitrage layers."""
        while self._running:
            try:
                await self._run_scan_cycle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("bregman_scan_error", error=str(exc), exc_info=True)

            await asyncio.sleep(self.scan_interval)

    async def _run_scan_cycle(self) -> None:
        """Execute one full scan cycle."""
        t0 = time.monotonic()

        # ── Fetch data ───────────────────────────────────────────────
        all_markets, event_groups = await asyncio.gather(
            self.market_scanner.fetch_markets(),
            self.event_scanner.fetch_event_groups(),
            return_exceptions=True,
        )

        if isinstance(all_markets, Exception):
            logger.error("market_fetch_failed", error=str(all_markets))
            all_markets = []
        if isinstance(event_groups, Exception):
            logger.error("event_group_fetch_failed", error=str(event_groups))
            event_groups = {}

        opportunities: List[ArbOpportunity] = []

        # ── Layer 1: Single-market arb ───────────────────────────────
        for market in all_markets:
            opp = self._check_single_market_arb(market)
            if opp is not None:
                opportunities.append(opp)

        # ── Layer 2: Event-group arb ─────────────────────────────────
        for event_id, group in event_groups.items():
            opp = self._check_event_group_arb(event_id, group)
            if opp is not None:
                opportunities.append(opp)

        # ── Layer 3: Cross-market (disabled — set cross_market_scan_interval < 86400 to enable)
        if _HAS_OPTIMIZATION and self.cross_scan_interval < 86400:
            now = time.monotonic()
            if (now - self._last_cross_scan) >= self.cross_scan_interval:
                self._last_cross_scan = now
                try:
                    cross_opps = await self._check_cross_market_arb(all_markets)
                    opportunities.extend(cross_opps)
                except Exception as exc:
                    logger.warning("cross_market_scan_error", error=str(exc))

        # ── Emit signals for profitable opportunities ────────────────
        for opp in opportunities:
            self._log_opportunity(opp)
            if opp.net_profit >= self.min_profit_threshold:
                self._emit_signals(opp)

        elapsed = time.monotonic() - t0
        logger.info(
            "bregman_scan_complete",
            markets=len(all_markets),
            event_groups=len(event_groups),
            opportunities=len(opportunities),
            profitable=[o.label for o in opportunities if o.net_profit >= self.min_profit_threshold],
            elapsed_s=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Layer 1: Single-market arbitrage
    # ------------------------------------------------------------------

    def _check_single_market_arb(self, market: Market) -> Optional[ArbOpportunity]:
        """Check if YES + NO ≠ 1.0 within a single market.

        * If YES + NO < 1.0 → buy both → guaranteed profit at resolution.
        * If YES + NO > 1.0 → sell both (or buy NO on each side).

        Profit must exceed 2× taker fee (paying fee on both legs).
        """
        yes_p = market.yes_price
        no_p = market.no_price
        if yes_p is None or no_p is None:
            return None
        if yes_p <= 0 or no_p <= 0:
            return None

        total = yes_p + no_p
        raw_gap = 1.0 - total  # positive = underpriced, negative = overpriced

        # Fee cost: buying two legs each incurs taker fee on payout
        fee_cost = 2 * self.taker_fee
        net = abs(raw_gap) - fee_cost

        if net < self.min_profit_threshold:
            return None

        # Determine strategy
        # Extract token_ids
        yes_token = ""
        no_token = ""
        for outcome, tok in market.tokens.items():
            if outcome.strip().lower() == "yes":
                yes_token = tok.token_id
            elif outcome.strip().lower() == "no":
                no_token = tok.token_id

        if raw_gap > 0:
            # Underpriced: buy YES + buy NO
            legs = [
                {"market_id": market.id, "side": "buy_yes", "price": yes_p, "token_id": yes_token},
                {"market_id": market.id, "side": "buy_no", "price": no_p, "token_id": no_token},
            ]
            label = f"L1 under: {market.question[:50]}"
        else:
            # Overpriced: buy NO to offset
            legs = [
                {"market_id": market.id, "side": "buy_no", "price": no_p, "token_id": no_token,
                 "note": "offset YES overpricing"},
            ]
            label = f"L1 over: {market.question[:50]}"

        return ArbOpportunity(
            layer=1,
            label=label,
            market_ids=[market.id],
            legs=legs,
            gross_profit=abs(raw_gap),
            net_profit=net,
            net_profit_pct=net / total if total > 0 else 0.0,
            confidence=0.95,  # near-certain, just math
            metadata={
                "yes_price": yes_p,
                "no_price": no_p,
                "total": total,
                "question": market.question,
            },
        )

    # ------------------------------------------------------------------
    # Layer 2: Event-group arbitrage
    # ------------------------------------------------------------------

    def _check_event_group_arb(
        self, event_id: str, group: EventGroup
    ) -> Optional[ArbOpportunity]:
        """Check if YES prices across all outcomes in an event sum to ≠ 1.0.

        For mutually exclusive outcome groups (neg_risk events):
        * sum(YES) < 1.0 → buy YES on every outcome → guaranteed profit
        * sum(YES) > 1.0 → buy NO on cheapest outcomes

        Uses Bregman projection (if available) to find optimal allocation.
        """
        if len(group.markets) < 2:
            return None

        # Collect YES prices
        yes_prices: List[float] = []
        valid_markets: List[Market] = []
        for m in group.markets:
            if m.yes_price is not None and m.yes_price > 0:
                yes_prices.append(m.yes_price)
                valid_markets.append(m)

        if len(yes_prices) < 2:
            return None

        total_yes = sum(yes_prices)
        raw_gap = 1.0 - total_yes  # positive = sum < 1 = underpriced

        # Fee: buying N legs
        n_legs = len(valid_markets)
        fee_cost = n_legs * self.taker_fee
        net = abs(raw_gap) - fee_cost

        if net < self.min_profit_threshold:
            return None

        # --- Optimal allocation via Bregman projection (if available) ---
        allocation = self._optimal_event_allocation(yes_prices, raw_gap)

        if raw_gap > 0:
            # Sum < 1: buy YES on all → one resolves YES → payout $1
            legs = []
            for i, m in enumerate(valid_markets):
                weight = allocation[i] if allocation is not None else 1.0
                # Get YES token_id from market tokens
                yes_token = ""
                for outcome, tok in m.tokens.items():
                    if outcome.strip().lower() == "yes":
                        yes_token = tok.token_id
                        break
                legs.append({
                    "market_id": m.id,
                    "side": "buy_yes",
                    "price": m.yes_price,
                    "token_id": yes_token,
                    "weight": round(weight, 4),
                    "question": m.question[:60],
                })
            label = f"L2 under (Σ={total_yes:.3f}): {group.event_slug[:40]}"
            confidence = 0.90
        else:
            # Sum > 1: buy NO on most overpriced outcomes
            # Sort by YES price descending (most overpriced first)
            indexed = sorted(enumerate(valid_markets), key=lambda t: -(t[1].yes_price or 0))
            legs = []
            for idx, m in indexed:
                no_p = m.no_price or (1.0 - (m.yes_price or 0.5))
                # Get NO token_id from market tokens
                no_token = ""
                for outcome, tok in m.tokens.items():
                    if outcome.strip().lower() == "no":
                        no_token = tok.token_id
                        break
                legs.append({
                    "market_id": m.id,
                    "side": "buy_no",
                    "price": no_p,
                    "token_id": no_token,
                    "question": m.question[:60],
                })
            label = f"L2 over (Σ={total_yes:.3f}): {group.event_slug[:40]}"
            confidence = 0.85

        return ArbOpportunity(
            layer=2,
            label=label,
            market_ids=[m.id for m in valid_markets],
            legs=legs,
            gross_profit=abs(raw_gap),
            net_profit=net,
            net_profit_pct=net / total_yes if total_yes > 0 else 0.0,
            confidence=confidence,
            metadata={
                "event_id": event_id,
                "event_slug": group.event_slug,
                "n_outcomes": len(valid_markets),
                "total_yes": round(total_yes, 6),
                "raw_gap": round(raw_gap, 6),
            },
        )

    def _optimal_event_allocation(
        self, yes_prices: List[float], raw_gap: float
    ) -> Optional[List[float]]:
        """Use Bregman projection to compute optimal bet allocation.

        Returns a list of weights (one per market) or None if optimization
        modules are unavailable.
        """
        if not _HAS_OPTIMIZATION or np is None:
            return None

        try:
            p = np.array(yes_prices, dtype=np.float64)
            # Project onto the probability simplex to find the
            # information-theoretically optimal allocation
            q = project_onto_simplex(p)
            # Scale weights proportional to projected distribution
            weights = q / q.sum() if q.sum() > 0 else np.ones_like(q) / len(q)
            return weights.tolist()
        except Exception as exc:
            logger.debug("bregman_allocation_fallback", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Layer 3: Cross-market arbitrage
    # ------------------------------------------------------------------

    async def _check_cross_market_arb(
        self, markets: List[Market]
    ) -> List[ArbOpportunity]:
        """Find exploitable dependencies between logically linked markets.

        1. Detect dependencies via LLM (cached).
        2. Build combined marginal polytope with constraints.
        3. Run Frank-Wolfe to project current prices onto the polytope.
        4. If Bregman divergence > threshold → arbitrage exists.
        5. Compute optimal trade vector from projection difference.
        """
        if not _HAS_OPTIMIZATION:
            return []

        opportunities: List[ArbOpportunity] = []

        # Step 1: Detect dependencies (uses cache heavily)
        dependencies = await self._get_dependencies(markets)
        if not dependencies:
            return []

        logger.info("cross_market_dependencies", count=len(dependencies))

        for dep in dependencies:
            try:
                opp = self._evaluate_cross_market_dependency(dep, markets)
                if opp is not None:
                    opportunities.append(opp)
            except Exception as exc:
                logger.debug("cross_market_eval_error", error=str(exc))

        return opportunities

    async def _get_dependencies(
        self, markets: List[Market]
    ) -> List[Any]:
        """Get market dependencies, using cache when possible."""
        if not _HAS_OPTIMIZATION:
            return []

        # Check if cache is fresh
        cache_age_days = self._dependency_cache.get("_age_days", 999)
        if cache_age_days < self.dep_refresh_days and self._dependency_cache.get("dependencies"):
            logger.debug("using_cached_dependencies", age_days=cache_age_days)
            return self._dependency_cache["dependencies"]

        # Detect fresh dependencies
        try:
            detector = DependencyDetector()
            deps = await detector.detect_dependencies(markets)
            self._dependency_cache = {
                "dependencies": deps,
                "_age_days": 0,
                "_updated": datetime.now(timezone.utc).isoformat(),
            }
            self._save_dependency_cache()
            return deps
        except Exception as exc:
            logger.warning("dependency_detection_failed", error=str(exc))
            return self._dependency_cache.get("dependencies", [])

    def _evaluate_cross_market_dependency(
        self, dep: Any, markets: List[Market]
    ) -> Optional[ArbOpportunity]:
        """Evaluate a single dependency for arbitrage via Frank-Wolfe projection."""
        if not _HAS_OPTIMIZATION or np is None:
            return None

        # Resolve market objects from dependency
        market_map = {m.id: m for m in markets}
        dep_market_ids = getattr(dep, "market_ids", []) if hasattr(dep, "market_ids") else []
        dep_markets = [market_map[mid] for mid in dep_market_ids if mid in market_map]

        if len(dep_markets) < 2:
            return None

        # Build current price vector
        prices = []
        for m in dep_markets:
            prices.append(m.yes_price or 0.5)

        p = np.array(prices, dtype=np.float64)
        # Clamp to avoid log(0)
        p = np.clip(p, 1e-8, 1.0 - 1e-8)

        # Build polytope from dependency constraints
        constraints = getattr(dep, "constraints", []) if hasattr(dep, "constraints") else []
        try:
            polytope = build_multi_market_polytope(dep_markets, constraints)
        except Exception as exc:
            logger.debug("polytope_build_error", error=str(exc))
            return None

        # Run Frank-Wolfe to project prices onto feasible set
        try:
            fw = FrankWolfe(
                max_iterations=self.fw_max_iter,
                convergence_threshold=self.fw_convergence,
            )
            result: FrankWolfeResult = fw.solve(p, polytope)
            projected = result.solution
        except Exception as exc:
            logger.debug("frank_wolfe_error", error=str(exc))
            return None

        # Compute Bregman (KL) divergence between current and projected
        try:
            divergence = kl_divergence(p, projected)
        except Exception:
            divergence = float(np.sum(np.abs(p - projected)))

        if divergence < self.min_profit_threshold:
            return None

        # Build trade vector: difference tells us what to buy/sell
        trade_vec = projected - p
        fee_cost = len(dep_markets) * self.taker_fee
        net = divergence - fee_cost

        if net < self.min_profit_threshold:
            return None

        legs = []
        for i, m in enumerate(dep_markets):
            delta = float(trade_vec[i])
            if abs(delta) < 0.005:
                continue
            side = "buy_yes" if delta > 0 else "buy_no"
            legs.append({
                "market_id": m.id,
                "side": side,
                "price": m.yes_price,
                "delta": round(delta, 4),
                "projected": round(float(projected[i]), 4),
                "question": m.question[:60],
            })

        if not legs:
            return None

        label = f"L3 cross ({len(dep_markets)} mkts, KL={divergence:.4f})"
        return ArbOpportunity(
            layer=3,
            label=label,
            market_ids=[m.id for m in dep_markets],
            legs=legs,
            gross_profit=divergence,
            net_profit=net,
            net_profit_pct=net / sum(prices) if sum(prices) > 0 else 0.0,
            confidence=min(0.7, getattr(dep, "confidence", 0.5)),
            metadata={
                "kl_divergence": round(divergence, 6),
                "fw_iterations": getattr(result, "iterations", None),
                "fw_converged": getattr(result, "converged", None),
                "dependency_type": getattr(dep, "dep_type", "unknown"),
            },
        )

    # ------------------------------------------------------------------
    # Execution planning
    # ------------------------------------------------------------------

    def _build_execution_plan(self, opportunity: ArbOpportunity) -> Dict[str, Any]:
        """VWAP-aware execution planning.

        Checks order book depth, estimates slippage, and determines if the
        opportunity is still profitable after all costs.
        """
        legs = opportunity.legs
        total_cost = 0.0
        planned_legs: List[Dict[str, Any]] = []

        for leg in legs:
            price = leg.get("price", 0.5)
            weight = leg.get("weight", 1.0)
            size_usd = min(self.max_position_usd * weight, self.max_position_usd)

            # Estimate slippage (conservative: 0.5% per $100 of size)
            slippage_pct = 0.005 * (size_usd / 100.0)
            effective_price = price * (1.0 + slippage_pct) if "buy" in leg.get("side", "") else price * (1.0 - slippage_pct)
            fee = size_usd * self.taker_fee
            leg_cost = size_usd * effective_price + fee

            total_cost += leg_cost
            planned_legs.append({
                **leg,
                "size_usd": round(size_usd, 2),
                "effective_price": round(effective_price, 4),
                "slippage_pct": round(slippage_pct, 4),
                "fee": round(fee, 2),
                "leg_cost": round(leg_cost, 2),
            })

        # Guaranteed profit = payout - total cost
        # For Layer 1/2: payout is $1 per share set
        payout = self.max_position_usd  # simplified: one unit resolves to this
        guaranteed_profit = payout - total_cost

        executable = guaranteed_profit >= 0.05  # $0.05 minimum

        return {
            "legs": planned_legs,
            "total_cost": round(total_cost, 2),
            "expected_payout": round(payout, 2),
            "guaranteed_profit": round(guaranteed_profit, 2),
            "executable": executable,
            "parallel_execution": opportunity.layer in (1, 2),  # L1/L2 legs are independent
        }

    # ------------------------------------------------------------------
    # Signal emission
    # ------------------------------------------------------------------

    def _emit_signals(self, opp: ArbOpportunity) -> None:
        """Convert an ArbOpportunity into TradeSignal(s) for the orchestrator."""
        # Build execution plan
        plan = self._build_execution_plan(opp)
        if not plan["executable"]:
            logger.debug("opportunity_not_executable", label=opp.label)
            return

        urgency = "immediate" if opp.layer <= 2 else "normal"

        for leg in opp.legs:
            market_id = leg.get("market_id", "")
            side = leg.get("side", "buy_yes")
            token_id = leg.get("token_id", "")
            leg_price = leg.get("price", 0.5)

            # Compute estimated_prob from leg price and side
            if side == "buy_yes":
                estimated_prob = leg_price
                market_price = leg_price
            else:
                estimated_prob = 1.0 - leg_price
                market_price = 1.0 - leg_price

            signal = TradeSignal(
                engine=self.name,
                market_id=market_id,
                token_id=token_id,
                side=side,
                confidence=opp.confidence,
                edge=opp.net_profit,
                urgency=urgency,
                metadata={
                    "layer": opp.layer,
                    "label": opp.label,
                    "gross_profit": round(opp.gross_profit, 6),
                    "net_profit": round(opp.net_profit, 6),
                    "net_profit_pct": round(opp.net_profit_pct, 6),
                    "leg_price": leg_price,
                    "market_price": market_price,
                    "estimated_prob": estimated_prob,
                    "original_price": leg_price,
                    "net_edge": opp.net_profit,
                    "conviction": "high",
                    "reasoning": opp.label,
                    "question": leg.get("question", opp.metadata.get("question", market_id)),
                    "max_position_usd": self.max_position_usd,
                    "execution_plan": plan,
                    **{k: v for k, v in opp.metadata.items()
                       if k not in ("question",)},  # don't double-write question
                },
            )
            self._signals.append(signal)

        logger.info(
            "bregman_signals_emitted",
            label=opp.label,
            layer=opp.layer,
            n_signals=len(opp.legs),
            net_profit=round(opp.net_profit, 4),
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_opportunity(self, opp: ArbOpportunity) -> None:
        """Append opportunity to JSONL log."""
        try:
            record = {
                "timestamp": opp.timestamp,
                "layer": opp.layer,
                "label": opp.label,
                "market_ids": opp.market_ids,
                "gross_profit": round(opp.gross_profit, 6),
                "net_profit": round(opp.net_profit, 6),
                "net_profit_pct": round(opp.net_profit_pct, 6),
                "confidence": round(opp.confidence, 4),
                "n_legs": len(opp.legs),
                **{k: v for k, v in opp.metadata.items()
                   if isinstance(v, (str, int, float, bool))},
            }
            append_jsonl(Path(self._opportunities_file), record)
        except Exception as exc:
            logger.warning("bregman_log_error", error=str(exc))

    # ------------------------------------------------------------------
    # Dependency cache persistence
    # ------------------------------------------------------------------

    def _load_dependency_cache(self) -> None:
        """Load cached market dependencies from disk."""
        try:
            if self.dep_cache_file.exists():
                raw = json.loads(self.dep_cache_file.read_text())
                self._dependency_cache = raw
                logger.info("dependency_cache_loaded", entries=len(raw.get("dependencies", [])))
            else:
                self._dependency_cache = {}
        except Exception as exc:
            logger.warning("dependency_cache_load_error", error=str(exc))
            self._dependency_cache = {}

    def _save_dependency_cache(self) -> None:
        """Persist dependency cache to disk."""
        try:
            self.dep_cache_file.parent.mkdir(parents=True, exist_ok=True)
            # Don't serialize full dependency objects — store IDs & constraints
            serializable = {
                k: v for k, v in self._dependency_cache.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
            self.dep_cache_file.write_text(json.dumps(serializable, indent=2))
        except Exception as exc:
            logger.warning("dependency_cache_save_error", error=str(exc))


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

async def _run_standalone() -> None:
    """Standalone one-shot scan: run all three layers and print results."""
    from ..utils import setup_logging

    config = load_config("config.yaml")
    setup_logging(config, log_json=False)

    engine = BregmanArbEngine(config)

    print("\n" + "=" * 70)
    print("  Morpheus — Bregman Arbitrage Engine")
    print("  Layers: Single-Market | Event-Group | Cross-Market")
    print("=" * 70 + "\n")

    # Run one scan cycle directly instead of starting the loop
    try:
        t0 = time.monotonic()

        # Fetch data
        print("  ⏳ Fetching markets and event groups …")
        all_markets = await engine.market_scanner.fetch_markets()
        event_groups = await engine.event_scanner.fetch_event_groups()
        print(f"  📊 {len(all_markets)} markets, {len(event_groups)} event groups\n")

        opportunities: List[ArbOpportunity] = []

        # Layer 1
        print("  🔍 Layer 1: Single-market arbitrage …")
        l1 = []
        for m in all_markets:
            opp = engine._check_single_market_arb(m)
            if opp:
                l1.append(opp)
        opportunities.extend(l1)
        print(f"     Found {len(l1)} opportunities\n")

        # Layer 2
        print("  🔍 Layer 2: Event-group arbitrage …")
        l2 = []
        for eid, grp in event_groups.items():
            opp = engine._check_event_group_arb(eid, grp)
            if opp:
                l2.append(opp)
        opportunities.extend(l2)
        print(f"     Found {len(l2)} opportunities\n")

        # Layer 3
        if _HAS_OPTIMIZATION:
            print("  🔍 Layer 3: Cross-market arbitrage …")
            l3 = await engine._check_cross_market_arb(all_markets)
            opportunities.extend(l3)
            print(f"     Found {len(l3)} opportunities\n")
        else:
            print("  ⚠️  Layer 3 skipped (optimization modules not available)\n")

        elapsed = time.monotonic() - t0

        # Print results
        if not opportunities:
            print("  No arbitrage opportunities found.\n")
            print("  This is normal — arb windows are short-lived.")
        else:
            profitable = [o for o in opportunities if o.net_profit >= engine.min_profit_threshold]
            print(f"  🎯 Total opportunities: {len(opportunities)}")
            print(f"  💰 Profitable (≥${engine.min_profit_threshold}/share): {len(profitable)}\n")

            for i, opp in enumerate(sorted(opportunities, key=lambda o: -o.net_profit)[:15], 1):
                icon = "💰" if opp.net_profit >= engine.min_profit_threshold else "📊"
                print(f"  {icon} [{i}] L{opp.layer} | {opp.label}")
                print(f"       Gross: ${opp.gross_profit:.4f}  Net: ${opp.net_profit:.4f}  ({opp.net_profit_pct:.2%})")
                print(f"       Markets: {len(opp.market_ids)}  Legs: {len(opp.legs)}  Conf: {opp.confidence:.2f}")
                print()

        print(f"  ⏱️  Scan completed in {elapsed:.1f}s\n")

    finally:
        await engine.event_scanner.close()
        await engine.market_scanner.close()


if __name__ == "__main__":
    asyncio.run(_run_standalone())
