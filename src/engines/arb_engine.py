"""Cross-platform arbitrage engine: Polymarket ↔ Kalshi.

Scans for pricing discrepancies on equivalent markets across platforms.
If the combined cost of YES on one platform + NO on the other < $1.00,
that's a risk-free arbitrage opportunity.

Runnable standalone:
    python3 -m src.engines.arb_engine
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from ..kalshi_client import KalshiClient, KalshiMarket
from ..markets import Market as PolyMarket
from ..markets_scanner import MarketScanner
from ..utils import BotConfig, append_jsonl, load_config
from .arb_matcher import ArbMatcher, MarketMatch
from .base import BaseEngine
from .signals import TradeSignal

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Opportunity dataclass
# ---------------------------------------------------------------------------

@dataclass
class ArbOpportunity:
    """A detected cross-platform arbitrage opportunity."""

    poly_market_id: str
    poly_question: str
    kalshi_ticker: str
    kalshi_title: str
    match_confidence: float

    # Leg 1: Buy YES Poly + Buy NO Kalshi
    poly_yes_price: float
    kalshi_no_price: float
    leg1_cost: float  # poly_yes + kalshi_no
    leg1_gross_profit: float  # 1.0 - leg1_cost
    leg1_net_profit: float  # after fees

    # Leg 2: Buy NO Poly + Buy YES Kalshi
    poly_no_price: float
    kalshi_yes_price: float
    leg2_cost: float
    leg2_gross_profit: float
    leg2_net_profit: float

    # Best leg
    best_leg: int  # 1 or 2
    best_net_profit: float
    best_net_profit_pct: float

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ArbEngine(BaseEngine):
    """Cross-platform arbitrage scanner."""

    name: str = "arb_cross_platform"

    def __init__(self, config: BotConfig):
        self.config = config

        # Arbitrage config
        arb_cfg = getattr(config, "arbitrage", None) or {}
        if not isinstance(arb_cfg, dict):
            arb_cfg = {}

        self.scan_interval = float(arb_cfg.get("scan_interval_seconds", 120))
        self.min_profit_pct = float(arb_cfg.get("min_profit_pct", 0.02))
        self.poly_taker_fee = float(arb_cfg.get("polymarket_taker_fee", 0.02))
        self.kalshi_fee = float(arb_cfg.get("kalshi_fee", 0.07))
        self.min_volume = int(arb_cfg.get("min_volume", 5000))
        self.max_position_usd = float(arb_cfg.get("max_position_usd", 50))
        self.match_threshold = float(arb_cfg.get("match_confidence_threshold", 0.80))
        self.opportunities_file = arb_cfg.get(
            "opportunities_file", "state/arb_opportunities.jsonl"
        )

        # Sub-components
        self.kalshi = KalshiClient(config)
        self.scanner = MarketScanner(config)
        self.matcher = ArbMatcher(confidence_threshold=self.match_threshold)

        # Pending signals
        self._signals: List[TradeSignal] = []

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self.kalshi.start()
        logger.info("arb_engine_started")

    async def stop(self) -> None:
        await self.kalshi.stop()
        await self.scanner.close()
        logger.info("arb_engine_stopped")

    async def get_signals(self) -> List[TradeSignal]:
        """Return and clear any pending trade signals."""
        signals = list(self._signals)
        self._signals.clear()
        return signals

    # ------------------------------------------------------------------
    # Core scanning logic
    # ------------------------------------------------------------------

    def _calculate_opportunity(self, match: MarketMatch) -> Optional[ArbOpportunity]:
        """Calculate arb opportunity for a matched market pair."""
        pm = match.poly_market
        km = match.kalshi_market

        poly_yes = pm.yes_price or 0.0
        poly_no = pm.no_price or 0.0
        kalshi_yes = km.yes_price
        kalshi_no = km.no_price

        # Skip markets with no real pricing
        if poly_yes <= 0 or poly_no <= 0:
            return None
        if kalshi_yes <= 0 and kalshi_no <= 0:
            return None

        # Leg 1: Buy YES on Poly + Buy NO on Kalshi
        leg1_cost = poly_yes + kalshi_no
        leg1_gross = 1.0 - leg1_cost

        # Leg 2: Buy NO on Poly + Buy YES on Kalshi
        leg2_cost = poly_no + kalshi_yes
        leg2_gross = 1.0 - leg2_cost

        # Fee calculation:
        # Polymarket taker fee applies to the payout side (worst case)
        # Kalshi fee applies to profit
        def _net_profit(gross: float, cost: float) -> float:
            if gross <= 0:
                return gross
            # Polymarket fee on the Polymarket leg payout (taker)
            poly_fee = self.poly_taker_fee * (1.0 - cost + gross)
            # Kalshi fee on Kalshi leg profit portion
            kalshi_fee = self.kalshi_fee * gross
            return gross - poly_fee - kalshi_fee

        leg1_net = _net_profit(leg1_gross, leg1_cost)
        leg2_net = _net_profit(leg2_gross, leg2_cost)

        # Pick best leg
        if leg1_net >= leg2_net:
            best_leg, best_net = 1, leg1_net
        else:
            best_leg, best_net = 2, leg2_net

        best_cost = leg1_cost if best_leg == 1 else leg2_cost
        best_pct = best_net / best_cost if best_cost > 0 else 0.0

        return ArbOpportunity(
            poly_market_id=pm.id,
            poly_question=pm.question,
            kalshi_ticker=km.ticker,
            kalshi_title=km.title,
            match_confidence=match.confidence,
            poly_yes_price=poly_yes,
            kalshi_no_price=kalshi_no,
            leg1_cost=leg1_cost,
            leg1_gross_profit=leg1_gross,
            leg1_net_profit=leg1_net,
            poly_no_price=poly_no,
            kalshi_yes_price=kalshi_yes,
            leg2_cost=leg2_cost,
            leg2_gross_profit=leg2_gross,
            leg2_net_profit=leg2_net,
            best_leg=best_leg,
            best_net_profit=best_net,
            best_net_profit_pct=best_pct,
        )

    async def scan(self) -> List[ArbOpportunity]:
        """Run a full arbitrage scan.

        1. Fetch Kalshi event categories (for filtering/display)
        2. Fetch Kalshi markets
        3. Fetch Polymarket markets
        4. Match equivalent markets
        5. Calculate arb opportunities
        6. Generate trade signals for profitable ones
        """
        # Fetch Kalshi categories
        try:
            await self.kalshi.fetch_event_categories()
        except Exception as exc:
            logger.warning("kalshi_categories_error", error=str(exc))

        # Fetch markets from both platforms concurrently
        # Use fetch_markets_by_events for Kalshi — the bulk /markets endpoint
        # returns mostly sports parlays, not the political/economics markets
        # that overlap with Polymarket.
        kalshi_markets, poly_markets = await asyncio.gather(
            self.kalshi.fetch_markets_by_events(
                categories=["Politics", "Economics", "Elections", "World",
                             "Financials", "Climate and Weather", "Tech"],
                min_volume=self.min_volume,
                max_markets=500,
            ),
            self.scanner.fetch_markets(),
            return_exceptions=True,
        )

        if isinstance(kalshi_markets, Exception):
            logger.error("kalshi_fetch_failed", error=str(kalshi_markets))
            return []
        if isinstance(poly_markets, Exception):
            logger.error("polymarket_fetch_failed", error=str(poly_markets))
            return []

        logger.info(
            "arb_scan_fetched",
            kalshi_count=len(kalshi_markets),
            poly_count=len(poly_markets),
        )

        # Match equivalent markets
        matches = self.matcher.match_markets(poly_markets, kalshi_markets)

        # Calculate opportunities
        opportunities: List[ArbOpportunity] = []
        for match in matches:
            opp = self._calculate_opportunity(match)
            if opp is None:
                continue

            # Log every opportunity (even unprofitable ones)
            self._log_opportunity(opp)

            if opp.best_net_profit_pct >= self.min_profit_pct:
                opportunities.append(opp)
                self._emit_signal(opp, match)

                logger.info(
                    "arb_opportunity_found",
                    poly_question=opp.poly_question[:80],
                    kalshi_title=opp.kalshi_title[:80],
                    best_leg=opp.best_leg,
                    net_profit_pct=f"{opp.best_net_profit_pct:.4f}",
                    match_confidence=f"{opp.match_confidence:.2f}",
                )

        logger.info(
            "arb_scan_complete",
            matches=len(matches),
            profitable_opportunities=len(opportunities),
        )

        return opportunities

    def _emit_signal(self, opp: ArbOpportunity, match: MarketMatch) -> None:
        """Convert an arb opportunity into a TradeSignal."""
        pm = match.poly_market

        if opp.best_leg == 1:
            # Buy YES on Poly
            side = "buy_yes"
            edge = opp.leg1_net_profit
        else:
            # Buy NO on Poly
            side = "buy_no"
            edge = opp.leg2_net_profit

        # Pick the token_id for the Polymarket side
        token_id = ""
        wanted = "yes" if side == "buy_yes" else "no"
        for outcome, tok in pm.tokens.items():
            if outcome.strip().lower() == wanted:
                token_id = tok.token_id
                break

        urgency = "immediate" if opp.best_net_profit_pct >= self.min_profit_pct else "low"

        signal = TradeSignal(
            engine=self.name,
            market_id=pm.id,
            token_id=token_id,
            side=side,
            confidence=opp.match_confidence,
            edge=edge,
            urgency=urgency,
            metadata={
                "question": pm.question,
                "kalshi_ticker": opp.kalshi_ticker,
                "kalshi_title": opp.kalshi_title,
                "best_leg": opp.best_leg,
                "net_profit_pct": round(opp.best_net_profit_pct, 6),
                "leg1_cost": round(opp.leg1_cost, 4),
                "leg2_cost": round(opp.leg2_cost, 4),
                "poly_yes": round(opp.poly_yes_price, 4),
                "poly_no": round(opp.poly_no_price, 4),
                "kalshi_yes": round(opp.kalshi_yes_price, 4),
                "kalshi_no": round(opp.kalshi_no_price, 4),
                "max_position_usd": self.max_position_usd,
            },
        )
        self._signals.append(signal)

    def _log_opportunity(self, opp: ArbOpportunity) -> None:
        """Append opportunity to the JSONL log file."""
        try:
            record = {
                "timestamp": opp.timestamp,
                "poly_market_id": opp.poly_market_id,
                "poly_question": opp.poly_question[:120],
                "kalshi_ticker": opp.kalshi_ticker,
                "kalshi_title": opp.kalshi_title[:120],
                "match_confidence": round(opp.match_confidence, 4),
                "poly_yes": round(opp.poly_yes_price, 4),
                "poly_no": round(opp.poly_no_price, 4),
                "kalshi_yes": round(opp.kalshi_yes_price, 4),
                "kalshi_no": round(opp.kalshi_no_price, 4),
                "leg1_cost": round(opp.leg1_cost, 4),
                "leg1_net": round(opp.leg1_net_profit, 6),
                "leg2_cost": round(opp.leg2_cost, 4),
                "leg2_net": round(opp.leg2_net_profit, 6),
                "best_leg": opp.best_leg,
                "best_net_pct": round(opp.best_net_profit_pct, 6),
            }
            append_jsonl(Path(self.opportunities_file), record)
        except Exception as exc:
            logger.warning("arb_log_error", error=str(exc))


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

async def _run_standalone() -> None:
    """Standalone scan: fetch, match, print opportunities."""
    import logging
    import sys

    from ..utils import setup_logging

    config = load_config("config.yaml")
    setup_logging(config, log_json=False)

    engine = ArbEngine(config)
    await engine.start()

    try:
        print("\n" + "=" * 70)
        print("  Morpheus — Cross-Platform Arbitrage Scanner")
        print("  Polymarket ↔ Kalshi")
        print("=" * 70 + "\n")

        opportunities = await engine.scan()

        signals = await engine.get_signals()

        if not opportunities:
            print("\n  No profitable arbitrage opportunities found.\n")
            print("  This is normal — true arb is rare and fleeting.")
            print("  The scanner found and logged all market matches.\n")
        else:
            print(f"\n  🎯 Found {len(opportunities)} arbitrage opportunities:\n")
            for i, opp in enumerate(opportunities, 1):
                leg = opp.best_leg
                if leg == 1:
                    desc = f"Buy YES Poly (${opp.poly_yes_price:.2f}) + Buy NO Kalshi (${opp.kalshi_no_price:.2f})"
                else:
                    desc = f"Buy NO Poly (${opp.poly_no_price:.2f}) + Buy YES Kalshi (${opp.kalshi_yes_price:.2f})"

                print(f"  [{i}] {opp.poly_question[:70]}")
                print(f"      Kalshi: {opp.kalshi_title[:70]}")
                print(f"      Match confidence: {opp.match_confidence:.2f}")
                print(f"      Strategy: {desc}")
                print(f"      Cost: ${opp.leg1_cost if leg == 1 else opp.leg2_cost:.4f}")
                print(f"      Net profit: {opp.best_net_profit_pct:.2%}")
                print()

        if signals:
            print(f"  📡 Generated {len(signals)} trade signals\n")

    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(_run_standalone())
