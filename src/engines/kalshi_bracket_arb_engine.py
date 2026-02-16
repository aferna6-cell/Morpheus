"""Kalshi bracket arbitrage scanner engine.

Exploits mispricings in bracket markets where the sum of all bracket
YES asks within an event is less than $1.00. Buying the full set
guarantees a profit at settlement regardless of outcome.

Example:
  Event "NYC High Temp Feb 14" has 8 brackets summing to $0.93.
  Buy 1 contract of each for $0.93 → guaranteed $1.00 payout = 7.5% risk-free.

Risk: zero directional risk. Only execution risk (partial fills).
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from ..engines.base import BaseEngine
from ..engines.signals import TradeSignal
from ..kalshi_client import KalshiClient, KalshiMarket
from ..market_filters import MarketFilters
from ..utils import BotConfig


# Bracket ticker pattern: -B followed by a digit (e.g., KXHIGHAUS-26FEB16-B74.5)
_BRACKET_RE = re.compile(r'-B\d')


def _is_bracket_ticker(ticker: str) -> bool:
    """Check if a ticker is a bracket market (-B{number} pattern)."""
    return bool(_BRACKET_RE.search(ticker.upper()))


def _extract_event_key(ticker: str, event_ticker: str) -> str:
    """Extract the event grouping key (same event = same bracket set).

    Brackets within the same event share the event_ticker.
    Fallback: strip the bracket range suffix from the ticker.
    """
    if event_ticker:
        return event_ticker

    # Fallback: strip last segment (the bracket range) from the ticker
    # e.g., KXHIGHTEMPB-NYC-26FEB14-B68-70 → KXHIGHTEMPB-NYC-26FEB14
    parts = ticker.split("-")
    if len(parts) >= 3:
        # Find the bracket specifier and remove it + range
        for i, p in enumerate(parts):
            if p.startswith("B") and i > 0:
                return "-".join(parts[:i])
    return ticker


@dataclass
class BracketSet:
    """A complete set of brackets within one event."""

    event_key: str
    brackets: List[KalshiMarket] = field(default_factory=list)
    sum_yes_asks: float = 0.0
    close_time: Optional[datetime] = None

    @property
    def margin(self) -> float:
        """Profit margin: 1.0 - sum_yes_asks."""
        return 1.0 - self.sum_yes_asks

    @property
    def margin_pct(self) -> float:
        """Margin as percentage."""
        return self.margin * 100.0

    @property
    def is_complete(self) -> bool:
        """True if all brackets have valid asks (no gaps)."""
        return all(m.yes_ask > 0 for m in self.brackets)

    @property
    def n_brackets(self) -> int:
        return len(self.brackets)


class KalshiBracketArbEngine(BaseEngine):
    """Bracket arbitrage scanner — risk-free profit from mispriced bracket sets."""

    name = "kalshi_bracket_arb"

    def __init__(
        self,
        config: BotConfig,
        kalshi_client: KalshiClient,
    ):
        self.config = config
        self.kalshi_client = kalshi_client
        self.logger = structlog.get_logger()
        self._filters = MarketFilters(config)

        arb_cfg = getattr(config, "bracket_arb", None) or {}
        if not isinstance(arb_cfg, dict):
            arb_cfg = {}

        self._scan_interval = float(arb_cfg.get("scan_interval_seconds", 300))
        self._min_margin_pct = float(arb_cfg.get("min_margin_pct", 3.0))
        self._max_active_sets = int(arb_cfg.get("max_active_sets", 2))
        self._max_days_to_close = int(arb_cfg.get("max_days_to_close", 3))
        self._min_brackets = int(arb_cfg.get("min_brackets", 3))

        self._pending_signals: List[TradeSignal] = []
        self._active_sets: Dict[str, BracketSet] = {}  # event_key → active arb set
        self._scan_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())
        self.logger.info(
            "bracket_arb_engine_started",
            scan_interval=self._scan_interval,
            min_margin_pct=self._min_margin_pct,
            max_active_sets=self._max_active_sets,
        )

    async def stop(self) -> None:
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        self.logger.info("bracket_arb_engine_stopped")

    async def get_signals(self) -> List[TradeSignal]:
        signals = list(self._pending_signals)
        self._pending_signals.clear()
        return signals

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        """Periodically scan for bracket arbitrage opportunities."""
        while self._running:
            try:
                await self._scan_for_arb()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error("bracket_arb_scan_error", error=str(e))
            await asyncio.sleep(self._scan_interval)

    async def _scan_for_arb(self) -> None:
        """Fetch bracket markets, group by event, check for arb."""
        # Fetch all markets closing within our window
        try:
            all_markets = await self.kalshi_client.fetch_markets_by_close_date(
                max_days=self._max_days_to_close,
                min_volume=0,  # brackets can be thin — we need all of them
            )
        except Exception as e:
            self.logger.error("bracket_arb_fetch_error", error=str(e))
            return

        # Filter to bracket markets only
        bracket_markets = [m for m in all_markets if _is_bracket_ticker(m.ticker)]

        if not bracket_markets:
            self.logger.debug("bracket_arb_no_brackets_found")
            return

        # Group by event
        event_groups: Dict[str, List[KalshiMarket]] = defaultdict(list)
        for m in bracket_markets:
            key = _extract_event_key(m.ticker, m.event_ticker)
            event_groups[key].append(m)

        # Analyze each event group
        opportunities: List[BracketSet] = []

        for event_key, markets in event_groups.items():
            if len(markets) < self._min_brackets:
                continue  # Need at least N brackets to form a complete set

            bracket_set = BracketSet(
                event_key=event_key,
                brackets=sorted(markets, key=lambda m: m.yes_ask),
                close_time=markets[0].close_time,
            )

            # Check completeness: all brackets must have valid asks
            if not bracket_set.is_complete:
                self.logger.debug(
                    "bracket_arb_incomplete_set",
                    event_key=event_key,
                    n_brackets=len(markets),
                    missing_asks=[m.ticker for m in markets if m.yes_ask <= 0],
                )
                continue

            # Sum all YES ask prices
            bracket_set.sum_yes_asks = sum(m.yes_ask for m in markets)

            # Check if profitable
            if bracket_set.margin_pct >= self._min_margin_pct:
                opportunities.append(bracket_set)
                self.logger.info(
                    "bracket_arb_opportunity",
                    event_key=event_key,
                    n_brackets=bracket_set.n_brackets,
                    sum_asks=round(bracket_set.sum_yes_asks, 4),
                    margin_pct=round(bracket_set.margin_pct, 2),
                    tickers=[m.ticker for m in markets],
                    asks=[round(m.yes_ask, 3) for m in markets],
                )
            elif bracket_set.margin_pct > 0:
                # Near-miss: sum is <$1 but margin below threshold
                # Validates that mispricings exist (research claim)
                self.logger.info(
                    "bracket_arb_near_miss",
                    event_key=event_key,
                    n_brackets=bracket_set.n_brackets,
                    sum_asks=round(bracket_set.sum_yes_asks, 4),
                    margin_pct=round(bracket_set.margin_pct, 2),
                    threshold_pct=self._min_margin_pct,
                )

        # Sort by margin (best first)
        opportunities.sort(key=lambda s: s.margin, reverse=True)

        # Log summary
        self.logger.info(
            "bracket_arb_scan_complete",
            total_brackets=len(bracket_markets),
            event_groups=len(event_groups),
            opportunities=len(opportunities),
            best_margin=round(opportunities[0].margin_pct, 2) if opportunities else 0,
        )

        # Emit signals for top opportunities (respecting max_active_sets)
        active_count = len(self._active_sets)
        for opp in opportunities:
            if active_count >= self._max_active_sets:
                break
            if opp.event_key in self._active_sets:
                continue  # Already trading this set

            self._emit_arb_signals(opp)
            self._active_sets[opp.event_key] = opp
            active_count += 1

    def _emit_arb_signals(self, bracket_set: BracketSet) -> None:
        """Emit one TradeSignal per bracket in the arb set."""
        for market in bracket_set.brackets:
            # Each leg is a buy_yes at the ask price
            entry_cost = market.yes_ask
            signal = TradeSignal(
                engine=self.name,
                market_id=market.ticker,
                token_id=market.ticker,
                side="buy_yes",
                confidence=0.95,  # Structural arb — near-certain
                edge=bracket_set.margin,  # The arb margin is the edge
                urgency="normal",
                metadata={
                    "estimated_prob": entry_cost,  # Not really a prob — it's the cost
                    "market_price": market.yes_price,
                    "net_edge": bracket_set.margin,
                    "conviction": "high",
                    "reasoning": (
                        f"Bracket arb: {bracket_set.n_brackets} brackets sum to "
                        f"${bracket_set.sum_yes_asks:.3f} (<$1.00). "
                        f"Margin: {bracket_set.margin_pct:.1f}%"
                    ),
                    "question": market.title,
                    "kalshi_ticker": market.ticker,
                    "kalshi_yes_ask": market.yes_ask,
                    "kalshi_no_ask": market.no_ask,
                    "platform": "kalshi",
                    "strategy": "bracket_arb",
                    "signal_source": "structural",
                    "arb_event_key": bracket_set.event_key,
                    "arb_n_brackets": bracket_set.n_brackets,
                    "arb_sum_asks": bracket_set.sum_yes_asks,
                    "arb_margin_pct": bracket_set.margin_pct,
                    # Force 1 contract per bracket leg
                    "_force_size_usd": entry_cost,
                },
            )
            self._pending_signals.append(signal)

        self.logger.info(
            "bracket_arb_signals_emitted",
            event_key=bracket_set.event_key,
            n_signals=bracket_set.n_brackets,
            total_cost=round(bracket_set.sum_yes_asks, 3),
            margin_pct=round(bracket_set.margin_pct, 2),
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_active_set(self, event_key: str) -> None:
        """Remove an active arb set (e.g., after settlement)."""
        self._active_sets.pop(event_key, None)

    def trigger_rescan(self) -> None:
        """Force a rescan on next cycle (called after resolutions free capital)."""
        self._active_sets.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return arb engine statistics."""
        return {
            "active_sets": len(self._active_sets),
            "sets": {
                key: {
                    "n_brackets": s.n_brackets,
                    "sum_asks": round(s.sum_yes_asks, 3),
                    "margin_pct": round(s.margin_pct, 2),
                }
                for key, s in self._active_sets.items()
            },
        }
