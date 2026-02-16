"""Kalshi longshot seller engine — exploit favorite-longshot bias.

Academic research and empirical data consistently show that binary contracts
priced below ~10c resolve YES far less often than their price implies:
  - YES priced at 5c resolves YES ~1% (not 5%)
  - YES priced at 10c resolves YES ~4% (not 10%)

This engine buys NO on extreme longshots, capturing the systematic
overpricing of unlikely outcomes.  No LLM calls — pure statistical bias.

Risk: occasional upsets.  Managed via conservative sizing and diversification
across many independent markets.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from ..engines.base import BaseEngine
from ..engines.signals import TradeSignal
from ..kalshi_client import KalshiClient, KalshiMarket
from ..market_filters import MarketFilters
from ..utils import BotConfig


class KalshiLongshotSeller(BaseEngine):
    """Sell longshots by buying NO on extremely low-priced YES contracts."""

    name = "kalshi_longshot"

    def __init__(
        self,
        config: BotConfig,
        kalshi_client: KalshiClient,
    ):
        self.config = config
        self.kalshi_client = kalshi_client
        self.logger = structlog.get_logger()
        self._filters = MarketFilters(config)

        ls_cfg = getattr(config, "longshot_seller", None) or {}
        if not isinstance(ls_cfg, dict):
            ls_cfg = {}

        self._interval = float(ls_cfg.get("scan_interval_seconds", 600))
        self._max_yes_cents = int(ls_cfg.get("max_yes_cents", 12))  # YES must be <= 12c
        self._min_yes_cents = int(ls_cfg.get("min_yes_cents", 2))   # skip dust (1c = settled)
        self._min_volume = int(ls_cfg.get("min_volume", 1000))
        self._min_hours_to_close = float(ls_cfg.get("min_hours_to_close", 4.0))
        self._max_days_to_close = int(ls_cfg.get("max_days_to_close", 7))
        self._max_positions = int(ls_cfg.get("max_positions", 5))
        self._max_per_trade_usd = float(ls_cfg.get("max_per_trade_usd", 2.0))
        self._min_profit_pct = float(ls_cfg.get("min_profit_pct", 0.03))  # 3% minimum

        self._pending: List[TradeSignal] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Cooldown: don't re-evaluate same market within 2 hours
        self._recently_evaluated: Dict[str, float] = {}
        self._eval_cooldown = 7200.0

        # Track active position count (set externally)
        self._active_positions = 0

        # Stats
        self._scanned = 0
        self._signals_emitted = 0

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._scan_loop())
        self.logger.info(
            "kalshi_longshot_seller_started",
            max_yes_cents=self._max_yes_cents,
            max_positions=self._max_positions,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info(
            "kalshi_longshot_seller_stopped",
            scanned=self._scanned,
            signals_emitted=self._signals_emitted,
        )

    async def get_signals(self) -> List[TradeSignal]:
        out = list(self._pending)
        self._pending.clear()
        return out

    async def _scan_loop(self) -> None:
        await asyncio.sleep(60)  # Let other engines start first
        while self._running:
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.error("longshot_scan_error", error=str(exc))
            await asyncio.sleep(self._interval)

    async def _scan_once(self) -> None:
        if self._active_positions >= self._max_positions:
            return

        markets = await self.kalshi_client.fetch_markets_by_close_date(
            max_days=self._max_days_to_close,
            min_volume=500,
        )

        now_ts = time.monotonic()
        candidates: List[KalshiMarket] = []

        for km in markets:
            self._scanned += 1

            # Cooldown check
            prev = self._recently_evaluated.get(km.ticker)
            if prev and (now_ts - prev) < self._eval_cooldown:
                continue

            # Price check: YES must be in the longshot range
            yes_cents = int(km.yes_price * 100)
            if yes_cents < self._min_yes_cents or yes_cents > self._max_yes_cents:
                continue

            # Volume check
            if km.volume_24h < self._min_volume:
                continue

            # Time check: not too close, not too far
            if not km.close_time:
                continue
            hours_left = (km.close_time - datetime.now(timezone.utc)).total_seconds() / 3600
            if hours_left < self._min_hours_to_close or hours_left < 0:
                continue

            # Profit check: NO ask must leave enough profit
            no_ask = km.no_ask if km.no_ask > 0 else km.no_price
            if no_ask <= 0 or no_ask >= 1.0:
                continue
            profit_pct = (1.0 - no_ask) / no_ask
            if profit_pct < self._min_profit_pct:
                continue

            # Ticker prefix filter (junk markets)
            prefix_result = self._filters.check_ticker_prefix(km.ticker)
            if not prefix_result.passed:
                continue

            # Sports filter
            sports_result = self._filters.check_sports(
                market_id=km.ticker,
                title=km.title,
                category=km.category,
                is_live=False,
            )
            if not sports_result.passed:
                continue

            candidates.append(km)

        # Score: prefer lower YES price (higher bias) and higher volume (better fills)
        def _score(km: KalshiMarket) -> float:
            yes_c = int(km.yes_price * 100)
            # Lower YES = higher bias edge.  Scale: 2c → 1.0, 12c → 0.17
            bias_score = 1.0 / max(yes_c, 1)
            vol_score = min(km.volume_24h / 10000.0, 1.0)
            return bias_score * 0.7 + vol_score * 0.3

        candidates.sort(key=_score, reverse=True)

        max_new = self._max_positions - self._active_positions
        emitted = 0

        for km in candidates:
            if emitted >= max_new:
                break

            self._recently_evaluated[km.ticker] = now_ts

            no_ask = km.no_ask if km.no_ask > 0 else km.no_price
            yes_cents = int(km.yes_price * 100)
            no_ask_cents = int(no_ask * 100)
            profit_pct = (1.0 - no_ask) / no_ask

            # Edge estimate based on favorite-longshot bias research:
            # Actual YES prob ≈ (implied_prob)^1.5 for extreme longshots
            implied_prob = km.yes_price
            estimated_actual_prob = implied_prob ** 1.5
            edge = (1.0 - estimated_actual_prob) - no_ask  # P(NO wins) - cost

            # Confidence: higher for lower YES price (stronger bias)
            if yes_cents <= 5:
                confidence = 0.85
            elif yes_cents <= 8:
                confidence = 0.75
            else:
                confidence = 0.65

            signal = TradeSignal(
                engine=self.name,
                market_id=km.ticker,
                token_id=f"{km.ticker}:no",
                side="buy_no",
                confidence=confidence,
                edge=edge,
                urgency="normal",
                metadata={
                    "platform": "kalshi",
                    "strategy": "longshot_sell",
                    "signal_source": "longshot_bias",
                    "estimated_prob": 1.0 - estimated_actual_prob,
                    "market_price": km.yes_price,
                    "net_edge": edge,
                    "conviction": "medium",
                    "reasoning": (
                        f"Longshot bias: YES at {yes_cents}c implies {implied_prob*100:.1f}% "
                        f"but actual prob ~{estimated_actual_prob*100:.1f}%. "
                        f"Buying NO at {no_ask_cents}c for {profit_pct*100:.1f}% return."
                    ),
                    "question": km.title,
                    "kalshi_ticker": km.ticker,
                    "kalshi_yes_ask": km.yes_ask,
                    "kalshi_no_ask": km.no_ask,
                    "kalshi_yes_bid": km.yes_bid,
                    "kalshi_no_bid": km.no_bid,
                    "_force_size_usd": min(self._max_per_trade_usd, no_ask * 2),
                },
            )
            self._pending.append(signal)
            self._signals_emitted += 1
            emitted += 1

            self.logger.info(
                "longshot_signal",
                ticker=km.ticker,
                yes_cents=yes_cents,
                no_ask_cents=no_ask_cents,
                profit_pct=round(profit_pct, 4),
                edge=round(edge, 4),
                volume=km.volume_24h,
            )

        # Clean stale cooldowns
        stale_cutoff = now_ts - (self._eval_cooldown * 2)
        for k in [k for k, ts in self._recently_evaluated.items() if ts < stale_cutoff]:
            del self._recently_evaluated[k]

        self.logger.info(
            "longshot_scan_summary",
            markets_fetched=len(markets),
            candidates=len(candidates),
            emitted=emitted,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "scanned": self._scanned,
            "signals_emitted": self._signals_emitted,
            "active_positions": self._active_positions,
        }
