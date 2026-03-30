"""Kalshi Rules-Based Engine — time decay and price consistency signals.

Ports Neo's RulesBasedStrategy to the Morpheus engine pattern.

Two rules are implemented:

  1. **Time Decay**: Markets within 24h of expiry priced > 85c or < 15c.
     The price is converging to 0 or 100 — buy in that direction.
     Momentum filter: skip YES signal if price is falling; skip NO if rising.
     Confidence: 50-75% scaled by time-to-expiry and extremity.

  2. **Price Consistency**: If YES + NO < 92, both sides are underpriced.
     Buy the cheaper side.  Edge = (100 - sum) / 2 cents.

Design notes:
- Skips markets within 2h of close (bonding engine handles those)
- Low weight (0.10) — this complements higher-tier engines, not replaces them
- Zero LLM cost, zero external API calls
- Scans active markets every ``scan_interval_seconds`` (default 300s)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import structlog

from ..kalshi_client import KalshiClient, KalshiMarket
from ..market_filters import MarketFilters
from ..utils import BotConfig
from .base import BaseEngine
from .signals import TradeSignal


class KalshiRulesEngine(BaseEngine):
    """Emit time-decay and price-consistency signals for Kalshi markets."""

    name = "kalshi_rules"

    # Strategy weight in ensemble (low — rules supplement higher-tier engines)
    STRATEGY_WEIGHT = 0.10

    def __init__(
        self,
        config: BotConfig,
        kalshi_client: KalshiClient,
    ) -> None:
        self.config = config
        self.kalshi_client = kalshi_client
        self.logger = structlog.get_logger()
        self._filters = MarketFilters(config)

        rules_cfg = getattr(config, "rules_engine", None) or {}
        if isinstance(rules_cfg, dict):
            self._interval = float(rules_cfg.get("scan_interval_seconds", 300))
            self._min_volume = int(rules_cfg.get("min_volume", 500))
            self._time_decay_min_price = int(rules_cfg.get("time_decay_min_price_cents", 85))
            self._time_decay_max_price = int(rules_cfg.get("time_decay_max_price_cents", 15))
            self._consistency_threshold = int(rules_cfg.get("consistency_threshold", 92))
            self._min_deviation = int(rules_cfg.get("min_deviation_cents", 8))
        else:
            self._interval = 300
            self._min_volume = 500
            self._time_decay_min_price = 85
            self._time_decay_max_price = 15
            self._consistency_threshold = 92
            self._min_deviation = 8

        self._pending: List[TradeSignal] = []
        self._task: Optional[asyncio.Task] = None

        # Cooldown: don't re-signal the same market+side within 30 minutes
        self._cooldown: Dict[str, float] = {}  # "ticker:side" -> monotonic ts
        self._cooldown_seconds = 1800.0

        # Stats
        self._scans = 0
        self._signals_generated = 0

    async def start(self) -> None:
        await self.kalshi_client.start()
        self._task = asyncio.create_task(self._scan_loop())
        self.logger.info(
            "kalshi_rules_engine_started",
            interval_seconds=self._interval,
            min_volume=self._min_volume,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info(
            "kalshi_rules_engine_stopped",
            scans=self._scans,
            signals_generated=self._signals_generated,
        )

    async def get_signals(self) -> List[TradeSignal]:
        out = list(self._pending)
        self._pending.clear()
        return out

    # ------------------------------------------------------------------
    # Public scan interface (callable directly for one-shot use)
    # ------------------------------------------------------------------

    async def scan(self) -> List[KalshiMarket]:
        """Fetch and filter candidate markets.  Returns filtered list."""
        raw = await self.kalshi_client.fetch_markets(
            status="open",
            limit=500,
            min_volume=self._min_volume,
        )
        candidates = []
        for km in raw:
            # Ticker prefix / junk filter
            if not self._filters.check_ticker_prefix(km.ticker).passed:
                continue
            # Sports filter
            if not self._filters.check_sports(
                market_id=km.ticker,
                title=km.title,
                category=km.category,
                is_live=False,
            ).passed:
                continue
            candidates.append(km)
        return candidates

    async def generate_signals(self, markets: List[KalshiMarket]) -> List[TradeSignal]:
        """Apply rules to a list of markets and return any signals produced."""
        signals: List[TradeSignal] = []
        now_mono = time.monotonic()

        for km in markets:
            # Rule 1: time decay
            sig = self._check_time_decay(km, now_mono)
            if sig:
                signals.append(sig)
                continue  # one signal per market per cycle

            # Rule 2: price consistency
            sig = self._check_price_consistency(km, now_mono)
            if sig:
                signals.append(sig)

        return signals

    # ------------------------------------------------------------------
    # Background scan loop
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        await asyncio.sleep(60)  # let other engines start first
        while True:
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.error("kalshi_rules_scan_error", error=str(exc))
            await asyncio.sleep(self._interval)

    async def _scan_once(self) -> None:
        self._scans += 1
        markets = await self.scan()
        signals = await self.generate_signals(markets)

        for sig in signals:
            self._pending.append(sig)
            self._signals_generated += 1

        self.logger.info(
            "kalshi_rules_scan_complete",
            markets_scanned=len(markets),
            signals=len(signals),
            total_generated=self._signals_generated,
        )

    # ------------------------------------------------------------------
    # Rule implementations
    # ------------------------------------------------------------------

    def _check_time_decay(
        self, km: KalshiMarket, now_mono: float
    ) -> Optional[TradeSignal]:
        """Rule 1: Near-expiry market with strongly directional price.

        Logic:
        - Must be within 24h of close (exclusive) and > 2h away (bonding handles < 2h)
        - YES price >= 85c → buy YES; YES price <= 15c → buy NO
        - Confidence scales from 50% (at 24h) to 75% (at 2h), weighted by price extremity
        - Momentum filter: YES signal suppressed if yes_price is trending down
          (proxy: yes_ask - yes_bid spread is inverted, i.e., no_ask < yes_ask).
          Without a price history we use a simple heuristic: if no_ask > yes_bid
          the market is leaning NO (implicit downward momentum on YES).
        """
        if km.close_time is None:
            return None

        now_dt = datetime.now(timezone.utc)
        hours_to_close = (km.close_time - now_dt).total_seconds() / 3600.0

        if hours_to_close <= 0 or hours_to_close > 24:
            return None

        # Bonding engine handles the < 2h window — stay out of its territory
        if hours_to_close < 2.0:
            return None

        yes_cents = round(km.yes_price * 100)
        no_cents = round(km.no_price * 100)

        if yes_cents >= self._time_decay_min_price:
            # BUY YES signal
            # Momentum heuristic: skip if the NO side ask is significantly
            # cheaper than YES bid (suggests market is drifting toward NO).
            if km.no_ask > 0 and km.yes_bid > 0:
                if km.no_ask < (1.0 - km.yes_bid) - 0.05:
                    self.logger.debug(
                        "rules_time_decay_yes_suppressed_momentum",
                        ticker=km.ticker,
                        yes_bid=km.yes_bid,
                        no_ask=km.no_ask,
                    )
                    return None

            side = "buy_yes"
            token_id = km.ticker
            edge = float(100 - yes_cents)
            if edge < 3:
                return None  # too little room to profit
        elif yes_cents <= self._time_decay_max_price:
            # BUY NO signal
            # Momentum heuristic: skip if yes_ask is trending up.
            if km.yes_ask > 0 and km.no_bid > 0:
                if km.yes_ask < (1.0 - km.no_bid) - 0.05:
                    self.logger.debug(
                        "rules_time_decay_no_suppressed_momentum",
                        ticker=km.ticker,
                        yes_ask=km.yes_ask,
                        no_bid=km.no_bid,
                    )
                    return None

            side = "buy_no"
            token_id = f"{km.ticker}:no"
            edge = float(yes_cents)
            if edge < 3:
                return None
        else:
            return None

        # Cooldown check
        cooldown_key = f"{km.ticker}:{side}"
        last_signal = self._cooldown.get(cooldown_key, 0.0)
        if now_mono - last_signal < self._cooldown_seconds:
            return None

        # Confidence: base 50, scales up to 75 based on:
        # - time proximity: closer to close = higher confidence (max +15)
        # - price extremity: further from 50 = higher confidence (max +10)
        time_factor = max(0.0, (24.0 - hours_to_close) / 22.0)  # 0→1 as we near close
        if side == "buy_yes":
            extremity_factor = (yes_cents - self._time_decay_min_price) / (100.0 - self._time_decay_min_price)
        else:
            extremity_factor = (self._time_decay_max_price - yes_cents) / self._time_decay_max_price

        confidence = 0.50 + time_factor * 0.15 + extremity_factor * 0.10
        confidence = max(0.50, min(0.75, confidence))

        self._cooldown[cooldown_key] = now_mono

        self.logger.info(
            "rules_time_decay_signal",
            ticker=km.ticker,
            side=side,
            yes_cents=yes_cents,
            hours_to_close=round(hours_to_close, 1),
            edge=edge,
            confidence=round(confidence, 3),
        )

        return TradeSignal(
            engine=self.name,
            market_id=km.ticker,
            token_id=token_id,
            side=side,
            confidence=round(confidence, 3),
            edge=round(edge / 100.0, 4),  # normalize to 0-1 scale like other engines
            urgency="normal" if hours_to_close > 6 else "immediate",
            metadata={
                "platform": "kalshi",
                "strategy": "rules_based",
                "signal_source": "rules_time_decay",
                "rule": "time_decay",
                "yes_price_cents": yes_cents,
                "no_price_cents": no_cents,
                "hours_to_close": round(hours_to_close, 2),
                "edge_cents": edge,
                "weight": self.STRATEGY_WEIGHT,
                "kalshi_ticker": km.ticker,
                "kalshi_yes_bid": km.yes_bid,
                "kalshi_yes_ask": km.yes_ask,
                "kalshi_no_bid": km.no_bid,
                "kalshi_no_ask": km.no_ask,
                "title": km.title,
                "question": km.title,
                "time_to_close_hours": round(hours_to_close, 2),
            },
        )

    def _check_price_consistency(
        self, km: KalshiMarket, now_mono: float
    ) -> Optional[TradeSignal]:
        """Rule 2: YES + NO should sum to ~100c; large deviations are mispricings.

        If sum < (100 - self._consistency_threshold), both sides are underpriced.
        Buy the cheaper side. Edge = (100 - sum) / 2.

        Threshold defaults to 92 (i.e., signal fires when sum < 92, deviation > 8c).
        """
        if km.yes_price <= 0 or km.no_price <= 0:
            return None

        yes_cents = round(km.yes_price * 100)
        no_cents = round(km.no_price * 100)
        total = yes_cents + no_cents

        deviation = 100 - total
        if total >= self._consistency_threshold or deviation < self._min_deviation:
            return None

        # Buy the cheaper side
        if yes_cents <= no_cents:
            side = "buy_yes"
            token_id = km.ticker
        else:
            side = "buy_no"
            token_id = f"{km.ticker}:no"

        # Cooldown check
        cooldown_key = f"{km.ticker}:{side}"
        last_signal = self._cooldown.get(cooldown_key, 0.0)
        if now_mono - last_signal < self._cooldown_seconds:
            return None

        edge = deviation / 2.0  # each side gets half the total arbitrage

        # Confidence: scales with deviation magnitude
        # 8c dev → 50%, 16c dev → 60%, 24c+ → 70%
        confidence = min(0.70, 0.50 + (deviation - self._min_deviation) / (24.0 - self._min_deviation) * 0.20)
        confidence = max(0.50, confidence)

        self._cooldown[cooldown_key] = now_mono

        # Determine time to close for urgency
        now_dt = datetime.now(timezone.utc)
        hours_to_close = 24.0
        if km.close_time:
            hours_to_close = max(0.0, (km.close_time - now_dt).total_seconds() / 3600.0)

        self.logger.info(
            "rules_price_consistency_signal",
            ticker=km.ticker,
            side=side,
            yes_cents=yes_cents,
            no_cents=no_cents,
            total=total,
            deviation=deviation,
            edge_cents=edge,
            confidence=round(confidence, 3),
        )

        return TradeSignal(
            engine=self.name,
            market_id=km.ticker,
            token_id=token_id,
            side=side,
            confidence=round(confidence, 3),
            edge=round(edge / 100.0, 4),
            urgency="normal",
            metadata={
                "platform": "kalshi",
                "strategy": "rules_based",
                "signal_source": "rules_price_consistency",
                "rule": "price_consistency",
                "yes_price_cents": yes_cents,
                "no_price_cents": no_cents,
                "total_cents": total,
                "deviation_cents": deviation,
                "edge_cents": edge,
                "weight": self.STRATEGY_WEIGHT,
                "kalshi_ticker": km.ticker,
                "kalshi_yes_bid": km.yes_bid,
                "kalshi_yes_ask": km.yes_ask,
                "kalshi_no_bid": km.no_bid,
                "kalshi_no_ask": km.no_ask,
                "title": km.title,
                "question": km.title,
                "time_to_close_hours": round(hours_to_close, 2),
            },
        )
