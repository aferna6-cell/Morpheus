"""Kalshi Orderbook Imbalance (OBI) Engine.

Ports Neo's OrderbookImbalanceStrategy to the Morpheus engine pattern.

OBI measures the ratio of buying pressure (bid depth) to selling pressure (ask
depth) on the YES side:

    OBI = (yes_bid_depth - no_bid_depth) / (yes_bid_depth + no_bid_depth)

A strongly positive OBI (OBI > 0.3) indicates that more YES bids are stacked
relative to NO bids — buying pressure → signal BUY_YES.

A strongly negative OBI (OBI < -0.3) indicates the inverse → signal BUY_NO.

Depth fallback:
When actual depth data (``yes_bid_depth``, ``no_bid_depth``) is unavailable
from the API, the engine uses bid/ask prices as a proxy:

    yes_bid_depth  ← yes_bid  (price buyers are willing to pay)
    no_bid_depth   ← 100 - yes_ask  (implied distance of ask from maximum)

This is a rough heuristic but captures the same directional signal.

Relationship to VPIN (kalshi_orderflow_engine):
- VPIN measures *taker* flow imbalance (aggressor-side pressure)
- OBI measures *resting order* imbalance (passive book depth)
- Together they give a more complete picture of order-flow dynamics
- Use both engines, let the orchestrator's multi-engine consensus boost
  confidence when they agree.

Design:
- Zero LLM cost, zero external API calls
- Scans every ``scan_interval_seconds`` (default 120s)
- Per-market cooldown of 15 minutes to avoid spamming the same signal
- Confidence: 50-95% range scaled with |OBI| - 0.3
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


# Minimum absolute OBI to generate a signal
OBI_THRESHOLD = 0.3


class KalshiOrderbookEngine(BaseEngine):
    """Detect directional pressure from resting-order depth imbalance."""

    name = "kalshi_orderbook"

    def __init__(
        self,
        config: BotConfig,
        kalshi_client: KalshiClient,
    ) -> None:
        self.config = config
        self.kalshi_client = kalshi_client
        self.logger = structlog.get_logger()
        self._filters = MarketFilters(config)

        ob_cfg = getattr(config, "orderbook_engine", None) or {}
        if isinstance(ob_cfg, dict):
            self._interval = float(ob_cfg.get("scan_interval_seconds", 120))
            self._min_volume = int(ob_cfg.get("min_volume", 200))
            self._obi_threshold = float(ob_cfg.get("obi_threshold", OBI_THRESHOLD))
            self._cooldown_seconds = float(ob_cfg.get("signal_cooldown_seconds", 900))
            self._max_hours_to_close = float(ob_cfg.get("max_hours_to_close", 24))
        else:
            self._interval = 120
            self._min_volume = 200
            self._obi_threshold = OBI_THRESHOLD
            self._cooldown_seconds = 900
            self._max_hours_to_close = 24

        self._pending: List[TradeSignal] = []
        self._task: Optional[asyncio.Task] = None

        # Per-market signal cooldown: "ticker:side" -> monotonic ts
        self._cooldown: Dict[str, float] = {}

        # Stats
        self._scans = 0
        self._signals_generated = 0

    async def start(self) -> None:
        await self.kalshi_client.start()
        self._task = asyncio.create_task(self._scan_loop())
        self.logger.info(
            "kalshi_orderbook_engine_started",
            interval_seconds=self._interval,
            obi_threshold=self._obi_threshold,
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
            "kalshi_orderbook_engine_stopped",
            scans=self._scans,
            signals_generated=self._signals_generated,
        )

    async def get_signals(self) -> List[TradeSignal]:
        out = list(self._pending)
        self._pending.clear()
        return out

    # ------------------------------------------------------------------
    # Public interface (callable for one-shot use)
    # ------------------------------------------------------------------

    def compute_obi(self, km: KalshiMarket) -> Optional[float]:
        """Compute the Orderbook Imbalance for a single market.

        Returns OBI in [-1, 1] or None if insufficient data.

        Uses actual depth fields if available (``yes_bid_depth``,
        ``no_bid_depth``); otherwise falls back to bid/ask prices as proxy.
        """
        # Prefer actual depth data if provided (some API responses include it)
        # KalshiMarket doesn't have depth fields by default — check metadata dict
        # For now we always use the price proxy (the SDK doesn't expose depth).

        yes_bid = km.yes_bid
        yes_ask = km.yes_ask
        no_bid = km.no_bid

        if yes_bid <= 0 or yes_ask <= 0:
            return None

        # Price-based depth proxy:
        #   yes_bid_depth  = how high buyers are bidding on YES  (0-1 scale)
        #   no_bid_depth   = how far the yes_ask is from 1.0
        #                    (a high ask means buyers of NO can afford it)
        yes_bid_depth = yes_bid
        # no_bid_depth proxy: the NO side's buying pressure = 1.0 - yes_ask
        # (when yes_ask is low, NO buyers have lots of room → high NO demand)
        no_bid_depth = max(0.0, 1.0 - yes_ask)

        # Alternative: if we have no_bid directly, use it
        if no_bid > 0:
            no_bid_depth = no_bid

        total = yes_bid_depth + no_bid_depth
        if total == 0:
            return None

        obi = (yes_bid_depth - no_bid_depth) / total
        return obi

    def signal_from_obi(
        self, km: KalshiMarket, obi: float, now_mono: float
    ) -> Optional[TradeSignal]:
        """Convert an OBI value into a TradeSignal, or None if below threshold."""
        abs_obi = abs(obi)
        if abs_obi < self._obi_threshold:
            return None

        direction = "YES" if obi > 0 else "NO"
        side = "buy_yes" if direction == "YES" else "buy_no"
        token_id = km.ticker if direction == "YES" else f"{km.ticker}:no"

        # Cooldown check
        cooldown_key = f"{km.ticker}:{side}"
        last_signal = self._cooldown.get(cooldown_key, 0.0)
        if now_mono - last_signal < self._cooldown_seconds:
            return None

        # Confidence: 50% at OBI=0.3, scaling up to 95% at OBI=1.0
        #   conf = 50 + (|OBI| - threshold) / (1.0 - threshold) * 45
        conf_raw = 50.0 + (abs_obi - self._obi_threshold) / (1.0 - self._obi_threshold) * 45.0
        confidence = max(0.50, min(0.95, conf_raw / 100.0))

        # Edge: OBI magnitude scaled to cents
        # An OBI of 0.5 → ~10c edge estimate; OBI of 1.0 → 20c
        edge_cents = abs_obi * 20.0

        # Time to close for urgency classification
        now_dt = datetime.now(timezone.utc)
        hours_to_close = 24.0
        if km.close_time:
            hours_to_close = max(0.0, (km.close_time - now_dt).total_seconds() / 3600.0)

        urgency = "immediate" if hours_to_close < 2.0 else "normal"

        self._cooldown[cooldown_key] = now_mono

        yes_cents = round(km.yes_price * 100)
        no_cents = round(km.no_price * 100)

        self.logger.info(
            "orderbook_obi_signal",
            ticker=km.ticker,
            obi=round(obi, 4),
            direction=direction,
            confidence=round(confidence, 3),
            edge_cents=round(edge_cents, 1),
            yes_cents=yes_cents,
            no_cents=no_cents,
            hours_to_close=round(hours_to_close, 1),
        )

        return TradeSignal(
            engine=self.name,
            market_id=km.ticker,
            token_id=token_id,
            side=side,
            confidence=round(confidence, 3),
            edge=round(edge_cents / 100.0, 4),
            urgency=urgency,
            metadata={
                "platform": "kalshi",
                "strategy": "orderbook_imbalance",
                "signal_source": "orderbook_obi",
                "obi": round(obi, 4),
                "abs_obi": round(abs_obi, 4),
                "direction": direction,
                "yes_bid": km.yes_bid,
                "yes_ask": km.yes_ask,
                "no_bid": km.no_bid,
                "no_ask": km.no_ask,
                "yes_price_cents": yes_cents,
                "no_price_cents": no_cents,
                "edge_cents": round(edge_cents, 2),
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

    # ------------------------------------------------------------------
    # Background scan loop
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        await asyncio.sleep(30)  # brief startup delay
        while True:
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.error("kalshi_orderbook_scan_error", error=str(exc))
            await asyncio.sleep(self._interval)

    async def _scan_once(self) -> None:
        self._scans += 1

        # Fetch same-day markets (OBI is most meaningful for short-dated markets)
        try:
            markets = await self.kalshi_client.fetch_markets_by_close_date(
                max_days=1, min_volume=self._min_volume,
            )
        except Exception as exc:
            self.logger.error("kalshi_orderbook_fetch_error", error=str(exc))
            return

        now_mono = time.monotonic()
        signals_this_cycle = 0

        for km in markets:
            # Ticker prefix filter
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

            # Must close within max_hours_to_close
            if km.close_time:
                now_dt = datetime.now(timezone.utc)
                hours_to_close = (km.close_time - now_dt).total_seconds() / 3600.0
                if hours_to_close > self._max_hours_to_close or hours_to_close < 0:
                    continue

            # Compute OBI
            obi = self.compute_obi(km)
            if obi is None:
                continue

            # Generate signal
            signal = self.signal_from_obi(km, obi, now_mono)
            if signal is None:
                continue

            self._pending.append(signal)
            self._signals_generated += 1
            signals_this_cycle += 1

        # Prune stale cooldown entries
        stale_cutoff = now_mono - self._cooldown_seconds * 2
        stale_keys = [k for k, ts in self._cooldown.items() if ts < stale_cutoff]
        for k in stale_keys:
            del self._cooldown[k]

        self.logger.info(
            "kalshi_orderbook_scan_complete",
            markets_scanned=len(markets),
            signals_this_cycle=signals_this_cycle,
            total_signals=self._signals_generated,
        )
