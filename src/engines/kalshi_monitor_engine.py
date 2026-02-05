"""Kalshi real-time price monitor — detects price spikes via orderbook polling.

Polls Kalshi orderbooks at high frequency for tracked markets and detects
significant price movements.  Emits mean-reversion signals on downward spikes
(buy the dip) and momentum signals on upward spikes.

Spike tiers:
- Fast:   >3% move detected between consecutive polls
- Medium: >5% move over 5 minutes (rolling window)
- Large:  >10% move over 30 minutes

No LLM cost — pure math / event-driven.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import structlog

from ..kalshi_client import KalshiClient, KalshiMarket
from ..utils import BotConfig
from .base import BaseEngine
from .signals import TradeSignal

logger = structlog.get_logger()

MAX_TICK_HISTORY = 120


@dataclass
class _Tick:
    price: float
    timestamp: float  # monotonic time.time()


@dataclass
class _MarketState:
    ticker: str
    title: str
    volume: int
    yes_bid: float = 0.0
    yes_ask: float = 0.0
    ticks: Deque[_Tick] = field(default_factory=lambda: deque(maxlen=MAX_TICK_HISTORY))
    last_signal_ts: float = 0.0


@dataclass(frozen=True)
class _SpikeTier:
    name: str
    pct: float        # minimum absolute % move
    window_s: float   # look-back window in seconds
    urgency: str


class KalshiMonitorEngine(BaseEngine):
    """Monitors Kalshi orderbooks for price spikes."""

    name = "kalshi_monitor"

    def __init__(self, config: BotConfig, kalshi_client: KalshiClient):
        self.config = config
        self.kalshi_client = kalshi_client

        mon_cfg = getattr(config, "kalshi_monitor", None) or {}
        if not isinstance(mon_cfg, dict):
            mon_cfg = {}

        self._poll_interval: float = float(mon_cfg.get("poll_interval_seconds", 15))
        self._market_refresh_interval: float = float(mon_cfg.get("market_refresh_seconds", 300))
        self._min_volume: int = int(mon_cfg.get("min_volume", 20000))
        self._max_tracked: int = int(mon_cfg.get("max_tracked_markets", 50))
        self._cooldown_s: float = float(mon_cfg.get("cooldown_seconds", 120))

        # Spike tiers
        self._tiers = [
            _SpikeTier(
                name="fast",
                pct=float(mon_cfg.get("fast_spike_pct", 0.03)),
                window_s=float(mon_cfg.get("fast_spike_window_seconds", 30)),
                urgency="immediate",
            ),
            _SpikeTier(
                name="medium",
                pct=float(mon_cfg.get("medium_spike_pct", 0.05)),
                window_s=float(mon_cfg.get("medium_spike_window_seconds", 300)),
                urgency="normal",
            ),
            _SpikeTier(
                name="large",
                pct=float(mon_cfg.get("large_spike_pct", 0.10)),
                window_s=float(mon_cfg.get("large_spike_window_seconds", 1800)),
                urgency="normal",
            ),
        ]

        # State
        self._markets: Dict[str, _MarketState] = {}
        self._pending: List[TradeSignal] = []
        self._poll_task: Optional[asyncio.Task] = None
        self._refresh_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_refresh: float = 0.0

        # Stats
        self._spikes_detected = 0

    async def start(self) -> None:
        await self.kalshi_client.start()
        self._running = True

        # Initial market list
        await self._refresh_market_list()

        self._poll_task = asyncio.create_task(self._poll_loop())
        self._refresh_task = asyncio.create_task(self._refresh_loop())

        logger.info(
            "kalshi_monitor_engine_started",
            tracked_markets=len(self._markets),
            poll_interval=self._poll_interval,
            tiers=[t.name for t in self._tiers],
        )

    async def stop(self) -> None:
        self._running = False
        for task in [self._poll_task, self._refresh_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info(
            "kalshi_monitor_engine_stopped",
            spikes_detected=self._spikes_detected,
        )

    async def get_signals(self) -> List[TradeSignal]:
        out = list(self._pending)
        self._pending.clear()
        return out

    # ------------------------------------------------------------------
    # Market list management
    # ------------------------------------------------------------------

    async def _refresh_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._market_refresh_interval)
            try:
                await self._refresh_market_list()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("kalshi_monitor_refresh_error", error=str(exc))

    async def _refresh_market_list(self) -> None:
        """Fetch high-volume Kalshi markets to monitor."""
        markets = await self.kalshi_client.fetch_markets_by_close_date(
            max_days=30,
            min_volume=self._min_volume,
        )

        # Sort by volume descending, take top N
        markets.sort(key=lambda m: m.volume, reverse=True)
        top = markets[:self._max_tracked]

        # Add new markets, keep existing state for already-tracked ones
        new_tickers = set()
        for km in top:
            new_tickers.add(km.ticker)
            if km.ticker not in self._markets:
                self._markets[km.ticker] = _MarketState(
                    ticker=km.ticker,
                    title=km.title,
                    volume=km.volume,
                    yes_bid=km.yes_bid,
                    yes_ask=km.yes_ask,
                )
            else:
                # Update volume
                self._markets[km.ticker].volume = km.volume

        # Remove markets no longer in the top list
        stale = [t for t in self._markets if t not in new_tickers]
        for t in stale:
            del self._markets[t]

        logger.info("kalshi_monitor_markets_refreshed", tracked=len(self._markets))

    # ------------------------------------------------------------------
    # Price polling
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._poll_prices()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("kalshi_monitor_poll_error", error=str(exc))
            await asyncio.sleep(self._poll_interval)

    async def _poll_prices(self) -> None:
        """Poll current prices for all tracked markets."""
        now = time.time()
        tickers = list(self._markets.keys())

        for ticker in tickers:
            state = self._markets.get(ticker)
            if state is None:
                continue

            try:
                km = await self.kalshi_client.fetch_market(ticker)
                if km is None:
                    continue

                price = km.yes_price
                if price <= 0 or price >= 1:
                    continue

                state.yes_bid = km.yes_bid
                state.yes_ask = km.yes_ask
                state.ticks.append(_Tick(price=price, timestamp=now))

                # Check for spikes
                if len(state.ticks) >= 2:
                    self._check_spikes(state, price, now)

            except Exception as exc:
                logger.debug("kalshi_monitor_tick_error", ticker=ticker, error=str(exc))

            # Brief pause between markets to respect rate limits
            await asyncio.sleep(0.15)

    # ------------------------------------------------------------------
    # Spike detection
    # ------------------------------------------------------------------

    def _check_spikes(self, state: _MarketState, current_price: float, now: float) -> None:
        """Check all spike tiers for a market."""
        # Cooldown check
        if (now - state.last_signal_ts) < self._cooldown_s:
            return

        # Check tiers largest first
        for tier in reversed(self._tiers):
            magnitude = self._detect_spike(state, tier, now)
            if magnitude is not None:
                signal = self._build_signal(state, tier, magnitude, current_price)
                self._pending.append(signal)
                state.last_signal_ts = now
                self._spikes_detected += 1

                logger.info(
                    "kalshi_spike_detected",
                    tier=tier.name,
                    ticker=state.ticker,
                    magnitude=f"{magnitude:+.4f}",
                    price=current_price,
                    side=signal.side,
                )
                break  # One signal per update

    def _detect_spike(
        self, state: _MarketState, tier: _SpikeTier, now: float
    ) -> Optional[float]:
        """Return signed magnitude if spike detected, else None."""
        cutoff = now - tier.window_s

        reference: Optional[_Tick] = None
        for tick in state.ticks:
            if tick.timestamp >= cutoff:
                reference = tick
                break

        if reference is None or reference.price <= 0:
            return None

        current = state.ticks[-1]
        magnitude = (current.price - reference.price) / reference.price

        if abs(magnitude) >= tier.pct:
            return magnitude
        return None

    def _build_signal(
        self,
        state: _MarketState,
        tier: _SpikeTier,
        magnitude: float,
        current_price: float,
    ) -> TradeSignal:
        """Create a TradeSignal from a detected spike."""
        # Mean-reversion: buy on downward spikes
        if magnitude < 0:
            side = "buy_yes"
            confidence = min(0.8, abs(magnitude) * 5)
        else:
            # Upward spike — could be momentum, buy NO as mean-reversion
            side = "buy_no"
            confidence = min(0.7, abs(magnitude) * 3)

        edge = max(0.05, abs(magnitude) * 0.5)  # min 5% edge to meet threshold

        return TradeSignal(
            engine=self.name,
            market_id=state.ticker,
            token_id=state.ticker if side == "buy_yes" else f"{state.ticker}:no",
            side=side,
            confidence=round(confidence, 4),
            edge=round(edge, 6),
            urgency=tier.urgency,
            metadata={
                "platform": "kalshi",
                "spike_tier": tier.name,
                "spike_magnitude": round(magnitude, 6),
                "spike_window_seconds": tier.window_s,
                "current_price": round(current_price, 6),
                "yes_bid": state.yes_bid,
                "yes_ask": state.yes_ask,
                "volume": state.volume,
                "title": state.title,
                "kalshi_ticker": state.ticker,
            },
            timestamp=datetime.now(timezone.utc),
        )
