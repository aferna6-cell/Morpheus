"""Spike Detection Engine — monitors the real-time WebSocket feed for
significant price movements and emits trade signals.

Spike tiers (configurable via ``config.yaml`` → ``spike_detection``):
- **Fast**:   >2 % move in <30 s
- **Medium**: >5 % move in <5 min
- **Large**:  >10 % move in <30 min

Strategy:
- Downward spikes on high-volume markets → buy (mean-reversion)
- Upward spikes → potential momentum / sell existing position
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import structlog

from ..ws_feed import PriceBookEntry, WebSocketFeed
from .base import BaseEngine
from .signals import TradeSignal


logger = structlog.get_logger()

# Maximum number of ticks to keep per asset
MAX_TICK_HISTORY = 100


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _Tick:
    """A single price observation."""

    price: float
    timestamp: float  # monotonic ``time.time()``


@dataclass
class _AssetState:
    """Rolling price history and cooldown tracking for one asset."""

    market_id: str
    token_id: str
    volume_24h: float
    ticks: Deque[_Tick] = field(default_factory=lambda: deque(maxlen=MAX_TICK_HISTORY))
    last_signal_ts: float = 0.0  # monotonic; used for cooldown


# ---------------------------------------------------------------------------
# Spike tiers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SpikeTier:
    name: str
    pct: float        # minimum absolute % move
    window_s: float   # look-back window in seconds
    urgency: str      # urgency tag for the resulting TradeSignal


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SpikeEngine(BaseEngine):
    """Detects price spikes from the WebSocket feed and emits TradeSignals."""

    name = "spike"

    def __init__(
        self,
        ws_feed: WebSocketFeed,
        config: Dict[str, Any],
        signal_queue: Optional[asyncio.Queue[TradeSignal]] = None,
    ) -> None:
        self._feed = ws_feed
        self._cfg = config
        self._queue: asyncio.Queue[TradeSignal] = signal_queue or asyncio.Queue()

        # Parse spike tiers from config
        self._tiers = self._build_tiers(config)
        self._min_volume = float(config.get("min_volume_24h", 10_000))
        self._cooldown_s = float(config.get("cooldown_after_signal_seconds", 60))

        # Per-asset rolling state
        self._assets: Dict[str, _AssetState] = {}

        self._running = False

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._feed.on_price_update(self._on_price_update)
        logger.info(
            "spike_engine_started",
            tiers=[t.name for t in self._tiers],
            min_volume=self._min_volume,
            cooldown_s=self._cooldown_s,
        )

    async def stop(self) -> None:
        self._running = False
        logger.info("spike_engine_stopped")

    async def get_signals(self) -> List[TradeSignal]:
        """Drain all pending signals from the internal queue."""
        signals: List[TradeSignal] = []
        while not self._queue.empty():
            try:
                signals.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return signals

    # ------------------------------------------------------------------
    # Asset registration
    # ------------------------------------------------------------------

    def track_asset(
        self,
        asset_id: str,
        market_id: str,
        volume_24h: float = 0.0,
    ) -> None:
        """Register an asset for spike monitoring.

        Should be called when the market scanner discovers active markets,
        passing each token's ``asset_id`` (CLOB token ID) and its parent
        ``market_id``.
        """
        if asset_id in self._assets:
            # Update volume in case it changed
            self._assets[asset_id].volume_24h = volume_24h
            return

        self._assets[asset_id] = _AssetState(
            market_id=market_id,
            token_id=asset_id,
            volume_24h=volume_24h,
        )

    # ------------------------------------------------------------------
    # Price update callback (registered with WebSocketFeed)
    # ------------------------------------------------------------------

    async def _on_price_update(self, asset_id: str, entry: PriceBookEntry) -> None:
        """Called by the WebSocket feed on every price change."""
        if not self._running:
            return

        state = self._assets.get(asset_id)
        if state is None:
            return  # Not tracked

        # Determine the representative price (midpoint preferred)
        price = entry.midpoint
        if price is None or price <= 0:
            return

        now = time.time()
        state.ticks.append(_Tick(price=price, timestamp=now))

        # Need at least 2 ticks for comparison
        if len(state.ticks) < 2:
            return

        # Check cooldown
        if (now - state.last_signal_ts) < self._cooldown_s:
            return

        # Volume gate
        if state.volume_24h < self._min_volume:
            return

        # Evaluate spike tiers (largest first for priority)
        for tier in reversed(self._tiers):
            spike = self._detect_spike(state, tier, now)
            if spike is not None:
                signal = self._build_signal(state, tier, spike, price, now)
                self._queue.put_nowait(signal)
                state.last_signal_ts = now

                logger.info(
                    "spike_detected",
                    tier=tier.name,
                    asset_id=asset_id,
                    market_id=state.market_id,
                    magnitude=f"{spike:+.4f}",
                    price=price,
                    side=signal.side,
                    confidence=signal.confidence,
                )
                break  # One signal per update

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------

    def _detect_spike(
        self, state: _AssetState, tier: _SpikeTier, now: float
    ) -> Optional[float]:
        """Return the signed magnitude if a spike is detected, else None.

        Magnitude is ``(current - reference) / reference``.
        Positive = price went up, negative = price went down.
        """
        cutoff = now - tier.window_s

        # Find the oldest tick within the window
        reference: Optional[_Tick] = None
        for tick in state.ticks:
            if tick.timestamp >= cutoff:
                reference = tick
                break

        if reference is None:
            return None

        current = state.ticks[-1]
        if reference.price <= 0:
            return None

        magnitude = (current.price - reference.price) / reference.price

        if abs(magnitude) >= tier.pct:
            return magnitude

        return None

    # ------------------------------------------------------------------
    # Signal construction
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        state: _AssetState,
        tier: _SpikeTier,
        magnitude: float,
        current_price: float,
        now: float,
    ) -> TradeSignal:
        """Create a :class:`TradeSignal` from a detected spike."""

        # Strategy heuristic:
        #   - Downward spike → mean-reversion buy opportunity
        #   - Upward spike → momentum or sell existing
        if magnitude < 0:
            side = "buy_yes"  # mean-reversion: buy the dip
            # Confidence scales with spike magnitude & volume
            confidence = min(1.0, abs(magnitude) * 5)
        else:
            side = "sell"  # upward spike — take profit / avoid chasing
            confidence = min(1.0, abs(magnitude) * 3)

        # Edge estimate: magnitude itself is a rough proxy
        edge = abs(magnitude) * 0.5  # conservative: expect half the move to revert

        return TradeSignal(
            engine=self.name,
            market_id=state.market_id,
            token_id=state.token_id,
            side=side,
            confidence=round(confidence, 4),
            edge=round(edge, 6),
            urgency=tier.urgency,
            metadata={
                "spike_tier": tier.name,
                "spike_magnitude": round(magnitude, 6),
                "spike_window_seconds": tier.window_s,
                "current_price": round(current_price, 6),
                "volume_24h": state.volume_24h,
                "tick_count": len(state.ticks),
            },
            timestamp=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tiers(cfg: Dict[str, Any]) -> List[_SpikeTier]:
        """Build spike tiers from the ``spike_detection`` config section."""
        return [
            _SpikeTier(
                name="fast",
                pct=float(cfg.get("fast_spike_pct", 0.02)),
                window_s=float(cfg.get("fast_spike_window_seconds", 30)),
                urgency="immediate",
            ),
            _SpikeTier(
                name="medium",
                pct=float(cfg.get("medium_spike_pct", 0.05)),
                window_s=float(cfg.get("medium_spike_window_seconds", 300)),
                urgency="normal",
            ),
            _SpikeTier(
                name="large",
                pct=float(cfg.get("large_spike_pct", 0.10)),
                window_s=float(cfg.get("large_spike_window_seconds", 1800)),
                urgency="low",
            ),
        ]
