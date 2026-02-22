"""Kalshi order-flow engine — detect informed money via VPIN analysis.

Monitors the public trade tape for markets settling today. Computes a
Volume-Synchronized Probability of Informed Trading (VPIN) metric per market
by bucketing trades into fixed-volume bins and tracking the imbalance between
YES-taker and NO-taker volume.

When VPIN spikes above a z-score threshold, it suggests informed traders are
moving directionally. The engine emits a signal in the direction of the
informed flow.

Design:
- Zero LLM cost — purely data-driven
- Uses public GET /markets/trades endpoint (no auth required)
- Scans only same-day markets with sufficient volume
- Rate-limited to avoid API throttling (30s between full scans)
- Rolling VPIN computed over a configurable bucket window
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

import structlog

from ..kalshi_client import KalshiClient, KalshiMarket
from ..market_filters import MarketFilters
from ..utils import BotConfig
from .base import BaseEngine
from .signals import TradeSignal


class _VPINBucket:
    """A fixed-volume bucket tracking taker-side imbalance."""

    __slots__ = ("volume_target", "yes_volume", "no_volume", "filled")

    def __init__(self, volume_target: int):
        self.volume_target = volume_target
        self.yes_volume = 0
        self.no_volume = 0
        self.filled = False

    @property
    def total_volume(self) -> int:
        return self.yes_volume + self.no_volume

    @property
    def imbalance(self) -> float:
        """Absolute imbalance in this bucket: |V_buy - V_sell| / V_total."""
        total = self.total_volume
        if total == 0:
            return 0.0
        return abs(self.yes_volume - self.no_volume) / total

    @property
    def direction(self) -> str:
        """Dominant direction: 'yes' or 'no'."""
        return "yes" if self.yes_volume >= self.no_volume else "no"


class _MarketVPIN:
    """Rolling VPIN tracker for a single market."""

    def __init__(self, bucket_size: int = 50, n_buckets: int = 20):
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets
        self._buckets: Deque[_VPINBucket] = deque(maxlen=n_buckets)
        self._current = _VPINBucket(bucket_size)
        self._last_trade_id: Optional[str] = None
        self._total_volume_seen = 0

    def add_trade(self, trade_id: str, taker_side: str, count: int) -> None:
        """Ingest a trade and update VPIN buckets."""
        if trade_id == self._last_trade_id:
            return  # dedup
        self._last_trade_id = trade_id
        self._total_volume_seen += count

        remaining = count
        while remaining > 0:
            space = self._current.volume_target - self._current.total_volume
            chunk = min(remaining, space)

            if taker_side == "yes":
                self._current.yes_volume += chunk
            else:
                self._current.no_volume += chunk
            remaining -= chunk

            if self._current.total_volume >= self._current.volume_target:
                self._current.filled = True
                self._buckets.append(self._current)
                self._current = _VPINBucket(self.bucket_size)

    @property
    def vpin(self) -> float:
        """VPIN = average absolute imbalance across filled buckets."""
        if len(self._buckets) < 2:
            return 0.0
        return sum(b.imbalance for b in self._buckets) / len(self._buckets)

    @property
    def directional_vpin(self) -> Tuple[float, str]:
        """Signed VPIN: (magnitude, dominant direction).

        Unlike raw VPIN which just measures absolute imbalance, this tracks
        which side is accumulating — 'yes' or 'no'.
        """
        if len(self._buckets) < 2:
            return 0.0, "yes"

        yes_total = sum(b.yes_volume for b in self._buckets)
        no_total = sum(b.no_volume for b in self._buckets)
        total = yes_total + no_total
        if total == 0:
            return 0.0, "yes"

        imbalance = (yes_total - no_total) / total
        direction = "yes" if imbalance > 0 else "no"
        return abs(imbalance), direction

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)


class KalshiOrderFlowEngine(BaseEngine):
    """Detect informed money on Kalshi via VPIN analysis of the public trade tape."""

    name = "kalshi_orderflow"

    def __init__(
        self,
        config: BotConfig,
        kalshi_client: KalshiClient,
    ):
        self.config = config
        self.kalshi_client = kalshi_client
        self.logger = structlog.get_logger()
        self._filters = MarketFilters(config)

        # Config
        of_cfg = getattr(config, "orderflow", None) or {}
        if isinstance(of_cfg, dict):
            self._scan_interval = float(of_cfg.get("scan_interval_seconds", 60))
            self._bucket_size = int(of_cfg.get("bucket_size", 50))
            self._n_buckets = int(of_cfg.get("n_buckets", 20))
            self._min_vpin = float(of_cfg.get("min_vpin", 0.60))
            self._min_z_score = float(of_cfg.get("min_z_score", 2.0))
            self._min_volume = int(of_cfg.get("min_volume", 200))
            self._signal_cooldown = float(of_cfg.get("signal_cooldown_seconds", 600))
            self._max_signals_per_cycle = int(of_cfg.get("max_signals_per_cycle", 3))
            self._min_confidence = float(of_cfg.get("min_confidence", 0.55))
            self._max_hours_to_close = float(of_cfg.get("max_hours_to_close", 24))
        else:
            self._scan_interval = 60
            self._bucket_size = 50
            self._n_buckets = 20
            self._min_vpin = 0.60
            self._min_z_score = 2.0
            self._min_volume = 200
            self._signal_cooldown = 600
            self._max_signals_per_cycle = 3
            self._min_confidence = 0.55
            self._max_hours_to_close = 24

        # State
        self._trackers: Dict[str, _MarketVPIN] = {}
        self._market_cache: Dict[str, KalshiMarket] = {}  # ticker -> market metadata
        self._signal_timestamps: Dict[str, float] = {}  # ticker -> last signal time
        self._pending: List[TradeSignal] = []
        self._task: Optional[asyncio.Task] = None

        # Baseline VPIN stats for z-score (running mean/std)
        self._vpin_history: Deque[float] = deque(maxlen=200)

        # Stats
        self._scans = 0
        self._trades_ingested = 0
        self._signals_generated = 0

    async def start(self) -> None:
        await self.kalshi_client.start()
        self._task = asyncio.create_task(self._scan_loop())
        self.logger.info(
            "kalshi_orderflow_engine_started",
            scan_interval=self._scan_interval,
            bucket_size=self._bucket_size,
            n_buckets=self._n_buckets,
            min_vpin=self._min_vpin,
            min_z_score=self._min_z_score,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info(
            "kalshi_orderflow_engine_stopped",
            scans=self._scans,
            trades_ingested=self._trades_ingested,
            signals_generated=self._signals_generated,
        )

    async def get_signals(self) -> List[TradeSignal]:
        out = list(self._pending)
        self._pending.clear()
        return out

    # ------------------------------------------------------------------
    # Main scan loop
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        # Let other engines initialize first
        await asyncio.sleep(30)
        while True:
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.error("orderflow_scan_error", error=str(exc))
            await asyncio.sleep(self._scan_interval)

    async def _scan_once(self) -> None:
        """Fetch trades for active same-day markets and compute VPIN."""
        self._scans += 1

        # Step 1: Get same-day markets (reuse kalshi_client's cached market list)
        markets = await self.kalshi_client.fetch_markets_by_close_date(max_days=1)
        if not markets:
            return

        # Filter to markets with decent volume
        active_markets = [
            m for m in markets
            if m.volume_24h >= self._min_volume
            and not self._filters.check_ticker_prefix(m.ticker)
        ]

        # Cache market metadata
        for m in active_markets:
            self._market_cache[m.ticker] = m

        # Step 2: For each active market, fetch recent trades
        # Rate-limit: max ~10 markets per scan to stay under API limits
        scan_tickers = [m.ticker for m in active_markets[:10]]

        for ticker in scan_tickers:
            try:
                # Fetch last 200 trades for this market
                trades = await self.kalshi_client.get_public_trades(
                    ticker=ticker, limit=200,
                )
                self._ingest_trades(ticker, trades)
            except Exception as exc:
                self.logger.debug("orderflow_fetch_error", ticker=ticker, error=str(exc))
            # Small delay between fetches to avoid rate limits
            await asyncio.sleep(0.5)

        # Step 3: Evaluate all tracked markets for VPIN signals
        self._evaluate_signals()

    def _ingest_trades(self, ticker: str, trades: List[Dict[str, Any]]) -> None:
        """Feed trades into the VPIN tracker for a market."""
        if ticker not in self._trackers:
            self._trackers[ticker] = _MarketVPIN(
                bucket_size=self._bucket_size,
                n_buckets=self._n_buckets,
            )

        tracker = self._trackers[ticker]
        for t in trades:
            trade_id = str(t.get("trade_id", ""))
            taker_side = t.get("taker_side", "")
            count = int(t.get("count", 1))

            if taker_side not in ("yes", "no"):
                continue

            tracker.add_trade(trade_id, taker_side, count)
            self._trades_ingested += 1

    def _evaluate_signals(self) -> None:
        """Check all tracked markets for VPIN spikes and emit signals."""
        now = time.monotonic()
        signals_this_cycle = 0

        # Collect VPIN values from all trackers with enough data
        vpins = []
        for ticker, tracker in self._trackers.items():
            if tracker.bucket_count >= 5:  # need at least 5 buckets
                vpins.append(tracker.vpin)

        # Update running VPIN history for z-score baseline
        self._vpin_history.extend(vpins)

        if len(self._vpin_history) < 10:
            return  # need baseline data before generating signals

        # Compute baseline stats
        mean_vpin = sum(self._vpin_history) / len(self._vpin_history)
        variance = sum((v - mean_vpin) ** 2 for v in self._vpin_history) / len(self._vpin_history)
        std_vpin = math.sqrt(variance) if variance > 0 else 0.01  # floor at 0.01

        # Evaluate each market
        candidates: List[Tuple[float, str, _MarketVPIN]] = []  # (z_score, ticker, tracker)

        for ticker, tracker in self._trackers.items():
            if tracker.bucket_count < 5:
                continue

            vpin = tracker.vpin
            z_score = (vpin - mean_vpin) / std_vpin

            if vpin < self._min_vpin or z_score < self._min_z_score:
                continue

            # Cooldown check
            last_signal = self._signal_timestamps.get(ticker, 0)
            if now - last_signal < self._signal_cooldown:
                continue

            candidates.append((z_score, ticker, tracker))

        # Sort by z-score descending, take top N
        candidates.sort(key=lambda x: x[0], reverse=True)

        for z_score, ticker, tracker in candidates[:self._max_signals_per_cycle]:
            if signals_this_cycle >= self._max_signals_per_cycle:
                break

            market = self._market_cache.get(ticker)
            if not market:
                continue

            # Determine direction from directional VPIN
            magnitude, direction = tracker.directional_vpin
            side = "buy_yes" if direction == "yes" else "buy_no"

            # Confidence scales with z-score: 0.55 base + 0.10 per z above threshold
            confidence = min(0.90, self._min_confidence + 0.10 * (z_score - self._min_z_score))

            # Edge estimate: VPIN magnitude as proxy
            # Informed traders typically push price ~3-8% beyond fair value
            edge = min(0.15, magnitude * 0.20)

            if edge < 0.02:
                continue  # not enough edge to bother

            # Use market YES price to estimate actual edge
            market_price = market.yes_price / 100.0 if direction == "yes" else market.no_price / 100.0
            if market_price <= 0 or market_price >= 1:
                continue

            signal = TradeSignal(
                engine=self.name,
                market_id=ticker,
                token_id=ticker,
                side=side,
                confidence=round(confidence, 3),
                edge=round(edge, 4),
                urgency="normal",
                metadata={
                    "strategy": "kalshi_orderflow",
                    "signal_source": "orderflow_vpin",
                    "vpin": round(tracker.vpin, 4),
                    "z_score": round(z_score, 2),
                    "directional_magnitude": round(magnitude, 4),
                    "dominant_direction": direction,
                    "bucket_count": tracker.bucket_count,
                    "total_volume": tracker._total_volume_seen,
                    "market_title": market.title,
                    "market_price_yes": market.yes_price,
                    "market_price_no": market.no_price,
                    "_market": {
                        "close_time": market.close_time,
                        "end_date": market.close_time,
                    },
                },
            )

            self._pending.append(signal)
            self._signal_timestamps[ticker] = now
            self._signals_generated += 1
            signals_this_cycle += 1

            self.logger.info(
                "orderflow_signal",
                ticker=ticker,
                side=side,
                vpin=round(tracker.vpin, 4),
                z_score=round(z_score, 2),
                direction=direction,
                magnitude=round(magnitude, 4),
                confidence=round(confidence, 3),
                edge=round(edge, 4),
                title=market.title[:60],
            )

        # Cleanup stale trackers (markets no longer in active list)
        active_tickers = set(self._market_cache.keys())
        stale = [t for t in self._trackers if t not in active_tickers]
        for t in stale:
            del self._trackers[t]

        if self._scans % 10 == 0:
            self.logger.info(
                "orderflow_status",
                tracked_markets=len(self._trackers),
                total_trades_ingested=self._trades_ingested,
                total_signals=self._signals_generated,
                baseline_mean_vpin=round(mean_vpin, 4),
                baseline_std_vpin=round(std_vpin, 4),
            )
