"""15-minute crypto engine — directional momentum on KXBTC15M.

Exploits short-term BTC momentum using real-time Binance WebSocket price data.
Every 15 minutes, Kalshi resolves binary markets ("BTC up or down?"). This
engine detects momentum early in each window and bets on continuation.

Strategy:
  - MOMENTUM (minutes 3-7): BTC moved >0.15% from window start → bet continuation
  - MEAN REVERSION (minutes 8-13): BTC moved >0.50% → bet reversal (overextended)

Uses maker orders to minimize fees (0.0175 * C * P * (1-P) per contract).
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from ..engines.base import BaseEngine
from ..engines.signals import TradeSignal
from ..kalshi_client import KalshiClient, KalshiMarket
from ..utils import BotConfig


@dataclass
class CryptoWindow:
    """Tracked state for a single 15-min window."""

    ticker: str
    title: str
    window_start_ts: float  # Unix timestamp of window start
    close_time: datetime  # When Kalshi resolves this market
    yes_bid: float  # 0-1
    yes_ask: float
    volume: int
    btc_price_at_start: Optional[float] = None  # BTC price at window open
    last_signal_time: float = 0.0  # monotonic — cooldown tracking
    signaled: bool = False  # Whether we already signaled this window


class KalshiCryptoEngine(BaseEngine):
    """15-minute crypto engine with Binance real-time price feed."""

    name = "kalshi_crypto"

    def __init__(
        self,
        config: BotConfig,
        kalshi_client: KalshiClient,
        price_feed: Any,  # BinancePriceFeed
    ):
        self.config = config
        self.kalshi_client = kalshi_client
        self.price_feed = price_feed
        self.logger = structlog.get_logger()

        # Config
        crypto_cfg = getattr(config, "crypto_engine", None) or {}
        if not isinstance(crypto_cfg, dict):
            crypto_cfg = {}

        self._scan_interval = float(crypto_cfg.get("scan_interval_seconds", 60))
        self._signal_interval = float(crypto_cfg.get("signal_interval_seconds", 15))
        self._momentum_threshold = float(crypto_cfg.get("momentum_threshold_pct", 0.15))
        self._reversion_threshold = float(crypto_cfg.get("reversion_threshold_pct", 0.50))
        self._min_elapsed_min = float(crypto_cfg.get("min_elapsed_minutes", 3))
        self._momentum_max_min = float(crypto_cfg.get("momentum_window_minutes", 7))
        self._reversion_max_min = float(crypto_cfg.get("reversion_window_minutes", 13))
        self._max_position_usd = float(crypto_cfg.get("max_position_usd", 5.0))
        self._kelly_momentum = float(crypto_cfg.get("kelly_fraction_momentum", 0.67))
        self._kelly_reversion = float(crypto_cfg.get("kelly_fraction_reversion", 0.50))
        self._min_confidence = float(crypto_cfg.get("min_confidence", 0.45))
        self._cooldown_seconds = float(crypto_cfg.get("cooldown_per_window_seconds", 120))
        self._maker_fee_coeff = float(crypto_cfg.get("maker_fee_coefficient", 0.0175))
        self._ticker_prefix = crypto_cfg.get("ticker_prefix", "KXBTC15M")

        # State
        self._windows: Dict[str, CryptoWindow] = {}
        self._pending_signals: List[TradeSignal] = []
        self._scan_task: Optional[asyncio.Task] = None
        self._signal_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())
        self._signal_task = asyncio.create_task(self._signal_loop())
        self.logger.info(
            "crypto_engine_started",
            ticker_prefix=self._ticker_prefix,
            momentum_threshold=self._momentum_threshold,
            signal_interval=self._signal_interval,
        )

    async def stop(self) -> None:
        self._running = False
        for task in [self._scan_task, self._signal_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.logger.info("crypto_engine_stopped")

    async def get_signals(self) -> List[TradeSignal]:
        signals = list(self._pending_signals)
        self._pending_signals.clear()
        return signals

    # ------------------------------------------------------------------
    # Market scanning
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        """Periodically scan for active 15-min crypto markets."""
        while self._running:
            try:
                await self._scan_markets()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error("crypto_scan_error", error=str(e))
            await asyncio.sleep(self._scan_interval)

    async def _scan_markets(self) -> None:
        """Find active KXBTC15M markets."""
        try:
            # Fetch markets closing within 1 day (15-min markets are always near-term)
            all_markets = await self.kalshi_client.fetch_markets_by_close_date(
                max_days=1,
                min_volume=0,  # 15M markets may have varying volume
            )
        except Exception as e:
            self.logger.error("crypto_fetch_error", error=str(e))
            return

        now = datetime.now(timezone.utc)
        now_ts = now.timestamp()
        active_tickers = set()

        for m in all_markets:
            # Only KXBTC15M markets
            if not m.ticker.upper().startswith(self._ticker_prefix):
                continue

            # Must be active/open and have a close time
            if m.status not in ("open", "active") or not m.close_time:
                continue

            # Must close within 20 minutes (current or next window)
            hours_to_close = (m.close_time - now).total_seconds() / 3600
            if hours_to_close < 0 or hours_to_close > 0.35:  # ~21 min
                continue

            active_tickers.add(m.ticker)

            if m.ticker not in self._windows:
                # Parse window start time (close_time - 15 min)
                window_start_ts = m.close_time.timestamp() - 900  # 15 min = 900s

                # Get BTC price at window start (or current if window just started)
                btc_at_start = self.price_feed.price_at(window_start_ts)
                if btc_at_start is None:
                    # Window may have just started — use current price
                    btc_at_start = self.price_feed.price if self.price_feed.price > 0 else None

                self._windows[m.ticker] = CryptoWindow(
                    ticker=m.ticker,
                    title=m.title,
                    window_start_ts=window_start_ts,
                    close_time=m.close_time,
                    yes_bid=m.yes_bid,
                    yes_ask=m.yes_ask,
                    volume=m.volume,
                    btc_price_at_start=btc_at_start,
                )
                self.logger.info(
                    "crypto_window_tracked",
                    ticker=m.ticker,
                    close=m.close_time.isoformat(),
                    btc_at_start=btc_at_start,
                    volume=m.volume,
                )
            else:
                # Update bid/ask/volume
                w = self._windows[m.ticker]
                w.yes_bid = m.yes_bid
                w.yes_ask = m.yes_ask
                w.volume = m.volume

                # Update BTC start price if we didn't have it
                if w.btc_price_at_start is None:
                    w.btc_price_at_start = self.price_feed.price_at(w.window_start_ts)
                    if w.btc_price_at_start is None and self.price_feed.price > 0:
                        w.btc_price_at_start = self.price_feed.price

        # Prune expired windows
        expired = [t for t in self._windows if t not in active_tickers]
        for t in expired:
            del self._windows[t]
            self.logger.debug("crypto_window_expired", ticker=t)

        if self._windows:
            self.logger.debug(
                "crypto_scan_complete",
                active_windows=len(self._windows),
                tickers=list(self._windows.keys()),
            )

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    async def _signal_loop(self) -> None:
        """Generate momentum/reversion signals on active windows."""
        while self._running:
            try:
                await self._generate_signals()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error("crypto_signal_error", error=str(e))
            await asyncio.sleep(self._signal_interval)

    async def _generate_signals(self) -> None:
        """Check each active window for tradeable momentum."""
        if not self.price_feed.is_connected:
            return

        now_ts = time.time()
        now_mono = time.monotonic()

        for ticker, window in list(self._windows.items()):
            # Skip if already signaled this window (1 signal per window)
            if window.signaled:
                continue

            # Cooldown check
            if now_mono - window.last_signal_time < self._cooldown_seconds:
                continue

            # Need BTC price at window start
            if window.btc_price_at_start is None or window.btc_price_at_start <= 0:
                continue

            # Compute elapsed time in window
            elapsed_seconds = now_ts - window.window_start_ts
            elapsed_minutes = elapsed_seconds / 60.0

            # Too early or too late
            if elapsed_minutes < self._min_elapsed_min:
                continue
            if elapsed_minutes > self._reversion_max_min:
                continue

            # Current BTC price
            current_btc = self.price_feed.price
            if current_btc <= 0:
                continue

            # Price change since window start
            change_pct = (current_btc - window.btc_price_at_start) / window.btc_price_at_start * 100.0
            abs_change = abs(change_pct)

            signal_type = None
            side = None
            confidence = 0.0

            # MOMENTUM: minutes 3-7, change > threshold
            if (
                elapsed_minutes <= self._momentum_max_min
                and abs_change >= self._momentum_threshold
            ):
                signal_type = "momentum"
                # Bet on continuation: if BTC is up, bet YES (up); if down, bet NO (down)
                side = "buy_yes" if change_pct > 0 else "buy_no"
                # Confidence scales with move size
                confidence = min(0.78, 0.48 + abs_change * 40)

            # MEAN REVERSION: minutes 8-13, change > reversion threshold
            elif (
                elapsed_minutes > self._momentum_max_min
                and elapsed_minutes <= self._reversion_max_min
                and abs_change >= self._reversion_threshold
            ):
                signal_type = "mean_reversion"
                # Bet against the move: if BTC is up a lot, bet NO (revert down)
                side = "buy_no" if change_pct > 0 else "buy_yes"
                # Confidence: moderate, scales with overextension
                confidence = min(0.70, 0.42 + (abs_change - self._reversion_threshold) * 25)

            if signal_type is None or side is None:
                continue

            if confidence < self._min_confidence:
                continue

            # Fetch fresh orderbook for this market
            try:
                fresh = await self.kalshi_client.fetch_market(ticker)
                if fresh is not None:
                    window.yes_bid = fresh.yes_bid
                    window.yes_ask = fresh.yes_ask
            except Exception:
                pass  # Use cached bid/ask

            # Compute edge: our implied probability vs market price
            # Market midpoint
            mid = (window.yes_bid + window.yes_ask) / 2
            if mid <= 0.02 or mid >= 0.98:
                # Orderbook not developed yet — skip
                continue

            # Our estimated probability
            if side == "buy_yes":
                our_prob = min(0.85, 0.50 + abs_change * 0.15)  # Scale with momentum
                entry_price = window.yes_ask  # We buy YES at ask
                edge = our_prob - entry_price
            else:
                our_prob = max(0.15, 0.50 - abs_change * 0.15)
                entry_price = 1.0 - window.yes_bid  # NO price = 1 - YES bid
                edge = (1.0 - our_prob) - entry_price

            # Maker fee adjustment
            fee_per_contract = self._maker_fee_coeff * entry_price * (1.0 - entry_price)
            net_edge = edge - fee_per_contract

            if net_edge <= 0.01:  # Need at least 1% net edge
                self.logger.debug(
                    "crypto_edge_too_low",
                    ticker=ticker,
                    signal_type=signal_type,
                    edge=round(edge, 4),
                    net_edge=round(net_edge, 4),
                    fee=round(fee_per_contract, 4),
                )
                continue

            # Position sizing: Kelly fraction
            kelly_frac = self._kelly_momentum if signal_type == "momentum" else self._kelly_reversion
            kelly_raw = (net_edge * confidence) / max(1.0 - net_edge, 0.01)
            size_usd = min(kelly_raw * kelly_frac * 130.0, self._max_position_usd)  # $130 approx bankroll
            size_usd = max(0.50, round(size_usd, 2))  # Minimum $0.50

            # Compute contracts and price
            price_cents = int(entry_price * 100)
            if price_cents <= 0 or price_cents >= 100:
                continue
            contracts = max(1, int(size_usd / (price_cents / 100.0)))

            self.logger.info(
                "crypto_signal",
                ticker=ticker,
                signal_type=signal_type,
                side=side,
                change_pct=round(change_pct, 3),
                elapsed_min=round(elapsed_minutes, 1),
                confidence=round(confidence, 3),
                edge=round(edge, 4),
                net_edge=round(net_edge, 4),
                size_usd=size_usd,
                contracts=contracts,
                btc_start=window.btc_price_at_start,
                btc_now=current_btc,
            )

            # Build TradeSignal
            signal = TradeSignal(
                engine=self.name,
                market_id=ticker,
                token_id=ticker,
                side=side,
                confidence=confidence,
                edge=net_edge,
                urgency="immediate",  # 15-min windows = time-sensitive
                metadata={
                    "estimated_prob": our_prob,
                    "market_price": mid,
                    "net_edge": net_edge,
                    "conviction": "medium" if confidence >= 0.55 else "low",
                    "reasoning": (
                        f"Crypto {signal_type}: BTC {change_pct:+.3f}% in {elapsed_minutes:.0f}min "
                        f"(start=${window.btc_price_at_start:.0f}, now=${current_btc:.0f})"
                    ),
                    "question": window.title,
                    "kalshi_ticker": ticker,
                    "kalshi_yes_ask": window.yes_ask,
                    "kalshi_no_ask": 1.0 - window.yes_bid if window.yes_bid > 0 else None,
                    "platform": "kalshi",
                    "strategy": "crypto",
                    "signal_source": "binance_direct",
                    "signal_type": signal_type,
                    "_force_size_usd": size_usd,
                    "volume": window.volume,
                },
            )
            self._pending_signals.append(signal)

            # Mark window as signaled + update cooldown
            window.signaled = True
            window.last_signal_time = now_mono
