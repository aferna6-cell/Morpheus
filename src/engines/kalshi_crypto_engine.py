"""15-minute crypto engine — latency arbitrage on KXBTC15M.

Wave 32 rewrite: Exploits 30-90 second price lag between Coinbase real-time
prices and Kalshi 15-minute BTC markets. NOT predicting — following confirmed
momentum from the exchange. The edge is SPEED, not accuracy.

Strategy (latency_arb mode):
  - Connect to Coinbase websocket for real-time BTC/USD prices
  - Monitor KXBTC15M Kalshi orderbook
  - When BTC confirms directional move (>0.15% in 2 min), trade Kalshi side
  - Hold to settlement (15 min markets auto-settle)

Legacy (llm_prediction mode): Disabled. Was the old momentum/mean-reversion
approach that lost money in Wave 22.

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
    """15-minute crypto engine — latency arbitrage via Coinbase websocket."""

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

        self._mode = crypto_cfg.get("mode", "latency_arb")
        self._scan_interval = float(crypto_cfg.get("scan_interval_seconds", 60))
        self._signal_interval = float(crypto_cfg.get("signal_interval_seconds", 15))
        self._momentum_threshold = float(crypto_cfg.get("momentum_threshold_pct", 0.15))
        self._max_position_usd = float(crypto_cfg.get("max_position_usd", 5.0))
        self._min_confidence = float(crypto_cfg.get("min_confidence", 0.50))
        self._cooldown_seconds = float(crypto_cfg.get("cooldown_per_window_seconds", 120))
        self._maker_fee_coeff = float(crypto_cfg.get("maker_fee_coefficient", 0.0175))
        self._ticker_prefix = crypto_cfg.get("ticker_prefix", "KXBTC15M")

        # Latency arb specific: how many seconds of recent price data to check
        self._confirmation_window_sec = float(crypto_cfg.get("confirmation_window_seconds", 120))
        # Minimum elapsed minutes before signaling (let the window develop)
        self._min_elapsed_min = float(crypto_cfg.get("min_elapsed_minutes", 2))
        # Max elapsed: don't signal too late in the window
        self._max_elapsed_min = float(crypto_cfg.get("max_elapsed_minutes", 10))

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
            mode=self._mode,
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
        """Find active KXBTC15M markets via targeted series_ticker query."""
        try:
            all_markets = await self.kalshi_client.fetch_markets_by_series(
                self._ticker_prefix,
            )
        except Exception as e:
            self.logger.error("crypto_fetch_error", error=str(e))
            return

        now = datetime.now(timezone.utc)
        active_tickers = set()

        for m in all_markets:
            if m.status not in ("open", "active") or not m.close_time:
                continue

            # Must close within 20 minutes (current or next window)
            hours_to_close = (m.close_time - now).total_seconds() / 3600
            if hours_to_close < 0 or hours_to_close > 0.35:  # ~21 min
                continue

            active_tickers.add(m.ticker)

            if m.ticker not in self._windows:
                window_start_ts = m.close_time.timestamp() - 900  # 15 min = 900s

                btc_at_start = self.price_feed.price_at(window_start_ts)
                if btc_at_start is None:
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
                w = self._windows[m.ticker]
                w.yes_bid = m.yes_bid
                w.yes_ask = m.yes_ask
                w.volume = m.volume

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
    # Signal generation — latency arbitrage
    # ------------------------------------------------------------------

    async def _signal_loop(self) -> None:
        """Generate latency arb signals on active windows."""
        while self._running:
            try:
                await self._generate_signals()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error("crypto_signal_error", error=str(e))
            await asyncio.sleep(self._signal_interval)

    async def _generate_signals(self) -> None:
        """Latency arb: check Coinbase price movement vs Kalshi orderbook.

        Key difference from old approach: we're NOT predicting. We're following
        confirmed momentum from the exchange. The edge is that Kalshi prices
        lag real exchange prices by 30-90 seconds.
        """
        if not self.price_feed.is_connected:
            return

        now_ts = time.time()
        now_mono = time.monotonic()

        for ticker, window in list(self._windows.items()):
            if window.signaled:
                continue

            if now_mono - window.last_signal_time < self._cooldown_seconds:
                continue

            if window.btc_price_at_start is None or window.btc_price_at_start <= 0:
                continue

            # Compute elapsed time in window
            elapsed_seconds = now_ts - window.window_start_ts
            elapsed_minutes = elapsed_seconds / 60.0

            if elapsed_minutes < self._min_elapsed_min:
                continue
            if elapsed_minutes > self._max_elapsed_min:
                continue

            current_btc = self.price_feed.price
            if current_btc <= 0:
                continue

            # LATENCY ARB LOGIC:
            # Check confirmed movement over recent confirmation window (default 2 min)
            recent_change = self.price_feed.price_change_pct(self._confirmation_window_sec)
            if recent_change is None:
                continue

            abs_change = abs(recent_change)
            if abs_change < self._momentum_threshold:
                continue

            # Also check overall direction since window start for consistency
            overall_change = (current_btc - window.btc_price_at_start) / window.btc_price_at_start * 100.0

            # Confirmed direction: recent move AND overall direction agree
            if recent_change > 0 and overall_change > 0:
                side = "buy_yes"  # BTC is up → bet on "up" market
            elif recent_change < 0 and overall_change < 0:
                side = "buy_no"  # BTC is down → bet on "down" (buy NO on "up" market)
            else:
                # Recent and overall disagree — skip, direction is unclear
                continue

            # Confidence scales with move magnitude and agreement
            agreement_factor = min(abs(overall_change), abs_change) / max(abs(overall_change), abs_change, 0.01)
            confidence = min(0.78, 0.50 + abs_change * 30 + agreement_factor * 0.05)

            if confidence < self._min_confidence:
                continue

            # Fetch fresh orderbook
            try:
                fresh = await self.kalshi_client.fetch_market(ticker)
                if fresh is not None:
                    window.yes_bid = fresh.yes_bid
                    window.yes_ask = fresh.yes_ask
            except Exception:
                pass  # Use cached bid/ask

            # Compute edge vs market price
            mid = (window.yes_bid + window.yes_ask) / 2
            if mid <= 0.02 or mid >= 0.98:
                continue

            # Our estimated probability based on confirmed momentum
            # Higher move = higher probability of continuation
            if side == "buy_yes":
                our_prob = min(0.85, 0.50 + abs(overall_change) * 0.12)
                entry_price = window.yes_ask
                edge = our_prob - entry_price
            else:
                our_prob = max(0.15, 0.50 - abs(overall_change) * 0.12)
                entry_price = 1.0 - window.yes_bid
                edge = (1.0 - our_prob) - entry_price

            # Maker fee
            fee_per_contract = self._maker_fee_coeff * entry_price * (1.0 - entry_price)
            net_edge = edge - fee_per_contract

            if net_edge <= 0.01:
                self.logger.debug(
                    "crypto_latency_edge_too_low",
                    ticker=ticker,
                    edge=round(edge, 4),
                    net_edge=round(net_edge, 4),
                    fee=round(fee_per_contract, 4),
                )
                continue

            # Size: conservative fixed sizing
            size_usd = min(self._max_position_usd, max(0.50, net_edge * confidence * 50))

            # Compute contracts
            price_cents = int(entry_price * 100)
            if price_cents <= 0 or price_cents >= 100:
                continue
            contracts = max(1, int(size_usd / (price_cents / 100.0)))

            self.logger.info(
                "crypto_latency_arb_signal",
                ticker=ticker,
                side=side,
                recent_change_pct=round(recent_change, 3),
                overall_change_pct=round(overall_change, 3),
                elapsed_min=round(elapsed_minutes, 1),
                confidence=round(confidence, 3),
                edge=round(edge, 4),
                net_edge=round(net_edge, 4),
                size_usd=size_usd,
                contracts=contracts,
                btc_start=window.btc_price_at_start,
                btc_now=current_btc,
            )

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
                        f"Crypto latency arb: BTC {recent_change:+.3f}% in {self._confirmation_window_sec:.0f}s, "
                        f"overall {overall_change:+.3f}% (start=${window.btc_price_at_start:.0f}, "
                        f"now=${current_btc:.0f})"
                    ),
                    "question": window.title,
                    "kalshi_ticker": ticker,
                    "kalshi_yes_ask": window.yes_ask,
                    "kalshi_no_ask": 1.0 - window.yes_bid if window.yes_bid > 0 else None,
                    "platform": "kalshi",
                    "strategy": "crypto_latency",
                    "signal_source": "coinbase_latency_arb",
                    "signal_type": "latency_arb",
                    "_force_size_usd": size_usd,
                    "volume": window.volume,
                },
            )
            self._pending_signals.append(signal)

            window.signaled = True
            window.last_signal_time = now_mono
