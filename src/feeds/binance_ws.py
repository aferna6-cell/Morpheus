"""Real-time BTC price feed via Coinbase WebSocket (primary) or REST polling (fallback).

Maintains a rolling buffer of (timestamp, price) ticks so consumers can query
current price, price-at-time, and percentage change over any interval.

Primary: Coinbase Exchange WebSocket (US-friendly, no geo-restrictions)
Fallback: Binance US REST API polling every 2s (if WebSocket fails)

Usage:
    feed = BinancePriceFeed()
    await feed.start()
    print(feed.price)                 # current BTC/USD
    print(feed.price_change_pct(300)) # change over last 5 min
    await feed.stop()
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Optional, Tuple

import structlog

logger = structlog.get_logger()

# How many seconds of history to keep (20 minutes)
_BUFFER_SECONDS = 1200

# Coinbase Exchange WebSocket (US-friendly, free, no API key)
_COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"
_COINBASE_SUBSCRIBE = json.dumps({
    "type": "subscribe",
    "channels": [{"name": "ticker", "product_ids": ["BTC-USD"]}],
})

# Binance US REST fallback
_BINANCE_US_REST = "https://api.binance.us/api/v3/ticker/price"


class BinancePriceFeed:
    """Real-time BTC/USD price via Coinbase WebSocket + Binance US REST fallback."""

    def __init__(
        self,
        ws_url: str = _COINBASE_WS_URL,
        reconnect_delay: float = 5.0,
    ):
        self._ws_url = ws_url
        self._reconnect_delay = reconnect_delay

        # Rolling buffer: (monotonic_ts, unix_ts, price)
        self._ticks: deque[Tuple[float, float, float]] = deque()
        self._current_price: float = 0.0
        self._last_update: float = 0.0  # monotonic

        self._ws_task: Optional[asyncio.Task] = None
        self._running = False
        self._connected = False
        self._ws_failures = 0  # Track consecutive WS failures

    async def start(self) -> None:
        """Start the price feed in background."""
        self._running = True
        self._ws_task = asyncio.create_task(self._run_forever())
        logger.info("btc_price_feed_starting", ws_url=self._ws_url)

    async def stop(self) -> None:
        """Disconnect cleanly."""
        self._running = False
        self._connected = False
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        logger.info("btc_price_feed_stopped")

    @property
    def price(self) -> float:
        """Current BTC/USD price (0.0 if not yet connected)."""
        return self._current_price

    @property
    def is_connected(self) -> bool:
        """Whether the feed is alive and recent data received."""
        if not self._connected:
            return False
        # Consider stale if no update in 30 seconds
        return (time.monotonic() - self._last_update) < 30.0

    def price_at(self, unix_ts: float) -> Optional[float]:
        """Find closest price to a given Unix timestamp.

        Returns None if no data near that timestamp.
        """
        if not self._ticks:
            return None

        best_price = None
        best_diff = float("inf")
        for _, ts, px in self._ticks:
            diff = abs(ts - unix_ts)
            if diff < best_diff:
                best_diff = diff
                best_price = px

        # Only return if within 60 seconds of requested time
        if best_diff > 60.0:
            return None
        return best_price

    def price_change_pct(self, seconds_ago: float) -> Optional[float]:
        """Percentage change from N seconds ago to now.

        Returns None if insufficient data.
        """
        if self._current_price <= 0 or not self._ticks:
            return None

        target_mono = time.monotonic() - seconds_ago
        best_price = None
        best_diff = float("inf")
        for mono, _, px in self._ticks:
            diff = abs(mono - target_mono)
            if diff < best_diff:
                best_diff = diff
                best_price = px

        if best_price is None or best_price <= 0:
            return None
        # Must be within 30s of target
        if best_diff > 30.0:
            return None

        return (self._current_price - best_price) / best_price * 100.0

    def _record_tick(self, price: float, unix_ts: Optional[float] = None) -> None:
        """Record a price tick into the rolling buffer."""
        now_mono = time.monotonic()
        if unix_ts is None:
            unix_ts = time.time()

        self._current_price = price
        self._last_update = now_mono
        self._connected = True

        # Thin: keep ~1 tick per second
        if not self._ticks or now_mono - self._ticks[-1][0] >= 1.0:
            self._ticks.append((now_mono, unix_ts, price))

        # Prune old ticks
        cutoff = now_mono - _BUFFER_SECONDS
        while self._ticks and self._ticks[0][0] < cutoff:
            self._ticks.popleft()

    async def _run_forever(self) -> None:
        """Reconnect loop with WebSocket primary + REST fallback."""
        while self._running:
            # After 3 consecutive WS failures, switch to REST polling
            if self._ws_failures >= 3:
                logger.info(
                    "btc_feed_switching_to_rest",
                    ws_failures=self._ws_failures,
                )
                await self._rest_poll_loop()
                # If REST loop exits, reset and try WS again
                self._ws_failures = 0
                continue

            try:
                await self._coinbase_ws_loop()
                self._ws_failures = 0  # Reset on clean exit
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._connected = False
                self._ws_failures += 1
                logger.warning(
                    "btc_ws_error",
                    error=str(e),
                    failures=self._ws_failures,
                    reconnect_in=self._reconnect_delay,
                )

            if self._running:
                await asyncio.sleep(self._reconnect_delay)

    async def _coinbase_ws_loop(self) -> None:
        """Coinbase Exchange WebSocket — primary feed."""
        try:
            import websockets
        except ImportError:
            logger.warning("websockets_not_installed_using_rest_fallback")
            self._ws_failures = 999  # Force REST fallback
            return

        async with websockets.connect(  # type: ignore[attr-defined]
            self._ws_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            # Subscribe to BTC-USD ticker
            await ws.send(_COINBASE_SUBSCRIBE)
            self._connected = True
            self._ws_failures = 0
            logger.info("coinbase_ws_connected")

            async for msg in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(msg)
                    if data.get("type") != "ticker":
                        continue

                    price = float(data["price"])
                    # Coinbase sends ISO timestamp, convert to Unix
                    ts_str = data.get("time", "")
                    if ts_str:
                        from datetime import datetime, timezone
                        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        unix_ts = dt.timestamp()
                    else:
                        unix_ts = time.time()

                    self._record_tick(price, unix_ts)
                except (KeyError, ValueError):
                    continue

    async def _rest_poll_loop(self) -> None:
        """Binance US REST API polling fallback (every 2s)."""
        import httpx

        logger.info("btc_rest_poll_started", url=_BINANCE_US_REST)
        async with httpx.AsyncClient(timeout=10.0) as client:
            while self._running:
                try:
                    resp = await client.get(
                        _BINANCE_US_REST,
                        params={"symbol": "BTCUSDT"},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        price = float(data["price"])
                        self._record_tick(price)
                    elif resp.status_code == 429:
                        await asyncio.sleep(10.0)  # Rate limited, back off
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
                await asyncio.sleep(2.0)
