"""Real-time BTC price feed via Binance WebSocket.

Maintains a rolling buffer of (timestamp, price) ticks so consumers can query
current price, price-at-time, and percentage change over any interval.

Usage:
    feed = BinancePriceFeed()
    await feed.start()
    print(feed.price)                 # current BTC/USDT
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


class BinancePriceFeed:
    """Real-time BTC/USDT price via Binance trade stream."""

    def __init__(
        self,
        ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade",
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

    async def start(self) -> None:
        """Start the WebSocket connection in background."""
        self._running = True
        self._ws_task = asyncio.create_task(self._run_forever())
        logger.info("binance_feed_starting", url=self._ws_url)

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
        logger.info("binance_feed_stopped")

    @property
    def price(self) -> float:
        """Current BTC/USDT price (0.0 if not yet connected)."""
        return self._current_price

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket is alive and recent data received."""
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

        # Binary search would be ideal but deque is small enough for linear
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
        # Find tick closest to target time
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

    async def _run_forever(self) -> None:
        """Reconnect loop."""
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._connected = False
                logger.warning(
                    "binance_ws_error",
                    error=str(e),
                    reconnect_in=self._reconnect_delay,
                )
            if self._running:
                await asyncio.sleep(self._reconnect_delay)

    async def _connect_and_listen(self) -> None:
        """Single WebSocket connection lifecycle."""
        try:
            import websockets
        except ImportError:
            # Fallback: use httpx polling if websockets not installed
            logger.warning("websockets_not_installed_using_poll_fallback")
            await self._poll_fallback()
            return

        async with websockets.connect(  # type: ignore[attr-defined]
            self._ws_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self._connected = True
            logger.info("binance_ws_connected")

            async for msg in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(msg)
                    price = float(data["p"])  # Trade price
                    trade_time = float(data["T"]) / 1000.0  # Unix seconds

                    now_mono = time.monotonic()
                    self._current_price = price
                    self._last_update = now_mono

                    # Append to buffer (thin: keep ~1 tick per second)
                    if (
                        not self._ticks
                        or now_mono - self._ticks[-1][0] >= 1.0
                    ):
                        self._ticks.append((now_mono, trade_time, price))

                    # Prune old ticks
                    cutoff = now_mono - _BUFFER_SECONDS
                    while self._ticks and self._ticks[0][0] < cutoff:
                        self._ticks.popleft()
                except (KeyError, ValueError):
                    continue  # Malformed message, skip

    async def _poll_fallback(self) -> None:
        """HTTP polling fallback if websockets library unavailable.

        Polls Binance REST API every 2 seconds for current price.
        Less ideal but functional.
        """
        import httpx

        logger.info("binance_poll_fallback_started")
        async with httpx.AsyncClient(timeout=10.0) as client:
            while self._running:
                try:
                    resp = await client.get(
                        "https://api.binance.com/api/v3/ticker/price",
                        params={"symbol": "BTCUSDT"},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        price = float(data["price"])
                        now_mono = time.monotonic()
                        now_unix = time.time()

                        self._current_price = price
                        self._last_update = now_mono
                        self._connected = True

                        self._ticks.append((now_mono, now_unix, price))

                        # Prune
                        cutoff = now_mono - _BUFFER_SECONDS
                        while self._ticks and self._ticks[0][0] < cutoff:
                            self._ticks.popleft()
                except Exception:
                    pass
                await asyncio.sleep(2.0)
