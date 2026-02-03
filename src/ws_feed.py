"""Real-time WebSocket price feed for Polymarket CLOB.

Connects to ``wss://ws-subscriptions-clob.polymarket.com/ws/market``,
subscribes to the *market* channel for active token IDs, and maintains an
in-memory price book (best bid, best ask, last trade) per asset.

Features:
- Auto-reconnect with exponential backoff
- Thread-safe price access via ``get_price_book()``
- Async event callbacks so engines can react to price updates in real time

Requires the ``websockets`` library (``pip install websockets``).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import structlog

# websockets is not in the default Poetry deps — install manually or add to
# pyproject.toml:  websockets = "^13.0"
try:
    import websockets
    import websockets.asyncio.client as ws_client
except ImportError:  # pragma: no cover
    websockets = None  # type: ignore
    ws_client = None  # type: ignore


logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PriceTick:
    """A single price observation for a token."""

    best_bid: float
    best_ask: float
    last_price: Optional[float]
    timestamp: datetime


@dataclass
class PriceBookEntry:
    """Current price state for a single asset/token."""

    asset_id: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    last_price: Optional[float] = None
    last_trade_size: Optional[float] = None
    last_trade_side: Optional[str] = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def midpoint(self) -> Optional[float]:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2.0
        return self.last_price


# Callback type: async callable receiving (asset_id, PriceBookEntry)
PriceCallback = Callable[[str, PriceBookEntry], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# WebSocket Feed
# ---------------------------------------------------------------------------

class WebSocketFeed:
    """Manages a persistent WebSocket connection to the Polymarket market channel."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._endpoint: str = config.get(
            "endpoint",
            "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        )
        self._reconnect_delay: float = float(config.get("reconnect_delay_seconds", 5))
        self._max_reconnect_delay: float = float(config.get("max_reconnect_delay_seconds", 300))
        self._heartbeat_interval: float = float(config.get("heartbeat_interval_seconds", 30))

        # Internal state — guarded by ``_lock`` for thread-safe reads
        self._lock = threading.Lock()
        self._price_book: Dict[str, PriceBookEntry] = {}

        # Subscriptions
        self._subscribed_assets: Set[str] = set()

        # Async event listeners
        self._callbacks: List[PriceCallback] = []

        # Control
        self._running = False
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._ws: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_price_update(self, callback: PriceCallback) -> None:
        """Register an async callback invoked on every price update."""
        self._callbacks.append(callback)

    def get_price_book(self) -> Dict[str, PriceBookEntry]:
        """Return a *snapshot* of the current price book (thread-safe)."""
        with self._lock:
            return dict(self._price_book)

    def get_price(self, asset_id: str) -> Optional[PriceBookEntry]:
        """Return price info for a single asset (thread-safe)."""
        with self._lock:
            return self._price_book.get(asset_id)

    async def subscribe(self, asset_ids: List[str]) -> None:
        """Subscribe to price updates for the given token IDs.

        Can be called before or after ``start()`` — queued subscriptions are
        sent as soon as the WebSocket connection is established.
        """
        new_ids = set(asset_ids) - self._subscribed_assets
        if not new_ids:
            return
        self._subscribed_assets.update(new_ids)
        logger.info("ws_subscribe_queued", count=len(new_ids))

        # If we already have an open connection, send subscription immediately
        if self._ws is not None:
            await self._send_subscribe(list(new_ids))

    async def unsubscribe(self, asset_ids: List[str]) -> None:
        """Unsubscribe from price updates for the given token IDs."""
        to_remove = set(asset_ids) & self._subscribed_assets
        if not to_remove:
            return
        self._subscribed_assets -= to_remove
        if self._ws is not None:
            try:
                msg = json.dumps({
                    "assets_ids": list(to_remove),
                    "type": "unsubscribe",
                })
                await self._ws.send(msg)
                logger.info("ws_unsubscribed", count=len(to_remove))
            except Exception as exc:
                logger.warning("ws_unsubscribe_error", error=str(exc))

    async def start(self) -> None:
        """Start the WebSocket feed (spawns a background task)."""
        if websockets is None:
            raise ImportError(
                "The 'websockets' package is required for the real-time feed. "
                "Install it with: pip install websockets"
            )
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("ws_feed_started", endpoint=self._endpoint)

    async def stop(self) -> None:
        """Gracefully stop the feed."""
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ws_feed_stopped")

    # ------------------------------------------------------------------
    # Connection loop (auto-reconnect with exponential backoff)
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Outer loop: reconnect on failure with exponential backoff."""
        delay = self._reconnect_delay

        while self._running:
            try:
                await self._connect_and_listen()
                # Clean disconnect — reset delay
                delay = self._reconnect_delay
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "ws_connection_lost",
                    error=str(exc),
                    reconnect_in=delay,
                )

            if not self._running:
                break

            await asyncio.sleep(delay)
            delay = min(delay * 2, self._max_reconnect_delay)

    async def _connect_and_listen(self) -> None:
        """Single connection lifecycle: connect → subscribe → listen."""
        logger.info("ws_connecting", endpoint=self._endpoint)

        async with ws_client.connect(
            self._endpoint,
            ping_interval=self._heartbeat_interval,
            ping_timeout=self._heartbeat_interval,
            close_timeout=10,
        ) as ws:
            self._ws = ws
            logger.info("ws_connected")

            # Subscribe to all tracked assets
            if self._subscribed_assets:
                await self._send_subscribe(list(self._subscribed_assets))

            async for raw_msg in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(raw_msg)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.warning("ws_invalid_json", msg=str(raw_msg)[:200])
                except Exception as exc:
                    logger.warning("ws_handle_error", error=str(exc))

        self._ws = None

    # ------------------------------------------------------------------
    # Subscription helpers
    # ------------------------------------------------------------------

    async def _send_subscribe(self, asset_ids: List[str]) -> None:
        """Send a subscription message for the given asset IDs."""
        if not self._ws or not asset_ids:
            return
        msg = json.dumps({
            "assets_ids": asset_ids,
            "type": "market",
            "custom_feature_enabled": True,
        })
        await self._ws.send(msg)
        logger.info("ws_subscribed", count=len(asset_ids))

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    async def _handle_message(self, data: Dict[str, Any]) -> None:
        event_type = data.get("event_type")

        if event_type == "book":
            await self._handle_book(data)
        elif event_type == "price_change":
            await self._handle_price_change(data)
        elif event_type == "last_trade_price":
            await self._handle_last_trade(data)
        elif event_type == "best_bid_ask":
            await self._handle_best_bid_ask(data)
        # tick_size_change and new_market are informational; ignore for now

    async def _handle_book(self, data: Dict[str, Any]) -> None:
        """Full orderbook snapshot — extract best bid/ask."""
        asset_id = data.get("asset_id", "")
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 0.0

        await self._update_entry(asset_id, best_bid=best_bid, best_ask=best_ask)

    async def _handle_price_change(self, data: Dict[str, Any]) -> None:
        """Incremental price change — update best bid/ask from change objects."""
        for change in data.get("price_changes", []):
            asset_id = change.get("asset_id", "")
            best_bid = _safe_float(change.get("best_bid"))
            best_ask = _safe_float(change.get("best_ask"))

            if asset_id and (best_bid is not None or best_ask is not None):
                await self._update_entry(
                    asset_id,
                    best_bid=best_bid,
                    best_ask=best_ask,
                )

    async def _handle_last_trade(self, data: Dict[str, Any]) -> None:
        """Trade event — update last price and trade info."""
        asset_id = data.get("asset_id", "")
        price = _safe_float(data.get("price"))
        size = _safe_float(data.get("size"))
        side = data.get("side")

        if asset_id and price is not None:
            await self._update_entry(
                asset_id,
                last_price=price,
                last_trade_size=size,
                last_trade_side=side,
            )

    async def _handle_best_bid_ask(self, data: Dict[str, Any]) -> None:
        """Custom feature: explicit best bid/ask update."""
        asset_id = data.get("asset_id", "")
        best_bid = _safe_float(data.get("best_bid"))
        best_ask = _safe_float(data.get("best_ask"))

        if asset_id:
            await self._update_entry(asset_id, best_bid=best_bid, best_ask=best_ask)

    # ------------------------------------------------------------------
    # Internal state management
    # ------------------------------------------------------------------

    async def _update_entry(
        self,
        asset_id: str,
        *,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None,
        last_price: Optional[float] = None,
        last_trade_size: Optional[float] = None,
        last_trade_side: Optional[str] = None,
    ) -> None:
        """Merge an update into the price book and notify listeners."""
        now = datetime.now(timezone.utc)

        with self._lock:
            entry = self._price_book.get(asset_id)
            if entry is None:
                entry = PriceBookEntry(asset_id=asset_id)
                self._price_book[asset_id] = entry

            if best_bid is not None:
                entry.best_bid = best_bid
            if best_ask is not None:
                entry.best_ask = best_ask
            if last_price is not None:
                entry.last_price = last_price
            if last_trade_size is not None:
                entry.last_trade_size = last_trade_size
            if last_trade_side is not None:
                entry.last_trade_side = last_trade_side
            entry.updated_at = now

            # Snapshot for callbacks (outside lock)
            snapshot = PriceBookEntry(
                asset_id=entry.asset_id,
                best_bid=entry.best_bid,
                best_ask=entry.best_ask,
                last_price=entry.last_price,
                last_trade_size=entry.last_trade_size,
                last_trade_side=entry.last_trade_side,
                updated_at=entry.updated_at,
            )

        # Fire callbacks (outside lock to avoid deadlocks)
        for cb in self._callbacks:
            try:
                await cb(asset_id, snapshot)
            except Exception as exc:
                logger.warning("ws_callback_error", error=str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """Parse a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
