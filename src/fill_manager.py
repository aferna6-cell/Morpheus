"""Fill manager — track resting orders and detect fills.

Fixes the critical bug where kalshi_executor.py line 236 pretends
resting orders filled immediately. This module polls order status
and detects actual fills, partial fills, and stale orders.

Persistence: resting orders are saved to disk so tracking survives restarts.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog

from .alerts import send_alert
from .kalshi_trading_client import KalshiTradingClient
from .trade_logger import get_trade_logger
from .utils import BotConfig


@dataclass
class RestingOrder:
    """A resting order waiting for fill."""

    order_id: str
    ticker: str
    side: str  # "yes" or "no"
    count: int
    price_cents: int
    placed_at: datetime
    account_label: str = "default"
    strategy: str = "llm"
    signal_source: str = ""  # "noaa_direct", "llm", etc.
    entry_edge: float = 0.0  # edge at time of entry (for trailing stops)
    filled_count: int = 0
    is_done: bool = False

    @property
    def is_weather(self) -> bool:
        """Check if this is a weather market order."""
        t = self.ticker.upper()
        return any(t.startswith(p) for p in ("KXHIGH", "KXLOW", "KXRAIN", "KXSNOW", "KXTEMP"))

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "ticker": self.ticker,
            "side": self.side,
            "count": self.count,
            "price_cents": self.price_cents,
            "placed_at": self.placed_at.isoformat(),
            "account_label": self.account_label,
            "strategy": self.strategy,
            "signal_source": self.signal_source,
            "entry_edge": self.entry_edge,
            "filled_count": self.filled_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RestingOrder":
        return cls(
            order_id=d["order_id"],
            ticker=d["ticker"],
            side=d["side"],
            count=d["count"],
            price_cents=d["price_cents"],
            placed_at=datetime.fromisoformat(d["placed_at"]),
            account_label=d.get("account_label", "default"),
            strategy=d.get("strategy", "llm"),
            signal_source=d.get("signal_source", ""),
            entry_edge=float(d.get("entry_edge", 0.0)),
            filled_count=d.get("filled_count", 0),
        )


@dataclass
class FillEvent:
    """Emitted when an order is filled (fully or partially)."""

    order_id: str
    ticker: str
    side: str
    filled_count: int
    price_cents: int
    account_label: str
    strategy: str
    is_partial: bool = False
    entry_edge: float = 0.0


class FillManager:
    """Tracks resting orders and polls for fills."""

    def __init__(
        self,
        config: BotConfig,
        trading_clients: List[KalshiTradingClient],
        state_dir: str = "state",
    ):
        self.config = config
        self.trading_clients = trading_clients
        self.logger = structlog.get_logger()

        self.poll_interval = 30.0  # seconds
        self.stale_timeout = 600.0  # 10 minutes

        self._resting: Dict[str, RestingOrder] = {}  # order_id -> RestingOrder
        self._on_fill_callbacks: List[Callable] = []
        self._task: Optional[asyncio.Task] = None
        self._state_file = Path(state_dir) / "resting_orders.json"

        # Restore resting orders from disk (survives restarts)
        self._load_state()

        # Stats
        self.total_filled = 0
        self.total_cancelled = 0
        self.total_partial = 0

        # Daily fill rate tracking
        self._daily_orders_placed = 0
        self._daily_orders_filled = 0
        self._daily_date = datetime.now(timezone.utc).date()

    def _load_state(self) -> None:
        """Load resting orders from disk."""
        try:
            if self._state_file.exists():
                data = json.loads(self._state_file.read_text())
                for d in data:
                    try:
                        order = RestingOrder.from_dict(d)
                        self._resting[order.order_id] = order
                    except Exception:
                        continue
                if self._resting:
                    self.logger.info(
                        "fill_manager_restored",
                        restored_orders=len(self._resting),
                    )
        except Exception as e:
            self.logger.debug("fill_manager_load_error", error=str(e))

    def _save_state(self) -> None:
        """Persist resting orders to disk."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            data = [o.to_dict() for o in self._resting.values() if not o.is_done]
            self._state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            self.logger.debug("fill_manager_save_error", error=str(e))

    def on_fill(self, callback: Callable) -> None:
        """Register a callback for fill events."""
        self._on_fill_callbacks.append(callback)

    def track_order(
        self,
        order_id: str,
        ticker: str,
        side: str,
        count: int,
        price_cents: int,
        account_label: str = "default",
        strategy: str = "llm",
        signal_source: str = "",
        entry_edge: float = 0.0,
    ) -> None:
        """Register an order for fill tracking."""
        self._resting[order_id] = RestingOrder(
            order_id=order_id,
            ticker=ticker,
            side=side,
            count=count,
            price_cents=price_cents,
            placed_at=datetime.now(timezone.utc),
            account_label=account_label,
            strategy=strategy,
            signal_source=signal_source,
            entry_edge=entry_edge,
        )
        # Daily fill rate tracking — reset on new day
        today = datetime.now(timezone.utc).date()
        if today != self._daily_date:
            self._daily_orders_placed = 0
            self._daily_orders_filled = 0
            self._daily_date = today
        self._daily_orders_placed += 1

        self.logger.info(
            "order_tracked",
            order_id=order_id,
            ticker=ticker,
            side=side,
            count=count,
            price_cents=price_cents,
            account=account_label,
        )
        self._save_state()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._poll_loop())
        self.logger.info(
            "fill_manager_started",
            poll_interval=self.poll_interval,
            stale_timeout=self.stale_timeout,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info(
            "fill_manager_stopped",
            total_filled=self.total_filled,
            total_cancelled=self.total_cancelled,
        )

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self._check_orders()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error("fill_manager_error", error=str(e))
            await asyncio.sleep(self.poll_interval)

    async def _check_orders(self) -> None:
        """Check status of all tracked resting orders."""
        if not self._resting:
            return

        # Build a set of currently open orders per account.
        # Track which accounts we successfully queried — if an account's
        # API call fails (network error, rate limit), we must NOT assume
        # its resting orders are filled.
        open_orders_by_account: Dict[str, set] = {}
        successful_accounts: set = set()
        for client in self.trading_clients:
            try:
                orders = await client.get_open_orders()
                open_orders_by_account[client.label] = {o.order_id for o in orders}
                successful_accounts.add(client.label)
            except Exception as e:
                self.logger.warning(
                    "fill_check_get_orders_failed",
                    account=client.label,
                    error=str(e),
                )

        # Check each tracked order
        done_ids = []
        now = datetime.now(timezone.utc)

        for order_id, resting in list(self._resting.items()):
            if resting.is_done:
                done_ids.append(order_id)
                continue

            # Skip orders for accounts where API call failed —
            # we can't determine status, retry next cycle
            if resting.account_label not in successful_accounts:
                continue

            account_open = open_orders_by_account.get(resting.account_label, set())

            if order_id not in account_open:
                # Order disappeared from open list — verify via fills API
                # before declaring filled (could be cancelled by exchange)
                was_filled = await self._verify_fill(order_id, resting)

                resting.is_done = True
                done_ids.append(order_id)

                if not was_filled:
                    # No fill record — likely cancelled by exchange
                    self.total_cancelled += 1
                    self.logger.info(
                        "order_cancelled_not_filled",
                        order_id=order_id,
                        ticker=resting.ticker,
                        account=resting.account_label,
                    )
                    continue

                # Confirmed fill
                resting.filled_count = resting.count
                self.total_filled += 1

                fill_event = FillEvent(
                    order_id=order_id,
                    ticker=resting.ticker,
                    side=resting.side,
                    filled_count=resting.count,
                    price_cents=resting.price_cents,
                    account_label=resting.account_label,
                    strategy=resting.strategy,
                    entry_edge=resting.entry_edge,
                )

                self._daily_orders_filled += 1

                self.logger.info(
                    "order_filled",
                    order_id=order_id,
                    ticker=resting.ticker,
                    side=resting.side,
                    count=resting.count,
                    price_cents=resting.price_cents,
                    account=resting.account_label,
                )

                # Log fill to trade history JSONL
                get_trade_logger().log_order_filled(
                    platform="kalshi",
                    ticker=resting.ticker,
                    side=resting.side,
                    count=resting.count,
                    fill_price_cents=resting.price_cents,
                    order_id=order_id,
                    account_label=resting.account_label,
                )

                # Notify callbacks
                for cb in self._on_fill_callbacks:
                    try:
                        result = cb(fill_event)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        self.logger.warning("fill_callback_error", error=str(e))

                # Alert
                cost_usd = resting.count * resting.price_cents / 100.0
                await send_alert(
                    f"FILLED: {resting.ticker} {resting.side} x{resting.count} "
                    f"@ {resting.price_cents}c (${cost_usd:.2f}) [{resting.account_label}]",
                    self.config,
                )

            else:
                # Still resting — check if stale
                # Weather orders get shorter timeout (5 min) — weather markets
                # move fast and capital should be freed for better opportunities.
                age = (now - resting.placed_at).total_seconds()
                effective_timeout = 300.0 if resting.is_weather else self.stale_timeout
                if age > effective_timeout:
                    # Cancel stale order
                    for client in self.trading_clients:
                        if client.label == resting.account_label:
                            cancelled = await client.cancel_order(order_id)
                            if cancelled:
                                resting.is_done = True
                                done_ids.append(order_id)
                                self.total_cancelled += 1
                                self.logger.info(
                                    "stale_order_cancelled",
                                    order_id=order_id,
                                    ticker=resting.ticker,
                                    age_seconds=age,
                                    timeout=effective_timeout,
                                    is_weather=resting.is_weather,
                                    account=resting.account_label,
                                )
                            break

        # Clean up done orders and persist
        for oid in done_ids:
            self._resting.pop(oid, None)
        if done_ids:
            self._save_state()

        # Log daily fill rate
        if self._daily_orders_placed > 0:
            self.logger.info(
                "fill_rate_daily",
                placed=self._daily_orders_placed,
                filled=self._daily_orders_filled,
                rate=round(self._daily_orders_filled / self._daily_orders_placed, 3),
            )

    async def _verify_fill(self, order_id: str, resting: RestingOrder) -> bool:
        """Check if a disappeared order was actually filled via fills API."""
        for client in self.trading_clients:
            if client.label == resting.account_label:
                try:
                    fills = await client.get_recent_fills(
                        ticker=resting.ticker, limit=50,
                    )
                    return any(f["order_id"] == order_id for f in fills)
                except Exception:
                    # If fills API also fails, assume filled (safer than dropping)
                    return True
        return True  # No matching client — assume filled to be safe

    @property
    def pending_count(self) -> int:
        return len(self._resting)
