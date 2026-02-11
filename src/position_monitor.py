"""Position monitor — enforce stop-loss, take-profit, and time-based exits.

The critical missing piece in Morpheus v1: risk.py has should_close_position()
but it was NEVER called from anywhere. This module runs as a background task
and actively monitors all positions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

import httpx

from .alerts import send_alert
from .kalshi_trading_client import KalshiTradingClient, KalshiPosition
from .risk import RiskManager
from .utils import BotConfig


@dataclass
class TrackedPosition:
    """Enriched position with entry metadata for P&L tracking."""

    ticker: str
    side: str  # "yes" or "no"
    count: int
    entry_price_cents: int
    entry_time: datetime
    strategy: str = "llm"
    order_id: str = ""
    account_label: str = "default"


class PositionMonitor:
    """Actively monitors positions and triggers exits."""

    def __init__(
        self,
        config: BotConfig,
        trading_clients: List[KalshiTradingClient],
        risk_manager: RiskManager,
    ):
        self.config = config
        self.trading_clients = trading_clients
        self.risk_manager = risk_manager
        self.logger = structlog.get_logger()

        risk_cfg = config.risk
        self.check_interval = float(risk_cfg.get("position_check_interval_minutes", 5)) * 60
        self.stop_loss_pct = float(risk_cfg.get("stop_loss_pct", 0.35))
        self.take_profit_pct = float(risk_cfg.get("take_profit_pct", 0.30))
        self.max_hold_hours = float(risk_cfg.get("max_position_hold_hours", 24))

        # Track positions with entry metadata
        self._tracked: Dict[str, TrackedPosition] = {}
        self._task: Optional[asyncio.Task] = None

        # Exit retry tracking: key -> (attempt_count, last_attempt_time)
        self._exit_retries: Dict[str, tuple[int, float]] = {}
        self._max_exit_retries = 3

        # Stale market cache: ticker -> (check_time, is_closed)
        self._market_status_cache: Dict[str, tuple[float, bool]] = {}
        self._market_status_ttl = 600.0  # 10 min cache

        # Kalshi API base URL for market status checks
        kalshi_cfg = getattr(config, "kalshi", None) or {}
        self._kalshi_base_url = (
            kalshi_cfg.get("base_url", "https://api.elections.kalshi.com/trade-api/v2")
            if isinstance(kalshi_cfg, dict)
            else "https://api.elections.kalshi.com/trade-api/v2"
        )

    def track_position(
        self,
        ticker: str,
        side: str,
        count: int,
        entry_price_cents: int,
        strategy: str = "llm",
        order_id: str = "",
        account_label: str = "default",
    ) -> None:
        """Register a new position for monitoring."""
        key = f"{account_label}:{ticker}"
        self._tracked[key] = TrackedPosition(
            ticker=ticker,
            side=side,
            count=count,
            entry_price_cents=entry_price_cents,
            entry_time=datetime.now(timezone.utc),
            strategy=strategy,
            order_id=order_id,
            account_label=account_label,
        )
        self.logger.info(
            "position_tracked",
            ticker=ticker,
            side=side,
            count=count,
            entry_price_cents=entry_price_cents,
            account=account_label,
        )

    async def start(self) -> None:
        self._task = asyncio.create_task(self._monitor_loop())
        self.logger.info(
            "position_monitor_started",
            check_interval=self.check_interval,
            stop_loss=self.stop_loss_pct,
            take_profit=self.take_profit_pct,
            max_hold_hours=self.max_hold_hours,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("position_monitor_stopped")

    async def liquidate_all(self) -> int:
        """Sell all existing positions across all accounts. Returns count liquidated."""
        total_liquidated = 0

        for client in self.trading_clients:
            try:
                positions = await client.get_positions()
                for pos in positions:
                    if pos.count == 0:
                        continue

                    # Determine sell side: if we're long YES (count > 0), sell YES
                    # On Kalshi, selling is done via action="sell" or buying opposite side
                    side = "yes" if pos.count > 0 else "no"
                    count = abs(pos.count)

                    # Place market-like order (price=1 for selling YES, price=99 for selling NO)
                    # Use aggressive limit price to simulate market order
                    sell_price = 1 if side == "yes" else 99

                    self.logger.info(
                        "liquidating_position",
                        ticker=pos.ticker,
                        side=side,
                        count=count,
                        account=client.label,
                    )

                    result = await client.place_order(
                        ticker=pos.ticker,
                        side=side,
                        count=count,
                        price_cents=sell_price,
                        order_type="limit",
                        is_exit=True,
                    )

                    if result:
                        total_liquidated += 1
                        await send_alert(
                            f"LIQUIDATED: {pos.ticker} {side} x{count} @ {sell_price}c "
                            f"[{client.label}]",
                            self.config,
                        )

            except Exception as e:
                self.logger.error(
                    "liquidation_error",
                    account=client.label,
                    error=str(e),
                )

        if total_liquidated > 0:
            await send_alert(
                f"Startup liquidation complete: {total_liquidated} positions sold",
                self.config,
            )
        else:
            self.logger.info("no_positions_to_liquidate")

        return total_liquidated

    async def _monitor_loop(self) -> None:
        while True:
            try:
                await self._check_all_positions()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error("position_monitor_error", error=str(e))
            await asyncio.sleep(self.check_interval)

    async def _check_all_positions(self) -> None:
        """Check all positions across all accounts for exit conditions."""
        # Auto-resume halted clients if balance has recovered
        for client in self.trading_clients:
            if client.is_halted:
                await client.check_and_resume()

        for client in self.trading_clients:
            try:
                positions = await client.get_positions()
                for pos in positions:
                    if pos.count == 0:
                        continue
                    await self._check_position(client, pos)
            except Exception as e:
                self.logger.error(
                    "position_check_error",
                    account=client.label,
                    error=str(e),
                )

    async def _is_market_closed(self, ticker: str) -> bool:
        """Check if a market has already closed/settled via Kalshi public API.

        Caches results for 10 minutes to avoid hammering the API.
        """
        import time as _time
        now = _time.monotonic()

        cached = self._market_status_cache.get(ticker)
        if cached:
            check_time, is_closed = cached
            if now - check_time < self._market_status_ttl:
                return is_closed

        try:
            async with httpx.AsyncClient(
                base_url=self._kalshi_base_url,
                timeout=10.0,
                headers={"Accept": "application/json"},
            ) as client:
                r = await client.get(f"/markets/{ticker}")
                if r.status_code == 200:
                    data = r.json()
                    market_data = data.get("market", data)
                    status = (market_data.get("status") or "").lower()
                    is_closed = status in ("closed", "settled", "finalized")
                    self._market_status_cache[ticker] = (now, is_closed)
                    return is_closed
                elif r.status_code == 404:
                    # Market doesn't exist anymore — treat as closed
                    self._market_status_cache[ticker] = (now, True)
                    return True
        except Exception as e:
            self.logger.debug("market_status_check_error", ticker=ticker, error=str(e))

        return False

    async def _check_position(
        self,
        client: KalshiTradingClient,
        pos: KalshiPosition,
    ) -> None:
        """Check a single position for exit conditions."""
        key = f"{client.label}:{pos.ticker}"
        tracked = self._tracked.get(key)

        if not tracked:
            # Position exists but we're not tracking it (pre-existing or from restart)
            # Track it now with estimated entry
            self.logger.info(
                "untracked_position_found",
                ticker=pos.ticker,
                count=pos.count,
                account=client.label,
            )
            # Use market_exposure as rough entry estimate
            entry_est = int(pos.market_exposure / abs(pos.count) * 100) if pos.count != 0 else 50
            self.track_position(
                ticker=pos.ticker,
                side="yes" if pos.count > 0 else "no",
                count=abs(pos.count),
                entry_price_cents=entry_est,
                account_label=client.label,
            )
            tracked = self._tracked[key]

        # Check if market has already closed — stale position (dead capital)
        is_closed = await self._is_market_closed(pos.ticker)
        if is_closed:
            self.logger.warning(
                "stale_position_detected",
                ticker=pos.ticker,
                count=pos.count,
                account=client.label,
                msg="Market closed/settled but position still open — attempting exit",
            )
            await self._exit_position(client, pos, tracked, "Market closed/settled (stale position)")
            return

        # Get current market price for P&L calculation
        # We don't have direct price access here, so use exposure-based estimate
        # This is approximate — the fill manager will have more precise data
        entry_price = tracked.entry_price_cents / 100.0
        current_price = pos.market_exposure / abs(pos.count) if pos.count != 0 else entry_price

        should_exit, reason = self.risk_manager.should_close_position(
            entry_price=entry_price,
            current_price=current_price,
            entry_time=tracked.entry_time,
            position_amount=pos.market_exposure,
        )

        if should_exit:
            await self._exit_position(client, pos, tracked, reason)

    async def _exit_position(
        self,
        client: KalshiTradingClient,
        pos: KalshiPosition,
        tracked: TrackedPosition,
        reason: str,
    ) -> None:
        """Exit a position by selling. Max 3 retries with exponential backoff."""
        import time as _time

        key = f"{client.label}:{pos.ticker}"

        # Check retry count — stop retrying after max attempts
        retry_count, last_attempt = self._exit_retries.get(key, (0, 0.0))
        if retry_count >= self._max_exit_retries:
            self.logger.warning(
                "exit_retry_exhausted",
                ticker=pos.ticker,
                account=client.label,
                attempts=retry_count,
                reason=reason,
            )
            return

        # Exponential backoff: wait 5min, 15min, 45min between retries
        if retry_count > 0:
            backoff_seconds = 300 * (3 ** (retry_count - 1))  # 300, 900, 2700
            elapsed = _time.monotonic() - last_attempt
            if elapsed < backoff_seconds:
                return  # Not enough time has passed since last attempt

        side = "yes" if pos.count > 0 else "no"
        count = abs(pos.count)

        # Aggressive limit price to ensure fill
        sell_price = 1 if side == "yes" else 99

        self.logger.info(
            "exiting_position",
            ticker=pos.ticker,
            side=side,
            count=count,
            reason=reason,
            account=client.label,
            attempt=retry_count + 1,
        )

        try:
            result = await client.place_order(
                ticker=pos.ticker,
                side=side,
                count=count,
                price_cents=sell_price,
                order_type="limit",
                is_exit=True,
            )
        except Exception as e:
            error_str = str(e).lower()
            # Detect insufficient_balance — stop retrying immediately
            if "insufficient" in error_str or "balance" in error_str:
                self.logger.error(
                    "exit_insufficient_balance",
                    ticker=pos.ticker,
                    account=client.label,
                    error=str(e),
                )
                # Max out retry counter to prevent further attempts
                self._exit_retries[key] = (self._max_exit_retries, _time.monotonic())
                return
            # Other errors — record retry attempt
            self._exit_retries[key] = (retry_count + 1, _time.monotonic())
            self.logger.warning(
                "exit_order_failed",
                ticker=pos.ticker,
                account=client.label,
                attempt=retry_count + 1,
                error=str(e),
            )
            return

        if result:
            # Calculate approximate P&L
            entry_cost = tracked.entry_price_cents * tracked.count / 100.0
            exit_cost = pos.market_exposure
            pnl = exit_cost - entry_cost

            self._tracked.pop(key, None)
            self._exit_retries.pop(key, None)  # Clear retry tracker on success

            # Update daily P&L
            self.risk_manager.update_daily_pnl(pnl)

            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            emoji = "PROFIT" if pnl >= 0 else "LOSS"
            await send_alert(
                f"{emoji}: {pos.ticker} {side} x{count} "
                f"| P&L: {pnl_str} | Reason: {reason} [{client.label}]",
                self.config,
            )

            self.logger.info(
                "position_exited",
                ticker=pos.ticker,
                pnl=pnl,
                reason=reason,
                account=client.label,
            )
        else:
            # Order returned None/falsy — record as failed attempt
            self._exit_retries[key] = (retry_count + 1, _time.monotonic())
            self.logger.warning(
                "exit_order_no_result",
                ticker=pos.ticker,
                account=client.label,
                attempt=retry_count + 1,
            )
