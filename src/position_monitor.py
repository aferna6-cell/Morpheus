"""Position monitor — enforce stop-loss, take-profit, and time-based exits.

The critical missing piece in Morpheus v1: risk.py has should_close_position()
but it was NEVER called from anywhere. This module runs as a background task
and actively monitors all positions.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import structlog

import httpx

from .alerts import send_alert
from .kalshi_trading_client import KalshiTradingClient, KalshiPosition
from .risk import RiskManager
from .utils import BotConfig

_WEATHER_PREFIXES = ("KXHIGH", "KXLOW", "KXRAIN", "KXSNOW", "KXTEMP")


def _is_weather_ticker(ticker: str) -> bool:
    return ticker.upper().startswith(_WEATHER_PREFIXES)


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
    entry_edge: float = 0.0  # edge at time of entry (for trailing stops)


class PositionMonitor:
    """Actively monitors positions and triggers exits."""

    def __init__(
        self,
        config: BotConfig,
        trading_clients: List[KalshiTradingClient],
        risk_manager: RiskManager,
        state_dir: str = "state",
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

        # Persistent set of known-closed markets (survives restarts)
        self._state_path = Path(state_dir)
        self._closed_markets_file = self._state_path / "closed_markets.json"
        self._known_closed: Set[str] = self._load_known_closed()

        # Stale market cache: ticker -> (check_time, is_closed)
        self._market_status_cache: Dict[str, tuple[float, bool]] = {}
        self._market_status_ttl = 600.0  # 10 min cache

        # Permanent cache: ticker -> market question text (titles never change)
        self._market_question_cache: Dict[str, str] = {}

        # Callback to trigger engine rescans when an account resumes
        self._on_resume_callbacks: List = []

        # Kalshi API base URL for market status checks
        kalshi_cfg = getattr(config, "kalshi", None) or {}
        self._kalshi_base_url = (
            kalshi_cfg.get("base_url", "https://api.elections.kalshi.com/trade-api/v2")
            if isinstance(kalshi_cfg, dict)
            else "https://api.elections.kalshi.com/trade-api/v2"
        )

    def _load_known_closed(self) -> Set[str]:
        """Load known-closed market keys from disk."""
        try:
            if self._closed_markets_file.exists():
                data = json.loads(self._closed_markets_file.read_text())
                return set(data)
        except Exception:
            pass
        return set()

    def _save_known_closed(self) -> None:
        """Persist known-closed market keys to disk."""
        try:
            self._state_path.mkdir(parents=True, exist_ok=True)
            self._closed_markets_file.write_text(
                json.dumps(sorted(self._known_closed), indent=2)
            )
        except Exception as e:
            self.logger.debug("save_closed_markets_error", error=str(e))

    def on_resume(self, callback) -> None:
        """Register a callback to fire when a halted account resumes trading."""
        self._on_resume_callbacks.append(callback)

    def track_position(
        self,
        ticker: str,
        side: str,
        count: int,
        entry_price_cents: int,
        strategy: str = "llm",
        order_id: str = "",
        account_label: str = "default",
        entry_edge: float = 0.0,
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
            entry_edge=entry_edge,
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
                resumed = await client.check_and_resume()
                if resumed:
                    self.logger.info(
                        "account_resumed_trading",
                        label=client.label,
                        msg="Triggering engine rescans",
                    )
                    for cb in self._on_resume_callbacks:
                        try:
                            cb()
                        except Exception as e:
                            self.logger.debug("resume_callback_error", error=str(e))

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

    async def _get_market_question(self, ticker: str) -> Optional[str]:
        """Fetch market title/question from Kalshi public API. Permanently cached."""
        if ticker in self._market_question_cache:
            return self._market_question_cache[ticker]

        try:
            async with httpx.AsyncClient(
                base_url=self._kalshi_base_url,
                timeout=10.0,
                headers={"Accept": "application/json"},
            ) as client:
                r = await client.get(f"/markets/{ticker}")
                if r.status_code == 200:
                    data = r.json()
                    market = data.get("market", data)
                    title = market.get("title") or market.get("question") or ""
                    if title:
                        self._market_question_cache[ticker] = title
                        return title
        except Exception as e:
            self.logger.debug("market_question_fetch_error", ticker=ticker, error=str(e))

        return None

    async def _check_weather_repricing(
        self,
        client: KalshiTradingClient,
        pos: KalshiPosition,
        tracked: TrackedPosition,
    ) -> Optional[str]:
        """Check if NOAA forecast has shifted against our weather position.

        Returns exit reason string if should exit, None if hold.
        """
        from .structured_data import compute_weather_probability

        # Feature flag
        weather_repricing_enabled = self.config.risk.get("weather_repricing_enabled", True)
        if not weather_repricing_enabled:
            return None

        question = await self._get_market_question(pos.ticker)
        if not question:
            self.logger.debug("weather_repricing_no_question", ticker=pos.ticker)
            return None

        result = await compute_weather_probability(question, pos.ticker)

        if result is None:
            # Ambiguous NOAA signal — don't exit, let P&L thresholds handle it
            self.logger.debug(
                "weather_repricing_ambiguous",
                ticker=pos.ticker,
                msg="NOAA signal ambiguous, skipping repricing",
            )
            return None

        p_yes, confidence, reasoning = result
        entry_price = tracked.entry_price_cents / 100.0

        # Compute current edge for our held side
        if tracked.side == "yes":
            current_edge = p_yes - entry_price
        else:
            current_edge = entry_price - p_yes

        # Exit thresholds from config
        exit_threshold = float(self.config.risk.get("weather_exit_threshold", -0.05))
        exit_threshold_hc = float(self.config.risk.get("weather_exit_threshold_high_conf", -0.03))

        # High-confidence NOAA signals get tighter exit threshold
        threshold = exit_threshold_hc if confidence >= 0.85 else exit_threshold

        # Trailing stop: if edge has improved significantly since entry,
        # tighten exit threshold to protect profit (lock in at break-even)
        if tracked.entry_edge > 0 and current_edge > tracked.entry_edge + 0.15:
            threshold = 0.0  # exit if edge drops below break-even
            self.logger.info(
                "weather_trailing_stop_tightened",
                ticker=pos.ticker,
                entry_edge=round(tracked.entry_edge, 4),
                current_edge=round(current_edge, 4),
                new_threshold=threshold,
            )

        self.logger.info(
            "weather_repricing_check",
            ticker=pos.ticker,
            side=tracked.side,
            entry_price=entry_price,
            p_yes=round(p_yes, 4),
            confidence=round(confidence, 3),
            current_edge=round(current_edge, 4),
            threshold=threshold,
            should_exit=current_edge < threshold,
            account=client.label,
        )

        if current_edge < threshold:
            return (
                f"Weather forecast exit: edge={current_edge:+.1%} < {threshold:+.1%} "
                f"(NOAA p_yes={p_yes:.3f}, conf={confidence:.2f}, side={tracked.side})"
            )

        return None

    async def _check_position(
        self,
        client: KalshiTradingClient,
        pos: KalshiPosition,
    ) -> None:
        """Check a single position for exit conditions."""
        key = f"{client.label}:{pos.ticker}"

        # Skip markets we already know are closed (persisted across restarts)
        if key in self._known_closed:
            return

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

        # Weather repricing: exit if NOAA forecast shifted against our position
        if _is_weather_ticker(pos.ticker):
            repricing_reason = await self._check_weather_repricing(client, pos, tracked)
            if repricing_reason:
                await self._exit_position(client, pos, tracked, repricing_reason)
                # Trigger engine rescans for potential opposite-side entry
                for cb in self._on_resume_callbacks:
                    try:
                        cb()
                    except Exception:
                        pass
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

        # Linear backoff: 60s, 120s, 180s between retries (was 300, 900, 2700)
        if retry_count > 0:
            backoff_seconds = 60 * retry_count  # 60, 120, 180
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
                self._exit_retries[key] = (self._max_exit_retries, _time.monotonic())
                return
            # Detect market_closed — Kalshi will settle automatically, stop retrying
            if "market_closed" in error_str or "market closed" in error_str:
                self.logger.info(
                    "exit_market_already_closed",
                    ticker=pos.ticker,
                    account=client.label,
                    msg="Market settled by Kalshi — no exit needed",
                )
                self._exit_retries[key] = (self._max_exit_retries, _time.monotonic())
                self._known_closed.add(key)
                self._save_known_closed()
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
