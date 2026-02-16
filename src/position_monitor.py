"""Position monitor — enforce stop-loss, take-profit, and time-based exits.

The critical missing piece in Morpheus v1: risk.py has should_close_position()
but it was NEVER called from anywhere. This module runs as a background task
and actively monitors all positions.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import structlog

import httpx

from .alerts import send_alert
from .kalshi_trading_client import KalshiTradingClient, KalshiPosition
from .risk import RiskManager
from .trade_logger import get_trade_logger
from .utils import BotConfig

_WEATHER_PREFIXES = ("KXHIGH", "KXLOW", "KXRAIN", "KXSNOW", "KXTEMP", "KXWIND")


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
    close_time: Optional[datetime] = None  # market close time (for dynamic max hold)
    signal_source: str = ""  # "noaa_direct", "llm", "yahoo_direct"


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
        self.take_profit_pct = float(risk_cfg.get("take_profit_pct", 0.60))
        self.max_hold_hours = float(risk_cfg.get("max_position_hold_hours", 24))

        # Track positions with entry metadata
        self._tracked: Dict[str, TrackedPosition] = {}
        self._task: Optional[asyncio.Task] = None

        # Exit retry tracking: key -> (attempt_count, last_attempt_time)
        self._exit_retries: Dict[str, tuple[int, float]] = {}
        self._max_exit_retries = 3

        # Pending exit orders: key -> (order_id, monotonic_time, sell_price_cents)
        # We don't treat a place_order() response as a confirmed fill — we wait
        # for the position to disappear from get_positions() before booking PnL.
        self._pending_exits: Dict[str, tuple[str, float, int]] = {}

        # Persistent set of known-closed markets (survives restarts)
        self._state_path = Path(state_dir)
        self._closed_markets_file = self._state_path / "closed_markets.json"
        self._known_closed: Set[str] = self._load_known_closed()

        # Stale market cache: ticker -> (check_time, is_closed)
        self._market_status_cache: Dict[str, tuple[float, bool]] = {}
        self._market_status_ttl = 600.0  # 10 min cache

        # Live price cache for SL/TP: ticker -> (monotonic_time, price)
        self._price_cache: Dict[str, tuple[float, float]] = {}
        self._price_cache_ttl = 120.0  # 2 min cache

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

        # Shared httpx client for market API calls (connection pooling)
        self._http_client: Optional[httpx.AsyncClient] = None

    def _load_known_closed(self) -> Set[str]:
        """Load known-closed market keys from disk, cleaning stale entries (48h TTL)."""
        try:
            if self._closed_markets_file.exists():
                data = json.loads(self._closed_markets_file.read_text())
                if isinstance(data, dict):
                    # New format: {key: timestamp_iso}
                    cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
                    fresh = {k for k, ts in data.items() if ts > cutoff}
                    if len(fresh) < len(data):
                        self.logger.info("known_closed_cleanup", removed=len(data) - len(fresh), kept=len(fresh))
                    return fresh
                # Legacy format: [key, key, ...]
                return set(data) if isinstance(data, list) else set()
        except Exception:
            pass
        return set()

    def _save_known_closed(self) -> None:
        """Persist known-closed market keys with timestamps."""
        try:
            self._state_path.mkdir(parents=True, exist_ok=True)
            now_iso = datetime.now(timezone.utc).isoformat()
            existing = {}
            if self._closed_markets_file.exists():
                try:
                    raw = json.loads(self._closed_markets_file.read_text())
                    if isinstance(raw, dict):
                        existing = raw
                except Exception:
                    pass
            merged = {k: existing.get(k, now_iso) for k in self._known_closed}
            self._closed_markets_file.write_text(json.dumps(merged, indent=2))
        except Exception as e:
            self.logger.debug("save_closed_markets_error", error=str(e))

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get shared httpx client (connection pooling)."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self._kalshi_base_url,
                timeout=10.0,
                headers={"Accept": "application/json"},
            )
        return self._http_client

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
        close_time: Optional[datetime] = None,
        signal_source: str = "",
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
            close_time=close_time,
            signal_source=signal_source,
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
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
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
                # Check pending exit orders before individual position checks
                await self._check_pending_exits(client, positions)
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
            http = await self._get_http_client()
            r = await http.get(f"/markets/{ticker}")
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

    async def _get_current_yes_price(self, ticker: str) -> Optional[float]:
        """Fetch current YES midpoint price from Kalshi public API. Cached 2min."""
        import time as _time
        now = _time.monotonic()
        cached = self._price_cache.get(ticker)
        is_weather = _is_weather_ticker(ticker)
        ttl = 30.0 if is_weather else self._price_cache_ttl
        if cached and now - cached[0] < ttl:
            return cached[1]
        try:
            http = await self._get_http_client()
            r = await http.get(f"/markets/{ticker}")
            if r.status_code == 200:
                data = r.json()
                m = data.get("market", data)
                yb = (m.get("yes_bid") or 0) / 100.0
                ya = (m.get("yes_ask") or 0) / 100.0
                lp = (m.get("last_price") or 0) / 100.0
                price = (yb + ya) / 2.0 if yb > 0 and ya > 0 else lp if lp > 0 else None
                if price:
                    self._price_cache[ticker] = (now, price)
                return price
        except Exception as e:
            self.logger.debug("price_fetch_error", ticker=ticker, error=str(e))
        return None

    async def _get_market_question(self, ticker: str) -> Optional[str]:
        """Fetch market title/question from Kalshi public API. Permanently cached."""
        if ticker in self._market_question_cache:
            return self._market_question_cache[ticker]

        try:
            http = await self._get_http_client()
            r = await http.get(f"/markets/{ticker}")
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

        # Min hold time: don't reprice positions entered < 2 min ago
        hold_seconds = (datetime.now(timezone.utc) - tracked.entry_time).total_seconds()
        if hold_seconds < 120:
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

        # Use current market price for repricing edge (not entry price)
        # If NOAA says p_yes=0.80 but market moved to 0.88, real edge is -0.08
        yes_price = None
        try:
            yes_price = await self._get_current_yes_price(pos.ticker)
        except Exception:
            pass
        if yes_price is None:
            # Fall back to entry price if market price unavailable
            yes_price = tracked.entry_price_cents / 100.0

        # Compute current edge for our held side vs market price
        if tracked.side == "yes":
            current_edge = p_yes - yes_price
        else:
            current_edge = (1.0 - p_yes) - (1.0 - yes_price)

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
            entry_price=tracked.entry_price_cents / 100.0,
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

    async def _check_pending_exits(
        self,
        client: KalshiTradingClient,
        positions: List[KalshiPosition],
    ) -> None:
        """Confirm pending exit fills by checking if positions disappeared."""
        import time as _time
        now = _time.monotonic()

        active_tickers = {pos.ticker for pos in positions if pos.count != 0}

        # Find pending exits belonging to this client
        prefix = f"{client.label}:"
        client_pending = {
            k: v for k, v in self._pending_exits.items()
            if k.startswith(prefix)
        }
        if not client_pending:
            return

        # Fetch open orders once (only if needed for non-timed-out checks)
        open_order_ids: Optional[set] = None
        has_young_pending = any(
            k.split(":", 1)[1] in active_tickers and now - v[1] < 600
            for k, v in client_pending.items()
        )
        if has_young_pending:
            try:
                open_orders = await client.get_open_orders()
                open_order_ids = {o.order_id for o in open_orders}
            except Exception:
                pass  # Can't check orders — leave pending, will timeout at 10min

        keys_to_remove: List[str] = []
        for key, (order_id, placed_time, sell_price_cents) in client_pending.items():
            ticker = key.split(":", 1)[1]

            if ticker not in active_tickers:
                # Position gone — exit filled! Book PnL.
                tracked = self._tracked.get(key)
                if tracked:
                    entry_cost = tracked.entry_price_cents * tracked.count / 100.0
                    exit_proceeds = sell_price_cents * tracked.count / 100.0
                    pnl = exit_proceeds - entry_cost

                    self.risk_manager.update_daily_pnl(pnl)

                    side = tracked.side
                    count = tracked.count
                    pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                    emoji = "PROFIT" if pnl >= 0 else "LOSS"
                    await send_alert(
                        f"{emoji}: {ticker} {side} x{count} "
                        f"| P&L: {pnl_str} | Exit confirmed [{client.label}]",
                        self.config,
                    )

                    self.logger.info(
                        "position_exited",
                        ticker=ticker,
                        pnl=pnl,
                        reason="exit_order_filled",
                        account=client.label,
                    )
                    get_trade_logger().log_position_closed(
                        platform="kalshi",
                        ticker=ticker,
                        side=side,
                        count=count,
                        entry_price_cents=tracked.entry_price_cents,
                        exit_price_cents=sell_price_cents,
                        pnl_usd=pnl,
                        account_label=client.label,
                    )

                self._tracked.pop(key, None)
                keys_to_remove.append(key)
                continue

            # Position still exists
            age_seconds = now - placed_time
            if age_seconds > 600:
                # 10-minute timeout — cancel stale exit order and retry next cycle
                if order_id:
                    await client.cancel_order(order_id)
                self.logger.info(
                    "pending_exit_timeout",
                    ticker=ticker,
                    account=client.label,
                    age_seconds=int(age_seconds),
                )
                keys_to_remove.append(key)
                continue

            # Order < 10 min — check if still resting
            if open_order_ids is not None and order_id and order_id not in open_order_ids:
                # Order gone but position still exists — cancelled externally
                # or partially filled. Remove from pending; will re-evaluate next cycle.
                self.logger.info(
                    "pending_exit_order_gone",
                    ticker=ticker,
                    account=client.label,
                    order_id=order_id,
                )
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self._pending_exits.pop(key, None)

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

        # Skip positions with a pending exit order (wait for fill confirmation)
        if key in self._pending_exits:
            self.logger.debug("skip_pending_exit", ticker=pos.ticker, account=client.label)
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

        # Get current market price for P&L calculation (Wave 10: real price)
        entry_price = tracked.entry_price_cents / 100.0
        yes_price = await self._get_current_yes_price(pos.ticker)
        if yes_price is not None:
            current_price = yes_price if tracked.side == "yes" else (1.0 - yes_price)
        else:
            # Fallback to static exposure (SL/TP won't trigger, but max-hold still works)
            current_price = pos.market_exposure / abs(pos.count) if pos.count != 0 else entry_price

        should_exit, reason = self.risk_manager.should_close_position(
            entry_price=entry_price,
            current_price=current_price,
            entry_time=tracked.entry_time,
            position_amount=pos.market_exposure,
            close_time=tracked.close_time,
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

        # Smart exit pricing: cross spread by 3c instead of posting at extreme 1c/99c
        yes_price = await self._get_current_yes_price(pos.ticker)
        if yes_price is not None:
            if side == "yes":
                sell_price = max(1, int(yes_price * 100) - 3)
            else:
                no_price = 1.0 - yes_price
                sell_price = max(1, int(no_price * 100) - 3)
        else:
            sell_price = 1 if side == "yes" else 99  # fallback

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
                # Permanently skip this position — let market settle naturally
                self._known_closed.add(key)
                self._save_known_closed()
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
            # Extract order_id from result
            order_id = ""
            if isinstance(result, dict):
                inner = result.get("order", result)
                if isinstance(inner, dict):
                    order_id = inner.get("order_id", "")

            # Don't pop from _tracked or log position_exited yet.
            # The order may sit unfilled — wait for the position to disappear
            # from get_positions() in _check_pending_exits() before booking PnL.
            self._pending_exits[key] = (order_id, _time.monotonic(), sell_price)
            self._exit_retries.pop(key, None)

            self.logger.info(
                "exit_order_placed",
                ticker=pos.ticker,
                side=side,
                count=count,
                sell_price=sell_price,
                reason=reason,
                account=client.label,
                order_id=order_id,
            )
        else:
            # Order returned None/falsy — record as failed attempt
            self._exit_retries[key] = (retry_count + 1, _time.monotonic())
            # If client is halted (insufficient balance), skip permanently
            if client.is_halted:
                self._exit_retries[key] = (self._max_exit_retries, _time.monotonic())
                self._known_closed.add(key)
                self._save_known_closed()
                self.logger.warning(
                    "exit_skipped_account_halted",
                    ticker=pos.ticker,
                    account=client.label,
                    msg="Account halted (insufficient balance) — skipping exit, will settle naturally",
                )
                return
            self.logger.warning(
                "exit_order_no_result",
                ticker=pos.ticker,
                account=client.label,
                attempt=retry_count + 1,
            )
