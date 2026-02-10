"""Telegram bot command handler — /status for partner visibility.

Runs a lightweight polling loop alongside the main bot.
Responds to /status with current balance, positions, survival mode,
recent trades, and LLM cost summary.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import structlog

from .utils import BotConfig

logger = structlog.get_logger()


class TelegramBot:
    """Handles incoming Telegram commands via long polling."""

    def __init__(
        self,
        config: BotConfig,
        state_dir: str = "state",
    ):
        self.config = config
        self._state_path = Path(state_dir)

        self._bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self._base_url = f"https://api.telegram.org/bot{self._bot_token}"
        self._last_update_id = 0
        self._task: Optional[asyncio.Task] = None

        # References set externally
        self._perf_tracker = None
        self._cost_tracker = None
        self._survival_tracker = None
        self._trading_clients: List = []

    def set_perf_tracker(self, tracker) -> None:
        self._perf_tracker = tracker

    def set_cost_tracker(self, tracker) -> None:
        self._cost_tracker = tracker

    def set_survival_tracker(self, tracker) -> None:
        self._survival_tracker = tracker

    def set_trading_clients(self, clients: List) -> None:
        self._trading_clients = clients

    async def start(self) -> None:
        if not self._bot_token or not self._chat_id:
            logger.info("telegram_bot_disabled", reason="no token or chat_id")
            return
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("telegram_bot_started")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _poll_loop(self) -> None:
        """Poll for new messages every 5 seconds."""
        await asyncio.sleep(10)  # let other services start first
        while True:
            try:
                await self._check_updates()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("telegram_poll_error", error=str(e))
            await asyncio.sleep(5)

    async def _check_updates(self) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self._base_url}/getUpdates",
                params={
                    "offset": self._last_update_id + 1,
                    "timeout": 1,
                    "allowed_updates": '["message"]',
                },
            )
            if resp.status_code != 200:
                return

            data = resp.json()
            for update in data.get("result", []):
                self._last_update_id = update.get("update_id", self._last_update_id)
                message = update.get("message", {})
                text = message.get("text", "")
                chat_id = str(message.get("chat", {}).get("id", ""))

                # Only respond to our configured chat
                if chat_id != self._chat_id:
                    continue

                if text.startswith("/status"):
                    await self._handle_status(client)
                elif text.startswith("/help"):
                    await self._send(client, "Commands:\n/status — Bot status dashboard\n/help — This message")

    async def _handle_status(self, client: httpx.AsyncClient) -> None:
        """Build and send a comprehensive status message."""
        lines = ["📊 *Morpheus Status*", ""]

        # Balances
        total_balance = 0.0
        for i, tc in enumerate(self._trading_clients):
            try:
                bal = await tc.get_balance()
                label = getattr(tc, "label", f"account_{i}")
                lines.append(f"💰 {label}: ${bal:.2f}")
                total_balance += bal
            except Exception:
                lines.append(f"💰 account_{i}: error")
        if self._trading_clients:
            lines.append(f"💰 *Total: ${total_balance:.2f}*")
            lines.append("")

        # Survival mode
        if self._survival_tracker:
            try:
                mode = self._survival_tracker.evaluate()
                mult = self._survival_tracker.multiplier
                status_line = self._survival_tracker.format_status_line()
                lines.append(f"🛡 Survival: {status_line}")
                lines.append("")
            except Exception:
                pass

        # LLM costs
        if self._cost_tracker:
            try:
                summary = self._cost_tracker.get_summary()
                daily_spend = summary.get("daily_spend", 0)
                total_spend = summary.get("total_spend", 0)
                remaining = summary.get("budget_remaining", 0)
                lines.append(f"🤖 LLM spend today: ${daily_spend:.2f}")
                lines.append(f"🤖 LLM spend total: ${total_spend:.2f}")
                lines.append(f"🤖 Budget remaining: ${remaining:.2f}")
                lines.append("")
            except Exception:
                pass

        # Performance stats
        if self._perf_tracker:
            try:
                stats_1d = self._perf_tracker.get_stats(days=1)
                stats_7d = self._perf_tracker.get_stats(days=7)
                lines.append(f"📈 Today: {stats_1d['total_trades']} trades, P&L ${stats_1d['total_pnl']:.2f}")
                lines.append(f"📈 7-day: {stats_7d['total_trades']} trades, P&L ${stats_7d['total_pnl']:.2f}")
                lines.append("")
            except Exception:
                pass

        # Open positions (from trade history)
        try:
            positions = self._count_open_positions()
            if positions:
                lines.append(f"📋 Open positions: {len(positions)}")
                for ticker, info in list(positions.items())[:5]:
                    lines.append(f"  • {ticker} ({info['side']}, {info['count']} contracts)")
                if len(positions) > 5:
                    lines.append(f"  ... and {len(positions) - 5} more")
            else:
                lines.append("📋 No open positions")
            lines.append("")
        except Exception:
            pass

        # Predictions pending resolution
        try:
            pred_path = self._state_path / "predictions.jsonl"
            res_path = self._state_path / "resolutions.jsonl"
            pred_count = sum(1 for _ in open(pred_path)) if pred_path.exists() else 0
            res_count = sum(1 for _ in open(res_path)) if res_path.exists() else 0
            lines.append(f"🎯 Predictions: {pred_count} total, {res_count} resolved")
        except Exception:
            pass

        # Uptime
        lines.append(f"\n🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

        await self._send(client, "\n".join(lines))

    def _count_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """Count open positions from trade history."""
        trade_log = self._state_path / "trade_history.jsonl"
        if not trade_log.exists():
            return {}

        positions: Dict[str, Dict[str, Any]] = {}
        try:
            with open(trade_log) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        t = json.loads(line.strip())
                        if t.get("event") == "order_placed":
                            ticker = t.get("ticker", "")
                            side = t.get("side", "?")
                            count = int(t.get("count", 0) or 0)
                            if ticker:
                                if ticker not in positions:
                                    positions[ticker] = {"side": side, "count": 0}
                                positions[ticker]["count"] += count
                    except (json.JSONDecodeError, ValueError):
                        continue
        except Exception:
            pass

        return positions

    async def _send(self, client: httpx.AsyncClient, text: str) -> None:
        try:
            await client.post(
                f"{self._base_url}/sendMessage",
                json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
            )
        except Exception as e:
            logger.debug("telegram_send_error", error=str(e))
