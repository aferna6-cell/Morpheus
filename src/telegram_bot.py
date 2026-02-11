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
                elif text.startswith("/positions"):
                    await self._handle_positions(client)
                elif text.startswith("/pnl"):
                    await self._handle_pnl(client)
                elif text.startswith("/predictions"):
                    await self._handle_predictions(client)
                elif text.startswith("/help"):
                    await self._send(client, (
                        "Commands:\n"
                        "/status — Bot status dashboard\n"
                        "/positions — Live positions from Kalshi\n"
                        "/pnl — Detailed P&L breakdown\n"
                        "/predictions — Pending predictions\n"
                        "/help — This message"
                    ))

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

    async def _handle_positions(self, client: httpx.AsyncClient) -> None:
        """Show live positions from Kalshi API across all accounts."""
        lines = ["📋 *Live Positions*", ""]

        total_exposure = 0.0
        total_positions = 0

        for tc in self._trading_clients:
            label = getattr(tc, "label", "unknown")
            try:
                bal = await tc.get_balance()
                positions = await tc.get_positions()
                active = [p for p in positions if p.count != 0]
                lines.append(f"*{label}* (${bal:.2f} available)")

                if not active:
                    lines.append("  No positions")
                else:
                    for p in active:
                        side = "YES" if p.count > 0 else "NO"
                        count = abs(p.count)
                        exp = p.market_exposure
                        total_exposure += exp
                        total_positions += 1
                        lines.append(f"  {p.ticker}: {side} x{count} (${exp:.2f})")

                lines.append("")
            except Exception as e:
                lines.append(f"*{label}*: error ({e})")
                lines.append("")

        lines.append(f"*Total:* {total_positions} positions, ${total_exposure:.2f} deployed")
        await self._send(client, "\n".join(lines))

    async def _handle_pnl(self, client: httpx.AsyncClient) -> None:
        """Detailed P&L breakdown by market category."""
        lines = ["📊 *P&L Breakdown*", ""]

        try:
            res_path = self._state_path / "resolutions.jsonl"
            if not res_path.exists():
                await self._send(client, "No resolutions yet.")
                return

            resolutions: List[Dict[str, Any]] = []
            with open(res_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            resolutions.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue

            if not resolutions:
                await self._send(client, "No resolutions yet.")
                return

            # Overall stats
            total_pnl = sum(r.get("pnl_usd", 0) for r in resolutions)
            wins = [r for r in resolutions if r.get("pnl_usd", 0) > 0]
            losses = [r for r in resolutions if r.get("pnl_usd", 0) < 0]
            even = [r for r in resolutions if r.get("pnl_usd", 0) == 0]
            n_decided = len(wins) + len(losses)
            win_rate = len(wins) / n_decided if n_decided > 0 else 0

            lines.append(f"*Overall:* {len(resolutions)} resolved")
            lines.append(f"W/L: {len(wins)}/{len(losses)} ({win_rate:.0%} win rate)")
            lines.append(f"Total P&L: *${total_pnl:+.2f}*")
            if wins:
                lines.append(f"Avg win: ${sum(r['pnl_usd'] for r in wins)/len(wins):.2f}")
            if losses:
                lines.append(f"Avg loss: ${sum(r['pnl_usd'] for r in losses)/len(losses):.2f}")
            lines.append("")

            # By category
            by_cat: Dict[str, Dict[str, Any]] = {}
            for r in resolutions:
                mid = r.get("market_id", "")
                prefix = mid.split("-")[0] if mid else "unknown"
                if prefix not in by_cat:
                    by_cat[prefix] = {"pnl": 0.0, "wins": 0, "losses": 0, "count": 0}
                by_cat[prefix]["pnl"] += r.get("pnl_usd", 0)
                by_cat[prefix]["count"] += 1
                if r.get("pnl_usd", 0) > 0:
                    by_cat[prefix]["wins"] += 1
                elif r.get("pnl_usd", 0) < 0:
                    by_cat[prefix]["losses"] += 1

            lines.append("*By Category:*")
            for cat, stats in sorted(by_cat.items(), key=lambda x: -x[1]["count"]):
                w = stats["wins"]
                l = stats["losses"]
                wr = w / (w + l) if (w + l) > 0 else 0
                pnl = stats["pnl"]
                lines.append(f"  {cat}: {w}/{w+l} ({wr:.0%}) ${pnl:+.2f}")

        except Exception as e:
            lines.append(f"Error: {e}")

        await self._send(client, "\n".join(lines))

    async def _handle_predictions(self, client: httpx.AsyncClient) -> None:
        """Show pending predictions summary."""
        lines = ["🎯 *Pending Predictions*", ""]

        try:
            pred_path = self._state_path / "predictions.jsonl"
            res_path = self._state_path / "resolutions.jsonl"

            if not pred_path.exists():
                await self._send(client, "No predictions yet.")
                return

            predictions = []
            with open(pred_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            predictions.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue

            resolved_ids: set = set()
            if res_path.exists():
                with open(res_path) as f:
                    for line in f:
                        if line.strip():
                            try:
                                r = json.loads(line.strip())
                                resolved_ids.add(r.get("market_id", ""))
                            except json.JSONDecodeError:
                                continue

            pending = [p for p in predictions if p.get("market_id", "") not in resolved_ids]

            # Group by category
            by_cat: Dict[str, int] = {}
            for p in pending:
                mid = p.get("market_id", "")
                prefix = mid.split("-")[0] if mid else "unknown"
                by_cat[prefix] = by_cat.get(prefix, 0) + 1

            lines.append(f"Total: {len(predictions)} predictions")
            lines.append(f"Resolved: {len(resolved_ids)}")
            lines.append(f"Pending: {len(pending)}")
            lines.append("")
            lines.append("*Pending by category:*")
            for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
                lines.append(f"  {cat}: {n}")

        except Exception as e:
            lines.append(f"Error: {e}")

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
