"""Performance tracker — real-time P&L, win rate, and daily Telegram summary.

Reads trade history from trade_logger JSONL, computes stats, and sends
daily summaries via Telegram. Also provides on-demand stats for the
orchestrator and position monitor.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from .alerts import send_alert
from .utils import BotConfig


class PerfTracker:
    """Tracks trading performance and sends periodic reports."""

    def __init__(
        self,
        config: BotConfig,
        state_dir: str = "state",
    ):
        self.config = config
        self.logger = structlog.get_logger()

        self._state_path = Path(state_dir)
        self._trade_log = self._state_path / "trade_history.jsonl"
        self._summary_path = self._state_path / "perf_summary.json"

        alerts_cfg = getattr(config, "alerts", None) or {}
        if not isinstance(alerts_cfg, dict):
            alerts_cfg = {}
        self._daily_summary_hour = int(alerts_cfg.get("daily_summary_hour", 23))
        self._summary_interval = 3600.0  # check hourly

        self._task: Optional[asyncio.Task] = None
        self._last_summary_date: Optional[str] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._summary_loop())
        self.logger.info("perf_tracker_started")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("perf_tracker_stopped")

    async def _summary_loop(self) -> None:
        while True:
            try:
                now = datetime.now(timezone.utc)
                today_str = now.strftime("%Y-%m-%d")

                # Send daily summary at configured hour
                if now.hour == self._daily_summary_hour and self._last_summary_date != today_str:
                    await self._send_daily_summary()
                    self._last_summary_date = today_str

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error("perf_tracker_error", error=str(e))

            await asyncio.sleep(self._summary_interval)

    # ------------------------------------------------------------------
    # Stats computation
    # ------------------------------------------------------------------

    def get_stats(self, days: int = 1) -> Dict[str, Any]:
        """Compute trading stats for the last N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        trades = self._read_trades(since=cutoff)

        if not trades:
            return {
                "period_days": days,
                "total_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "avg_edge": 0.0,
                "total_cost": 0.0,
                "total_fees": 0.0,
            }

        # Separate orders placed vs fills vs exits
        orders = [t for t in trades if t.get("event") == "order_placed"]
        exits = [t for t in trades if t.get("event") == "position_exit"]

        total_cost = sum(float(t.get("cost_usd", 0)) for t in orders)
        total_fees = sum(float(t.get("fee_usd", 0)) for t in orders)
        total_pnl = sum(float(t.get("pnl", 0)) for t in exits)

        wins = sum(1 for t in exits if float(t.get("pnl", 0)) > 0)
        losses = sum(1 for t in exits if float(t.get("pnl", 0)) <= 0)
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

        edges = [float(t.get("edge", 0)) for t in orders if t.get("edge")]
        avg_edge = sum(edges) / len(edges) if edges else 0.0

        # Per-strategy breakdown
        by_strategy: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "pnl": 0.0, "cost": 0.0})
        for t in orders:
            strat = t.get("strategy", t.get("platform", "unknown"))
            by_strategy[strat]["count"] += 1
            by_strategy[strat]["cost"] += float(t.get("cost_usd", 0))

        for t in exits:
            strat = t.get("strategy", "unknown")
            by_strategy[strat]["pnl"] += float(t.get("pnl", 0))

        return {
            "period_days": days,
            "total_trades": len(orders),
            "total_exits": len(exits),
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 3),
            "wins": wins,
            "losses": losses,
            "avg_edge": round(avg_edge, 4),
            "total_cost": round(total_cost, 2),
            "total_fees": round(total_fees, 2),
            "by_strategy": dict(by_strategy),
        }

    def _read_trades(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Read trade log entries."""
        if not self._trade_log.exists():
            return []

        trades = []
        try:
            with open(self._trade_log, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if since:
                            ts_str = entry.get("timestamp", "")
                            if ts_str:
                                try:
                                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                                    if ts < since:
                                        continue
                                except (ValueError, TypeError):
                                    pass
                        trades.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            self.logger.warning("trade_log_read_error", error=str(e))

        return trades

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    async def _send_daily_summary(self) -> None:
        """Send a daily P&L summary via Telegram."""
        stats_1d = self.get_stats(days=1)
        stats_7d = self.get_stats(days=7)
        stats_30d = self.get_stats(days=30)

        pnl_1d = stats_1d["total_pnl"]
        pnl_7d = stats_7d["total_pnl"]
        pnl_30d = stats_30d["total_pnl"]

        pnl_emoji = "+" if pnl_1d >= 0 else ""
        week_emoji = "+" if pnl_7d >= 0 else ""

        msg = (
            f"DAILY REPORT\n"
            f"---\n"
            f"Today: {pnl_emoji}${pnl_1d:.2f} | "
            f"Trades: {stats_1d['total_trades']} | "
            f"Win: {stats_1d['win_rate']*100:.0f}%\n"
            f"7d: {week_emoji}${pnl_7d:.2f} | "
            f"Trades: {stats_7d['total_trades']}\n"
            f"30d: ${pnl_30d:.2f}\n"
        )

        # Strategy breakdown for today
        if stats_1d.get("by_strategy"):
            msg += "---\n"
            for strat, data in stats_1d["by_strategy"].items():
                s_pnl = data.get("pnl", 0)
                msg += f"{strat}: {data['count']} trades, ${s_pnl:+.2f}\n"

        msg += f"---\nAvg edge: {stats_1d['avg_edge']:+.3f} | Cost: ${stats_1d['total_cost']:.2f}"

        await send_alert(msg, self.config)
        self.logger.info("daily_summary_sent", pnl_1d=pnl_1d, trades_1d=stats_1d["total_trades"])

        # Save summary to disk
        try:
            summary = {
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "stats_1d": stats_1d,
                "stats_7d": stats_7d,
                "stats_30d": stats_30d,
            }
            with open(self._summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning("summary_save_error", error=str(e))

    # ------------------------------------------------------------------
    # On-demand profit alert
    # ------------------------------------------------------------------

    async def alert_profit(
        self,
        ticker: str,
        side: str,
        pnl: float,
        reason: str,
        account: str = "default",
    ) -> None:
        """Send an immediate profit/loss alert."""
        emoji = "PROFIT" if pnl >= 0 else "LOSS"
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

        msg = (
            f"{emoji} [{account}]\n"
            f"{ticker} {side} | {pnl_str}\n"
            f"Reason: {reason}"
        )
        await send_alert(msg, self.config)
