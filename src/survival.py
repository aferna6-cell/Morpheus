"""Survival mode — explicit self-sustainability gate.

Tracks rolling P&L vs. costs (LLM + VPS) and reduces/halts aggression
when unprofitable. Three modes:

  NORMAL  — 7d profit >= 0 → full sizing (1.0x)
  REDUCED — 7d profit < 0, 14d profit >= 0 → half sizing (0.5x)
  HALTED  — both 7d AND 14d profit < 0 → no new trades (0.0x)

Grace period on first startup: don't penalize during ramp-up.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from .cost_tracker import CostTracker
from .perf_tracker import PerfTracker
from .utils import BotConfig, load_json_state, save_json_state


class SurvivalMode(str, Enum):
    NORMAL = "normal"
    REDUCED = "reduced"
    HALTED = "halted"


_MULTIPLIERS = {
    SurvivalMode.NORMAL: 1.0,
    SurvivalMode.REDUCED: 0.5,
    SurvivalMode.HALTED: 0.0,
}

# Singleton instance
_instance: Optional[SurvivalTracker] = None


class SurvivalTracker:
    """Tracks P&L vs. costs and gates trading aggression."""

    def __init__(
        self,
        config: BotConfig,
        perf_tracker: PerfTracker,
        cost_tracker: CostTracker,
        *,
        state_dir: str = "state",
    ):
        self.logger = structlog.get_logger()

        survival_cfg = getattr(config, "survival", None) or {}
        if not isinstance(survival_cfg, dict):
            survival_cfg = {}

        self.enabled = bool(survival_cfg.get("enabled", True))
        self.vps_cost_daily = float(survival_cfg.get("vps_cost_daily", 0.33))
        self.reduced_window_days = int(survival_cfg.get("reduced_window_days", 7))
        self.halted_window_days = int(survival_cfg.get("halted_window_days", 14))
        self.reduced_multiplier = float(survival_cfg.get("reduced_multiplier", 0.5))
        self.grace_period_days = int(survival_cfg.get("grace_period_days", 7))

        self._perf_tracker = perf_tracker
        self._cost_tracker = cost_tracker

        self._state_path = Path(state_dir) / "survival.json"
        self._mode = SurvivalMode.NORMAL
        self._first_run_date: Optional[str] = None

        self._load_state()

        self.logger.info(
            "survival_tracker_init",
            enabled=self.enabled,
            mode=self._mode.value,
            first_run_date=self._first_run_date,
            grace_period_days=self.grace_period_days,
            vps_cost_daily=self.vps_cost_daily,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def mode(self) -> SurvivalMode:
        return self._mode

    @property
    def multiplier(self) -> float:
        if not self.enabled:
            return 1.0
        return _MULTIPLIERS.get(self._mode, 1.0)

    def evaluate(self) -> SurvivalMode:
        """Re-evaluate survival mode based on current P&L and costs.

        Call this at the top of each orchestrator cycle.
        Returns the current mode.
        """
        if not self.enabled:
            return SurvivalMode.NORMAL

        # Grace period: don't penalize during ramp-up
        if self._in_grace_period():
            self._mode = SurvivalMode.NORMAL
            return self._mode

        # Compute profit for each window
        profit_short = self._compute_profit(self.reduced_window_days)
        profit_long = self._compute_profit(self.halted_window_days)

        old_mode = self._mode

        if profit_short < 0 and profit_long < 0:
            self._mode = SurvivalMode.HALTED
        elif profit_short < 0:
            self._mode = SurvivalMode.REDUCED
        else:
            self._mode = SurvivalMode.NORMAL

        if self._mode != old_mode:
            self.logger.warning(
                "survival_mode_changed",
                old_mode=old_mode.value,
                new_mode=self._mode.value,
                profit_7d=round(profit_short, 2),
                profit_14d=round(profit_long, 2),
            )
            self._save_state()

        return self._mode

    def format_status_line(self) -> str:
        """Format a status line for the Telegram daily summary."""
        if not self.enabled:
            return "Survival: disabled"

        profit_short = self._compute_profit(self.reduced_window_days)
        profit_long = self._compute_profit(self.halted_window_days)

        cost_summary = self._cost_tracker.get_summary()
        llm_spend = cost_summary.get("total_spend", 0.0)

        mode_icons = {
            SurvivalMode.NORMAL: "NORMAL",
            SurvivalMode.REDUCED: "REDUCED",
            SurvivalMode.HALTED: "HALTED",
        }

        grace = " (grace)" if self._in_grace_period() else ""
        icon = mode_icons.get(self._mode, "?")
        multiplier_pct = int(self.multiplier * 100)

        return (
            f"Survival: {icon}{grace} ({multiplier_pct}% sizing)\n"
            f"  {self.reduced_window_days}d profit: ${profit_short:+.2f} | "
            f"{self.halted_window_days}d profit: ${profit_long:+.2f}\n"
            f"  LLM spend (month): ${llm_spend:.2f}"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_profit(self, days: int) -> float:
        """Compute profit = trading_pnl - (llm_costs + vps_costs) over N days."""
        stats = self._perf_tracker.get_stats(days=days)
        trading_pnl = float(stats.get("total_pnl", 0.0))

        # LLM costs: use monthly total prorated to the window
        # (cost_tracker only tracks current month, so we approximate)
        cost_summary = self._cost_tracker.get_summary()
        monthly_llm = float(cost_summary.get("total_spend", 0.0))

        # Prorate monthly LLM spend to the window
        now = datetime.now(timezone.utc)
        days_into_month = now.day
        if days_into_month > 0:
            daily_llm_rate = monthly_llm / days_into_month
        else:
            daily_llm_rate = 0.0
        llm_cost_window = daily_llm_rate * days

        vps_cost_window = self.vps_cost_daily * days

        profit = trading_pnl - llm_cost_window - vps_cost_window
        return profit

    def _in_grace_period(self) -> bool:
        """Check if we're still within the grace period from first run."""
        if not self._first_run_date:
            return True  # shouldn't happen, but safe default

        try:
            first_run = datetime.fromisoformat(self._first_run_date)
            if first_run.tzinfo is None:
                first_run = first_run.replace(tzinfo=timezone.utc)
            elapsed = datetime.now(timezone.utc) - first_run
            return elapsed.days < self.grace_period_days
        except (ValueError, TypeError):
            return False

    def _load_state(self) -> None:
        state = load_json_state(self._state_path)
        if state:
            self._first_run_date = state.get("first_run_date")
            mode_str = state.get("last_mode", "normal")
            try:
                self._mode = SurvivalMode(mode_str)
            except ValueError:
                self._mode = SurvivalMode.NORMAL
        else:
            # First run ever
            self._first_run_date = datetime.now(timezone.utc).isoformat()
            self._mode = SurvivalMode.NORMAL
            self._save_state()

    def _save_state(self) -> None:
        save_json_state(
            {
                "first_run_date": self._first_run_date,
                "last_mode": self._mode.value,
                "last_update": datetime.now(timezone.utc).isoformat(),
            },
            self._state_path,
        )


def get_survival_tracker(
    config: BotConfig,
    perf_tracker: PerfTracker,
    cost_tracker: CostTracker,
    *,
    state_dir: str = "state",
) -> SurvivalTracker:
    """Get or create the singleton SurvivalTracker."""
    global _instance
    if _instance is None:
        _instance = SurvivalTracker(
            config, perf_tracker, cost_tracker, state_dir=state_dir,
        )
    return _instance
