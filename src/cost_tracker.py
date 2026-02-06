"""OpenAI API cost tracking and monthly budget enforcement.

Tracks estimated spend per model and halts LLM calls when budget is exceeded.
State persists to <state_dir>/cost_tracker.json.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger()

# Approximate costs per 1M tokens (USD) — update as pricing changes.
MODEL_COSTS: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "o3": {"input": 10.00, "output": 40.00},
    # Anthropic models
    "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
}

DEFAULT_COST = {"input": 5.00, "output": 15.00}  # fallback for unknown models


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a single API call."""
    rates = MODEL_COSTS.get(model, DEFAULT_COST)
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


class CostTracker:
    """Tracks cumulative OpenAI spend and enforces a monthly budget cap."""

    def __init__(self, monthly_budget: float = 100.0, daily_budget: float = 5.0, state_path: Optional[Path] = None):
        self.monthly_budget = monthly_budget
        self.daily_budget = daily_budget
        self.state_path = state_path or Path("state/cost_tracker.json")

        self._current_month: str = ""
        self._current_day: str = ""
        self._total_spend: float = 0.0
        self._daily_spend: float = 0.0
        self._call_count: int = 0
        self._spend_by_model: Dict[str, float] = {}

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def budget_remaining(self) -> float:
        self._maybe_reset_month()
        return max(0.0, self.monthly_budget - self._total_spend)

    @property
    def is_budget_exceeded(self) -> bool:
        self._maybe_reset_month()
        self._maybe_reset_day()
        return self._total_spend >= self.monthly_budget or self._daily_spend >= self.daily_budget

    def record_call(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record an API call. Returns estimated cost of this call."""
        self._maybe_reset_month()
        self._maybe_reset_day()
        cost = _estimate_cost(model, input_tokens, output_tokens)
        self._total_spend += cost
        self._daily_spend += cost
        self._call_count += 1
        self._spend_by_model[model] = self._spend_by_model.get(model, 0.0) + cost
        self._save()

        logger.debug(
            "cost_tracked",
            model=model,
            call_cost=round(cost, 5),
            total_spend=round(self._total_spend, 2),
            budget_remaining=round(self.budget_remaining, 2),
        )

        if self._total_spend >= self.monthly_budget * 0.9 and self._total_spend - cost < self.monthly_budget * 0.9:
            logger.warning(
                "cost_budget_90pct",
                total_spend=round(self._total_spend, 2),
                monthly_budget=self.monthly_budget,
            )

        return cost

    def check_budget(self) -> bool:
        """Return True if we can still make API calls. Logs warning if exceeded."""
        self._maybe_reset_day()
        if self._daily_spend >= self.daily_budget:
            logger.error(
                "cost_daily_budget_exceeded",
                daily_spend=round(self._daily_spend, 2),
                daily_budget=self.daily_budget,
                msg="LLM calls halted until tomorrow",
            )
            return False
        if self.is_budget_exceeded:
            logger.error(
                "cost_budget_exceeded",
                total_spend=round(self._total_spend, 2),
                monthly_budget=self.monthly_budget,
                msg="LLM calls halted until next month",
            )
            return False
        return True

    def get_summary(self) -> Dict[str, Any]:
        self._maybe_reset_month()
        self._maybe_reset_day()
        return {
            "month": self._current_month,
            "total_spend": round(self._total_spend, 2),
            "daily_spend": round(self._daily_spend, 2),
            "monthly_budget": self.monthly_budget,
            "daily_budget": self.daily_budget,
            "budget_remaining": round(self.budget_remaining, 2),
            "daily_remaining": round(max(0.0, self.daily_budget - self._daily_spend), 2),
            "call_count": self._call_count,
            "spend_by_model": {k: round(v, 4) for k, v in self._spend_by_model.items()},
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _current_month_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _current_day_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _maybe_reset_day(self) -> None:
        day = self._current_day_str()
        if day != self._current_day:
            if self._current_day:
                logger.info(
                    "cost_day_reset",
                    old_day=self._current_day,
                    old_daily_spend=round(self._daily_spend, 2),
                    new_day=day,
                )
            self._current_day = day
            self._daily_spend = 0.0

    def _maybe_reset_month(self) -> None:
        month = self._current_month_str()
        if month != self._current_month:
            if self._current_month:
                logger.info(
                    "cost_month_reset",
                    old_month=self._current_month,
                    old_spend=round(self._total_spend, 2),
                    new_month=month,
                )
            self._current_month = month
            self._total_spend = 0.0
            self._call_count = 0
            self._spend_by_model = {}
            self._save()

    def _load(self) -> None:
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text())
                self._current_month = data.get("month", "")
                self._current_day = data.get("day", "")
                self._total_spend = float(data.get("total_spend", 0.0))
                self._daily_spend = float(data.get("daily_spend", 0.0))
                self._call_count = int(data.get("call_count", 0))
                self._spend_by_model = data.get("spend_by_model", {})
                # Auto-reset if month/day changed
                self._maybe_reset_month()
                self._maybe_reset_day()
                logger.info(
                    "cost_tracker_loaded",
                    month=self._current_month,
                    total_spend=round(self._total_spend, 2),
                    budget_remaining=round(self.budget_remaining, 2),
                )
        except Exception as e:
            logger.warning("cost_tracker_load_error", error=str(e))
            self._current_month = self._current_month_str()

    def _save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({
                "month": self._current_month,
                "day": self._current_day,
                "total_spend": self._total_spend,
                "daily_spend": self._daily_spend,
                "call_count": self._call_count,
                "spend_by_model": self._spend_by_model,
                "monthly_budget": self.monthly_budget,
                "daily_budget": self.daily_budget,
            }, indent=2))
        except Exception as e:
            logger.warning("cost_tracker_save_error", error=str(e))
