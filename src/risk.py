"""Risk management — tuned for $200 capital, same-day markets.

Position sizing: bankroll * kelly * 0.33 (third-Kelly)
- Hard cap at 5% of bankroll per trade ($10 on $200)
- 3% minimum edge
- No conviction gating
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

import structlog

from .markets import Market
from .signals import SignalResult
from .utils import BotConfig, calculate_kelly_fraction, load_json_state, save_json_state


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionSize:
    amount_usd: float
    kelly_fraction: float
    risk_level: RiskLevel
    reasoning: str
    max_loss: float

    @property
    def is_valid(self) -> bool:
        return self.amount_usd > 0.0


@dataclass
class RiskMetrics:
    total_exposure: float
    max_position_size: float
    daily_pnl: float
    max_daily_loss: float
    open_positions: int
    available_capital: float
    risk_utilization: float

    @property
    def is_overexposed(self) -> bool:
        return self.risk_utilization > 0.9


class RiskManager:
    """Manages portfolio risk and position sizing — disciplined mode."""

    def __init__(self, config: BotConfig, *, state_dir: str = "."):
        self.config = config
        self.state_dir = state_dir
        self.logger = structlog.get_logger()

        risk_config = config.risk
        self.max_loss_per_trade = risk_config.get("max_loss_per_trade", 10.0)
        self.max_daily_loss = risk_config.get("max_daily_loss", 20.0)
        self.stop_loss_pct = risk_config.get("stop_loss_pct", 0.35)
        self.take_profit_pct = risk_config.get("take_profit_pct", 0.30)
        self.max_position_hold_hours = risk_config.get("max_position_hold_hours", 24)

        # Third-Kelly with 5% bankroll cap
        self.use_quarter_kelly = risk_config.get("use_quarter_kelly", False)
        self.max_bankroll_pct = risk_config.get("max_bankroll_pct", 0.05)  # 5%

        strategy_config = config.strategy
        self.kelly_fraction = strategy_config.get("kelly_fraction", 0.33)
        self.max_position_size = strategy_config.get("max_position_size", 15.0)
        self.max_total_exposure = strategy_config.get("max_total_exposure", 80.0)
        self.min_edge = strategy_config.get("min_edge", 0.03)
        self.min_conviction = strategy_config.get("min_conviction", "low")

        # HIGH conviction multiplier (reduced from 2x to 1.5x)
        self.high_conviction_multiplier = float(
            strategy_config.get("high_conviction_multiplier", 1.5)
        )

        from pathlib import Path
        self.state_file = str(Path(self.state_dir) / "risk_state.json")
        self.daily_pnl = 0.0
        self.daily_pnl_date = datetime.now(timezone.utc).date()
        self.trading_halted = False
        self._load_state()

    def calculate_position_size(
        self,
        signal: SignalResult,
        market: Market,
        available_capital: float,
        current_positions: Dict[str, float],
    ) -> PositionSize:
        try:
            if self.trading_halted:
                return PositionSize(0.0, 0.0, RiskLevel.CRITICAL,
                                    "Trading halted — daily loss limit", 0.0)

            # Check for forced size override (arb/copy signals set this)
            forced_size = None
            meta = getattr(signal, "metadata", None)
            if isinstance(meta, dict):
                forced_size = meta.get("_force_size_usd")

            market_price = signal.market_price
            if not market_price or market_price <= 0:
                return PositionSize(0.0, 0.0, RiskLevel.HIGH, "Invalid market price", 0.0)

            # Odds for Kelly
            odds = (1.0 / market_price) - 1.0 if market_price > 0 else 0.0
            if odds <= 0 and forced_size is None:
                return PositionSize(0.0, 0.0, RiskLevel.HIGH, "Invalid odds", 0.0)

            if forced_size is not None and forced_size > 0:
                # Use forced size directly (arb: mathematically sized)
                position_amount = float(forced_size)
                kelly_f = 0.0
            else:
                edge = abs(signal.edge)
                
                # Calculate Kelly fraction
                kelly_f = calculate_kelly_fraction(edge, odds, self.kelly_fraction)
                
                # Third-Kelly with bankroll cap
                kelly_bet = available_capital * kelly_f * self.kelly_fraction
                max_bet = available_capital * self.max_bankroll_pct
                position_amount = min(kelly_bet, max_bet)

            # Conviction-based max (string-based, not enum)
            conviction = getattr(signal, "conviction", "low")
            conv_str = conviction.value if hasattr(conviction, "value") else str(conviction).lower()
            effective_max = self.max_position_size
            if conv_str == "high":
                effective_max *= self.high_conviction_multiplier

            position_amount = min(
                position_amount,
                effective_max,
                self.max_loss_per_trade / self.stop_loss_pct if self.stop_loss_pct > 0 else effective_max,
            )

            # Exposure limit
            current_exposure = sum(current_positions.values())
            remaining_exposure = self.max_total_exposure - current_exposure
            position_amount = min(position_amount, max(0, remaining_exposure))

            # No confidence scaling — edge already accounts for uncertainty

            risk_level = self._assess_risk_level(
                position_amount, available_capital, current_exposure, signal
            )

            max_loss = position_amount * self.stop_loss_pct

            parts = [
                f"Kelly={kelly_f:.3f}",
                f"edge={abs(signal.edge):.3f}",
                f"conf={signal.confidence:.2f}",
                f"conviction={conv_str}",
                f"risk={risk_level.value}",
            ]
            parts = [p for p in parts if p]  # remove empty
            if forced_size is not None:
                parts.insert(0, f"forced=${forced_size:.2f}")

            self.logger.info(
                "position_sized",
                market_id=market.id,
                amount=position_amount,
                kelly=kelly_f,
                conviction=conv_str,
                forced=forced_size is not None,
            )

            return PositionSize(
                amount_usd=max(0.0, position_amount),
                kelly_fraction=kelly_f,
                risk_level=risk_level,
                reasoning=", ".join(parts),
                max_loss=max_loss,
            )

        except Exception as e:
            self.logger.error("position_sizing_error", market_id=market.id, error=str(e))
            return PositionSize(0.0, 0.0, RiskLevel.CRITICAL, f"Error: {e}", 0.0)

    def check_trade_approval(
        self, position_size: PositionSize, market: Market, signal: SignalResult
    ) -> bool:
        if not position_size.is_valid:
            return False
        if self.trading_halted:
            return False
        if position_size.risk_level == RiskLevel.CRITICAL:
            return False

        # Edge check using net_edge (already accounts for fees/slippage)
        net_edge = getattr(signal, "net_edge", None)
        if net_edge is not None:
            if abs(net_edge) < self.min_edge:
                self.logger.debug(
                    "trade_rejected_low_edge",
                    market_id=market.id,
                    net_edge=net_edge,
                    min_edge=self.min_edge,
                )
                return False

        # No conviction gating — all positive-edge trades approved
        # No closing-soon rejection — we trade same-day markets

        conviction = getattr(signal, "conviction", "low")
        conv_str = conviction.value if hasattr(conviction, "value") else str(conviction).lower()

        self.logger.info(
            "trade_approved",
            market_id=market.id,
            amount=position_size.amount_usd,
            conviction=conv_str,
        )
        return True

    def update_daily_pnl(self, pnl_change: float) -> None:
        current_date = datetime.now(timezone.utc).date()
        if current_date != self.daily_pnl_date:
            self.daily_pnl = 0.0
            self.daily_pnl_date = current_date
            self.trading_halted = False
            self.logger.info("new_trading_day", date=current_date)

        self.daily_pnl += pnl_change

        if self.daily_pnl <= -self.max_daily_loss:
            self.trading_halted = True
            self.logger.error("daily_loss_limit_hit",
                              daily_pnl=self.daily_pnl, limit=self.max_daily_loss)
        self._save_state()

    def get_risk_metrics(
        self, available_capital: float, current_positions: Dict[str, float]
    ) -> RiskMetrics:
        total_exposure = sum(current_positions.values())
        risk_utilization = (
            total_exposure / self.max_total_exposure if self.max_total_exposure > 0 else 0.0
        )
        return RiskMetrics(
            total_exposure=total_exposure,
            max_position_size=self.max_position_size,
            daily_pnl=self.daily_pnl,
            max_daily_loss=self.max_daily_loss,
            open_positions=len(current_positions),
            available_capital=available_capital,
            risk_utilization=risk_utilization,
        )

    def should_close_position(
        self,
        entry_price: float,
        current_price: float,
        entry_time: datetime,
        position_amount: float,
    ) -> tuple[bool, str]:
        now = datetime.now(timezone.utc)
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

        if pnl_pct <= -self.stop_loss_pct:
            return True, f"Stop loss: {pnl_pct:.1%}"
        if pnl_pct >= self.take_profit_pct:
            return True, f"Take profit: {pnl_pct:.1%}"

        hours_held = (now - entry_time).total_seconds() / 3600
        if hours_held >= self.max_position_hold_hours:
            return True, f"Max hold time: {hours_held:.1f}h"

        return False, "Within risk parameters"

    def _assess_risk_level(
        self,
        position_amount: float,
        available_capital: float,
        current_exposure: float,
        signal: SignalResult,
    ) -> RiskLevel:
        position_pct = position_amount / available_capital if available_capital > 0 else 1.0
        new_exposure = current_exposure + position_amount
        exposure_util = new_exposure / self.max_total_exposure if self.max_total_exposure > 0 else 1.0

        risk_factors = []
        if position_pct > 0.05:  # lowered from 0.25 to 0.05 (5%)
            risk_factors.append("large_position")
        if exposure_util > 0.70:  # lowered from 0.85 to 0.70
            risk_factors.append("high_exposure")
        if signal.confidence < 0.6:
            risk_factors.append("low_confidence")

        if len(risk_factors) >= 3:
            return RiskLevel.CRITICAL
        elif len(risk_factors) == 2:
            return RiskLevel.HIGH
        elif len(risk_factors) == 1:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _load_state(self) -> None:
        try:
            state = load_json_state(self.state_file)
            if state:
                self.daily_pnl = state.get("daily_pnl", 0.0)
                date_str = state.get("daily_pnl_date")
                if date_str:
                    self.daily_pnl_date = datetime.fromisoformat(date_str).date()
                self.trading_halted = state.get("trading_halted", False)
        except Exception as e:
            self.logger.warning("risk_state_load_error", error=str(e))

    def _save_state(self) -> None:
        try:
            save_json_state(
                {
                    "daily_pnl": self.daily_pnl,
                    "daily_pnl_date": self.daily_pnl_date.isoformat(),
                    "trading_halted": self.trading_halted,
                    "last_update": datetime.now(timezone.utc).isoformat(),
                },
                self.state_file,
            )
        except Exception as e:
            self.logger.warning("risk_state_save_error", error=str(e))
