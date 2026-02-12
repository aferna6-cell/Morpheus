"""Kalshi trade executor.

Takes approved :class:`TradeSignal` objects (from the Kalshi LLM engine)
and places orders via :class:`KalshiTradingClient`.  Handles order
confirmation, rejection, and Telegram alerting.

Kalshi contracts pay $1 each, so ``position_size_usd ≈ number of contracts``.
Fees are ~$0.07 per contract (flat, not percentage-based).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import structlog

from .alerts import send_alert
from .kalshi_trading_client import KalshiTradingClient
from .markets import Market
from .risk import PositionSize, RiskManager
from .signals.base import SignalResult, TradingSide
from .trade_logger import get_trade_logger
from .utils import BotConfig


@dataclass
class KalshiTradeExecution:
    """Result of a Kalshi trade attempt."""
    ticker: str
    side: str
    intended_contracts: int
    executed_contracts: int
    price_cents: int
    executed_amount_usd: float
    average_price: float       # 0-1 scale
    order_id: Optional[str]
    was_successful: bool
    reason: str
    execution_time: datetime

    # Alias for orchestrator compatibility
    @property
    def executed_amount(self) -> float:
        return self.executed_amount_usd


class KalshiExecutor:
    """Executes Kalshi trades from orchestrator signals."""

    def __init__(
        self,
        config: BotConfig,
        trading_client: KalshiTradingClient,
        risk_manager: RiskManager,
    ):
        self.config = config
        self.trading_client = trading_client
        self.risk_manager = risk_manager
        self.logger = structlog.get_logger()

        kalshi_cfg = getattr(config, "kalshi", None) or {}
        if isinstance(kalshi_cfg, dict):
            self._fee_per_contract = float(kalshi_cfg.get("fee_per_contract", 0.07))
            self._max_contracts = int(kalshi_cfg.get("max_position_contracts", 100))
        else:
            self._fee_per_contract = 0.07
            self._max_contracts = 100

    async def execute_signal(
        self,
        market: Market,
        signal: SignalResult,
        position_size: PositionSize,
    ) -> KalshiTradeExecution:
        """Place a Kalshi order for the given signal.

        Args:
            market: The Market object (converted from KalshiMarket).
            signal: The approved SignalResult.
            position_size: Risk-manager-approved position sizing.

        Returns:
            KalshiTradeExecution with fill details.
        """
        ticker = market.id  # This is the Kalshi ticker

        # Determine side and price
        meta = getattr(signal, "metadata", None) or {}
        signal_source = meta.get("signal_source") or getattr(signal, "signal_source", None)

        if signal.recommended_side == TradingSide.BUY_YES:
            side = "yes"
            # Use yes_ask from metadata if available, else market price
            ask_price = meta.get("kalshi_yes_ask") or signal.market_price or market.yes_price or 0.5
            price_cents = max(1, min(99, round(ask_price * 100)))
        elif signal.recommended_side == TradingSide.BUY_NO:
            side = "no"
            ask_price = meta.get("kalshi_no_ask") or (
                1.0 - (signal.market_price or market.yes_price or 0.5)
            )
            price_cents = max(1, min(99, round(ask_price * 100)))
        else:
            return KalshiTradeExecution(
                ticker=ticker,
                side="hold",
                intended_contracts=0,
                executed_contracts=0,
                price_cents=0,
                executed_amount_usd=0.0,
                average_price=0.0,
                order_id=None,
                was_successful=False,
                reason="Signal is HOLD",
                execution_time=datetime.now(timezone.utc),
            )

        # NOAA weather signals: cross the spread proportionally to improve fill rate.
        # Dynamic: min(spread/2, 3c) — adapts to actual book depth instead of flat +2c.
        if signal_source == "noaa_direct" and signal.confidence >= 0.70:
            # Compute spread from yes_ask and no_ask if available
            yes_ask = meta.get("kalshi_yes_ask")
            no_ask = meta.get("kalshi_no_ask")
            if yes_ask is not None and no_ask is not None:
                spread_cents = max(0, round((yes_ask + no_ask - 1.0) * 100))
                cross_amount = min(max(1, spread_cents // 2), 3)
            else:
                cross_amount = 2  # fallback to flat 2c if no spread data
            original_price = price_cents
            price_cents = min(99, price_cents + cross_amount)
            self.logger.info(
                "weather_spread_cross",
                ticker=ticker,
                side=side,
                original_price=original_price,
                crossed_price=price_cents,
                cross_amount=cross_amount,
                spread_cents=spread_cents if yes_ask and no_ask else "unknown",
            )

        # Calculate contracts: $1 per contract, so USD ≈ contracts
        # But we pay price_cents/100 per contract (cost = count * price / 100)
        entry_cost = price_cents / 100.0
        if entry_cost <= 0:
            return self._fail(ticker, side, 0, 0, "Invalid entry cost")

        raw_count = math.floor(position_size.amount_usd / entry_cost)
        count = min(max(raw_count, 1), self._max_contracts)

        # Dollar-cost guard: ensure actual cost doesn't exceed approved position size
        actual_cost = count * entry_cost
        if actual_cost > position_size.amount_usd * 1.1 and count > 1:
            count = max(1, math.floor(position_size.amount_usd / entry_cost))

        total_fee = count * self._fee_per_contract
        total_cost = count * entry_cost + total_fee

        # Fee-drag guard: reject if fees exceed 2x expected edge value
        net_edge = getattr(signal, "net_edge", signal.edge)
        expected_edge_value = count * entry_cost * abs(net_edge)
        if expected_edge_value > 0 and total_fee > 2.0 * expected_edge_value:
            return self._fail(
                ticker, side, count, price_cents,
                f"Fee drag too high: ${total_fee:.2f} fee vs ${expected_edge_value:.2f} edge",
            )

        self.logger.info(
            "kalshi_execute",
            ticker=ticker,
            side=side,
            count=count,
            price_cents=price_cents,
            cost_usd=total_cost,
            fee_usd=total_fee,
        )

        # Place the order
        result = await self.trading_client.place_order(
            ticker=ticker,
            side=side,
            count=count,
            price_cents=price_cents,
            order_type="limit",
        )

        # Get trade logger
        trade_logger = get_trade_logger()

        if not result:
            # Log failed order
            conviction = getattr(signal, "conviction", "medium")
            conv_str = conviction.value if hasattr(conviction, "value") else str(conviction)
            trade_logger.log_order_failed(
                platform="kalshi",
                ticker=ticker,
                side=side,
                count=count,
                price_cents=price_cents,
                error="Order placement failed - no response",
                conviction=conv_str,
                account_label=self.trading_client.label,
            )
            return self._fail(ticker, side, count, price_cents, "Order placement failed")

        order_id = (
            result.get("order_id")
            or result.get("order", {}).get("order_id", "unknown")
            if isinstance(result, dict) else "unknown"
        )
        status = (
            result.get("status")
            or result.get("order", {}).get("status", "unknown")
            if isinstance(result, dict) else "unknown"
        )

        # For limit orders, we consider it successful if it was accepted
        # (resting or executed). Full fill tracking would need polling.
        is_success = status in {"resting", "executed", "FILLED", "OPEN"}
        executed_count = count if status == "executed" else 0
        executed_usd = executed_count * entry_cost

        # Log the order to trade history
        conviction = getattr(signal, "conviction", "medium")
        conv_str = conviction.value if hasattr(conviction, "value") else str(conviction)
        net_edge = getattr(signal, "net_edge", signal.edge)

        if is_success:
            # Get entry probability for CLV tracking
            entry_probability = signal.estimated_prob if hasattr(signal, "estimated_prob") else entry_cost

            trade_logger.log_order_placed(
                platform="kalshi",
                ticker=ticker,
                side=side,
                count=count,
                price_cents=price_cents,
                order_id=str(order_id),
                order_type="limit",
                conviction=conv_str,
                edge=net_edge,
                cost_usd=total_cost,
                entry_probability=entry_probability,
                market_price=entry_cost,
                account_label=self.trading_client.label,
            )
        else:
            trade_logger.log_order_failed(
                platform="kalshi",
                ticker=ticker,
                side=side,
                count=count,
                price_cents=price_cents,
                error=f"Order rejected: {status}",
                conviction=conv_str,
                account_label=self.trading_client.label,
            )

        # FIXED: Don't pretend resting orders filled — track actual status
        # Resting orders will be monitored by FillManager
        execution = KalshiTradeExecution(
            ticker=ticker,
            side=side,
            intended_contracts=count,
            executed_contracts=executed_count,  # 0 if resting, count if executed
            price_cents=price_cents,
            executed_amount_usd=executed_usd,   # 0 if resting
            average_price=entry_cost,
            order_id=str(order_id),
            was_successful=is_success,
            reason=f"ok ({status})" if is_success else f"Order status: {status}",
            execution_time=datetime.now(timezone.utc),
        )

        # Send Telegram alert
        if is_success:
            acct = self.trading_client.label or "kalshi"
            alert_msg = (
                f"🎯 [{acct}] {side.upper()} {count}x {ticker}\n"
                f"Price: {price_cents}¢ | Cost: ${count * entry_cost:.2f} + ${total_fee:.2f} fee\n"
                f"Edge: {net_edge:+.3f} | Conv: {conv_str}\n"
                f"{market.question[:100]}"
            )
            try:
                await send_alert(alert_msg, self.config)
            except Exception as exc:
                self.logger.warning("kalshi_alert_failed", error=str(exc))

        self.logger.info(
            "kalshi_execution_result",
            ticker=ticker,
            side=side,
            count=count,
            success=is_success,
            order_id=str(order_id),
        )

        return execution

    def _fail(
        self, ticker: str, side: str, count: int, price_cents: int, reason: str,
    ) -> KalshiTradeExecution:
        self.logger.warning("kalshi_execution_failed", ticker=ticker, reason=reason)
        return KalshiTradeExecution(
            ticker=ticker,
            side=side,
            intended_contracts=count,
            executed_contracts=0,
            price_cents=price_cents,
            executed_amount_usd=0.0,
            average_price=0.0,
            order_id=None,
            was_successful=False,
            reason=reason,
            execution_time=datetime.now(timezone.utc),
        )
