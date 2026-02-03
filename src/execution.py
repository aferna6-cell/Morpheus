"""Order execution & portfolio integration.

Goals:
- Deterministic YES/NO token mapping (no unsafe fallbacks)
- Prefer limit orders with configurable slippage
- Reconcile orders (poll status/trades). Do not assume fills.
- Provide basic position lifecycle hooks (close positions).

This remains intentionally minimal; it uses best-effort py-clob-client wrappers.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import structlog

from .client import OrderSide, PolymarketClient
from .markets import Market
from .portfolio import Portfolio
from .risk import PositionSize
from .signals.base import SignalResult, TradingSide
from .trade_logger import get_trade_logger
from .utils import append_jsonl


class ExecutionResult(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class TradeExecution:
    market_id: str
    token_id: str
    side: str
    intended_amount_usd: float
    executed_amount_usd: float
    average_price: float
    execution_time: datetime
    order_id: Optional[str]
    result: ExecutionResult
    reason: str

    @property
    def was_successful(self) -> bool:
        return self.result == ExecutionResult.SUCCESS


def _clamp_price(p: float) -> float:
    return max(0.0001, min(0.9999, p))


class OrderManager:
    def __init__(self, config, client: PolymarketClient, portfolio: Portfolio,
                 state_dir: Optional[str] = None, dry_run: bool = False):
        self.config = config
        self.client = client
        self.portfolio = portfolio
        self.state_dir = state_dir
        self.dry_run = dry_run
        self.logger = structlog.get_logger()

    def _log_edge_decay(
        self,
        market: Market,
        signal: SignalResult,
        execution_price: float,
        decision_time: datetime,
    ) -> None:
        """Log edge decay between decision and execution to edge_decay.jsonl."""
        if not self.state_dir:
            return
        from pathlib import Path

        decision_price = signal.market_price or market.midpoint_price or 0.5
        predicted_p_yes = signal.estimated_prob
        side = signal.recommended_side.value
        now = datetime.now(timezone.utc)
        time_delta = (now - decision_time).total_seconds()

        edge_at_decision = abs(predicted_p_yes - decision_price)
        edge_at_execution = abs(predicted_p_yes - execution_price)

        record = {
            "market_id": market.id,
            "decision_price": round(decision_price, 6),
            "execution_price": round(execution_price, 6),
            "predicted_p_yes": round(predicted_p_yes, 4),
            "side": side,
            "time_delta_seconds": round(time_delta, 1),
            "edge_at_decision": round(edge_at_decision, 4),
            "edge_at_execution": round(edge_at_execution, 4),
            "timestamp": now.isoformat(),
        }
        append_jsonl(Path(self.state_dir) / "edge_decay.jsonl", record)

    def _log_bankroll_entry(
        self,
        market: Market,
        signal: SignalResult,
        executed_usd: float,
        avg_price: float,
    ) -> None:
        """Log trade entry cost to bankroll."""
        if not self.state_dir or self.dry_run:
            return
        try:
            from .bankroll import log_bankroll_event
            fee_pct = float(self.config.strategy.get("fee_pct", 0.02))
            log_bankroll_event(
                self.state_dir,
                market_id=market.id,
                side=signal.recommended_side.value,
                amount_usdc=executed_usd,
                price=avg_price,
                fee_estimate=round(executed_usd * fee_pct, 4),
                event_type="entry",
            )
        except Exception as e:
            self.logger.warning("bankroll_log_error", error=str(e))

    def pick_token_id(self, market: Market, side: TradingSide) -> Optional[str]:
        """Deterministically map BUY_YES/BUY_NO to the correct token.

        Gamma outcomes can come in any order. We require explicit outcome labels.
        """

        wanted = "yes" if side == TradingSide.BUY_YES else "no"
        for outcome, tok in market.tokens.items():
            if outcome and outcome.strip().lower() == wanted:
                return tok.token_id
        return None

    def _reference_price_for_side(
        self, market: Market, side: TradingSide
    ) -> Optional[float]:
        if side == TradingSide.BUY_YES:
            return market.yes_price
        if side == TradingSide.BUY_NO:
            return market.no_price
        return None

    async def _reconcile_order(
        self, *, order_id: str, token_id: str, timeout_s: float
    ) -> Tuple[float, float]:
        """Return (filled_shares, avg_price)."""

        deadline = asyncio.get_event_loop().time() + timeout_s
        filled_shares: float = 0.0
        avg_price: float = 0.0

        while asyncio.get_event_loop().time() < deadline:
            o = await self.client.get_order_by_id(order_id)
            status = (o or {}).get("status")

            # Best-effort fields (vary by version)
            if o:
                if o.get("filled_size") is not None:
                    try:
                        filled_shares = float(o.get("filled_size"))
                    except Exception:
                        pass
                if o.get("avg_price") is not None:
                    try:
                        avg_price = float(o.get("avg_price"))
                    except Exception:
                        pass

            if status in {"FILLED", "CLOSED", "DONE"}:
                break

            # Fallback: trades lookup (many versions include order_id on trade)
            trades = await self.client.get_trades(token_id=token_id)
            t_for_order = [
                t
                for t in trades
                if str(t.get("order_id") or t.get("orderId")) == str(order_id)
            ]
            if t_for_order:
                try:
                    sizes = [
                        float(t.get("size") or t.get("amount") or 0.0)
                        for t in t_for_order
                    ]
                    prices = [float(t.get("price") or 0.0) for t in t_for_order]
                    filled_shares = sum(sizes)
                    if filled_shares > 0:
                        avg_price = (
                            sum(s * p for s, p in zip(sizes, prices)) / filled_shares
                        )
                except Exception:
                    pass

            if filled_shares > 0 and avg_price > 0:
                # partial fill; keep waiting briefly for completion
                pass

            await asyncio.sleep(2.0)

        return filled_shares, avg_price

    async def execute_signal(
        self, market: Market, signal: SignalResult, position_size: PositionSize
    ) -> TradeExecution:
        decision_time = datetime.now(timezone.utc)
        token_id = self.pick_token_id(market, signal.recommended_side)
        if not token_id:
            return TradeExecution(
                market_id=market.id,
                token_id="",
                side=signal.recommended_side.value,
                intended_amount_usd=position_size.amount_usd,
                executed_amount_usd=0.0,
                average_price=0.0,
                execution_time=datetime.now(timezone.utc),
                order_id=None,
                result=ExecutionResult.FAILED,
                reason="No YES/NO token_id found (requires explicit outcomes)",
            )

        if position_size.amount_usd <= 0:
            return TradeExecution(
                market_id=market.id,
                token_id=token_id,
                side=signal.recommended_side.value,
                intended_amount_usd=position_size.amount_usd,
                executed_amount_usd=0.0,
                average_price=market.midpoint_price or 0.0,
                execution_time=datetime.now(timezone.utc),
                order_id=None,
                result=ExecutionResult.REJECTED,
                reason="Position size was 0",
            )

        if signal.recommended_side not in {TradingSide.BUY_YES, TradingSide.BUY_NO}:
            return TradeExecution(
                market_id=market.id,
                token_id=token_id,
                side=signal.recommended_side.value,
                intended_amount_usd=position_size.amount_usd,
                executed_amount_usd=0.0,
                average_price=market.midpoint_price or 0.0,
                execution_time=datetime.now(timezone.utc),
                order_id=None,
                result=ExecutionResult.REJECTED,
                reason="Signal HOLD",
            )

        exec_cfg = self.config.execution
        prefer_limit = bool(exec_cfg.get("prefer_limit_orders", True))
        slippage = float(exec_cfg.get("default_slippage", 0.0))
        timeout_s = float(exec_cfg.get("order_timeout_seconds", 60))

        # Maker-order config
        maker_offset = float(exec_cfg.get("maker_spread_offset", 0.005))
        urgent_allow_taker = bool(exec_cfg.get("urgent_allow_taker", True))
        limit_timeout_s = float(exec_cfg.get("limit_order_timeout_seconds", 30))

        # Detect urgency flag (set by orchestrator for spike/arb signals)
        is_urgent = getattr(signal, "metadata", None) and isinstance(
            signal.metadata, dict  # type: ignore[union-attr]
        ) and signal.metadata.get("urgent", False)  # type: ignore[union-attr]

        ref_price = self._reference_price_for_side(market, signal.recommended_side)
        if ref_price is None:
            ref_price = await self.client.get_midpoint(token_id)

        if not ref_price or ref_price <= 0:
            return TradeExecution(
                market_id=market.id,
                token_id=token_id,
                side=signal.recommended_side.value,
                intended_amount_usd=position_size.amount_usd,
                executed_amount_usd=0.0,
                average_price=0.0,
                execution_time=datetime.now(timezone.utc),
                order_id=None,
                result=ExecutionResult.FAILED,
                reason="No reference price available",
            )

        # Route: urgent signals use taker (market) orders when allowed
        use_taker = (not prefer_limit) or (is_urgent and urgent_allow_taker)

        if use_taker:
            return await self._execute_taker(
                market, signal, position_size, token_id, ref_price, decision_time,
            )

        # Default: maker (limit) order path — place inside spread for quick fill
        return await self._execute_maker(
            market, signal, position_size, token_id, ref_price,
            decision_time, maker_offset, limit_timeout_s, slippage,
            urgent_allow_taker,
        )

    # ------------------------------------------------------------------
    # Taker (market) order execution
    # ------------------------------------------------------------------

    async def _execute_taker(
        self,
        market: Market,
        signal: SignalResult,
        position_size: PositionSize,
        token_id: str,
        ref_price: float,
        decision_time: datetime,
    ) -> TradeExecution:
        resp = await self.client.place_market_order(
            token_id=token_id, amount=position_size.amount_usd, side=OrderSide.BUY,
            ref_price=ref_price,
        )
        if not resp:
            return TradeExecution(
                market_id=market.id,
                token_id=token_id,
                side=signal.recommended_side.value,
                intended_amount_usd=position_size.amount_usd,
                executed_amount_usd=0.0,
                average_price=ref_price,
                execution_time=datetime.now(timezone.utc),
                order_id=None,
                result=ExecutionResult.FAILED,
                reason="Taker order failed",
            )

        avg_price = float(resp.get("average_price", ref_price))
        executed_usd = float(resp.get("filled_amount", 0.0))
        order_id = resp.get("order_id") or resp.get("id")

        if executed_usd <= 0:
            return TradeExecution(
                market_id=market.id,
                token_id=token_id,
                side=signal.recommended_side.value,
                intended_amount_usd=position_size.amount_usd,
                executed_amount_usd=0.0,
                average_price=avg_price,
                execution_time=datetime.now(timezone.utc),
                order_id=order_id,
                result=ExecutionResult.FAILED,
                reason="No fill reported",
            )

        shares = executed_usd / max(avg_price, 1e-9)
        await self.portfolio.add_position(
            market=market,
            token_id=token_id,
            side=signal.recommended_side.value,
            entry_price=avg_price,
            amount=shares,
        )

        self.logger.info(
            "trade_executed",
            market_id=market.id,
            token_id=token_id,
            side=signal.recommended_side.value,
            usd=executed_usd,
            price=avg_price,
            order_id=order_id,
            order_type="taker",
        )

        self._log_edge_decay(market, signal, avg_price, decision_time)
        self._log_bankroll_entry(market, signal, executed_usd, avg_price)

        # Log to trade history
        trade_logger = get_trade_logger()
        trade_logger.log_order_placed(
            platform="polymarket",
            ticker=market.id,
            side=signal.recommended_side.value,
            count=int(shares),
            price_cents=int(avg_price * 100),
            order_id=str(order_id),
            order_type="taker",
            edge=signal.edge,
            cost_usd=executed_usd,
            entry_probability=signal.estimated_prob,
            market_price=avg_price,
        )

        return TradeExecution(
            market_id=market.id,
            token_id=token_id,
            side=signal.recommended_side.value,
            intended_amount_usd=position_size.amount_usd,
            executed_amount_usd=executed_usd,
            average_price=avg_price,
            execution_time=datetime.now(timezone.utc),
            order_id=order_id,
            result=ExecutionResult.SUCCESS,
            reason="ok (taker)",
        )

    # ------------------------------------------------------------------
    # Maker (limit) order execution — 0% fees
    # ------------------------------------------------------------------

    async def _execute_maker(
        self,
        market: Market,
        signal: SignalResult,
        position_size: PositionSize,
        token_id: str,
        ref_price: float,
        decision_time: datetime,
        maker_offset: float,
        limit_timeout_s: float,
        slippage: float,
        fallback_to_taker: bool,
    ) -> TradeExecution:
        # Place limit slightly inside the spread for quick fills
        # "Inside the spread" for a BUY means a bit above our ref (but below the ask)
        limit_price = _clamp_price(ref_price + maker_offset)
        # Also respect legacy slippage as a ceiling
        ceiling = _clamp_price(ref_price * (1.0 + slippage)) if slippage > 0 else limit_price
        limit_price = min(limit_price, ceiling) if ceiling > 0 else limit_price

        size_shares = position_size.amount_usd / limit_price

        resp = await self.client.place_limit_order(
            token_id=token_id,
            price=limit_price,
            size=size_shares,
            side=OrderSide.BUY,
        )
        if not resp:
            return TradeExecution(
                market_id=market.id,
                token_id=token_id,
                side=signal.recommended_side.value,
                intended_amount_usd=position_size.amount_usd,
                executed_amount_usd=0.0,
                average_price=0.0,
                execution_time=datetime.now(timezone.utc),
                order_id=None,
                result=ExecutionResult.FAILED,
                reason="Maker limit order failed",
            )

        order_id = resp.get("order_id") or resp.get("id")
        if not order_id:
            return TradeExecution(
                market_id=market.id,
                token_id=token_id,
                side=signal.recommended_side.value,
                intended_amount_usd=position_size.amount_usd,
                executed_amount_usd=0.0,
                average_price=0.0,
                execution_time=datetime.now(timezone.utc),
                order_id=None,
                result=ExecutionResult.FAILED,
                reason="Maker limit order returned no id",
            )

        # Wait for fill with the maker timeout
        filled_shares, avg_price = await self._reconcile_order(
            order_id=str(order_id), token_id=token_id, timeout_s=limit_timeout_s,
        )

        if filled_shares <= 0:
            # Cancel unfilled limit order
            await self.client.cancel_order(str(order_id))

            # Fallback: re-place as taker if allowed
            if fallback_to_taker:
                self.logger.info(
                    "maker_timeout_fallback_taker",
                    market_id=market.id,
                    order_id=str(order_id),
                    timeout_s=limit_timeout_s,
                )
                return await self._execute_taker(
                    market, signal, position_size, token_id, ref_price, decision_time,
                )

            return TradeExecution(
                market_id=market.id,
                token_id=token_id,
                side=signal.recommended_side.value,
                intended_amount_usd=position_size.amount_usd,
                executed_amount_usd=0.0,
                average_price=0.0,
                execution_time=datetime.now(timezone.utc),
                order_id=str(order_id),
                result=ExecutionResult.FAILED,
                reason="Maker limit not filled before timeout",
            )

        if avg_price <= 0:
            avg_price = limit_price

        executed_usd = filled_shares * avg_price

        await self.portfolio.add_position(
            market=market,
            token_id=token_id,
            side=signal.recommended_side.value,
            entry_price=avg_price,
            amount=filled_shares,
        )

        self.logger.info(
            "trade_executed",
            market_id=market.id,
            token_id=token_id,
            side=signal.recommended_side.value,
            usd=executed_usd,
            shares=filled_shares,
            price=avg_price,
            order_id=str(order_id),
            order_type="maker",
        )

        self._log_edge_decay(market, signal, avg_price, decision_time)
        self._log_bankroll_entry(market, signal, executed_usd, avg_price)

        # Log to trade history
        trade_logger = get_trade_logger()
        trade_logger.log_order_placed(
            platform="polymarket",
            ticker=market.id,
            side=signal.recommended_side.value,
            count=int(filled_shares),
            price_cents=int(avg_price * 100),
            order_id=str(order_id),
            order_type="maker",
            edge=signal.edge,
            cost_usd=executed_usd,
            entry_probability=signal.estimated_prob,
            market_price=avg_price,
        )

        return TradeExecution(
            market_id=market.id,
            token_id=token_id,
            side=signal.recommended_side.value,
            intended_amount_usd=position_size.amount_usd,
            executed_amount_usd=executed_usd,
            average_price=avg_price,
            execution_time=datetime.now(timezone.utc),
            order_id=str(order_id),
            result=ExecutionResult.SUCCESS,
            reason="ok (maker)",
        )

    async def close_position(self, position_id: str, *, reason: str) -> bool:
        """Attempt to close a position by placing a SELL limit order."""

        position = self.portfolio.positions.get(position_id)
        if not position or position.status.value != "open":
            return False

        exec_cfg = self.config.execution
        slippage = float(exec_cfg.get("default_slippage", 0.0))
        timeout_s = float(exec_cfg.get("order_timeout_seconds", 60))

        current_price = position.current_price
        if current_price is None:
            current_price = await self.client.get_midpoint(position.token_id)

        if not current_price or current_price <= 0:
            self.logger.warning("close_position_no_price", position_id=position_id)
            return False

        limit_price = _clamp_price(current_price * (1.0 - slippage))
        size_shares = float(position.entry_amount)

        resp = await self.client.place_limit_order(
            token_id=position.token_id,
            price=limit_price,
            size=size_shares,
            side=OrderSide.SELL,
        )
        if not resp:
            self.logger.warning("close_position_order_failed", position_id=position_id)
            return False

        order_id = resp.get("order_id") or resp.get("id")
        if not order_id:
            return False

        filled_shares, avg_price = await self._reconcile_order(
            order_id=str(order_id), token_id=position.token_id, timeout_s=timeout_s
        )
        if filled_shares <= 0:
            await self.client.cancel_order(str(order_id))
            return False

        if avg_price <= 0:
            avg_price = limit_price

        # Close in portfolio (best-effort). PnL uses share amount.
        realized_pnl = await self.portfolio.close_position(
            position_id=position_id,
            exit_price=avg_price,
            exit_amount=filled_shares,
            reason=reason,
        )

        self.logger.info(
            "position_closed",
            position_id=position_id,
            order_id=str(order_id),
            shares=filled_shares,
            price=avg_price,
            reason=reason,
        )

        # Log position closure to trade history
        trade_logger = get_trade_logger()
        trade_logger.log_position_closed(
            platform="polymarket",
            ticker=position.market_id,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=avg_price,
            shares=filled_shares,
            pnl_usd=realized_pnl,
            reason=reason,
        )

        return True
