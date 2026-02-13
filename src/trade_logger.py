"""Persistent trade activity logger.

Writes all trading activity to a JSONL file for historical analysis.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog


class TradeLogger:
    """Append-only trade log with JSONL output."""

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = Path(log_path or "state/trade_history.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = structlog.get_logger()

    def _write(self, record: Dict[str, Any]) -> None:
        """Append a record to the JSONL file."""
        record["logged_at"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self.logger.error("trade_log_write_failed", error=str(e))

    def log_order_placed(
        self,
        platform: str,
        ticker: str,
        side: str,
        count: int,
        price_cents: int,
        order_id: str,
        order_type: str = "limit",
        conviction: Optional[str] = None,
        edge: Optional[float] = None,
        cost_usd: Optional[float] = None,
        entry_probability: Optional[float] = None,
        market_price: Optional[float] = None,
        account_label: Optional[str] = None,
        signal_source: Optional[str] = None,
    ) -> None:
        """Log a successfully placed order with CLV tracking fields."""
        self._write({
            "event": "order_placed",
            "platform": platform,
            "account": account_label,
            "ticker": ticker,
            "side": side,
            "count": count,
            "price_cents": price_cents,
            "order_id": order_id,
            "order_type": order_type,
            "conviction": conviction,
            "edge": edge,
            "cost_usd": cost_usd,
            # CLV tracking fields
            "entry_probability": entry_probability,  # our model's prediction
            "market_price_at_entry": market_price,   # market price when we entered
            "signal_source": signal_source,
        })

    def log_order_failed(
        self,
        platform: str,
        ticker: str,
        side: str,
        count: int,
        price_cents: int,
        error: str,
        conviction: Optional[str] = None,
        account_label: Optional[str] = None,
    ) -> None:
        """Log a failed order attempt."""
        self._write({
            "event": "order_failed",
            "platform": platform,
            "account": account_label,
            "ticker": ticker,
            "side": side,
            "count": count,
            "price_cents": price_cents,
            "error": error,
            "conviction": conviction,
        })

    def log_order_filled(
        self,
        platform: str,
        ticker: str,
        side: str,
        count: int,
        fill_price_cents: int,
        order_id: str,
        account_label: Optional[str] = None,
        strategy: Optional[str] = None,
        signal_source: Optional[str] = None,
    ) -> None:
        """Log an order fill."""
        self._write({
            "event": "order_filled",
            "platform": platform,
            "account": account_label,
            "ticker": ticker,
            "side": side,
            "count": count,
            "fill_price_cents": fill_price_cents,
            "order_id": order_id,
            "strategy": strategy,
            "signal_source": signal_source,
        })

    def log_position_closed(
        self,
        platform: str,
        ticker: str,
        side: str,
        count: int,
        entry_price_cents: int,
        exit_price_cents: int,
        pnl_usd: float,
        account_label: Optional[str] = None,
    ) -> None:
        """Log a position closure with P&L."""
        self._write({
            "event": "position_closed",
            "platform": platform,
            "account": account_label,
            "ticker": ticker,
            "side": side,
            "count": count,
            "entry_price_cents": entry_price_cents,
            "exit_price_cents": exit_price_cents,
            "pnl_usd": pnl_usd,
        })

    def log_market_resolution(
        self,
        platform: str,
        ticker: str,
        outcome: str,
        held_side: Optional[str] = None,
        held_count: Optional[int] = None,
        pnl_usd: Optional[float] = None,
        closing_probability: Optional[float] = None,
        entry_probability: Optional[float] = None,
        clv: Optional[float] = None,
        account_label: Optional[str] = None,
    ) -> None:
        """Log a market resolution with CLV data."""
        self._write({
            "event": "market_resolved",
            "platform": platform,
            "account": account_label,
            "ticker": ticker,
            "outcome": outcome,
            "held_side": held_side,
            "held_count": held_count,
            "pnl_usd": pnl_usd,
            # CLV tracking
            "closing_probability": closing_probability,
            "entry_probability": entry_probability,
            "clv": clv,  # entry_prob - closing_prob (positive = ahead of market)
        })


# Singleton instance
_trade_logger: Optional[TradeLogger] = None


def get_trade_logger(log_path: Optional[str] = None) -> TradeLogger:
    """Get or create the singleton trade logger."""
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger(log_path)
    return _trade_logger
