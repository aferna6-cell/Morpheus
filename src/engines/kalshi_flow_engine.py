"""Kalshi flow-following engine — follows large trades on the public tape.

Kalshi doesn't expose individual trader positions, so instead of copy-trading
specific wallets we monitor the public trade tape for unusually large trades
and follow them.  Large trades (50+ contracts) often represent informed money.

Strategy:
- Poll the Kalshi public trades endpoint every N seconds
- Detect trades above a configurable contract threshold
- Emit a BUY signal in the same direction as the large trade
- Cooldown per ticker to avoid re-signaling on the same market

No LLM cost — pure event-driven.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set

import structlog

from ..kalshi_client import KalshiClient, KalshiMarket
from ..utils import BotConfig
from .base import BaseEngine
from .signals import TradeSignal

logger = structlog.get_logger()


class KalshiFlowEngine(BaseEngine):
    """Detects large trades on Kalshi and emits follow signals."""

    name = "kalshi_flow"

    def __init__(self, config: BotConfig, kalshi_client: KalshiClient):
        self.config = config
        self.kalshi_client = kalshi_client

        flow_cfg = getattr(config, "kalshi_flow", None) or {}
        if not isinstance(flow_cfg, dict):
            flow_cfg = {}

        self._poll_interval: float = float(flow_cfg.get("poll_interval_seconds", 30))
        self._min_contracts: int = int(flow_cfg.get("min_contracts", 50))
        self._min_volume: int = int(flow_cfg.get("min_volume", 10000))
        self._cooldown_s: float = float(flow_cfg.get("cooldown_seconds", 300))
        self._lookback_s: int = int(flow_cfg.get("lookback_seconds", 120))
        self._max_price: float = float(flow_cfg.get("max_price", 0.90))
        self._min_price: float = float(flow_cfg.get("min_price", 0.10))

        # Runtime state
        self._seen_trade_ids: Set[str] = set()
        self._last_signal_by_ticker: Dict[str, float] = {}
        self._pending: List[TradeSignal] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Stats
        self._trades_scanned = 0
        self._signals_generated = 0

    async def start(self) -> None:
        await self.kalshi_client.start()
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "kalshi_flow_engine_started",
            poll_interval=self._poll_interval,
            min_contracts=self._min_contracts,
            cooldown_s=self._cooldown_s,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(
            "kalshi_flow_engine_stopped",
            trades_scanned=self._trades_scanned,
            signals_generated=self._signals_generated,
        )

    async def get_signals(self) -> List[TradeSignal]:
        out = list(self._pending)
        self._pending.clear()
        return out

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("kalshi_flow_poll_error", error=str(exc))
            await asyncio.sleep(self._poll_interval)

    async def _poll_once(self) -> None:
        """Fetch recent public trades and look for large ones."""
        now_ts = int(datetime.now(timezone.utc).timestamp())
        min_ts = now_ts - self._lookback_s

        try:
            trades = await self.kalshi_client.get_public_trades(
                min_ts=min_ts,
                limit=200,
            )
        except Exception as exc:
            logger.warning("kalshi_flow_fetch_error", error=str(exc))
            return

        now_mono = time.monotonic()

        for trade in trades:
            self._trades_scanned += 1

            trade_id = trade.get("trade_id", "")
            if trade_id in self._seen_trade_ids:
                continue
            self._seen_trade_ids.add(trade_id)

            ticker = trade.get("ticker", "")
            count = int(trade.get("count", 0))
            taker_side = trade.get("taker_side", "")  # "yes" or "no"
            price = float(trade.get("price", 0))

            # Normalize price (Kalshi returns cents)
            if price > 1:
                price = price / 100.0

            # Filter: only large trades
            if count < self._min_contracts:
                continue

            # Filter: avoid extreme prices (likely settled or illiquid)
            if price < self._min_price or price > self._max_price:
                continue

            # Cooldown per ticker
            last_signal = self._last_signal_by_ticker.get(ticker, 0)
            if (now_mono - last_signal) < self._cooldown_s:
                continue

            # Build signal
            side = f"buy_{taker_side}" if taker_side in ("yes", "no") else None
            if not side:
                continue

            # Confidence scales with trade size
            confidence = min(0.7, 0.4 + (count / 500.0) * 0.3)
            edge = 0.05  # meets min_edge threshold

            signal = TradeSignal(
                engine=self.name,
                market_id=ticker,
                token_id=ticker if taker_side == "yes" else f"{ticker}:no",
                side=side,
                confidence=round(confidence, 4),
                edge=edge,
                urgency="high",
                metadata={
                    "platform": "kalshi",
                    "trade_id": trade_id,
                    "taker_side": taker_side,
                    "trade_count": count,
                    "trade_price": price,
                    "kalshi_ticker": ticker,
                },
            )

            self._pending.append(signal)
            self._last_signal_by_ticker[ticker] = now_mono
            self._signals_generated += 1

            logger.info(
                "kalshi_flow_signal",
                ticker=ticker,
                side=side,
                count=count,
                price=price,
                confidence=confidence,
            )

        # Prune old seen trade IDs (keep last 5000)
        if len(self._seen_trade_ids) > 5000:
            # Convert to list, keep last 2500
            recent = list(self._seen_trade_ids)[-2500:]
            self._seen_trade_ids = set(recent)
