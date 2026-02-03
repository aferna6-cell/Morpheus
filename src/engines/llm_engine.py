"""LLM engine — wraps the existing LLMSignal as a BaseEngine for the orchestrator.

Scans markets on a configurable interval and emits TradeSignal objects
for any market where the LLM finds actionable edge.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import structlog

from ..client import PolymarketClient
from ..cost_tracker import CostTracker
from ..markets import Market
from ..markets_scanner import MarketScanner
from ..signals.base import TradingSide
from ..signals.llm_signal import LLMSignal
from ..utils import BotConfig
from .base import BaseEngine
from .signals import TradeSignal


class LLMEngine(BaseEngine):
    """Wraps :class:`LLMSignal` as an async engine producing :class:`TradeSignal`."""

    name = "llm"

    def __init__(
        self,
        config: BotConfig,
        client: PolymarketClient,
        scanner: MarketScanner,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.config = config
        self.client = client
        self.scanner = scanner
        self.logger = structlog.get_logger()

        # Underlying signal
        self._llm_signal = LLMSignal(config=config)
        if cost_tracker:
            self._llm_signal.set_cost_tracker(cost_tracker)

        # Scan interval from config
        self._interval = float(config.timing.get("signal_refresh_interval_minutes", 60)) * 60.0

        # Pending signals buffer — drained by get_signals()
        self._pending: List[TradeSignal] = []
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._task = asyncio.create_task(self._scan_loop())
        self.logger.info("llm_engine_started", interval_seconds=self._interval)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("llm_engine_stopped")

    async def get_signals(self) -> List[TradeSignal]:
        out = list(self._pending)
        self._pending.clear()
        return out

    # ------------------------------------------------------------------
    # Background scanning
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        while True:
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.error("llm_engine_scan_error", error=str(exc))
            await asyncio.sleep(self._interval)

    async def _scan_once(self) -> None:
        markets = await self.scanner.fetch_markets()
        markets = await self.scanner.enrich_midpoints(self.client, markets)

        for m in markets:
            if m.midpoint_price is None:
                continue

            result = await self._llm_signal.evaluate(m)
            if result.recommended_side == TradingSide.HOLD:
                continue

            # Pick token_id for the recommended side
            wanted = "yes" if result.recommended_side == TradingSide.BUY_YES else "no"
            token_id = ""
            for outcome, tok in m.tokens.items():
                if outcome and outcome.strip().lower() == wanted:
                    token_id = tok.token_id
                    break

            if not token_id:
                continue

            conviction = getattr(result, "conviction", None)
            net_edge = getattr(result, "net_edge", result.edge)

            signal = TradeSignal(
                engine=self.name,
                market_id=m.id,
                token_id=token_id,
                side=result.recommended_side.value,
                confidence=result.confidence,
                edge=result.edge,
                urgency="normal",
                metadata={
                    "estimated_prob": result.estimated_prob,
                    "market_price": result.market_price,
                    "net_edge": net_edge,
                    "conviction": conviction.value if conviction else "medium",
                    "reasoning": result.reasoning,
                    "_market": m,
                },
            )
            self._pending.append(signal)
            self.logger.info(
                "llm_engine_signal",
                market_id=m.id,
                side=result.recommended_side.value,
                edge=result.edge,
                confidence=result.confidence,
            )

        # Periodic cost summary
        self._llm_signal.log_periodic_summary()
