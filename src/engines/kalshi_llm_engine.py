"""Kalshi LLM engine — scans Kalshi markets and evaluates via the LLM signal pipeline.

Mirrors :class:`LLMEngine` but operates on Kalshi markets instead of Polymarket.
Converts :class:`KalshiMarket` objects to the :class:`Market` format the LLM
signal expects and emits :class:`TradeSignal` objects for the orchestrator.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import structlog

from ..cost_tracker import CostTracker
from ..kalshi_client import KalshiClient, KalshiMarket
from ..markets import Market, TokenInfo
from ..signals.base import TradingSide
from ..signals.llm_signal import LLMSignal
from ..utils import BotConfig
from .base import BaseEngine
from .signals import TradeSignal


def _kalshi_to_market(km: KalshiMarket) -> Market:
    """Convert a :class:`KalshiMarket` to a :class:`Market` for the LLM signal.

    Kalshi prices are already 0-1 (converted in ``kalshi_client.py``).
    We synthesize token info with the ticker as both the market id and token id
    (Kalshi doesn't have separate token ids — the ticker IS the contract).
    """
    tokens = {
        "Yes": TokenInfo(
            token_id=km.ticker,           # ticker serves as the "token id"
            outcome="Yes",
            price=km.yes_price,
            volume_24h=float(km.volume_24h),
        ),
        "No": TokenInfo(
            token_id=f"{km.ticker}:no",   # synthetic id for NO side
            outcome="No",
            price=km.no_price,
            volume_24h=float(km.volume_24h),
        ),
    }

    return Market(
        id=km.ticker,
        question=km.title,
        description="",                   # Kalshi doesn't expose long descriptions
        category=km.category,
        end_date=km.close_time,
        volume_24h=float(km.volume_24h),
        liquidity=float(km.open_interest) * 1.0,  # rough proxy
        tokens=tokens,
    )


class KalshiLLMEngine(BaseEngine):
    """Kalshi-specific LLM engine producing :class:`TradeSignal` objects."""

    name = "kalshi_llm"

    def __init__(
        self,
        config: BotConfig,
        kalshi_client: KalshiClient,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.config = config
        self.kalshi_client = kalshi_client
        self.logger = structlog.get_logger()

        # Underlying signal — shared LLM pipeline
        self._llm_signal = LLMSignal(config=config)
        if cost_tracker:
            self._llm_signal.set_cost_tracker(cost_tracker)

        # Kalshi-specific config
        kalshi_cfg = getattr(config, "kalshi", None) or {}
        if isinstance(kalshi_cfg, dict):
            self._interval = float(kalshi_cfg.get("scan_interval_seconds", 60))
            self._min_volume = int(kalshi_cfg.get("min_volume", 100))
            self._categories = kalshi_cfg.get("categories", [
                "Politics", "Economics", "Elections", "World", "Climate and Weather",
            ])
            self._fee_per_contract = float(kalshi_cfg.get("fee_per_contract", 0.07))
            self._max_resolution_days = int(kalshi_cfg.get("max_resolution_days", 30))
        else:
            self._interval = 60
            self._min_volume = 100
            self._categories = ["Politics", "Economics", "Elections", "World", "Climate and Weather"]
            self._fee_per_contract = 0.07
            self._max_resolution_days = 30

        # Pending signals buffer — drained by get_signals()
        self._pending: List[TradeSignal] = []
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self.kalshi_client.start()
        self._task = asyncio.create_task(self._scan_loop())
        self.logger.info("kalshi_llm_engine_started", interval_seconds=self._interval)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.kalshi_client.stop()
        self.logger.info("kalshi_llm_engine_stopped")

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
                self.logger.error("kalshi_llm_engine_scan_error", error=str(exc))
            await asyncio.sleep(self._interval)

    async def _scan_once(self) -> None:
        # Fetch markets by close date — gets short-term markets across ALL categories
        kalshi_markets = await self.kalshi_client.fetch_markets_by_close_date(
            max_days=self._max_resolution_days,
            min_volume=self._min_volume,
        )

        self.logger.info("kalshi_llm_scan", markets_found=len(kalshi_markets))

        for km in kalshi_markets:
            # Skip markets with no useful price data (already filtered, but safety check)
            if km.yes_price <= 0 or km.yes_price >= 1:
                continue

            # Convert to Market format for the LLM
            market = _kalshi_to_market(km)

            result = await self._llm_signal.evaluate(market)
            if result.recommended_side == TradingSide.HOLD:
                continue

            # Pick the right "token_id" (ticker) for the side
            if result.recommended_side == TradingSide.BUY_YES:
                token_id = km.ticker
                side_str = "buy_yes"
            else:
                token_id = f"{km.ticker}:no"
                side_str = "buy_no"

            conviction = getattr(result, "conviction", None)
            net_edge = getattr(result, "net_edge", result.edge)

            signal = TradeSignal(
                engine=self.name,
                market_id=km.ticker,
                token_id=token_id,
                side=side_str,
                confidence=result.confidence,
                edge=result.edge,
                urgency="normal",
                metadata={
                    "platform": "kalshi",
                    "estimated_prob": result.estimated_prob,
                    "market_price": result.market_price,
                    "net_edge": net_edge,
                    "conviction": conviction.value if conviction else "medium",
                    "reasoning": result.reasoning,
                    "kalshi_ticker": km.ticker,
                    "kalshi_yes_bid": km.yes_bid,
                    "kalshi_yes_ask": km.yes_ask,
                    "kalshi_no_bid": km.no_bid,
                    "kalshi_no_ask": km.no_ask,
                    "kalshi_volume": km.volume,
                    "fee_per_contract": self._fee_per_contract,
                    "_market": market,
                },
            )
            self._pending.append(signal)
            self.logger.info(
                "kalshi_llm_engine_signal",
                ticker=km.ticker,
                side=side_str,
                edge=result.edge,
                confidence=result.confidence,
            )

        # Periodic cost summary
        self._llm_signal.log_periodic_summary()
