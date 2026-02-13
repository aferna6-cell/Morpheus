"""Kalshi contrarian engine — bet against overconfident crowds.

Scans for markets where crowd consensus is 80-95%, then uses a specialized
LLM prompt to identify cases where the crowd is wrong. Inspired by the
4chan Polymarket wallet strategy (87.9% win rate, 207 bets, $1.5M profit).

Key differences from the standard LLM engine:
- Only targets markets at 80-95% crowd consensus (one side)
- Uses contrarian-specific LLM prompt (why is the crowd wrong?)
- Requires high conviction + 10% minimum edge
- Larger position sizes ($25 vs $15) for asymmetric payoff
- Scans 1-7 day markets (not just same-day)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import structlog

from ..cost_tracker import CostTracker
from ..kalshi_client import KalshiClient, KalshiMarket
from ..market_filters import MarketFilters
from ..markets import Market, TokenInfo
from ..signals.base import TradingSide
from ..signals.ensemble_signal import EnsembleSignal, detect_market_type, _SKIP_TYPES
from ..utils import BotConfig
from .base import BaseEngine
from .signals import TradeSignal


def _kalshi_to_market(km: KalshiMarket) -> Market:
    """Convert KalshiMarket to Market for LLM signal."""
    tokens = {
        "Yes": TokenInfo(
            token_id=km.ticker,
            outcome="Yes",
            price=km.yes_price,
            volume_24h=float(km.volume_24h),
        ),
        "No": TokenInfo(
            token_id=f"{km.ticker}:no",
            outcome="No",
            price=km.no_price,
            volume_24h=float(km.volume_24h),
        ),
    }
    return Market(
        id=km.ticker,
        question=km.title,
        description="",
        category=km.category,
        end_date=km.close_time,
        volume_24h=float(km.volume_24h),
        liquidity=float(km.open_interest) * 1.0,
        tokens=tokens,
    )


class KalshiContrarianEngine(BaseEngine):
    """Contrarian engine — bets against overconfident crowds."""

    name = "kalshi_contrarian"

    def __init__(
        self,
        config: BotConfig,
        kalshi_client: KalshiClient,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.config = config
        self.kalshi_client = kalshi_client
        self.logger = structlog.get_logger()

        # Market filters
        self._filters = MarketFilters(config)

        # Reuse ensemble signal for contrarian evaluation
        self._signal = EnsembleSignal(config=config)
        if cost_tracker:
            self._signal.set_cost_tracker(cost_tracker)

        # Contrarian config
        contrarian_cfg = getattr(config, "contrarian", None) or {}
        if isinstance(contrarian_cfg, dict):
            self._interval = float(contrarian_cfg.get("scan_interval_seconds", 600))
            self._min_crowd = float(contrarian_cfg.get("min_crowd_confidence", 0.80))
            self._max_crowd = float(contrarian_cfg.get("max_crowd_confidence", 0.95))
            self._min_edge = float(contrarian_cfg.get("min_contrarian_edge", 0.10))
            self._min_conviction = contrarian_cfg.get("min_conviction", "high")
            self._max_resolution_days = int(contrarian_cfg.get("max_resolution_days", 7))
            self._min_volume = int(contrarian_cfg.get("min_volume_24h", 5000))
            self._categories = contrarian_cfg.get("categories", [
                "Politics", "Economics", "Elections", "Climate",
                "Tech", "Companies", "Financials",
            ])
        else:
            self._interval = 600
            self._min_crowd = 0.80
            self._max_crowd = 0.95
            self._min_edge = 0.10
            self._min_conviction = "high"
            self._max_resolution_days = 7
            self._min_volume = 5000
            self._categories = ["Politics", "Economics", "Elections"]

        self._pending: List[TradeSignal] = []
        self._task: Optional[asyncio.Task] = None

        # Cooldown: don't re-evaluate same market within 2 hours
        self._recently_evaluated: Dict[str, float] = {}  # ticker -> timestamp
        self._eval_cooldown = 7200.0  # 2 hours

        # Balance gate
        self._min_trade_balance: float = 1.0
        self._balance_checker = None

        # Rescan trigger: set externally when capital frees up
        self._rescan_event = asyncio.Event()

        # Stats
        self._markets_scanned = 0
        self._overconfident_found = 0
        self._signals_generated = 0

    async def start(self) -> None:
        await self.kalshi_client.start()
        self._task = asyncio.create_task(self._scan_loop())
        self.logger.info(
            "kalshi_contrarian_engine_started",
            interval_seconds=self._interval,
            min_crowd=self._min_crowd,
            max_crowd=self._max_crowd,
            min_edge=self._min_edge,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info(
            "kalshi_contrarian_engine_stopped",
            markets_scanned=self._markets_scanned,
            overconfident_found=self._overconfident_found,
            signals_generated=self._signals_generated,
        )

    async def get_signals(self) -> List[TradeSignal]:
        out = list(self._pending)
        self._pending.clear()
        return out

    async def _scan_loop(self) -> None:
        # Initial delay to let other engines start first
        await asyncio.sleep(30)
        while True:
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.error("kalshi_contrarian_scan_error", error=str(exc))

            # Wait for interval OR rescan trigger (whichever comes first)
            try:
                await asyncio.wait_for(self._rescan_event.wait(), timeout=self._interval)
                self._rescan_event.clear()
                self.logger.info("contrarian_rescan_triggered", msg="Capital freed — immediate rescan")
            except asyncio.TimeoutError:
                pass  # Normal interval elapsed

    def set_balance_checker(self, checker, min_balance: float = 1.0) -> None:
        """Set an async callable that returns total USD across all Kalshi accounts."""
        self._balance_checker = checker
        self._min_trade_balance = min_balance

    def trigger_rescan(self) -> None:
        """Signal the engine to run an immediate scan (e.g., when capital frees up)."""
        self._rescan_event.set()

    async def _scan_once(self) -> None:
        # Note: balance gate removed — budget check in evaluate_contrarian()
        # blocks costly LLM calls, and dispatcher handles insufficient balance.

        # Fetch markets closing within 1-7 days (wider than standard engine's same-day)
        kalshi_markets = await self.kalshi_client.fetch_markets_by_close_date(
            max_days=self._max_resolution_days,
            min_volume=500,  # Reduced API payload (was 100, Python filter is min_volume_24h)
        )

        self.logger.info("contrarian_scan_raw", markets_fetched=len(kalshi_markets))

        # Filter for overconfident crowd markets
        candidates: List[KalshiMarket] = []
        now_ts = time.monotonic()
        filter_stats: Dict[str, int] = {
            "cooldown": 0, "low_volume": 0, "invalid_price": 0,
            "not_overconfident": 0, "too_soon": 0, "ticker_prefix": 0,
            "price_filter": 0, "market_type": 0, "sports": 0,
        }

        for km in kalshi_markets:
            self._markets_scanned += 1

            # Skip recently evaluated
            prev_ts = self._recently_evaluated.get(km.ticker)
            if prev_ts and (now_ts - prev_ts) < self._eval_cooldown:
                filter_stats["cooldown"] += 1
                continue

            # Must have decent volume (real crowd consensus, not illiquid)
            if km.volume_24h < self._min_volume:
                filter_stats["low_volume"] += 1
                continue

            # Must have valid price
            if km.yes_price <= 0 or km.yes_price >= 1:
                filter_stats["invalid_price"] += 1
                continue

            # Check if crowd is overconfident on either side
            crowd_confidence = max(km.yes_price, km.no_price)
            if crowd_confidence < self._min_crowd or crowd_confidence > self._max_crowd:
                filter_stats["not_overconfident"] += 1
                continue

            # Must resolve in >6 hours (not imminent)
            hours_left = None
            if km.close_time:
                hours_left = (km.close_time - datetime.now(timezone.utc)).total_seconds() / 3600
                if hours_left < 6:
                    filter_stats["too_soon"] += 1
                    continue

            # Ticker prefix filter (junk markets — crypto ranges, mentions, etc.)
            prefix_result = self._filters.check_ticker_prefix(km.ticker)
            if not prefix_result.passed:
                filter_stats["ticker_prefix"] += 1
                continue

            # Price filter — skip extreme prices where LLM has no edge
            price_result = self._filters.check_price(km.ticker, km.yes_bid, km.yes_ask)
            if not price_result.passed:
                filter_stats["price_filter"] += 1
                continue

            # Market type skip (sports, coin flips)
            mtype = detect_market_type(km.title)
            if mtype in _SKIP_TYPES:
                filter_stats["market_type"] += 1
                continue

            # Apply sports filter
            filter_result = self._filters.check_sports(
                market_id=km.ticker,
                title=km.title,
                category=km.category,
                is_live=False,
            )
            if not filter_result.passed:
                filter_stats["sports"] += 1
                continue

            candidates.append(km)

        self._overconfident_found += len(candidates)
        self.logger.info(
            "contrarian_candidates",
            total=len(candidates),
            example_tickers=[c.ticker for c in candidates[:5]],
        )

        # Evaluate each candidate with the contrarian prompt
        screened_out = 0
        for km in candidates:
            market = _kalshi_to_market(km)

            # Cheap gpt-4o-mini screen before expensive evaluate_contrarian
            if self._signal.screening_enabled:
                mid_price = max(km.yes_price, km.no_price)
                try:
                    passed = await self._signal._screen_market(market, mid_price)
                except Exception:
                    passed = True  # Don't block on screening errors
                if not passed:
                    screened_out += 1
                    self.logger.debug("contrarian_screened_out", ticker=km.ticker)
                    continue

            result = await self._signal.evaluate_contrarian(market)

            # Record evaluation
            self._recently_evaluated[km.ticker] = now_ts

            if result.recommended_side == TradingSide.HOLD:
                continue

            net_edge = getattr(result, "net_edge", result.edge)
            conviction = getattr(result, "conviction", "low")
            conv_str = conviction.value if hasattr(conviction, "value") else str(conviction).lower()

            # Theta decay timing: adjust min edge based on time-to-resolution
            # Mean reversion peaks ~5-7 days out, prices become sticky near resolution
            effective_min_edge = self._min_edge  # default 10%
            if km.close_time:
                h_left = (km.close_time - datetime.now(timezone.utc)).total_seconds() / 3600
                if h_left > 120:       # 5+ days: peak mean reversion window
                    effective_min_edge = 0.08
                elif h_left > 48:      # 2-5 days
                    effective_min_edge = 0.09
                elif h_left > 24:      # 1-2 days
                    effective_min_edge = 0.10
                else:                  # <24 hours: prices sticky, need bigger edge
                    effective_min_edge = 0.13

            # Require minimum contrarian edge
            if abs(net_edge) < effective_min_edge:
                self.logger.debug(
                    "contrarian_skip_low_edge",
                    ticker=km.ticker,
                    net_edge=net_edge,
                    min_edge=effective_min_edge,
                )
                continue

            # Require minimum conviction
            conviction_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
            min_rank = conviction_rank.get(self._min_conviction, 3)
            actual_rank = conviction_rank.get(conv_str, 0)
            if actual_rank < min_rank:
                self.logger.debug(
                    "contrarian_skip_low_conviction",
                    ticker=km.ticker,
                    conviction=conv_str,
                    min_conviction=self._min_conviction,
                )
                continue

            # Build signal
            if result.recommended_side == TradingSide.BUY_YES:
                token_id = km.ticker
                side_str = "buy_yes"
            else:
                token_id = f"{km.ticker}:no"
                side_str = "buy_no"

            # Urgency: contrarian trades are not urgency-driven
            urgency = "normal"
            if market.time_to_close_hours is not None:
                if market.time_to_close_hours < 12:
                    urgency = "immediate"

            contrarian_thesis = getattr(result, "contrarian_thesis", "")
            crowd_wrong_reason = getattr(result, "crowd_wrong_reason", "")

            signal = TradeSignal(
                engine=self.name,
                market_id=km.ticker,
                token_id=token_id,
                side=side_str,
                confidence=result.confidence,
                edge=result.edge,
                urgency=urgency,
                metadata={
                    "platform": "kalshi",
                    "strategy": "contrarian",
                    "estimated_prob": result.estimated_prob,
                    "market_price": result.market_price,
                    "net_edge": net_edge,
                    "conviction": conv_str,
                    "reasoning": result.reasoning,
                    "signal_source": getattr(result, "signal_source", "llm"),
                    "contrarian_thesis": contrarian_thesis,
                    "crowd_wrong_reason": crowd_wrong_reason,
                    "crowd_confidence": max(km.yes_price, km.no_price),
                    "kalshi_ticker": km.ticker,
                    "kalshi_yes_bid": km.yes_bid,
                    "kalshi_yes_ask": km.yes_ask,
                    "kalshi_no_bid": km.no_bid,
                    "kalshi_no_ask": km.no_ask,
                    "kalshi_volume": km.volume,
                    "fee_per_contract": 0.0,
                    "_market": market,
                    "title": km.title,
                    "question": km.title,
                },
            )
            self._pending.append(signal)
            self._signals_generated += 1

            self.logger.info(
                "contrarian_signal_generated",
                ticker=km.ticker,
                side=side_str,
                edge=result.edge,
                net_edge=net_edge,
                conviction=conv_str,
                confidence=result.confidence,
                crowd_confidence=max(km.yes_price, km.no_price),
                crowd_wrong=crowd_wrong_reason[:80],
            )

        # Clean stale cooldown entries
        stale_cutoff = now_ts - (self._eval_cooldown * 2)
        stale_keys = [k for k, ts in self._recently_evaluated.items() if ts < stale_cutoff]
        for k in stale_keys:
            del self._recently_evaluated[k]

        # Scan summary
        self.logger.info(
            "contrarian_scan_summary",
            markets_fetched=len(kalshi_markets),
            candidates=len(candidates),
            screened_out=screened_out,
            signals_generated=self._signals_generated,
            filter_stats={k: v for k, v in filter_stats.items() if v > 0},
        )
