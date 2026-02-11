"""Kalshi LLM engine — ensemble-powered, same-day focus.

Applies:
- Volume ≥ $500
- Spread ≤ 8%
- Resolution ≤ 1 day (same-day markets)
- Sports blocking
- Edge ≥ 3%
- Parallel Claude + GPT-4o ensemble
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time
from typing import Dict, List, Optional, Tuple

import structlog

from ..cost_tracker import CostTracker
from ..kalshi_client import KalshiClient, KalshiMarket
from ..market_filters import MarketFilters
from ..markets import Market, TokenInfo
from ..signals.base import TradingSide
from ..signals.ensemble_signal import EnsembleSignal, detect_market_type
from ..structured_data import weather_forecast_changed, _previous_forecasts
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


class KalshiLLMEngine(BaseEngine):
    """Kalshi LLM engine with strict market filtering."""

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

        # Market filters
        self._filters = MarketFilters(config)

        # Ensemble signal (replaces LLMSignal)
        self._signal = EnsembleSignal(config=config)
        if cost_tracker:
            self._signal.set_cost_tracker(cost_tracker)

        # Kalshi config
        kalshi_cfg = getattr(config, "kalshi", None) or {}
        if isinstance(kalshi_cfg, dict):
            self._interval = float(kalshi_cfg.get("scan_interval_seconds", 300))
            self._min_volume = int(kalshi_cfg.get("min_volume", 500))
            self._categories = kalshi_cfg.get("categories", [
                "Politics", "Economics", "Elections", "Climate and Weather",
                "Science and Technology", "Companies", "Financials",
            ])
            self._fee_per_contract = float(kalshi_cfg.get("fee_per_contract", 0.0))
            self._max_resolution_days = int(kalshi_cfg.get("max_resolution_days", 1))
            self._max_spread_pct = float(kalshi_cfg.get("max_spread_pct", 0.08))
        else:
            self._interval = 300
            self._min_volume = 500
            self._categories = ["Politics", "Economics", "Elections"]
            self._fee_per_contract = 0.0
            self._max_resolution_days = 1
            self._max_spread_pct = 0.08

        # Strategy config — lowered thresholds
        strategy_cfg = getattr(config, "strategy", {}) or {}
        if isinstance(strategy_cfg, dict):
            self._min_edge = float(strategy_cfg.get("min_edge", 0.03))
        else:
            self._min_edge = 0.03

        self._pending: List[TradeSignal] = []
        self._task: Optional[asyncio.Task] = None

        # Balance gate
        self._min_trade_balance: float = 1.0
        self._balance_checker = None

        # Scan cooldown: skip recently evaluated markets (save LLM budget)
        import time as _time
        self._recently_evaluated: Dict[str, Tuple[float, float]] = {}  # ticker -> (timestamp, price_at_eval)
        self._eval_cooldown_seconds: float = 3600.0  # 60 minutes (budget-friendly)
        self._price_change_threshold: float = 0.05   # re-evaluate if price moved 5%+

        # Rescan trigger: set externally when capital frees up
        self._rescan_event = asyncio.Event()

        # Stats
        self._markets_scanned = 0
        self._markets_filtered = 0
        self._signals_generated = 0
        self._markets_cooldown_skipped = 0

    async def start(self) -> None:
        await self.kalshi_client.start()
        self._task = asyncio.create_task(self._scan_loop())
        self.logger.info(
            "kalshi_llm_engine_started",
            interval_seconds=self._interval,
            min_volume=self._min_volume,
            max_spread_pct=self._max_spread_pct,
            min_edge=self._min_edge,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.kalshi_client.stop()
        self.logger.info(
            "kalshi_llm_engine_stopped",
            markets_scanned=self._markets_scanned,
            markets_filtered=self._markets_filtered,
            signals_generated=self._signals_generated,
        )

    async def get_signals(self) -> List[TradeSignal]:
        out = list(self._pending)
        self._pending.clear()
        return out

    async def _scan_loop(self) -> None:
        while True:
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.error("kalshi_llm_engine_scan_error", error=str(exc))

            # Wait for interval OR rescan trigger (whichever comes first)
            try:
                await asyncio.wait_for(self._rescan_event.wait(), timeout=self._interval)
                self._rescan_event.clear()
                self.logger.info("kalshi_llm_rescan_triggered", msg="Capital freed — immediate rescan")
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
        # Balance gate — don't waste LLM API calls if there's no money to trade
        if self._balance_checker is not None:
            try:
                total_balance = await self._balance_checker()
                if total_balance < self._min_trade_balance:
                    self.logger.info(
                        "kalshi_llm_skip_unfunded",
                        total_balance=total_balance,
                        min_required=self._min_trade_balance,
                        msg="Skipping LLM evaluation — accounts unfunded",
                    )
                    return
            except Exception as exc:
                self.logger.warning("balance_check_failed", error=str(exc))

        # Fetch same-day markets
        kalshi_markets = await self.kalshi_client.fetch_markets_by_close_date(
            max_days=self._max_resolution_days,
            min_volume=100,  # low threshold, we filter below
        )

        self.logger.info("kalshi_llm_scan_raw", markets_fetched=len(kalshi_markets))

        filtered_markets = []
        filter_stats = {
            "price": 0,
            "volume": 0,
            "spread": 0,
            "resolution": 0,
            "sports": 0,
            "passed": 0,
        }

        for km in kalshi_markets:
            self._markets_scanned += 1

            # Apply strict filters
            filter_result = self._filters.check_all(
                market_id=km.ticker,
                title=km.title,
                category=km.category,
                volume=float(km.volume),
                bid=km.yes_bid,
                ask=km.yes_ask,
                close_time=km.close_time,
                is_live=False,  # Kalshi doesn't expose live status clearly
            )

            if not filter_result.passed:
                self._markets_filtered += 1
                # Track why markets are filtered
                reason = filter_result.reason.lower()
                if "price" in reason:
                    filter_stats["price"] += 1
                elif "volume" in reason:
                    filter_stats["volume"] += 1
                elif "spread" in reason:
                    filter_stats["spread"] += 1
                elif "resolves" in reason or "close" in reason:
                    filter_stats["resolution"] += 1
                elif "sports" in reason:
                    filter_stats["sports"] += 1
                continue

            filter_stats["passed"] += 1
            filtered_markets.append(km)

        self.logger.info(
            "kalshi_llm_scan_filtered",
            markets_passed=len(filtered_markets),
            filtered_by_price=filter_stats.get("price", 0),
            filtered_by_volume=filter_stats["volume"],
            filtered_by_spread=filter_stats["spread"],
            filtered_by_resolution=filter_stats["resolution"],
            filtered_by_sports=filter_stats["sports"],
        )

        # Sort by close time — soonest-closing markets get evaluated FIRST
        # This ensures we spend LLM budget on the most time-sensitive opportunities
        filtered_markets.sort(
            key=lambda m: m.close_time or datetime.max.replace(tzinfo=timezone.utc),
        )

        # Evaluate remaining markets with ensemble (soonest-closing first)
        # Cap evaluations per scan to control LLM budget
        max_evals_per_scan = 8
        evals_this_scan = 0
        cooldown_skipped = 0
        now_ts = time.monotonic()
        for km in filtered_markets:
            if km.yes_price <= 0 or km.yes_price >= 1:
                continue

            # Scan cooldown: skip recently evaluated markets unless price moved
            # or NOAA forecast changed for weather markets
            prev = self._recently_evaluated.get(km.ticker)
            if prev is not None:
                prev_ts, prev_price = prev
                elapsed = now_ts - prev_ts
                price_moved = abs(km.yes_price - prev_price)
                if elapsed < self._eval_cooldown_seconds and price_moved < self._price_change_threshold:
                    # Override cooldown for weather markets if forecast changed
                    mtype = detect_market_type(km.title)
                    if mtype == "weather":
                        # Check if any tracked forecast for this city changed
                        city_key = km.ticker  # will be checked on next eval
                        forecast_changed = any(
                            k.startswith(city_key.split("-")[0].lower()[-3:])
                            for k in _previous_forecasts
                        )
                        # Simple heuristic: just allow re-eval if enough time passed (10 min)
                        if elapsed > 600:
                            self.logger.info(
                                "weather_cooldown_override",
                                ticker=km.ticker,
                                elapsed=round(elapsed),
                                msg="Re-evaluating weather market (forecast may have updated)",
                            )
                        else:
                            cooldown_skipped += 1
                            continue
                    else:
                        cooldown_skipped += 1
                        continue

            # Budget cap: stop evaluating after N markets per scan
            if evals_this_scan >= max_evals_per_scan:
                self.logger.info(
                    "kalshi_llm_eval_cap_reached",
                    cap=max_evals_per_scan,
                    remaining=len(filtered_markets) - evals_this_scan - cooldown_skipped,
                )
                break

            market = _kalshi_to_market(km)
            result = await self._signal.evaluate(market)
            evals_this_scan += 1

            # Record evaluation timestamp and price
            self._recently_evaluated[km.ticker] = (now_ts, km.yes_price)

            if result.recommended_side == TradingSide.HOLD:
                continue

            # Check edge threshold (no conviction gating — all edges welcome)
            net_edge = getattr(result, "net_edge", result.edge)
            if abs(net_edge) < self._min_edge:
                self.logger.debug(
                    "kalshi_llm_skip_low_edge",
                    ticker=km.ticker,
                    net_edge=net_edge,
                    min_edge=self._min_edge,
                )
                continue

            # Build signal
            if result.recommended_side == TradingSide.BUY_YES:
                token_id = km.ticker
                side_str = "buy_yes"
            else:
                token_id = f"{km.ticker}:no"
                side_str = "buy_no"

            # Markets closing sooner get highest priority
            urgency = "low"
            if market.time_to_close_hours is not None:
                if market.time_to_close_hours < 3:
                    urgency = "immediate"   # closing very soon — top priority
                elif market.time_to_close_hours < 8:
                    urgency = "normal"
                # > 8h = "low"

            conviction = getattr(result, "conviction", "medium")

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
                    "estimated_prob": result.estimated_prob,
                    "market_price": result.market_price,
                    "net_edge": net_edge,
                    "conviction": conviction,
                    "reasoning": result.reasoning,
                    "signal_source": getattr(result, "signal_source", "llm"),
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
            self._signals_generated += 1

            self.logger.info(
                "kalshi_llm_engine_signal",
                ticker=km.ticker,
                side=side_str,
                edge=result.edge,
                net_edge=net_edge,
                conviction=conviction,
                confidence=result.confidence,
                urgency=urgency,
            )

        if cooldown_skipped:
            self._markets_cooldown_skipped += cooldown_skipped
            self.logger.info(
                "kalshi_llm_cooldown_skipped",
                skipped=cooldown_skipped,
                total_skipped=self._markets_cooldown_skipped,
            )

        # Clean up old cooldown entries (>2x cooldown period)
        stale_cutoff = now_ts - (self._eval_cooldown_seconds * 2)
        stale_keys = [k for k, (ts, _) in self._recently_evaluated.items() if ts < stale_cutoff]
        for k in stale_keys:
            del self._recently_evaluated[k]

        # Log periodic summary
        self._signal.log_periodic_summary()
