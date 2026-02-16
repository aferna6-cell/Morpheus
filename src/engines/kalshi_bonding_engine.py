"""Kalshi bonding engine — harvest near-certain outcomes for bond-like returns.

Scans for markets priced at 90-98c settling within 72 hours. Buying a 95c YES
contract that settles to $1 = 5.26% return in ≤3 days = 520%+ annualized.

Key design principles:
- Conservative: tail risk (a "certain" event not happening) is the main danger
- Verification: weather markets verified via NOAA, index via Yahoo Finance
- Consensus-gated: unverifiable markets require price > 95c AND volume > $5000
- Fixed sizing: $5-10 per bond (not Kelly — low-edge, high-probability)
- No LLM calls: zero cost, relies entirely on structured data + market consensus
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import structlog

from ..kalshi_client import KalshiClient, KalshiMarket
from ..market_filters import MarketFilters
from ..markets import Market, TokenInfo
from ..signals.ensemble_signal import detect_market_type, _SKIP_TYPES
from ..structured_data import compute_weather_probability, compute_stock_index_probability
from ..utils import BotConfig
from .base import BaseEngine
from .signals import TradeSignal


# Index ticker prefixes that have Yahoo Finance fast-path verification
_INDEX_PREFIXES = (
    "KXINXU", "KXINX-", "KXNASDAQ100",
    "KXSPY", "KXQQQ", "KXIWM", "KXDIA",
    "KXWTI", "KXGOLD",
)


def _kalshi_to_market(km: KalshiMarket) -> Market:
    """Convert KalshiMarket to Market for structured data lookups."""
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
        description=km.subtitle,
        category=km.category,
        end_date=km.close_time,
        volume_24h=float(km.volume_24h),
        liquidity=float(km.open_interest) * 1.0,
        tokens=tokens,
    )


class KalshiBondingEngine(BaseEngine):
    """Scan for near-certain outcomes to 'bond' — low-risk, low-return harvesting."""

    name = "kalshi_bonding"

    def __init__(
        self,
        config: BotConfig,
        kalshi_client: KalshiClient,
    ):
        self.config = config
        self.kalshi_client = kalshi_client
        self.logger = structlog.get_logger()

        # Market filters (ticker blocklist, sports, etc.)
        self._filters = MarketFilters(config)

        # Bonding config
        bonding_cfg = getattr(config, "bonding", None) or {}
        if isinstance(bonding_cfg, dict):
            self._interval = float(bonding_cfg.get("scan_interval_seconds", 600))
            self._min_price_cents = int(bonding_cfg.get("min_price_cents", 90))
            self._max_price_cents = int(bonding_cfg.get("max_price_cents", 98))
            self._max_hours = float(bonding_cfg.get("max_hours_to_settle", 72))
            self._min_volume = int(bonding_cfg.get("min_volume", 2000))
            self._max_position_usd = float(bonding_cfg.get("max_position_usd", 10.0))
            self._max_total_bonds = int(bonding_cfg.get("max_total_bonds", 5))
            self._min_profit_pct = float(bonding_cfg.get("min_profit_pct", 0.02))
            self._verify_with_data = bool(bonding_cfg.get("verify_with_data", True))
            # Consensus requirements for unverifiable markets
            self._consensus_min_price = int(bonding_cfg.get("consensus_min_price_cents", 95))
            self._consensus_min_volume = int(bonding_cfg.get("consensus_min_volume", 5000))
        else:
            self._interval = 600
            self._min_price_cents = 90
            self._max_price_cents = 98
            self._max_hours = 72
            self._min_volume = 2000
            self._max_position_usd = 10.0
            self._max_total_bonds = 5
            self._min_profit_pct = 0.02
            self._verify_with_data = True
            self._consensus_min_price = 95
            self._consensus_min_volume = 5000

        self._pending: List[TradeSignal] = []
        self._task: Optional[asyncio.Task] = None

        # Cooldown: don't re-evaluate same market within 1 hour
        self._recently_evaluated: Dict[str, float] = {}  # ticker -> monotonic timestamp
        self._eval_cooldown = 3600.0  # 1 hour

        # Track active bond count (set externally by orchestrator/fill manager)
        self._active_bond_count = 0

        # Rescan trigger
        self._rescan_event = asyncio.Event()

        # Stats
        self._markets_scanned = 0
        self._candidates_found = 0
        self._verified_count = 0
        self._signals_generated = 0

    async def start(self) -> None:
        await self.kalshi_client.start()
        self._task = asyncio.create_task(self._scan_loop())
        self.logger.info(
            "kalshi_bonding_engine_started",
            interval_seconds=self._interval,
            min_price_cents=self._min_price_cents,
            max_price_cents=self._max_price_cents,
            max_hours=self._max_hours,
            max_bonds=self._max_total_bonds,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info(
            "kalshi_bonding_engine_stopped",
            markets_scanned=self._markets_scanned,
            candidates_found=self._candidates_found,
            verified=self._verified_count,
            signals_generated=self._signals_generated,
        )

    async def get_signals(self) -> List[TradeSignal]:
        out = list(self._pending)
        self._pending.clear()
        return out

    def trigger_rescan(self) -> None:
        """Signal the engine to run an immediate scan (e.g., when capital frees up)."""
        self._rescan_event.set()

    async def _scan_loop(self) -> None:
        # Initial delay — let other engines start first
        await asyncio.sleep(45)
        while True:
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.error("kalshi_bonding_scan_error", error=str(exc))

            # Wait for interval OR rescan trigger
            try:
                await asyncio.wait_for(self._rescan_event.wait(), timeout=self._interval)
                self._rescan_event.clear()
                self.logger.info("bonding_rescan_triggered", msg="Capital freed — immediate rescan")
            except asyncio.TimeoutError:
                pass  # Normal interval elapsed

    async def _scan_once(self) -> None:
        """Scan for bond-like opportunities."""
        # Cap signals if we already have enough active bonds
        if self._active_bond_count >= self._max_total_bonds:
            self.logger.debug(
                "bonding_skip_at_capacity",
                active_bonds=self._active_bond_count,
                max_bonds=self._max_total_bonds,
            )
            return

        # Fetch markets settling within max_hours (convert to days, round up)
        max_days = max(1, int(self._max_hours / 24) + 1)
        kalshi_markets = await self.kalshi_client.fetch_markets_by_close_date(
            max_days=max_days,
            min_volume=500,  # Low bar at API level; Python-side filter is stricter
        )

        self.logger.info("bonding_scan_raw", markets_fetched=len(kalshi_markets))

        # Filter for bond candidates
        candidates: List[KalshiMarket] = []
        now_ts = time.monotonic()
        filter_stats: Dict[str, int] = {
            "cooldown": 0, "low_volume": 0, "invalid_price": 0,
            "not_near_certain": 0, "too_far": 0, "too_soon": 0,
            "too_thin": 0, "ticker_prefix": 0, "sports": 0,
            "skip_type": 0,
        }

        for km in kalshi_markets:
            self._markets_scanned += 1

            # Skip recently evaluated
            prev_ts = self._recently_evaluated.get(km.ticker)
            if prev_ts and (now_ts - prev_ts) < self._eval_cooldown:
                filter_stats["cooldown"] += 1
                continue

            # Must have valid price
            if km.yes_price <= 0 or km.yes_price >= 1:
                filter_stats["invalid_price"] += 1
                continue

            # Must settle within max_hours
            if km.close_time:
                hours_left = (km.close_time - datetime.now(timezone.utc)).total_seconds() / 3600
                if hours_left > self._max_hours:
                    filter_stats["too_far"] += 1
                    continue
                # Don't bond markets settling in < 1 hour (too close to expiry, execution risk)
                if hours_left < 1.0:
                    filter_stats["too_soon"] += 1
                    continue
            else:
                filter_stats["too_far"] += 1
                continue

            # Must have decent volume
            if km.volume_24h < self._min_volume:
                filter_stats["low_volume"] += 1
                continue

            # Check if near-certain on either side
            # YES side: yes_price >= min_price_cents/100
            # NO side: no_price >= min_price_cents/100 (equivalent to yes_price <= (100 - min_price_cents)/100)
            yes_cents = int(km.yes_price * 100)
            no_cents = int(km.no_price * 100)

            is_yes_bond = self._min_price_cents <= yes_cents <= self._max_price_cents
            is_no_bond = self._min_price_cents <= no_cents <= self._max_price_cents

            if not is_yes_bond and not is_no_bond:
                filter_stats["not_near_certain"] += 1
                continue

            # Check profit covers fees (Kalshi maker fee = 0, but check min_profit_pct)
            if is_yes_bond:
                buy_price = km.yes_ask if km.yes_ask > 0 else km.yes_price
                profit_pct = (1.0 - buy_price) / buy_price if buy_price > 0 else 0
            else:
                buy_price = km.no_ask if km.no_ask > 0 else km.no_price
                profit_pct = (1.0 - buy_price) / buy_price if buy_price > 0 else 0

            if profit_pct < self._min_profit_pct:
                filter_stats["too_thin"] += 1
                continue

            # Ticker prefix filter (junk markets)
            prefix_result = self._filters.check_ticker_prefix(km.ticker)
            if not prefix_result.passed:
                filter_stats["ticker_prefix"] += 1
                continue

            # Sports filter
            filter_result = self._filters.check_sports(
                market_id=km.ticker,
                title=km.title,
                category=km.category,
                is_live=False,
            )
            if not filter_result.passed:
                filter_stats["sports"] += 1
                continue

            # Skip types where we have no verification ability
            mtype = detect_market_type(km.title)
            if mtype in _SKIP_TYPES:
                filter_stats["skip_type"] += 1
                continue

            candidates.append(km)

        self._candidates_found += len(candidates)
        self.logger.info(
            "bonding_candidates",
            total=len(candidates),
            example_tickers=[c.ticker for c in candidates[:5]],
            filter_stats={k: v for k, v in filter_stats.items() if v > 0},
        )

        # Verify and emit signals for each candidate
        bonds_emitted = 0
        max_new_bonds = self._max_total_bonds - self._active_bond_count

        for km in candidates:
            if bonds_emitted >= max_new_bonds:
                break

            # Determine which side to bond
            yes_cents = int(km.yes_price * 100)
            no_cents = int(km.no_price * 100)
            is_yes_bond = self._min_price_cents <= yes_cents <= self._max_price_cents
            is_no_bond = self._min_price_cents <= no_cents <= self._max_price_cents

            # Prefer the side with higher certainty
            if is_yes_bond and is_no_bond:
                # Both sides qualify — pick the more certain one
                if yes_cents >= no_cents:
                    side = "buy_yes"
                    buy_price = km.yes_ask if km.yes_ask > 0 else km.yes_price
                else:
                    side = "buy_no"
                    buy_price = km.no_ask if km.no_ask > 0 else km.no_price
            elif is_yes_bond:
                side = "buy_yes"
                buy_price = km.yes_ask if km.yes_ask > 0 else km.yes_price
            else:
                side = "buy_no"
                buy_price = km.no_ask if km.no_ask > 0 else km.no_price

            # Verify with structured data if available
            verified, verification_source, data_prob = await self._verify_near_certainty(
                km, side,
            )

            # Record evaluation
            self._recently_evaluated[km.ticker] = now_ts

            if not verified:
                self.logger.debug(
                    "bonding_verification_failed",
                    ticker=km.ticker,
                    side=side,
                    price=buy_price,
                    source=verification_source,
                )
                continue

            self._verified_count += 1

            # Compute edge and confidence
            price_cents = int(buy_price * 100)
            profit_pct = (1.0 - buy_price) / buy_price if buy_price > 0 else 0
            edge = 1.0 - buy_price - buy_price  # data_prob (≈1.0) minus buy_price, simplified

            # Use data_prob if we have it for more accurate edge
            if data_prob is not None:
                edge = data_prob - buy_price
            else:
                # Market consensus: assume true prob ≈ buy_price (edge ≈ 0, but profit from settlement)
                edge = profit_pct  # The profit IS the edge for bonds

            # Confidence: based on verification source quality
            if verification_source in ("noaa", "yahoo_finance"):
                confidence = 0.90  # Hard data verification
            elif verification_source == "consensus_strong":
                confidence = 0.75  # Strong market consensus
            else:
                confidence = 0.60

            # Token ID
            if side == "buy_yes":
                token_id = km.ticker
            else:
                token_id = f"{km.ticker}:no"

            # Hours to settlement for urgency
            hours_left = 0.0
            if km.close_time:
                hours_left = (km.close_time - datetime.now(timezone.utc)).total_seconds() / 3600

            urgency = "normal"
            if hours_left < 6:
                urgency = "immediate"

            market = _kalshi_to_market(km)

            signal = TradeSignal(
                engine=self.name,
                market_id=km.ticker,
                token_id=token_id,
                side=side,
                confidence=confidence,
                edge=edge,
                urgency=urgency,
                metadata={
                    "platform": "kalshi",
                    "strategy": "bonding",
                    "signal_source": "bonding",
                    "buy_price": buy_price,
                    "price_cents": price_cents,
                    "profit_pct": round(profit_pct, 4),
                    "verification_source": verification_source,
                    "data_prob": data_prob,
                    "hours_to_settle": round(hours_left, 1),
                    "market_volume": km.volume,
                    "volume_24h": km.volume_24h,
                    "kalshi_ticker": km.ticker,
                    "kalshi_yes_bid": km.yes_bid,
                    "kalshi_yes_ask": km.yes_ask,
                    "kalshi_no_bid": km.no_bid,
                    "kalshi_no_ask": km.no_ask,
                    "fee_per_contract": 0.0,
                    "_force_size_usd": min(self._max_position_usd, 10.0),
                    "_market": market,
                    "title": km.title,
                    "question": km.title,
                },
            )
            self._pending.append(signal)
            self._signals_generated += 1
            bonds_emitted += 1

            self.logger.info(
                "bonding_signal_generated",
                ticker=km.ticker,
                side=side,
                buy_price=round(buy_price, 2),
                profit_pct=round(profit_pct, 4),
                edge=round(edge, 4),
                confidence=confidence,
                verification=verification_source,
                hours_to_settle=round(hours_left, 1),
            )

        # Clean stale cooldown entries
        stale_cutoff = now_ts - (self._eval_cooldown * 2)
        stale_keys = [k for k, ts in self._recently_evaluated.items() if ts < stale_cutoff]
        for k in stale_keys:
            del self._recently_evaluated[k]

        # Scan summary
        self.logger.info(
            "bonding_scan_summary",
            markets_fetched=len(kalshi_markets),
            candidates=len(candidates),
            verified=self._verified_count,
            signals_emitted=bonds_emitted,
            total_signals=self._signals_generated,
        )

    async def _verify_near_certainty(
        self,
        km: KalshiMarket,
        side: str,
    ) -> Tuple[bool, str, Optional[float]]:
        """Verify that a near-certain market is truly near-certain.

        Returns (verified: bool, source: str, data_probability: Optional[float]).

        Verification tiers:
        1. Weather markets → NOAA data (compute_weather_probability)
        2. Index markets → Yahoo Finance (compute_stock_index_probability)
        3. Other markets → require price >= 95c AND volume >= $5000 (strong consensus)
        """
        if not self._verify_with_data:
            return True, "verification_disabled", None

        mtype = detect_market_type(km.title)
        ticker_upper = km.ticker.upper()

        # Tier 1: Weather — verify with NOAA
        if mtype == "weather":
            try:
                weather_result = await compute_weather_probability(
                    km.title, km.ticker, km.close_time,
                )
                if weather_result is not None:
                    p_yes, w_confidence, w_reasoning = weather_result
                    # For YES bonds: data must agree outcome is very likely (p_yes > 0.85)
                    # For NO bonds: data must agree YES is very unlikely (p_yes < 0.15)
                    if side == "buy_yes" and p_yes >= 0.85:
                        return True, "noaa", p_yes
                    elif side == "buy_no" and p_yes <= 0.15:
                        return True, "noaa", 1.0 - p_yes
                    else:
                        # NOAA disagrees — this is NOT near-certain
                        self.logger.debug(
                            "bonding_noaa_disagrees",
                            ticker=km.ticker,
                            side=side,
                            p_yes=round(p_yes, 3),
                        )
                        return False, "noaa_disagrees", None
                # NOAA returned None (ambiguous) — fall through to consensus check
            except Exception as exc:
                self.logger.debug(
                    "bonding_weather_verify_error",
                    ticker=km.ticker,
                    error=str(exc),
                )
                # Fall through to consensus check

        # Tier 2: Stock index — verify with Yahoo Finance
        if any(ticker_upper.startswith(p) for p in _INDEX_PREFIXES):
            try:
                idx_result = await compute_stock_index_probability(
                    km.title, km.ticker, km.close_time,
                )
                if idx_result is not None:
                    p_yes, idx_confidence, idx_reasoning = idx_result
                    if side == "buy_yes" and p_yes >= 0.85:
                        return True, "yahoo_finance", p_yes
                    elif side == "buy_no" and p_yes <= 0.15:
                        return True, "yahoo_finance", 1.0 - p_yes
                    else:
                        self.logger.debug(
                            "bonding_index_disagrees",
                            ticker=km.ticker,
                            side=side,
                            p_yes=round(p_yes, 3),
                        )
                        return False, "yahoo_disagrees", None
                # Yahoo returned None — fall through to consensus check
            except Exception as exc:
                self.logger.debug(
                    "bonding_index_verify_error",
                    ticker=km.ticker,
                    error=str(exc),
                )
                # Fall through to consensus check

        # Tier 3: Market consensus — no structured data available
        # Require HIGHER bar: price >= 95c AND volume >= $5000
        dominant_price = km.yes_price if side == "buy_yes" else km.no_price
        dominant_cents = int(dominant_price * 100)

        if dominant_cents >= self._consensus_min_price and km.volume >= self._consensus_min_volume:
            return True, "consensus_strong", None

        # Not verified — either price too low or volume too thin for pure consensus
        self.logger.debug(
            "bonding_consensus_insufficient",
            ticker=km.ticker,
            side=side,
            price_cents=dominant_cents,
            volume=km.volume,
            min_price=self._consensus_min_price,
            min_volume=self._consensus_min_volume,
        )
        return False, "consensus_weak", None
