"""Kalshi theta decay engine — time-decay seller strategy.

Harvests the empirical mispricing in the final 2 hours before market close:

1. **Favorites (80-92c):** Whelan et al. (300K Kalshi contracts) show events priced
   at 80% actually win ~84% of the time — a persistent 4pp underpricing.
   We buy YES at maker price (bid+1c) on these underpriced favorites.

2. **Longshots (5-15c):** Longshot bias — bettors overpay for low-probability
   outcomes. Events priced at 15% win only ~10-12%. We buy NO to harvest
   the overpricing.

Near close, theta decay accelerates: uncertainty collapses as resolution
approaches, and prices converge toward 0/100. We capture this convergence.

Skips: weather, index, crypto (already covered by dedicated fast-paths),
junk/sports (no edge), and all markets not closing within 2 hours.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from ..engines.base import BaseEngine
from ..engines.signals import TradeSignal
from ..kalshi_client import KalshiClient, KalshiMarket
from ..utils import BotConfig


# Ticker prefixes already covered by fast-path engines — skip to avoid overlap
_FASTPATH_PREFIXES = (
    # Weather fast-paths
    "KXHIGH", "KXLOW", "KXRAIN", "KXSNOW", "KXTEMP", "KXWIND",
    # Index fast-paths
    "KXINXU", "KXINX-", "KXNASDAQ100",
    # Crypto
    "KXBTC", "KXETH", "KXDOGE", "KXSOL", "KXXRP",
    "KXBTCD", "KXETHD", "KXDOGED", "KXSOLD",
    "KXBTC15M", "KXETH15M", "KXSOL15M",
    "KXBNBD", "KXLTCD", "KXADAD", "KXDOTD", "KXAVAXD", "KXLINKD",
    "KXMATD", "KXUNIDD", "KXSHIB", "KXXRPD",
    # Commodity fast-paths
    "KXWTI", "KXWTIW", "KXGOLD",
)

# Sports / junk — never touch
_JUNK_PREFIXES = (
    "KXNBAMENTION", "KXNCAAB", "KXNFLMENTION", "KXFOXNEWSMENTION",
    "KXMLBMENTION", "KXNHLMENTION", "KXMLSMENTION", "KXCONGRESSMENTION",
    "KXTRUMPMENTION", "KXWOMENTION", "KXWMENTION",
    "KXSUPERBOWLAD", "KXRT", "KXSPOTIFY", "KXSPOTIFYD", "KXSPOTIFYGLOBALD",
    "KXSBADAPPEARANCES", "KXTOPSONG", "KXTOPALBUM", "KXALBUMDEBUT",
    "KXFIRSTSUPERBOWLSONG", "KXAAAGASW", "KXNEXTTEAMNFL",
    "KXNETFLIXRANK", "KXNETFLIX", "KXNASCARRACE", "KXNASCAR",
    "KXF1RACE", "KXINDYRACE", "KXLLM", "KXNBAALLSTAR",
    "KXEOWEEK", "KXTRUMPACT", "KXEXECORDER",
    # Sports categories
    "KXNFL", "KXNBA", "KXMLB", "KXNHL", "KXMLS",
    "KXSCOTTISHPREM", "KXPREM", "KXLALIGA", "KXSERIEA", "KXBUNDESLIGA",
    "KXCHAMPIONSLEAGUE", "KXNCAA", "KXCFB",
)

# All skip prefixes combined
_SKIP_PREFIXES = _FASTPATH_PREFIXES + _JUNK_PREFIXES


def _should_skip(ticker: str) -> bool:
    """Return True if this ticker is covered by other engines or is junk."""
    t = ticker.upper()
    for prefix in _SKIP_PREFIXES:
        if t.startswith(prefix):
            return True
    return False


def _hours_to_close(market: KalshiMarket) -> Optional[float]:
    """Return hours until market close, or None if no close_time."""
    if not market.close_time:
        return None
    now = datetime.now(timezone.utc)
    delta = market.close_time - now
    return delta.total_seconds() / 3600.0


class KalshiThetaEngine(BaseEngine):
    """Time-decay seller engine — harvests terminal mispricing near close.

    Scans for markets closing within 2 hours and exploits two empirical biases:
    - Favorite underpricing (80-92c YES → actual ~84-94% win rate)
    - Longshot overpricing (5-15c YES → actual ~2-10% win rate)
    """

    name = "kalshi_theta"

    def __init__(
        self,
        config: BotConfig,
        kalshi_client: KalshiClient,
    ):
        self.config = config
        self.kalshi_client = kalshi_client
        self.logger = structlog.get_logger()

        # Read config with sensible defaults
        theta_cfg = getattr(config, "theta_engine", None) or {}
        if not isinstance(theta_cfg, dict):
            theta_cfg = {}

        # Scan parameters
        self._scan_interval = float(theta_cfg.get("scan_interval_seconds", 300))
        self._max_hours_to_close = float(theta_cfg.get("max_hours_to_close", 2.0))
        self._min_hours_to_close = float(theta_cfg.get("min_hours_to_close", 0.05))  # ~3 min floor

        # Favorite parameters (buy YES on high-prob markets)
        self._fav_min_price = float(theta_cfg.get("fav_min_price", 0.80))
        self._fav_max_price = float(theta_cfg.get("fav_max_price", 0.92))
        self._fav_min_edge = float(theta_cfg.get("fav_min_edge", 0.03))  # 3% edge minimum

        # Longshot parameters (buy NO on low-prob markets)
        self._long_min_price = float(theta_cfg.get("long_min_price", 0.05))
        self._long_max_price = float(theta_cfg.get("long_max_price", 0.15))
        self._long_min_edge = float(theta_cfg.get("long_min_edge", 0.05))  # 5% edge minimum

        # Sizing & limits
        self._max_per_trade_usd = float(theta_cfg.get("max_per_trade_usd", 3.0))
        self._max_theta_positions = int(theta_cfg.get("max_theta_positions", 5))
        self._min_volume = int(theta_cfg.get("min_volume", 200))  # lower bar near close

        # Empirical calibration: Whelan et al. win rates by price bucket
        # price_bucket → actual_win_rate
        self._fav_calibration = {
            0.80: 0.84,
            0.82: 0.86,
            0.84: 0.87,
            0.86: 0.89,
            0.88: 0.91,
            0.90: 0.93,
            0.92: 0.94,
        }
        self._long_calibration = {
            0.05: 0.02,
            0.07: 0.04,
            0.10: 0.06,
            0.12: 0.08,
            0.15: 0.10,
        }

        # State
        self._pending_signals: List[TradeSignal] = []
        self._active_tickers: set = set()  # tickers with active theta positions
        self._scan_task: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())
        self.logger.info(
            "theta_engine_started",
            scan_interval=self._scan_interval,
            max_hours=self._max_hours_to_close,
            fav_range=f"{self._fav_min_price}-{self._fav_max_price}",
            long_range=f"{self._long_min_price}-{self._long_max_price}",
            max_positions=self._max_theta_positions,
        )

    async def stop(self) -> None:
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        self.logger.info("theta_engine_stopped")

    async def get_signals(self) -> List[TradeSignal]:
        signals = list(self._pending_signals)
        self._pending_signals.clear()
        return signals

    def trigger_rescan(self) -> None:
        """Force a rescan on next cycle (called on account resume)."""
        self._active_tickers.clear()

    # ------------------------------------------------------------------
    # Scan loop
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        """Periodically scan for theta decay opportunities."""
        while self._running:
            try:
                await self._scan_for_theta()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error("theta_scan_error", error=str(e))
            await asyncio.sleep(self._scan_interval)

    async def _scan_for_theta(self) -> None:
        """Fetch markets closing soon and evaluate for theta signals."""
        # Fetch markets closing within 1 day (we filter down to 2h ourselves)
        try:
            all_markets = await self.kalshi_client.fetch_markets_by_close_date(
                max_days=1,
                min_volume=self._min_volume,
            )
        except Exception as e:
            self.logger.error("theta_fetch_error", error=str(e))
            return

        # Filter to markets closing within our theta window
        candidates: List[KalshiMarket] = []
        for m in all_markets:
            # Must be open
            if m.status not in ("open", "active"):
                continue

            # Skip covered/junk tickers
            if _should_skip(m.ticker):
                continue

            # Already have a theta position here
            if m.ticker in self._active_tickers:
                continue

            # Check time to close
            hours = _hours_to_close(m)
            if hours is None:
                continue
            if hours > self._max_hours_to_close or hours < self._min_hours_to_close:
                continue

            candidates.append(m)

        if not candidates:
            self.logger.debug("theta_no_candidates", total_markets=len(all_markets))
            return

        # Separate into favorites and longshots
        favorites: List[KalshiMarket] = []
        longshots: List[KalshiMarket] = []

        for m in candidates:
            yes_price = m.yes_price
            if self._fav_min_price <= yes_price <= self._fav_max_price:
                favorites.append(m)
            elif self._long_min_price <= yes_price <= self._long_max_price:
                longshots.append(m)

        self.logger.info(
            "theta_scan_results",
            total_candidates=len(candidates),
            favorites=len(favorites),
            longshots=len(longshots),
            active_positions=len(self._active_tickers),
        )

        # Evaluate and emit signals (respect max position limit)
        available_slots = self._max_theta_positions - len(self._active_tickers)
        if available_slots <= 0:
            self.logger.debug("theta_max_positions_reached")
            return

        signals_emitted = 0

        # Process favorites — buy YES on underpriced high-prob markets
        for m in favorites:
            if signals_emitted >= available_slots:
                break
            signal = self._evaluate_favorite(m)
            if signal:
                self._pending_signals.append(signal)
                self._active_tickers.add(m.ticker)
                signals_emitted += 1

        # Process longshots — buy NO on overpriced low-prob markets
        for m in longshots:
            if signals_emitted >= available_slots:
                break
            signal = self._evaluate_longshot(m)
            if signal:
                self._pending_signals.append(signal)
                self._active_tickers.add(m.ticker)
                signals_emitted += 1

        if signals_emitted > 0:
            self.logger.info(
                "theta_signals_emitted",
                count=signals_emitted,
                tickers=[s.market_id for s in self._pending_signals[-signals_emitted:]],
            )

    # ------------------------------------------------------------------
    # Signal evaluation
    # ------------------------------------------------------------------

    def _estimate_true_prob_favorite(self, yes_price: float) -> float:
        """Estimate true win probability for a favorite using Whelan calibration.

        Interpolates between calibration points. Markets priced at 80c
        actually win ~84% of the time (4pp underpriced).
        """
        # Find surrounding calibration points
        buckets = sorted(self._fav_calibration.keys())
        if yes_price <= buckets[0]:
            return self._fav_calibration[buckets[0]]
        if yes_price >= buckets[-1]:
            return self._fav_calibration[buckets[-1]]

        # Linear interpolation
        for i in range(len(buckets) - 1):
            lo, hi = buckets[i], buckets[i + 1]
            if lo <= yes_price <= hi:
                frac = (yes_price - lo) / (hi - lo)
                return self._fav_calibration[lo] + frac * (
                    self._fav_calibration[hi] - self._fav_calibration[lo]
                )

        return yes_price  # fallback: no calibration edge

    def _estimate_true_prob_longshot(self, yes_price: float) -> float:
        """Estimate true win probability for a longshot using calibration.

        Markets priced at 15c actually win only ~10% of the time.
        Returns the actual YES probability (we buy NO, so our edge = yes_price - true_prob).
        """
        buckets = sorted(self._long_calibration.keys())
        if yes_price <= buckets[0]:
            return self._long_calibration[buckets[0]]
        if yes_price >= buckets[-1]:
            return self._long_calibration[buckets[-1]]

        # Linear interpolation
        for i in range(len(buckets) - 1):
            lo, hi = buckets[i], buckets[i + 1]
            if lo <= yes_price <= hi:
                frac = (yes_price - lo) / (hi - lo)
                return self._long_calibration[lo] + frac * (
                    self._long_calibration[hi] - self._long_calibration[lo]
                )

        return yes_price  # fallback

    def _evaluate_favorite(self, market: KalshiMarket) -> Optional[TradeSignal]:
        """Evaluate a favorite market (80-92c) for theta signal.

        Strategy: buy YES at maker price (yes_bid + 1c). The true win rate
        exceeds the market price by ~4pp, giving us a statistical edge.
        """
        yes_price = market.yes_price
        true_prob = self._estimate_true_prob_favorite(yes_price)

        # Entry price: bid + 1c (maker order, sits on book)
        entry_price = market.yes_bid + 0.01 if market.yes_bid > 0 else yes_price
        # Don't overshoot — cap at ask
        if market.yes_ask > 0:
            entry_price = min(entry_price, market.yes_ask)

        # Edge: true probability minus our cost
        edge = true_prob - entry_price

        # Require minimum edge
        if edge < self._fav_min_edge:
            self.logger.debug(
                "theta_fav_low_edge",
                ticker=market.ticker,
                yes_price=round(yes_price, 3),
                true_prob=round(true_prob, 3),
                entry=round(entry_price, 3),
                edge=round(edge, 4),
            )
            return None

        # Need valid bid/ask — no point signaling in a dead orderbook
        if market.yes_bid <= 0 or market.yes_ask <= 0:
            return None

        # Spread sanity: reject if spread > 10c (illiquid)
        spread = market.yes_ask - market.yes_bid
        if spread > 0.10:
            return None

        hours = _hours_to_close(market) or 2.0

        # Confidence: higher closer to close, higher with more edge
        confidence = min(0.85, 0.60 + edge * 3.0 + (2.0 - hours) * 0.05)

        self.logger.info(
            "theta_fav_signal",
            ticker=market.ticker,
            title=market.title[:60],
            yes_price=round(yes_price, 3),
            true_prob=round(true_prob, 3),
            entry=round(entry_price, 3),
            edge=round(edge, 4),
            hours_to_close=round(hours, 2),
            confidence=round(confidence, 3),
        )

        return TradeSignal(
            engine=self.name,
            market_id=market.ticker,
            token_id=market.ticker,
            side="buy_yes",
            confidence=confidence,
            edge=edge,
            urgency="normal",
            metadata={
                "estimated_prob": true_prob,
                "market_price": yes_price,
                "entry_price": entry_price,
                "net_edge": edge,
                "conviction": "medium",
                "reasoning": (
                    f"Theta favorite: {market.title[:50]}. "
                    f"Price {yes_price:.0%} but Whelan calibration says "
                    f"{true_prob:.0%} true win rate ({edge:.1%} edge). "
                    f"{hours:.1f}h to close."
                ),
                "question": market.title,
                "kalshi_ticker": market.ticker,
                "kalshi_yes_bid": market.yes_bid,
                "kalshi_yes_ask": market.yes_ask,
                "kalshi_no_bid": market.no_bid,
                "kalshi_no_ask": market.no_ask,
                "platform": "kalshi",
                "strategy": "theta_favorite",
                "signal_source": "theta_decay",
                "theta_type": "favorite",
                "hours_to_close": round(hours, 2),
                "_force_size_usd": min(self._max_per_trade_usd, entry_price * 5),
            },
        )

    def _evaluate_longshot(self, market: KalshiMarket) -> Optional[TradeSignal]:
        """Evaluate a longshot market (5-15c YES) for theta signal.

        Strategy: buy NO at maker price. The true YES probability is lower
        than the market price (longshot bias), so NO is underpriced.
        """
        yes_price = market.yes_price
        true_yes_prob = self._estimate_true_prob_longshot(yes_price)

        # We buy NO: our true win rate = 1 - true_yes_prob
        true_no_prob = 1.0 - true_yes_prob
        no_price = market.no_price  # current no mid

        # Entry price: no_bid + 1c (maker order)
        entry_price = market.no_bid + 0.01 if market.no_bid > 0 else no_price
        # Cap at ask
        if market.no_ask > 0:
            entry_price = min(entry_price, market.no_ask)

        # Edge: true NO probability minus our cost
        edge = true_no_prob - entry_price

        # Require minimum edge
        if edge < self._long_min_edge:
            self.logger.debug(
                "theta_long_low_edge",
                ticker=market.ticker,
                yes_price=round(yes_price, 3),
                true_yes_prob=round(true_yes_prob, 3),
                no_entry=round(entry_price, 3),
                edge=round(edge, 4),
            )
            return None

        # Need valid bid/ask
        if market.no_bid <= 0 or market.no_ask <= 0:
            return None

        # Spread sanity
        spread = market.no_ask - market.no_bid
        if spread > 0.10:
            return None

        hours = _hours_to_close(market) or 2.0

        # Confidence: longshots need higher bar (more uncertainty)
        confidence = min(0.80, 0.55 + edge * 2.5 + (2.0 - hours) * 0.05)

        self.logger.info(
            "theta_long_signal",
            ticker=market.ticker,
            title=market.title[:60],
            yes_price=round(yes_price, 3),
            true_yes_prob=round(true_yes_prob, 3),
            no_entry=round(entry_price, 3),
            edge=round(edge, 4),
            hours_to_close=round(hours, 2),
            confidence=round(confidence, 3),
        )

        return TradeSignal(
            engine=self.name,
            market_id=market.ticker,
            token_id=f"{market.ticker}:no",
            side="buy_no",
            confidence=confidence,
            edge=edge,
            urgency="normal",
            metadata={
                "estimated_prob": true_no_prob,
                "market_price": no_price,
                "entry_price": entry_price,
                "net_edge": edge,
                "conviction": "medium",
                "reasoning": (
                    f"Theta longshot: {market.title[:50]}. "
                    f"YES at {yes_price:.0%} but calibrated to {true_yes_prob:.0%} "
                    f"(longshot bias). Buy NO at {entry_price:.0%} for "
                    f"{edge:.1%} edge. {hours:.1f}h to close."
                ),
                "question": market.title,
                "kalshi_ticker": market.ticker,
                "kalshi_yes_bid": market.yes_bid,
                "kalshi_yes_ask": market.yes_ask,
                "kalshi_no_bid": market.no_bid,
                "kalshi_no_ask": market.no_ask,
                "platform": "kalshi",
                "strategy": "theta_longshot",
                "signal_source": "theta_decay",
                "theta_type": "longshot",
                "hours_to_close": round(hours, 2),
                "_force_size_usd": min(self._max_per_trade_usd, entry_price * 5),
            },
        )

    # ------------------------------------------------------------------
    # Position tracking helpers
    # ------------------------------------------------------------------

    def clear_position(self, ticker: str) -> None:
        """Remove a ticker from active theta positions (after resolution/exit)."""
        self._active_tickers.discard(ticker)

    def get_stats(self) -> Dict[str, Any]:
        """Return theta engine statistics."""
        return {
            "active_positions": len(self._active_tickers),
            "active_tickers": list(self._active_tickers),
            "max_positions": self._max_theta_positions,
            "scan_interval": self._scan_interval,
            "max_hours": self._max_hours_to_close,
        }
