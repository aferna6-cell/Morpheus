"""Market regime detector for Morpheus V2.

Analyses recent market_snapshots and P&L history to infer the current market
regime, then translates that into a Kelly multiplier that the risk pipeline
(or the orchestrator) can apply to position sizing.

Regimes
-------
NORMAL       — Baseline behaviour; multiplier = 1.0
TRENDING     — Strong directional edge visible; multiplier = 1.2
VOLATILE     — Price swings are large; reduce size; multiplier = 0.5
LOW_ACTIVITY — Volume and signal rate are depressed; multiplier = 0.7

The regime is re-evaluated every time detect() is called. The caller is
responsible for caching the result if performance matters.

Usage:
    detector = RegimeDetector(config, state_provider)
    regime, multiplier = await detector.detect()
    adjusted_kelly = base_kelly * multiplier
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import structlog

from .utils import BotConfig

# ---------------------------------------------------------------------------
# Multiplier bounds — hard-coded, not configurable
# ---------------------------------------------------------------------------

REGIME_MULTIPLIER_MIN: float = 0.5
REGIME_MULTIPLIER_MAX: float = 1.5

# ---------------------------------------------------------------------------
# Thresholds used in detection
# ---------------------------------------------------------------------------

# Minimum number of snapshots required before we attempt regime detection
MIN_SNAPSHOTS_FOR_DETECTION: int = 5

# Price volatility: if the average intraday yes_price range (max-min)
# across tracked markets exceeds this fraction of the mid-price, flag VOLATILE
VOLATILE_PRICE_RANGE_PCT: float = 0.15

# Trending: if the mean absolute P&L per recent trade >= this, flag TRENDING
TRENDING_MEAN_PNL_USD: float = 0.05

# Low activity: if the 24-h signal count is below this fraction of the normal
# rolling baseline, flag LOW_ACTIVITY
LOW_ACTIVITY_SIGNAL_FRACTION: float = 0.40


class Regime(str, Enum):
    NORMAL = "NORMAL"
    TRENDING = "TRENDING"
    VOLATILE = "VOLATILE"
    LOW_ACTIVITY = "LOW_ACTIVITY"


_REGIME_MULTIPLIERS: dict[Regime, float] = {
    Regime.NORMAL: 1.0,
    Regime.TRENDING: 1.2,
    Regime.VOLATILE: 0.5,
    Regime.LOW_ACTIVITY: 0.7,
}


@dataclass
class RegimeResult:
    """Output from RegimeDetector.detect()."""

    regime: Regime
    kelly_multiplier: float          # Already clamped to [0.5, 1.5]
    confidence: float                # 0.0–1.0; how certain we are
    reason: str                      # Human-readable explanation (for logs)


class RegimeDetector:
    """Detects current market regime from recent snapshots and P&L.

    Args:
        config: BotConfig loaded from config.yaml.
        state_provider: Object that exposes portfolio/market state.
            Required methods (sync or async):
              - get_recent_snapshots(n: int) -> list[dict]
                  Each dict has keys: ticker, yes_price, no_price, volume, snapshot_at
              - get_recent_pnl_records(n: int) -> list[dict]
                  Each dict has keys: pnl_cents (int), timestamp (str)
              - get_recent_signal_count(hours: int) -> int
                  Number of signals generated in the last N hours
              - get_baseline_signal_rate() -> float
                  Expected signals per hour under normal conditions
    """

    def __init__(self, config: BotConfig, state_provider: object) -> None:
        self.config = config
        self.state = state_provider
        self.logger = structlog.get_logger()

        # How many recent snapshots / P&L records to examine
        self._n_snapshots: int = 50
        self._n_pnl_records: int = 20
        self._signal_window_hours: int = 6

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def detect(self) -> RegimeResult:
        """Analyse recent state and return the current regime + Kelly multiplier.

        Returns NORMAL with multiplier=1.0 if there is insufficient data to
        make a determination (fail-safe — never returns < 0.5 by construction).
        """
        # Gather data
        snapshots = await self._call(self.state.get_recent_snapshots, self._n_snapshots)  # type: ignore[attr-defined]
        pnl_records = await self._call(self.state.get_recent_pnl_records, self._n_pnl_records)  # type: ignore[attr-defined]
        signal_count = await self._call(self.state.get_recent_signal_count, self._signal_window_hours)  # type: ignore[attr-defined]
        baseline_rate = await self._call(self.state.get_baseline_signal_rate)  # type: ignore[attr-defined]

        if not isinstance(snapshots, list):
            snapshots = []
        if not isinstance(pnl_records, list):
            pnl_records = []
        if not isinstance(signal_count, (int, float)):
            signal_count = 0
        if not isinstance(baseline_rate, (int, float)) or baseline_rate <= 0.0:
            baseline_rate = 1.0

        # Check for LOW_ACTIVITY first (overrides everything — no point trading)
        low_activity_result = self._check_low_activity(
            signal_count, baseline_rate, self._signal_window_hours
        )
        if low_activity_result is not None:
            return low_activity_result

        if len(snapshots) < MIN_SNAPSHOTS_FOR_DETECTION:
            return RegimeResult(
                regime=Regime.NORMAL,
                kelly_multiplier=1.0,
                confidence=0.0,
                reason=f"insufficient snapshots ({len(snapshots)} < {MIN_SNAPSHOTS_FOR_DETECTION})",
            )

        # VOLATILE check (takes precedence over TRENDING — don't size up into chaos)
        volatile_result = self._check_volatile(snapshots)
        if volatile_result is not None:
            return volatile_result

        # TRENDING check
        trending_result = self._check_trending(pnl_records)
        if trending_result is not None:
            return trending_result

        return RegimeResult(
            regime=Regime.NORMAL,
            kelly_multiplier=1.0,
            confidence=0.8,
            reason="all regime checks within normal bounds",
        )

    # ------------------------------------------------------------------
    # Regime detection sub-checks
    # ------------------------------------------------------------------

    def _check_volatile(
        self, snapshots: list[dict]
    ) -> Optional[RegimeResult]:
        """Detect VOLATILE regime from large intraday price ranges.

        For each ticker, compute the range of yes_price across snapshots.
        If the median range-to-mid ratio exceeds the threshold → VOLATILE.
        """
        ticker_prices: dict[str, list[float]] = {}
        for snap in snapshots:
            ticker = snap.get("ticker", "")
            price = snap.get("yes_price")
            if ticker and isinstance(price, (int, float)) and price is not None:
                ticker_prices.setdefault(ticker, []).append(float(price))

        if not ticker_prices:
            return None

        range_ratios: list[float] = []
        for ticker, prices in ticker_prices.items():
            if len(prices) < 2:
                continue
            price_range = max(prices) - min(prices)
            mid = statistics.mean(prices)
            if mid > 0.0:
                range_ratios.append(price_range / mid)

        if not range_ratios:
            return None

        median_ratio = statistics.median(range_ratios)

        if median_ratio >= VOLATILE_PRICE_RANGE_PCT:
            multiplier = _clamp(
                _REGIME_MULTIPLIERS[Regime.VOLATILE], REGIME_MULTIPLIER_MIN, REGIME_MULTIPLIER_MAX
            )
            confidence = min(1.0, median_ratio / VOLATILE_PRICE_RANGE_PCT - 1.0 + 0.5)
            self.logger.info(
                "regime_detected_volatile",
                median_range_ratio=round(median_ratio, 4),
                threshold=VOLATILE_PRICE_RANGE_PCT,
                kelly_multiplier=multiplier,
            )
            return RegimeResult(
                regime=Regime.VOLATILE,
                kelly_multiplier=multiplier,
                confidence=round(confidence, 3),
                reason=(
                    f"median intraday price range ratio={median_ratio:.3f} "
                    f">= threshold={VOLATILE_PRICE_RANGE_PCT}"
                ),
            )

        return None

    def _check_trending(
        self, pnl_records: list[dict]
    ) -> Optional[RegimeResult]:
        """Detect TRENDING regime from consistent positive recent P&L.

        If the mean absolute P&L across recent closed trades is high relative
        to TRENDING_MEAN_PNL_USD and the win rate >= 60%, we're trending.
        """
        if len(pnl_records) < 5:
            return None

        pnl_values: list[float] = []
        wins = 0
        for rec in pnl_records:
            pnl_cents = rec.get("pnl_cents")
            if isinstance(pnl_cents, (int, float)):
                pnl_usd = float(pnl_cents) / 100.0
                pnl_values.append(pnl_usd)
                if pnl_usd > 0:
                    wins += 1

        if not pnl_values:
            return None

        mean_abs_pnl = statistics.mean(abs(v) for v in pnl_values)
        win_rate = wins / len(pnl_values)

        if mean_abs_pnl >= TRENDING_MEAN_PNL_USD and win_rate >= 0.60:
            multiplier = _clamp(
                _REGIME_MULTIPLIERS[Regime.TRENDING], REGIME_MULTIPLIER_MIN, REGIME_MULTIPLIER_MAX
            )
            confidence = min(1.0, win_rate * (mean_abs_pnl / TRENDING_MEAN_PNL_USD) * 0.5)
            self.logger.info(
                "regime_detected_trending",
                mean_abs_pnl=round(mean_abs_pnl, 4),
                win_rate=round(win_rate, 3),
                kelly_multiplier=multiplier,
            )
            return RegimeResult(
                regime=Regime.TRENDING,
                kelly_multiplier=multiplier,
                confidence=round(confidence, 3),
                reason=(
                    f"mean_abs_pnl={mean_abs_pnl:.4f} >= {TRENDING_MEAN_PNL_USD} "
                    f"and win_rate={win_rate:.2%} >= 60%"
                ),
            )

        return None

    def _check_low_activity(
        self,
        signal_count: int,
        baseline_rate: float,
        window_hours: int,
    ) -> Optional[RegimeResult]:
        """Detect LOW_ACTIVITY regime from depressed signal generation.

        If the observed signal count over window_hours is below
        LOW_ACTIVITY_SIGNAL_FRACTION of the expected count from baseline_rate,
        we're in low-activity.
        """
        expected = baseline_rate * window_hours
        if expected <= 0.0:
            return None

        actual_fraction = signal_count / expected

        if actual_fraction < LOW_ACTIVITY_SIGNAL_FRACTION:
            multiplier = _clamp(
                _REGIME_MULTIPLIERS[Regime.LOW_ACTIVITY],
                REGIME_MULTIPLIER_MIN,
                REGIME_MULTIPLIER_MAX,
            )
            confidence = max(0.0, 1.0 - actual_fraction / LOW_ACTIVITY_SIGNAL_FRACTION)
            self.logger.info(
                "regime_detected_low_activity",
                signal_count=signal_count,
                expected=round(expected, 1),
                actual_fraction=round(actual_fraction, 3),
                kelly_multiplier=multiplier,
            )
            return RegimeResult(
                regime=Regime.LOW_ACTIVITY,
                kelly_multiplier=multiplier,
                confidence=round(confidence, 3),
                reason=(
                    f"signal_count={signal_count} is {actual_fraction:.1%} of "
                    f"expected={expected:.1f} (threshold={LOW_ACTIVITY_SIGNAL_FRACTION:.0%})"
                ),
            )

        return None

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    async def _call(fn: object, *args: object) -> object:
        """Call a method that may be sync or async, forwarding positional args."""
        import inspect
        result = fn(*args)  # type: ignore[operator]
        if inspect.isawaitable(result):
            return await result  # type: ignore[misc]
        return result


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
