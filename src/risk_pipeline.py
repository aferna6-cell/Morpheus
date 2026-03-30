"""Formal 8-layer sequential risk validation pipeline for Morpheus V2.

Each layer can independently reject a trade. The first rejection terminates
the pipeline and returns an unapproved RiskDecision. If all 8 layers pass,
Kelly criterion sizing is applied and a position size is returned.

Hard-coded ceilings are enforced as module-level constants and cannot be
overridden by any config value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import structlog

from .utils import BotConfig

# ---------------------------------------------------------------------------
# Hard-coded ceilings — cannot be overridden by config
# ---------------------------------------------------------------------------
ABSOLUTE_MAX_POSITION_SIZE: float = 200.0     # Never exceed $200 per market
ABSOLUTE_MAX_TOTAL_EXPOSURE: float = 10_000.0  # Never exceed $10K total open
ABSOLUTE_MAX_DAILY_LOSS: float = 500.0         # Never lose more than $500/day
ABSOLUTE_MAX_KELLY_FRACTION: float = 0.50      # Never use more than half-Kelly
ABSOLUTE_MIN_CONFIDENCE: float = 0.30          # Never trade below 30% confidence
ABSOLUTE_MIN_EDGE_CENTS: float = 1.0           # Never trade below 1c edge

# Kalshi approximate maker fee rate (0.0175 * P * (1 - P) per contract)
# At P=0.50 this works out to ~0.4375% of notional.
KALSHI_MAKER_FEE_COEFFICIENT: float = 0.0175

# Confidence floors by market category
_CONFIDENCE_FLOORS: dict[str, float] = {
    "crypto": 0.85,
    "politics": 0.70,
    "economics": 0.65,
    "sports": 0.60,
}
_DEFAULT_CONFIDENCE_FLOOR: float = 0.60

# Kelly boost multipliers by market type (from calibration config fallback)
_DEFAULT_KELLY_BOOSTS: dict[str, float] = {
    "index": 1.0,
    "weather": 1.0,
    "economics": 0.70,
    "politics": 0.50,
}
_DEFAULT_KELLY_BOOST: float = 1.0


@dataclass
class SignalForRisk:
    """Normalised signal passed into the risk pipeline by the orchestrator."""

    ticker: str
    side: str                    # 'BUY_YES' | 'BUY_NO'
    confidence: float
    edge_cents: float            # Expected P&L per contract in cents
    urgency: str                 # 'immediate' | 'normal' | 'low'
    strategy: str
    market_type: str             # 'index' | 'weather' | 'economics' | 'politics' | ...
    yes_price_cents: int         # Current YES ask/bid in cents (1-99)
    no_price_cents: int          # Current NO ask/bid in cents (1-99)
    volume: int                  # 24-hour volume in contracts

    # Optional position-size override (used by arb/mm strategies)
    forced_size_usd: Optional[float] = field(default=None)


@dataclass
class RiskDecision:
    """Result returned by RiskPipeline.validate()."""

    approved: bool
    position_size: float          # In USD; 0.0 when not approved
    reject_reason: Optional[str]  # None when approved
    layer: Optional[str]          # Which layer rejected (for logging/audit)
    kelly_fraction_used: float    # Actual Kelly fraction applied (audit trail)


class RiskPipeline:
    """8-layer sequential risk validation gate.

    Each layer returns an error string on failure or None on pass.  The first
    failure terminates the pipeline immediately (fail-closed).  All
    dependencies are injected — no global state.

    Args:
        config: BotConfig loaded from config.yaml.
        state_provider: Protocol object exposing portfolio state queries. Must
            implement the following async/sync methods:
              - get_daily_pnl() -> float
              - get_total_exposure() -> float
              - get_balance() -> float
              - get_recent_losses() -> int
              - get_portfolio_peak() -> float
              - get_current_equity() -> float
              - get_strategy_errors(strategy: str) -> int
    """

    def __init__(self, config: BotConfig, state_provider: object) -> None:
        self.config = config
        self.state = state_provider
        self.logger = structlog.get_logger()

        # Cache frequently-accessed config values (with hard-ceiling enforcement)
        strat = config.strategy
        risk = config.risk
        mf = config.market_filters

        raw_kelly = float(strat.get("kelly_fraction", 0.33))
        self._kelly_fraction: float = min(raw_kelly, ABSOLUTE_MAX_KELLY_FRACTION)

        raw_max_pos = float(strat.get("max_position_size", 25.0))
        self._max_position_size: float = min(raw_max_pos, ABSOLUTE_MAX_POSITION_SIZE)

        raw_max_exp = float(strat.get("max_total_exposure", 200.0))
        self._max_total_exposure: float = min(raw_max_exp, ABSOLUTE_MAX_TOTAL_EXPOSURE)

        raw_daily_loss = float(risk.get("max_daily_loss", 20.0))
        self._max_daily_loss: float = min(raw_daily_loss, ABSOLUTE_MAX_DAILY_LOSS)

        self._min_edge: float = float(strat.get("min_edge", 0.04))
        self._min_confidence: float = float(strat.get("min_confidence", 0.60))
        self._min_volume: int = int(mf.get("min_volume_24h", 1000))
        self._consecutive_loss_limit: int = int(risk.get("consecutive_loss_limit", 5))
        self._drawdown_halt_pct: float = float(risk.get("drawdown_halt_pct", 0.15))

        # Per-type Kelly boosts from calibration config
        cal = config.calibration
        self._kelly_boosts: dict[str, float] = {}
        per_type = cal.get("per_type", {})
        for mtype, vals in per_type.items():
            if isinstance(vals, dict):
                boost = float(vals.get("kelly_boost", _DEFAULT_KELLY_BOOST))
                self._kelly_boosts[mtype] = boost
        # Fill in any missing defaults
        for mtype, default_boost in _DEFAULT_KELLY_BOOSTS.items():
            if mtype not in self._kelly_boosts:
                self._kelly_boosts[mtype] = default_boost

        # Drawdown halt is latched until reset; use an in-memory flag.
        # The state_provider is the authoritative source for equity/peak.
        self._drawdown_halted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def validate(self, signal: SignalForRisk) -> RiskDecision:
        """Run all 8 layers sequentially.

        Returns a RiskDecision.  The first rejection short-circuits the
        pipeline. If all layers pass, Kelly sizing is applied.
        """
        self.logger.debug(
            "risk_pipeline_start",
            ticker=signal.ticker,
            side=signal.side,
            confidence=signal.confidence,
            edge_cents=signal.edge_cents,
            strategy=signal.strategy,
            market_type=signal.market_type,
        )

        # Track whether consecutive-loss breaker fired (reduces size but
        # does NOT reject).
        consecutive_loss_flag = False

        # Layers 1-4: synchronous checks (no I/O)
        sync_layers: list[tuple[str, object]] = [
            ("layer1_spread_cost", self._layer1_spread_cost),
            ("layer2_volume_liquidity", self._layer2_volume_liquidity),
            ("layer3_confidence_floor", self._layer3_confidence_floor),
            ("layer4_net_edge", self._layer4_net_edge),
        ]
        for layer_name, fn in sync_layers:
            reason = fn(signal)  # type: ignore[call-arg]
            if reason is not None:
                self.logger.info(
                    "risk_pipeline_rejected",
                    ticker=signal.ticker,
                    layer=layer_name,
                    reason=reason,
                )
                return RiskDecision(
                    approved=False,
                    position_size=0.0,
                    reject_reason=reason,
                    layer=layer_name,
                    kelly_fraction_used=0.0,
                )

        # Layers 5-8: async (require state queries)
        async_layers: list[tuple[str, object]] = [
            ("layer5_daily_loss", self._layer5_daily_loss),
            ("layer6_consecutive_loss", self._layer6_consecutive_loss),
            ("layer7_drawdown", self._layer7_drawdown),
            ("layer8_exposure_cap", self._layer8_exposure_cap),
        ]
        for layer_name, fn in async_layers:
            reason = await fn(signal)  # type: ignore[call-arg]
            if layer_name == "layer6_consecutive_loss":
                if reason == "_REDUCED_SIZING_":
                    # Layer 6 uses a sentinel rather than rejecting
                    consecutive_loss_flag = True
                    reason = None
            if reason is not None:
                self.logger.info(
                    "risk_pipeline_rejected",
                    ticker=signal.ticker,
                    layer=layer_name,
                    reason=reason,
                )
                return RiskDecision(
                    approved=False,
                    position_size=0.0,
                    reject_reason=reason,
                    layer=layer_name,
                    kelly_fraction_used=0.0,
                )

        # All layers passed — compute Kelly size
        size, kelly_fraction = self._kelly_size(signal, consecutive_loss_flag)

        self.logger.info(
            "risk_pipeline_approved",
            ticker=signal.ticker,
            side=signal.side,
            position_size=round(size, 4),
            kelly_fraction=round(kelly_fraction, 4),
            consecutive_loss_flag=consecutive_loss_flag,
        )
        return RiskDecision(
            approved=True,
            position_size=size,
            reject_reason=None,
            layer=None,
            kelly_fraction_used=kelly_fraction,
        )

    # ------------------------------------------------------------------
    # Layer 1: Spread cost — can we profit after fees?
    # ------------------------------------------------------------------

    def _layer1_spread_cost(self, signal: SignalForRisk) -> Optional[str]:
        """Can we profit after Kalshi maker fees?

        Fee formula: fee_per_contract = KALSHI_MAKER_FEE_COEFFICIENT * P * (1-P)
        where P is the price of the side we're buying in [0.01, 0.99].

        The fee is compared in cents against edge_cents.  Net edge must be
        strictly positive.
        """
        p = signal.yes_price_cents / 100.0 if signal.side == "BUY_YES" else signal.no_price_cents / 100.0
        # Clamp to valid range
        p = max(0.01, min(0.99, p))
        fee_per_contract_cents = KALSHI_MAKER_FEE_COEFFICIENT * p * (1.0 - p) * 100.0
        net_edge_cents = signal.edge_cents - fee_per_contract_cents
        if net_edge_cents <= 0.0:
            return (
                f"spread_cost: net edge after fees is {net_edge_cents:.2f}c "
                f"(edge={signal.edge_cents:.2f}c fee={fee_per_contract_cents:.2f}c)"
            )
        return None

    # ------------------------------------------------------------------
    # Layer 2: Volume / liquidity minimum
    # ------------------------------------------------------------------

    def _layer2_volume_liquidity(self, signal: SignalForRisk) -> Optional[str]:
        """Market must meet the configured minimum 24-hour volume."""
        if signal.volume < self._min_volume:
            return (
                f"volume_liquidity: volume={signal.volume} < min={self._min_volume}"
            )
        return None

    # ------------------------------------------------------------------
    # Layer 3: Confidence floor (category-specific)
    # ------------------------------------------------------------------

    def _layer3_confidence_floor(self, signal: SignalForRisk) -> Optional[str]:
        """Confidence must exceed both the category floor and the hard minimum."""
        # Hard-coded absolute floor
        if signal.confidence < ABSOLUTE_MIN_CONFIDENCE:
            return (
                f"confidence_floor: confidence={signal.confidence:.3f} < "
                f"absolute_min={ABSOLUTE_MIN_CONFIDENCE}"
            )

        # Category-specific floor
        mtype_lower = signal.market_type.lower()
        floor = _CONFIDENCE_FLOORS.get(mtype_lower, _DEFAULT_CONFIDENCE_FLOOR)
        if signal.confidence < floor:
            return (
                f"confidence_floor: confidence={signal.confidence:.3f} < "
                f"floor={floor} for market_type='{signal.market_type}'"
            )

        return None

    # ------------------------------------------------------------------
    # Layer 4: Net edge after fees must exceed threshold
    # ------------------------------------------------------------------

    def _layer4_net_edge(self, signal: SignalForRisk) -> Optional[str]:
        """Net edge must exceed both the config threshold and the hard minimum.

        edge_cents is used for the absolute floor; the fractional min_edge
        is checked against edge_cents expressed as a fraction of price.
        """
        # Absolute minimum (hard-coded)
        if signal.edge_cents < ABSOLUTE_MIN_EDGE_CENTS:
            return (
                f"net_edge: edge_cents={signal.edge_cents:.2f} < "
                f"absolute_min={ABSOLUTE_MIN_EDGE_CENTS}c"
            )

        # Config fractional minimum: edge / price >= min_edge
        p = signal.yes_price_cents / 100.0 if signal.side == "BUY_YES" else signal.no_price_cents / 100.0
        p = max(0.01, min(0.99, p))
        # edge_cents is in cents; price p is in [0,1]; to get fraction divide edge_cents by (p*100)
        edge_fraction = signal.edge_cents / (p * 100.0)
        if edge_fraction < self._min_edge:
            return (
                f"net_edge: edge_fraction={edge_fraction:.4f} < "
                f"min_edge={self._min_edge} "
                f"(edge_cents={signal.edge_cents:.2f}c price={p:.2f})"
            )

        return None

    # ------------------------------------------------------------------
    # Layer 5: Daily loss limit
    # ------------------------------------------------------------------

    async def _layer5_daily_loss(self, signal: SignalForRisk) -> Optional[str]:
        """Halt all new trades when today's realized P&L breaches the daily loss limit."""
        daily_pnl = await self._call(self.state.get_daily_pnl)  # type: ignore[attr-defined]
        if daily_pnl <= -self._max_daily_loss:
            return (
                f"daily_loss: daily_pnl={daily_pnl:.2f} <= "
                f"-max_daily_loss={-self._max_daily_loss:.2f}"
            )
        return None

    # ------------------------------------------------------------------
    # Layer 6: Consecutive loss circuit breaker
    # ------------------------------------------------------------------

    async def _layer6_consecutive_loss(self, signal: SignalForRisk) -> Optional[str]:
        """Reduce position sizing after a run of consecutive losses.

        This layer does NOT reject — it returns a sentinel string
        '_REDUCED_SIZING_' that the pipeline runner converts into a flag
        passed to _kelly_size().  Returning None means the full Kelly is used.
        """
        recent_losses = await self._call(self.state.get_recent_losses)  # type: ignore[attr-defined]
        if recent_losses >= self._consecutive_loss_limit:
            self.logger.warning(
                "consecutive_loss_breaker_triggered",
                consecutive_losses=recent_losses,
                limit=self._consecutive_loss_limit,
                action="half_position_size",
            )
            return "_REDUCED_SIZING_"
        return None

    # ------------------------------------------------------------------
    # Layer 7: Portfolio drawdown check
    # ------------------------------------------------------------------

    async def _layer7_drawdown(self, signal: SignalForRisk) -> Optional[str]:
        """Halt trading when portfolio has drawn down >= drawdown_halt_pct from peak."""
        peak = await self._call(self.state.get_portfolio_peak)  # type: ignore[attr-defined]
        current_equity = await self._call(self.state.get_current_equity)  # type: ignore[attr-defined]

        if peak <= 0.0:
            # No meaningful peak recorded yet — skip check
            return None

        drawdown = (peak - current_equity) / peak
        if drawdown >= self._drawdown_halt_pct:
            self._drawdown_halted = True
            return (
                f"drawdown: drawdown={drawdown:.2%} >= "
                f"halt_pct={self._drawdown_halt_pct:.2%} "
                f"(peak={peak:.2f} equity={current_equity:.2f})"
            )

        # Recovery: reset latch once equity recovers above threshold
        if self._drawdown_halted and drawdown < self._drawdown_halt_pct:
            self._drawdown_halted = False
            self.logger.info(
                "drawdown_halt_lifted",
                drawdown=round(drawdown, 4),
                peak=round(peak, 2),
                current_equity=round(current_equity, 2),
            )

        return None

    # ------------------------------------------------------------------
    # Layer 8: Total exposure cap
    # ------------------------------------------------------------------

    async def _layer8_exposure_cap(self, signal: SignalForRisk) -> Optional[str]:
        """Reject if adding this position would breach the total exposure cap."""
        total_exposure = await self._call(self.state.get_total_exposure)  # type: ignore[attr-defined]

        # Determine the candidate position size using Kelly (no flags yet;
        # we only need the upper bound for the exposure check).
        candidate_size, _ = self._kelly_size(signal, consecutive_loss_flag=False)

        if total_exposure + candidate_size >= self._max_total_exposure:
            return (
                f"exposure_cap: total_exposure={total_exposure:.2f} + "
                f"candidate_size={candidate_size:.2f} >= "
                f"max_total_exposure={self._max_total_exposure:.2f} "
                f"(absolute_ceiling={ABSOLUTE_MAX_TOTAL_EXPOSURE:.0f})"
            )

        # Also enforce hard ceiling unconditionally
        if total_exposure >= ABSOLUTE_MAX_TOTAL_EXPOSURE:
            return (
                f"exposure_cap: total_exposure={total_exposure:.2f} >= "
                f"absolute_ceiling={ABSOLUTE_MAX_TOTAL_EXPOSURE:.0f}"
            )

        return None

    # ------------------------------------------------------------------
    # Kelly criterion position sizing
    # ------------------------------------------------------------------

    def _kelly_size(
        self,
        signal: SignalForRisk,
        consecutive_loss_flag: bool = False,
    ) -> tuple[float, float]:
        """Compute Kelly position size in USD.

        Kelly formula: f* = edge * (1 + odds) / odds
        Where:
          - edge = signal.edge_cents / 100  (fractional edge)
          - odds = payout_odds for the side being traded

        Adjustments applied in order:
          1. Per-type Kelly boost (from calibration config)
          2. Base Kelly fraction (config, capped at ABSOLUTE_MAX_KELLY_FRACTION)
          3. Consecutive-loss half-sizing (if flag is set)
          4. Hard ceiling: min(result, ABSOLUTE_MAX_POSITION_SIZE)
          5. Config ceiling: min(result, max_position_size)
          6. Forced-size override for arb/mm strategies

        Returns:
            (position_size_usd, kelly_fraction_used)
        """
        # Forced-size override (arb/mm strategies use mathematically derived sizes)
        if signal.forced_size_usd is not None and signal.forced_size_usd > 0.0:
            forced = min(
                signal.forced_size_usd,
                self._max_position_size,
                ABSOLUTE_MAX_POSITION_SIZE,
            )
            return forced, 0.0  # Kelly fraction not applicable

        # Derive price and odds for the traded side
        if signal.side == "BUY_YES":
            price_cents = signal.yes_price_cents
        else:
            price_cents = signal.no_price_cents

        price_cents = max(1, min(99, price_cents))
        p = price_cents / 100.0  # Probability / cost per $1 payout

        # Payout odds: how much do we win per $1 risked?
        # If we pay p per contract and it pays $1 on win: odds = (1/p) - 1
        if p <= 0.0:
            return 0.0, 0.0
        odds = (1.0 / p) - 1.0

        edge = signal.edge_cents / 100.0  # Convert cents to fractional
        if edge <= 0.0 or odds <= 0.0:
            return 0.0, 0.0

        # Raw Kelly fraction (fraction of bankroll)
        raw_kelly_f = edge * (1.0 + odds) / odds

        # Per-type Kelly boost
        mtype_lower = signal.market_type.lower()
        kelly_boost = self._kelly_boosts.get(mtype_lower, _DEFAULT_KELLY_BOOST)

        # Fractional Kelly (third-Kelly by default), capped by hard ceiling
        effective_kelly = min(
            self._kelly_fraction * kelly_boost,
            ABSOLUTE_MAX_KELLY_FRACTION,
        )

        # Apply consecutive-loss half-sizing
        if consecutive_loss_flag:
            effective_kelly *= 0.5

        kelly_f = raw_kelly_f * effective_kelly

        # We need the balance to size in USD
        try:
            if hasattr(self.state, "get_balance"):
                # Attempt synchronous read first; if it's a coroutine we
                # fall back to a stored balance (state_provider should cache it)
                balance = self.state.get_balance()  # type: ignore[attr-defined]
                if hasattr(balance, "__await__"):
                    # Coroutine returned — cannot await here (sync context).
                    # Fall back to a conservative $10 floor so sizing still
                    # works without blocking. In async callers use _kelly_size_async.
                    balance = 10.0
            else:
                balance = 10.0
        except Exception:
            balance = 10.0

        if not isinstance(balance, (int, float)) or math.isnan(balance) or balance <= 0.0:
            balance = 10.0

        position_size = kelly_f * balance

        # Apply position size ceilings
        position_size = min(
            position_size,
            self._max_position_size,
            ABSOLUTE_MAX_POSITION_SIZE,
        )
        position_size = max(0.0, position_size)

        return position_size, effective_kelly

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    async def _call(fn: object) -> object:
        """Call a function that may be sync or async."""
        import asyncio
        import inspect
        result = fn()  # type: ignore[operator]
        if inspect.isawaitable(result):
            return await result  # type: ignore[misc]
        return result
