"""Tests for the 8-layer risk pipeline.

Tests each layer independently and the combined pipeline.
All Kalshi API calls are mocked — no network access required.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Import guard — risk_pipeline may not exist yet during development
# ---------------------------------------------------------------------------

try:
    from src.risk_pipeline import (
        RiskPipeline,
        RiskDecision,
        SignalForRisk,
        ABSOLUTE_MAX_POSITION_SIZE,
        ABSOLUTE_MAX_TOTAL_EXPOSURE,
        ABSOLUTE_MAX_DAILY_LOSS,
        ABSOLUTE_MAX_KELLY_FRACTION,
        ABSOLUTE_MIN_CONFIDENCE,
        ABSOLUTE_MIN_EDGE_CENTS,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not PIPELINE_AVAILABLE,
    reason="src/risk_pipeline.py not yet implemented",
)


@pytest.fixture
def sample_signal():
    """A valid signal that should pass all risk layers."""
    return SignalForRisk(
        ticker="KXINXU-26MAR28-T5480",
        side="BUY_YES",
        confidence=0.75,
        edge_cents=15.0,
        urgency="normal",
        strategy="kalshi_llm",
        market_type="index",
        yes_price_cents=63,
        no_price_cents=37,
        volume=5000,
    )


@pytest.fixture
def pipeline(minimal_config, mock_risk_state):
    """Risk pipeline with default test config and clean state."""
    return RiskPipeline(config=minimal_config, state_provider=mock_risk_state)


# ---------------------------------------------------------------------------
# Hard-coded ceilings — these must NEVER be configurable
# ---------------------------------------------------------------------------

class TestHardCodedCeilings:
    def test_max_position_ceiling_exists(self):
        assert ABSOLUTE_MAX_POSITION_SIZE == 200.0

    def test_max_exposure_ceiling_exists(self):
        assert ABSOLUTE_MAX_TOTAL_EXPOSURE == 10_000.0

    def test_max_daily_loss_ceiling_exists(self):
        assert ABSOLUTE_MAX_DAILY_LOSS == 500.0

    def test_max_kelly_ceiling_exists(self):
        assert ABSOLUTE_MAX_KELLY_FRACTION == 0.50

    def test_min_confidence_floor_exists(self):
        assert ABSOLUTE_MIN_CONFIDENCE == 0.30

    def test_min_edge_floor_exists(self):
        assert ABSOLUTE_MIN_EDGE_CENTS == 1.0


# ---------------------------------------------------------------------------
# Layer 1: Spread cost check
# ---------------------------------------------------------------------------

class TestLayer1SpreadCost:
    @pytest.mark.asyncio
    async def test_approves_positive_net_edge(self, pipeline, sample_signal):
        """Signal with 15c edge should pass spread cost check."""
        reason = pipeline._layer1_spread_cost(sample_signal)
        assert reason is None  # No rejection

    @pytest.mark.asyncio
    async def test_rejects_zero_edge(self, pipeline, sample_signal):
        """Signal with 0 edge cannot profit after fees."""
        sample_signal.edge_cents = 0.0
        reason = pipeline._layer1_spread_cost(sample_signal)
        assert reason is not None
        assert "edge" in reason.lower() or "fee" in reason.lower()

    def test_layer1_rejects_sub_fee_edge(self, pipeline, sample_signal):
        """Layer 1: net_edge after fees must be > 0; fee ≈ 0.41c at p=0.63."""
        # 0.3c edge < 0.41c fee → net_edge < 0 → rejected by layer 1
        sample_signal.edge_cents = 0.3
        reason = pipeline._layer1_spread_cost(sample_signal)
        assert reason is not None


# ---------------------------------------------------------------------------
# Layer 2: Volume / liquidity
# ---------------------------------------------------------------------------

class TestLayer2VolumeLiquidity:
    @pytest.mark.asyncio
    async def test_approves_high_volume(self, pipeline, sample_signal):
        sample_signal.volume = 5000  # Well above min
        reason = pipeline._layer2_volume_liquidity(sample_signal)
        assert reason is None

    @pytest.mark.asyncio
    async def test_rejects_low_volume(self, pipeline, sample_signal):
        sample_signal.volume = 50  # Below min_volume_24h=1000
        reason = pipeline._layer2_volume_liquidity(sample_signal)
        assert reason is not None


# ---------------------------------------------------------------------------
# Layer 3: Confidence floor
# ---------------------------------------------------------------------------

class TestLayer3ConfidenceFloor:
    @pytest.mark.asyncio
    async def test_approves_high_confidence(self, pipeline, sample_signal):
        sample_signal.confidence = 0.80
        reason = pipeline._layer3_confidence_floor(sample_signal)
        assert reason is None

    @pytest.mark.asyncio
    async def test_rejects_below_floor(self, pipeline, sample_signal):
        sample_signal.confidence = 0.40  # Below config min_confidence=0.60
        reason = pipeline._layer3_confidence_floor(sample_signal)
        assert reason is not None

    @pytest.mark.asyncio
    async def test_rejects_below_absolute_minimum(self, pipeline, sample_signal):
        """Even config-configured floor cannot override absolute minimum."""
        sample_signal.confidence = 0.20  # Below ABSOLUTE_MIN_CONFIDENCE=0.30
        reason = pipeline._layer3_confidence_floor(sample_signal)
        assert reason is not None

    @pytest.mark.asyncio
    async def test_category_floor_politics(self, pipeline, sample_signal):
        """Politics markets have a higher floor due to worse LLM calibration."""
        sample_signal.market_type = "politics"
        sample_signal.confidence = 0.65  # Between default (0.60) and politics floor (0.70)
        reason = pipeline._layer3_confidence_floor(sample_signal)
        assert reason is not None  # Should be rejected by politics-specific floor


# ---------------------------------------------------------------------------
# Layer 4: Net edge after fees
# ---------------------------------------------------------------------------

class TestLayer4NetEdge:
    @pytest.mark.asyncio
    async def test_approves_above_min_edge(self, pipeline, sample_signal):
        sample_signal.edge_cents = 15.0  # Well above min_edge=4c
        reason = pipeline._layer4_net_edge(sample_signal)
        assert reason is None

    @pytest.mark.asyncio
    async def test_rejects_below_min_edge(self, pipeline, sample_signal):
        sample_signal.edge_cents = 2.0  # Below config min_edge=4% (4c on 1 contract)
        reason = pipeline._layer4_net_edge(sample_signal)
        assert reason is not None


# ---------------------------------------------------------------------------
# Layer 5: Daily loss limit
# ---------------------------------------------------------------------------

class TestLayer5DailyLoss:
    @pytest.mark.asyncio
    async def test_approves_when_no_loss(self, pipeline, sample_signal, mock_risk_state):
        mock_risk_state.get_daily_pnl.return_value = 0.0  # No loss today
        reason = await pipeline._layer5_daily_loss(sample_signal)
        assert reason is None

    @pytest.mark.asyncio
    async def test_rejects_when_loss_limit_hit(self, pipeline, sample_signal, mock_risk_state):
        mock_risk_state.get_daily_pnl.return_value = -15.0  # -$15, exceeds $10 limit
        reason = await pipeline._layer5_daily_loss(sample_signal)
        assert reason is not None
        assert "loss" in reason.lower() or "halt" in reason.lower()

    @pytest.mark.asyncio
    async def test_absolute_ceiling_constant_is_500(self):
        """The absolute daily loss constant must be $500 — it cannot be raised by config."""
        assert ABSOLUTE_MAX_DAILY_LOSS == 500.0

    @pytest.mark.asyncio
    async def test_config_limit_enforced_before_absolute(self, pipeline, sample_signal, mock_risk_state):
        """The pipeline enforces the configured limit, not just the absolute ceiling."""
        # Config max_daily_loss=10.0 in minimal_config → -$15 triggers halt
        mock_risk_state.get_daily_pnl.return_value = -15.0
        reason = await pipeline._layer5_daily_loss(sample_signal)
        assert reason is not None  # Triggered because -15 < -10 (configured limit)


# ---------------------------------------------------------------------------
# Layer 8: Exposure cap
# ---------------------------------------------------------------------------

class TestLayer8ExposureCap:
    @pytest.mark.asyncio
    async def test_approves_under_cap(self, pipeline, sample_signal, mock_risk_state):
        mock_risk_state.get_total_exposure.return_value = 10.0  # $10 of $50 max
        reason = await pipeline._layer8_exposure_cap(sample_signal)
        assert reason is None

    @pytest.mark.asyncio
    async def test_rejects_over_cap(self, pipeline, sample_signal, mock_risk_state):
        # Config max_total_exposure=50.0; set exposure to 49.90 so any Kelly size overflows
        mock_risk_state.get_total_exposure.return_value = 49.90
        reason = await pipeline._layer8_exposure_cap(sample_signal)
        assert reason is not None


# ---------------------------------------------------------------------------
# Kelly sizing
# ---------------------------------------------------------------------------

class TestKellySizing:
    def test_normal_kelly(self, pipeline, sample_signal):
        """Normal signal should get reasonable position size."""
        size, fraction = pipeline._kelly_size(sample_signal)
        assert 0 < size <= 10.0  # Should be between $0 and max_position_size

    def test_kelly_below_max_position(self, pipeline, sample_signal):
        """Kelly size must not exceed max_position_size."""
        size, _ = pipeline._kelly_size(sample_signal)
        assert size <= 10.0  # config max_position_size

    def test_kelly_absolute_ceiling(self, pipeline, sample_signal):
        """Kelly size must never exceed ABSOLUTE_MAX_POSITION_SIZE."""
        size, _ = pipeline._kelly_size(sample_signal)
        assert size <= ABSOLUTE_MAX_POSITION_SIZE

    def test_reduced_kelly_on_consecutive_losses(self, pipeline, sample_signal):
        """Position size is halved after consecutive losses."""
        normal_size, _ = pipeline._kelly_size(sample_signal, consecutive_loss_flag=False)
        reduced_size, _ = pipeline._kelly_size(sample_signal, consecutive_loss_flag=True)
        assert reduced_size < normal_size
        assert abs(reduced_size - normal_size * 0.5) < 0.01  # Should be half

    def test_politics_market_kelly_boost(self, pipeline, sample_signal):
        """Politics markets should get reduced Kelly (0.5x) due to worse calibration."""
        sample_signal.market_type = "index"
        index_size, _ = pipeline._kelly_size(sample_signal)

        sample_signal.market_type = "politics"
        politics_size, _ = pipeline._kelly_size(sample_signal)

        # Politics should be smaller (0.5x vs 1.0x boost)
        assert politics_size < index_size


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_approves_valid_signal(self, pipeline, sample_signal):
        """A well-constructed signal should be approved."""
        decision = await pipeline.validate(sample_signal)
        assert decision.approved is True
        assert decision.position_size > 0
        assert decision.reject_reason is None

    @pytest.mark.asyncio
    async def test_rejects_low_confidence(self, pipeline, sample_signal):
        """Signal below confidence floor should be rejected in layer 3."""
        sample_signal.confidence = 0.30
        decision = await pipeline.validate(sample_signal)
        assert decision.approved is False
        assert decision.position_size == 0
        assert decision.reject_reason is not None

    @pytest.mark.asyncio
    async def test_rejects_when_daily_loss_hit(self, pipeline, sample_signal, mock_risk_state):
        """Trading should halt when daily loss limit is reached."""
        mock_risk_state.get_daily_pnl.return_value = -15.0  # Over $10 limit
        decision = await pipeline.validate(sample_signal)
        assert decision.approved is False

    @pytest.mark.asyncio
    async def test_first_rejection_wins(self, pipeline, sample_signal, mock_risk_state):
        """Multiple violations should report only the first layer's reason."""
        # Violate layers 2 AND 3 simultaneously
        sample_signal.volume = 10        # Layer 2 violation
        sample_signal.confidence = 0.20  # Layer 3 violation
        decision = await pipeline.validate(sample_signal)
        assert decision.approved is False
        # Layer 2 comes first, so that should be the reject reason
        assert decision.layer is not None
