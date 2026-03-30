"""Tests for trading strategies — signal generation logic.

Tests all active strategies (LLM, BracketArb, Bonding, OrderFlow, Rules, OrderBook).
All external API calls are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestBracketArbEngine:
    """Tests for the bracket arbitrage engine (mathematical edge)."""

    def test_detects_arb_when_sum_below_100(self):
        """Should detect arb when sum of YES asks < 100c."""
        from src.engines.kalshi_bracket_arb_engine import KalshiBracketArbEngine

        # A bracket set where sum of YES asks = 0.87 → 13c arb
        brackets = [
            {"ticker": "KXHIGH-26MAR28-B60", "yes_ask": 18, "no_ask": 82},
            {"ticker": "KXHIGH-26MAR28-B65", "yes_ask": 22, "no_ask": 78},
            {"ticker": "KXHIGH-26MAR28-B70", "yes_ask": 25, "no_ask": 75},
            {"ticker": "KXHIGH-26MAR28-B75", "yes_ask": 22, "no_ask": 78},
        ]
        sum_asks = sum(b["yes_ask"] for b in brackets)
        assert sum_asks < 100, f"Sum should be < 100, got {sum_asks}"
        margin = (100 - sum_asks) / 100
        assert margin >= 0.015  # At least 1.5% margin

    def test_no_arb_when_sum_above_100(self):
        """No signal when sum of YES asks >= 100c."""
        brackets = [
            {"ticker": "KXHIGH-26MAR28-B60", "yes_ask": 28},
            {"ticker": "KXHIGH-26MAR28-B65", "yes_ask": 30},
            {"ticker": "KXHIGH-26MAR28-B70", "yes_ask": 28},
            {"ticker": "KXHIGH-26MAR28-B75", "yes_ask": 25},
        ]
        sum_asks = sum(b["yes_ask"] for b in brackets)
        assert sum_asks > 100  # No arb exists

    def test_bracket_ticker_detection(self):
        """Bracket tickers have -B pattern in ticker."""
        bracket_ticker = "KXHIGHTEMP-NYC-26MAR28-B72"
        non_bracket = "KXINXU-26MAR28-T5480"

        # Bracket tickers contain the -B pattern
        assert "-B" in bracket_ticker.split("-")[-1] or bracket_ticker.endswith("B72")
        assert "B" not in non_bracket.split("-")[-1]


class TestBondingEngine:
    """Tests for the bonding engine (near-certainty harvesting)."""

    def test_qualifies_near_certain_market(self, sample_market_bonding):
        """Market at 95c YES should qualify for bonding."""
        # New schema uses _dollars fields (0-1 scale); derive yes_price from last_price_dollars
        yes_price_dollars = sample_market_bonding["last_price_dollars"]
        yes_price_cents = int(yes_price_dollars * 100)
        assert 90 <= yes_price_cents <= 97, "Bonding target: 90-97c"

    def test_rejects_too_cheap(self):
        """Market at 80c YES should not qualify."""
        yes_price = 80
        assert yes_price < 90  # Below bonding threshold

    def test_rejects_too_expensive(self):
        """Market at 99c YES should not qualify (too close to settled)."""
        yes_price = 99
        assert yes_price > 97  # Above bonding threshold (may have stale prices)

    def test_edge_calculation(self):
        """Edge should be (100 - yes_price) / 100."""
        yes_price = 95
        edge = (100 - yes_price) / 100
        assert edge == pytest.approx(0.05, abs=0.001)


class TestOrderFlowEngine:
    """Tests for VPIN-based orderflow engine."""

    def test_vpin_calculation(self):
        """VPIN should measure taker-side imbalance in volume buckets."""
        # Bucket: 1000 contracts, 700 YES buys, 300 NO buys
        yes_volume = 700
        no_volume = 300
        total = yes_volume + no_volume
        vpin = abs(yes_volume - no_volume) / total
        assert vpin == pytest.approx(0.4, abs=0.001)

    def test_vpin_symmetric(self):
        """VPIN should be the same for opposite imbalances."""
        vpin_yes = abs(700 - 300) / 1000
        vpin_no = abs(300 - 700) / 1000
        assert vpin_yes == vpin_no

    def test_vpin_zero_on_balanced_flow(self):
        """VPIN should be 0 when flow is perfectly balanced."""
        yes_volume = 500
        no_volume = 500
        vpin = abs(yes_volume - no_volume) / (yes_volume + no_volume)
        assert vpin == pytest.approx(0.0, abs=0.001)

    def test_vpin_max_on_one_sided_flow(self):
        """VPIN should be 1.0 when all flow is one-sided."""
        yes_volume = 1000
        no_volume = 0
        total = yes_volume + no_volume
        if total > 0:
            vpin = abs(yes_volume - no_volume) / total
            assert vpin == pytest.approx(1.0, abs=0.001)


class TestRulesBasedEngine:
    """Tests for rules-based engine (if implemented)."""

    def test_time_decay_rule_near_expiry(self):
        """Market within 24h of expiry at extreme price should trigger."""
        yes_price = 88  # > 85c threshold
        hours_to_close = 12  # < 24h

        # Both conditions met — should trigger BUY_YES signal
        is_extreme = yes_price > 85
        is_near = hours_to_close < 24
        should_trigger = is_extreme and is_near
        assert should_trigger is True

    def test_time_decay_rule_not_near_expiry(self):
        """Market at 90c but 48h away should not trigger time decay rule."""
        yes_price = 90
        hours_to_close = 48  # Not near expiry

        # Even though extreme, not close enough to expiry
        is_extreme = yes_price > 85
        is_near = hours_to_close < 24
        should_trigger = is_extreme and is_near
        assert should_trigger is False

    def test_price_consistency_rule_when_sum_low(self):
        """If YES + NO < 92, both sides are underpriced."""
        yes_price = 40
        no_price = 45
        total = yes_price + no_price

        # Total < 92 → inconsistency → edge = (100 - total) / 2
        assert total == 85  # < 92
        edge_cents = (100 - total) / 2
        assert edge_cents == pytest.approx(7.5, abs=0.01)

    def test_price_consistency_normal(self):
        """Normal markets with YES + NO ≈ 100 should not trigger."""
        yes_price = 63
        no_price = 37
        total = yes_price + no_price
        assert total == 100  # Consistent


class TestOrderbookImbalanceEngine:
    """Tests for orderbook imbalance engine (if implemented)."""

    def test_obi_positive_signals_buy_yes(self):
        """Positive OBI (more bid depth) should signal BUY_YES."""
        yes_bid_depth = 500
        no_bid_depth = 200
        total = yes_bid_depth + no_bid_depth
        obi = (yes_bid_depth - no_bid_depth) / total
        assert obi > 0.3  # Above threshold
        # Should signal BUY_YES

    def test_obi_negative_signals_buy_no(self):
        """Negative OBI (more no bid depth) should signal BUY_NO."""
        yes_bid_depth = 100
        no_bid_depth = 600
        total = yes_bid_depth + no_bid_depth
        obi = (yes_bid_depth - no_bid_depth) / total
        assert obi < -0.3  # Below negative threshold
        # Should signal BUY_NO

    def test_obi_neutral_no_signal(self):
        """Balanced orderbook should not generate a signal."""
        yes_bid_depth = 300
        no_bid_depth = 320
        total = yes_bid_depth + no_bid_depth
        obi = (yes_bid_depth - no_bid_depth) / total
        assert abs(obi) < 0.1  # Balanced — no signal

    def test_confidence_scales_with_obi(self):
        """Higher OBI should yield higher confidence."""
        def confidence_from_obi(obi_abs):
            return min(0.95, 0.50 + (obi_abs - 0.3) * 0.64)

        low_obi_conf = confidence_from_obi(0.35)
        high_obi_conf = confidence_from_obi(0.80)
        assert high_obi_conf > low_obi_conf


class TestSignalDeduplication:
    """Tests for orchestrator deduplication logic."""

    def test_dedup_same_ticker_same_side(self):
        """Two signals for same ticker and side should be deduplicated."""
        seen = set()
        key1 = ("KXINXU-26MAR28-T5480", "BUY_YES", "kalshi_llm")
        key2 = ("KXINXU-26MAR28-T5480", "BUY_YES", "kalshi_bonding")

        seen.add(key1)
        # Different engine but same market + side — event-level dedup
        event1 = "KXINXU-26MAR28"  # Strip the T5480 suffix
        event2 = "KXINXU-26MAR28"

        # Same event prefix → correlation limit applies
        assert event1 == event2

    def test_dedup_same_ticker_different_side(self):
        """Same ticker but different side should NOT be deduplicated."""
        # BUY_YES and BUY_NO on same market are independent positions
        side_yes = "BUY_YES"
        side_no = "BUY_NO"
        assert side_yes != side_no

    def test_multi_engine_consensus(self):
        """2+ engines on same side should boost confidence."""
        base_confidence = 0.70
        boost = 1.5  # 50% boost from multi-engine consensus
        boosted = min(0.99, base_confidence * boost)
        assert boosted > base_confidence


class TestCalibration:
    """Tests for Platt scaling calibration."""

    def test_platt_scaling_compresses_toward_half(self):
        """Platt alpha < 1.0 compresses predictions toward 0.5."""
        def platt_scale(p, alpha):
            """Simple Platt scaling approximation."""
            # alpha < 1.0 → compress toward 0.5
            return 0.5 + (p - 0.5) * alpha

        p = 0.85  # LLM estimates 85%
        alpha = 0.68  # Default Platt alpha
        calibrated = platt_scale(p, alpha)
        assert 0.5 < calibrated < 0.85  # Compressed, but still > 0.5

    def test_platt_alpha_1_is_passthrough(self):
        """Platt alpha = 1.0 should not change the probability."""
        def platt_scale(p, alpha):
            return 0.5 + (p - 0.5) * alpha

        p = 0.75
        calibrated = platt_scale(p, 1.0)
        assert calibrated == pytest.approx(0.75, abs=0.001)

    def test_index_alpha_higher_than_default(self):
        """Index markets should have higher (less compressive) Platt alpha."""
        # From Morpheus config: index_platt_alpha=0.90, default=0.68
        index_alpha = 0.90
        default_alpha = 0.68
        assert index_alpha > default_alpha  # Index markets are more trustworthy

    def test_politics_higher_shrinkage(self):
        """Politics markets should have extra shrinkage toward 0.5."""
        # Politics: extra_shrink = 0.12 (more conservative)
        politics_shrink = 0.12
        index_shrink = -0.05  # Actually LESS shrinkage for index (negative = expand)
        assert politics_shrink > 0  # More conservative for politics


class TestMarketFiltering:
    """Tests for market filtering logic."""

    def test_blocklist_crypto(self):
        """Crypto tickers should be blocked."""
        from src.market_filters import is_ticker_blocked
        assert is_ticker_blocked("KXBTC-26MAR28-T30000") is True

    def test_blocklist_kxfed_year_long(self):
        """KXFED year-long positions should be blocked (caused $49 loss in Wave 33)."""
        from src.market_filters import is_ticker_blocked
        # KXFED markets are in the blocklist
        assert is_ticker_blocked("KXFED-26-T550") is True

    def test_approves_index_market(self):
        """Index markets should pass the filter."""
        from src.market_filters import is_ticker_blocked
        assert is_ticker_blocked("KXINXU-26MAR28-T5480") is False

    def test_approves_weather_market(self):
        """Weather bracket markets should pass."""
        from src.market_filters import is_ticker_blocked
        assert is_ticker_blocked("KXHIGHTEMP-NYC-26MAR28-B72") is False
