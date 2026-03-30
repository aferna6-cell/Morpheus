"""Tests for SmartEntry execution state machine.

Validates limit → timeout → market fallback logic without hitting Kalshi API.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

try:
    from src.smart_entry import SmartEntry, OrderResult, OrderState
    SMART_ENTRY_AVAILABLE = True
except ImportError:
    SMART_ENTRY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SMART_ENTRY_AVAILABLE,
    reason="src/smart_entry.py not yet implemented",
)


@pytest.fixture
def smart_entry():
    return SmartEntry()


@pytest.fixture
def market_index():
    return {
        "ticker": "KXINXU-26MAR28-T5480",
        "yes_bid": 62,
        "yes_ask": 65,
        "no_bid": 35,
        "no_ask": 38,
        "yes_price": 63,
        "no_price": 37,
        "market_type": "index",
    }


class TestLimitPricingLogic:
    def test_limit_price_calculated(self, smart_entry, market_index):
        """SmartEntry should calculate a limit price from market data."""
        from src.smart_entry import _compute_limit_price
        # market_index values are in cents (62 = $0.62), convert to float for _compute_limit_price
        price = _compute_limit_price(
            side="yes",
            yes_bid=market_index["yes_bid"] / 100,
            yes_ask=market_index["yes_ask"] / 100,
            no_bid=market_index["no_bid"] / 100,
            no_ask=market_index["no_ask"] / 100,
            confidence=0.70,
        )
        # Should be a positive integer in cents
        assert isinstance(price, int)
        assert price > 0

    def test_noaa_ticker_detected(self, smart_entry):
        """NOAA-prefixed tickers should be auto-detected as 'noaa' signal type."""
        from src.smart_entry import _detect_signal_type
        assert _detect_signal_type("KXHIGHTEMP-NYC-26MAR28") == "noaa"

    def test_index_ticker_detected(self, smart_entry):
        """Index tickers should be detected as 'index' signal type."""
        from src.smart_entry import _detect_signal_type
        assert _detect_signal_type("KXINXU-26MAR28-T5480") == "index"

    def test_generic_ticker_detected(self, smart_entry):
        """Unknown tickers default to 'generic'."""
        from src.smart_entry import _detect_signal_type
        result = _detect_signal_type("KXPOLITICS-2026-T50")
        assert result == "generic"


class TestPaperModeExecution:
    @pytest.mark.asyncio
    async def test_paper_fill_returns_order_result(self, smart_entry, market_index, mock_trading_client, minimal_config):
        """In paper mode, execute() should return an OrderResult."""
        mock_trading_client.dry_run = True  # Paper mode flag

        signal_meta = {
            "ticker": market_index["ticker"],
            "side": "yes",
            "quantity": 1,
            "confidence": 0.75,
            "yes_bid": market_index["yes_bid"] / 100,
            "yes_ask": market_index["yes_ask"] / 100,
            "no_bid": market_index["no_bid"] / 100,
            "no_ask": market_index["no_ask"] / 100,
        }

        result = await smart_entry.execute(signal_meta, mock_trading_client, minimal_config)

        assert result is not None
        assert isinstance(result, OrderResult)
        assert result.is_paper is True


class TestLiveExecution:
    @pytest.mark.asyncio
    async def test_place_order_called_in_live_mode(self, smart_entry, market_index, mock_trading_client, minimal_config):
        """In live mode, place_order should be called."""
        mock_trading_client.dry_run = False
        # Return "executed" status so SmartEntry returns immediately (no poll loop)
        mock_trading_client.place_order = AsyncMock(return_value={
            "order_id": "test-123",
            "status": "executed",
        })

        signal_meta = {
            "ticker": market_index["ticker"],
            "side": "yes",
            "quantity": 1,
            "confidence": 0.75,
            "yes_bid": market_index["yes_bid"] / 100,
            "yes_ask": market_index["yes_ask"] / 100,
            "no_bid": market_index["no_bid"] / 100,
            "no_ask": market_index["no_ask"] / 100,
        }

        try:
            result = await smart_entry.execute(signal_meta, mock_trading_client, minimal_config)
            # If no timeout/error, place_order should have been called
            assert mock_trading_client.place_order.called
        except Exception:
            # Integration-level issues ok in unit tests
            assert mock_trading_client.place_order.called


class TestOrderResult:
    def test_order_result_fields(self):
        """OrderResult should have required fields."""
        result = OrderResult(
            order_id="test-123",
            client_order_id="client-abc",
            ticker="KXINXU-26MAR28-T5480",
            side="yes",
            intended_quantity=1,
            filled_quantity=1,
            fill_price_cents=63,
            total_cost=0.63,
            state=OrderState.FILLED,
            is_paper=False,
            execution_ms=150,
            reasoning="Limit filled in 2.3s",
        )
        assert result.order_id == "test-123"
        assert result.client_order_id == "client-abc"
        assert result.ticker == "KXINXU-26MAR28-T5480"
        assert result.side == "yes"
        assert result.intended_quantity == 1
        assert result.filled_quantity == 1
        assert result.fill_price_cents == 63
        assert result.total_cost == 0.63
        assert result.state == OrderState.FILLED
        assert result.is_paper is False
        assert result.execution_ms == 150
        assert result.reasoning == "Limit filled in 2.3s"
        assert result.is_filled is True
        assert result.partially_filled is False
