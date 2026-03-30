"""Shared test fixtures for Morpheus test suite."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Async event loop
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config():
    """Minimal BotConfig with safe defaults for testing."""
    from src.utils import BotConfig
    return BotConfig(
        strategy={
            "enabled_strategies": ["kalshi_llm", "kalshi_bracket_arb", "kalshi_bonding"],
            "min_edge": 0.04,
            "kelly_fraction": 0.25,
            "max_position_size": 10.0,
            "max_total_exposure": 50.0,
            "min_confidence": 0.60,
        },
        market_filters={
            "min_volume_24h": 1000,
            "min_price": 0.12,
            "max_price": 0.90,
            "max_spread_pct": 0.10,
        },
        risk={
            "max_daily_loss": 10.0,
            "stop_loss_pct": 0.40,
            "take_profit_pct": 0.60,
            "max_position_hold_hours": 24,
            "consecutive_loss_limit": 5,
            "drawdown_halt_pct": 0.15,
        },
        calibration={
            "default_platt_alpha": 0.68,
            "index_platt_alpha": 0.90,
            "weather_platt_alpha": 0.83,
        },
        kalshi={
            "base_url": "https://api.elections.kalshi.com/trade-api/v2",
            "scan_interval_seconds": 600,
        },
        dev={"dry_run": True, "paper_trading_balance": 200.0},
    )


# ---------------------------------------------------------------------------
# Database fixture
# ---------------------------------------------------------------------------

@pytest.fixture
async def test_db(tmp_path):
    """In-memory or temp SQLite database for tests."""
    db_path = tmp_path / "test_morpheus.db"

    try:
        from src.db import Database
        db = Database(str(db_path))
        await db.init()
        yield db
        await db.close()
    except ImportError:
        # db.py not yet implemented — yield a mock
        yield MagicMock()


# ---------------------------------------------------------------------------
# Mock Kalshi market data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_market_index():
    """A sample index (SPY) market dict."""
    return {
        "ticker": "KXINXU-26MAR28-T5480",
        "event_ticker": "KXINXU-26MAR28",
        "title": "Will S&P 500 be above 5480 at close on March 28?",
        "yes_bid": 62,
        "yes_ask": 65,
        "no_bid": 35,
        "no_ask": 38,
        "yes_price": 63,
        "no_price": 37,
        "volume": 5000,
        "open_interest": 1200,
        "close_time": "2026-03-28T21:00:00Z",
        "status": "open",
        "market_type": "index",
    }


@pytest.fixture
def sample_market_weather():
    """A sample weather market dict."""
    return {
        "ticker": "KXHIGHTEMP-NYC-26MAR28-B72",
        "event_ticker": "KXHIGHTEMP-NYC-26MAR28",
        "title": "Will NYC high temperature be above 72°F on March 28?",
        "yes_bid": 30,
        "yes_ask": 34,
        "no_bid": 66,
        "no_ask": 70,
        "yes_price": 32,
        "no_price": 68,
        "volume": 1500,
        "open_interest": 400,
        "close_time": "2026-03-28T23:59:00Z",
        "status": "open",
        "market_type": "weather",
    }


@pytest.fixture
def sample_market_bracket():
    """A sample bracket market where the sum of all bracket YES asks is < 1.00."""
    return {
        "ticker": "KXHIGHTEMP-NYC-26MAR28-B68",
        "event_ticker": "KXHIGHTEMP-NYC-26MAR28",
        "title": "Will NYC high temperature be 68-70°F on March 28?",
        "yes_bid": 18,
        "yes_ask": 21,
        "no_bid": 79,
        "no_ask": 82,
        "yes_price": 20,
        "no_price": 80,
        "volume": 800,
        "open_interest": 200,
        "close_time": "2026-03-28T23:59:00Z",
        "status": "open",
        "market_type": "weather",
        "is_bracket": True,
    }


@pytest.fixture
def sample_market_bonding():
    """A market priced at 95c YES — near-certainty for bonding engine."""
    return {
        "ticker": "KXINXU-26MAR28-T3000",
        "event_ticker": "KXINXU-26MAR28",
        "title": "Will S&P 500 be above 3000 at close on March 28?",
        "yes_bid": 94,
        "yes_ask": 95,
        "no_bid": 5,
        "no_ask": 6,
        "yes_price": 95,
        "no_price": 5,
        "volume": 2000,
        "open_interest": 500,
        "close_time": "2026-03-28T21:00:00Z",
        "status": "open",
        "market_type": "index",
    }


# ---------------------------------------------------------------------------
# Mock Kalshi API client
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_kalshi_client():
    """Mock KalshiClient that returns sample data without hitting the API."""
    client = MagicMock()
    client.get_markets = AsyncMock(return_value=[])
    client.get_market = AsyncMock(return_value=None)
    client.get_positions = AsyncMock(return_value=[])
    client.get_balance = AsyncMock(return_value=100.0)
    return client


@pytest.fixture
def mock_trading_client():
    """Mock KalshiTradingClient that simulates order placement."""
    client = MagicMock()
    client.place_order = AsyncMock(return_value={
        "order_id": "test-order-123",
        "status": "resting",
        "filled_count": 0,
    })
    client.cancel_order = AsyncMock(return_value=True)
    client.get_order = AsyncMock(return_value={
        "order_id": "test-order-123",
        "status": "filled",
        "filled_count": 1,
        "no_price": 35,
    })
    client.get_positions = AsyncMock(return_value=[])
    client.get_balance = AsyncMock(return_value=100.0)
    client._trading_halted = False
    return client


# ---------------------------------------------------------------------------
# Signal fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_trade_signal(sample_market_index):
    """A sample trade signal from the LLM engine."""
    try:
        from src.engines.signals import TradeSignal
        return TradeSignal(
            engine="kalshi_llm",
            market_id=sample_market_index["ticker"],
            token_id="yes_tok",
            side="buy_yes",
            confidence=0.75,
            edge=0.15,
            urgency="normal",
            metadata={
                "market_type": "index",
                "p_yes": 0.78,
                "market_price": 0.63,
            },
        )
    except ImportError:
        return MagicMock(
            engine="kalshi_llm",
            market_id=sample_market_index["ticker"],
            side="buy_yes",
            confidence=0.75,
            edge=0.15,
            urgency="normal",
        )


# ---------------------------------------------------------------------------
# Risk state mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_risk_state():
    """Mock risk state provider for the risk pipeline."""
    state = MagicMock()
    state.get_daily_pnl = AsyncMock(return_value=0.0)
    state.get_total_exposure = AsyncMock(return_value=0.0)
    state.get_balance = AsyncMock(return_value=100.0)
    state.get_recent_losses = AsyncMock(return_value=0)
    state.get_portfolio_peak = AsyncMock(return_value=100.0)
    state.get_current_equity = AsyncMock(return_value=100.0)
    state.get_strategy_errors = AsyncMock(return_value=0)
    return state
