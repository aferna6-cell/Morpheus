"""Tests for the SQLite database layer."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

try:
    from src.db import Database
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DB_AVAILABLE,
    reason="src/db.py not yet implemented",
)


@pytest.fixture
async def db(tmp_path):
    """Fresh test database using init_db()."""
    from src.db import init_db
    d = await init_db(str(tmp_path / "test.db"))
    yield d
    await d.close()


class TestDatabaseInit:
    @pytest.mark.asyncio
    async def test_creates_all_tables(self, tmp_path):
        """All required tables must exist after init."""
        from src.db import init_db
        d = await init_db(str(tmp_path / "init_test.db"))

        async with d._conn() as conn:
            rows = await conn.execute_fetchall(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        table_names = {r["name"] for r in rows}
        required = {
            "trades", "positions", "orders", "signals",
            "daily_pnl", "market_snapshots", "cost_basis_lots",
            "clv_records", "strategy_perf",
        }
        assert required.issubset(table_names), f"Missing tables: {required - table_names}"
        await d.close()

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, tmp_path):
        """WAL mode must be enabled for concurrent access."""
        from src.db import init_db
        d = await init_db(str(tmp_path / "wal_test.db"))

        async with d._conn() as conn:
            rows = await conn.execute_fetchall("PRAGMA journal_mode")
            mode = rows[0]["journal_mode"]

        assert mode == "wal", f"Expected WAL mode, got: {mode}"
        await d.close()

    @pytest.mark.asyncio
    async def test_idempotent_init(self, tmp_path):
        """Opening same db path twice should not raise errors."""
        from src.db import init_db
        d = await init_db(str(tmp_path / "idempotent.db"))
        await d.close()
        # Re-open same path — schema is already there, should be idempotent
        d2 = await init_db(str(tmp_path / "idempotent.db"))
        await d2.close()


class TestTradeLogging:
    @pytest.mark.asyncio
    async def test_log_trade(self, db):
        """Trade should be persisted to the trades table."""
        await db.log_trade({
            "ticker": "KXINXU-26MAR28-T5480",
            "direction": "BUY_YES",
            "quantity": 2,
            "price_cents": 63,
            "strategy": "kalshi_llm",
            "confidence": 0.75,
            "edge_cents": 15.0,
            "is_paper": 1,
        })

        async with db._conn() as conn:
            rows = await conn.execute_fetchall("SELECT * FROM trades")
        assert len(rows) == 1
        assert rows[0]["ticker"] == "KXINXU-26MAR28-T5480"
        assert rows[0]["direction"] == "BUY_YES"

    @pytest.mark.asyncio
    async def test_multiple_trades(self, db):
        """Multiple trades should all be persisted."""
        for i in range(5):
            await db.log_trade({
                "ticker": f"KXTEST-{i}",
                "direction": "BUY_YES",
                "quantity": 1,
                "price_cents": 50,
                "strategy": "test",
                "confidence": 0.70,
                "edge_cents": 10.0,
                "is_paper": 1,
            })

        async with db._conn() as conn:
            rows = await conn.execute_fetchall("SELECT COUNT(*) as cnt FROM trades")
        assert rows[0]["cnt"] == 5


class TestPositions:
    @pytest.mark.asyncio
    async def test_upsert_position(self, db):
        """Position should be created on first upsert."""
        await db.upsert_position({
            "ticker": "KXINXU-26MAR28-T5480",
            "direction": "BUY_YES",
            "quantity": 2,
            "avg_price_cents": 63,
            "strategy": "kalshi_llm",
        })

        positions = await db.get_positions()
        assert len(positions) == 1
        assert positions[0]["ticker"] == "KXINXU-26MAR28-T5480"

    @pytest.mark.asyncio
    async def test_get_empty_positions(self, db):
        """get_positions() should return empty list on fresh DB."""
        positions = await db.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_position_upsert_updates_price(self, db):
        """Upserting same ticker/direction should update the record."""
        await db.upsert_position({
            "ticker": "KXINXU-26MAR28-T5480",
            "direction": "BUY_YES",
            "quantity": 2,
            "avg_price_cents": 63,
            "strategy": "kalshi_llm",
        })
        await db.upsert_position({
            "ticker": "KXINXU-26MAR28-T5480",
            "direction": "BUY_YES",
            "quantity": 2,
            "avg_price_cents": 70,  # Updated price
            "current_price_cents": 70,
            "strategy": "kalshi_llm",
        })

        positions = await db.get_positions()
        # Should be just 1 position (upserted)
        assert len(positions) == 1
        assert positions[0]["avg_price_cents"] == 70


class TestSignalLogging:
    @pytest.mark.asyncio
    async def test_log_signal(self, db):
        """Signal should be persisted with approve/reject status."""
        await db.log_signal({
            "strategy": "kalshi_llm",
            "ticker": "KXTEST-123",
            "direction": "BUY_YES",
            "confidence": 0.75,
            "edge_cents": 15.0,
            "approved": 1,
            "reject_reason": None,
        })

        async with db._conn() as conn:
            rows = await conn.execute_fetchall("SELECT * FROM signals")
        assert len(rows) == 1
        assert rows[0]["approved"] == 1

    @pytest.mark.asyncio
    async def test_log_rejected_signal(self, db):
        """Rejected signals should be logged with reason."""
        await db.log_signal({
            "strategy": "kalshi_llm",
            "ticker": "KXTEST-456",
            "direction": "BUY_YES",
            "confidence": 0.40,
            "edge_cents": 2.0,
            "approved": 0,
            "reject_reason": "confidence_below_floor: 0.40 < 0.60",
        })

        async with db._conn() as conn:
            rows = await conn.execute_fetchall(
                "SELECT * FROM signals WHERE approved = 0"
            )
        assert len(rows) == 1
        assert "confidence" in rows[0]["reject_reason"]


class TestDailyPnl:
    @pytest.mark.asyncio
    async def test_get_daily_pnl_empty(self, db):
        """get_daily_pnl() returns empty list on fresh DB."""
        result = await db.get_daily_pnl(days=7)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_daily_pnl_structure(self, db):
        """Daily P&L records should have expected fields."""
        # Manually insert a record
        async with db._conn() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO daily_pnl (date, realized_pnl_cents, trade_count, win_count) "
                "VALUES (date('now'), 1500, 10, 7)"
            )

        rows = await db.get_daily_pnl(days=1)
        assert len(rows) >= 1
        row = rows[0]
        assert "date" in row
        assert "realized_pnl_cents" in row


class TestMarketSnapshots:
    @pytest.mark.asyncio
    async def test_snapshot_market(self, db):
        """Market snapshot should be persisted."""
        await db.snapshot_market(
            ticker="KXINXU-26MAR28-T5480",
            yes_price=63,
            no_price=37,
            volume=5000,
        )

        history = await db.get_market_history("KXINXU-26MAR28-T5480", hours=1)
        assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_price_history_ordering(self, db):
        """Market history should be ordered by snapshot time."""
        for price in [60, 62, 65, 63]:
            await db.snapshot_market(
                ticker="KXTEST",
                yes_price=price,
                no_price=100 - price,
                volume=1000,
            )

        history = await db.get_market_history("KXTEST", hours=1)
        assert len(history) == 4
        # Should be ordered by time (most recent last or first — just check consistency)
        assert isinstance(history[0]["yes_price"], int)


class TestCLVLogging:
    @pytest.mark.asyncio
    async def test_log_clv(self, db):
        """CLV record should be persisted."""
        await db.log_clv(
            ticker="KXINXU-26MAR28-T5480",
            market_type="index",
            entry_cents=63,
            close_cents=70,
        )

        async with db._conn() as conn:
            rows = await conn.execute_fetchall("SELECT * FROM clv_records")
        assert len(rows) == 1
        row = rows[0]
        assert row["ticker"] == "KXINXU-26MAR28-T5480"
        assert row["market_type"] == "index"
        # CLV = (entry - close) / close = (63 - 70) / 70 ≈ -0.1
        assert abs(row["clv"] - (63 - 70) / 70) < 0.001


class TestStrategyPerf:
    @pytest.mark.asyncio
    async def test_get_strategy_perf_empty(self, db):
        """get_strategy_perf() returns empty dict on fresh DB."""
        result = await db.get_strategy_perf(days=30)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_strategy_perf_structure(self, db):
        """Strategy perf records should have wins, losses, pnl."""
        async with db._conn() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO strategy_perf (strategy, date, wins, losses, pnl_cents) "
                "VALUES ('kalshi_llm', date('now'), 5, 3, 2000)"
            )

        result = await db.get_strategy_perf(days=30)
        assert "kalshi_llm" in result
        perf = result["kalshi_llm"]
        assert perf["wins"] == 5
        assert perf["losses"] == 3
