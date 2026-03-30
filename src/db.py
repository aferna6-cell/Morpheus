"""SQLite WAL database layer for Morpheus trading bot.

Provides async SQLite access via aiosqlite with full schema for trades,
positions, orders, signals, P&L, cost basis, CLV, and strategy performance.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog

try:
    import aiosqlite
except ImportError as e:
    raise ImportError(
        "aiosqlite is required for the database layer. "
        "Install it with: pip install aiosqlite"
    ) from e

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# DDL — all tables in a single string for clarity
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS trades (
    id               INTEGER PRIMARY KEY,
    timestamp        TEXT    NOT NULL,
    ticker           TEXT    NOT NULL,
    direction        TEXT    NOT NULL,
    quantity         INTEGER NOT NULL,
    price_cents      INTEGER NOT NULL,
    fill_price_cents INTEGER,
    strategy         TEXT    NOT NULL,
    confidence       REAL,
    edge_cents       REAL,
    pnl_cents        REAL,
    is_paper         BOOLEAN DEFAULT 0,
    order_id         TEXT,
    metadata         TEXT,
    UNIQUE(order_id)
);

CREATE TABLE IF NOT EXISTS positions (
    id                   INTEGER PRIMARY KEY,
    ticker               TEXT    NOT NULL,
    direction            TEXT    NOT NULL,
    quantity             INTEGER NOT NULL,
    avg_price_cents      INTEGER NOT NULL,
    current_price_cents  INTEGER,
    unrealized_pnl_cents INTEGER,
    opened_at            TEXT    NOT NULL,
    strategy             TEXT    NOT NULL,
    stop_loss_pct        REAL    DEFAULT 0.40,
    take_profit_pct      REAL    DEFAULT 0.60,
    UNIQUE(ticker, direction)
);

CREATE TABLE IF NOT EXISTS orders (
    id                  TEXT    PRIMARY KEY,
    ticker              TEXT    NOT NULL,
    side                TEXT    NOT NULL,
    quantity            INTEGER NOT NULL,
    price_cents         INTEGER NOT NULL,
    order_type          TEXT    NOT NULL,
    status              TEXT    NOT NULL,
    placed_at           TEXT    NOT NULL,
    timeout_at          TEXT,
    filled_quantity     INTEGER DEFAULT 0,
    fill_price_cents    INTEGER,
    exchange_order_id   TEXT,
    is_smart_entry      BOOLEAN DEFAULT 1
);

CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY,
    timestamp     TEXT    NOT NULL,
    strategy      TEXT    NOT NULL,
    ticker        TEXT    NOT NULL,
    direction     TEXT    NOT NULL,
    confidence    REAL,
    edge_cents    REAL,
    approved      BOOLEAN,
    reject_reason TEXT,
    metadata      TEXT
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    date                TEXT    PRIMARY KEY,
    realized_pnl_cents  INTEGER DEFAULT 0,
    unrealized_pnl_cents INTEGER DEFAULT 0,
    trade_count         INTEGER DEFAULT 0,
    win_count           INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id          INTEGER PRIMARY KEY,
    ticker      TEXT    NOT NULL,
    yes_price   INTEGER,
    no_price    INTEGER,
    volume      INTEGER,
    snapshot_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_market_snapshots_ticker_time
    ON market_snapshots(ticker, snapshot_at);

CREATE TABLE IF NOT EXISTS cost_basis_lots (
    id                   INTEGER PRIMARY KEY,
    ticker               TEXT    NOT NULL,
    quantity             INTEGER NOT NULL,
    purchase_price_cents INTEGER NOT NULL,
    purchase_date        TEXT    NOT NULL,
    disposed             BOOLEAN DEFAULT 0,
    disposal_date        TEXT,
    sale_price_cents     INTEGER,
    realized_gain_cents  INTEGER
);

CREATE TABLE IF NOT EXISTS clv_records (
    id                INTEGER PRIMARY KEY,
    ticker            TEXT    NOT NULL,
    market_type       TEXT,
    entry_price_cents INTEGER,
    close_price_cents INTEGER,
    clv               REAL,
    timestamp         TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_perf (
    strategy  TEXT    NOT NULL,
    date      TEXT    NOT NULL,
    wins      INTEGER DEFAULT 0,
    losses    INTEGER DEFAULT 0,
    pnl_cents INTEGER DEFAULT 0,
    PRIMARY KEY (strategy, date)
);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class _ConnectionContext:
    """Trivial async context manager wrapping an aiosqlite.Connection.

    Allows callers to write ``async with db._conn() as conn:`` without
    entering/exiting the connection itself (it stays open for the lifetime
    of the Database object).
    """

    def __init__(self, conn: "aiosqlite.Connection") -> None:
        self._conn = conn

    async def __aenter__(self) -> "aiosqlite.Connection":
        return self._conn

    async def __aexit__(self, *_: object) -> None:
        pass  # Do not close; the Database owns the connection lifetime.


class Database:
    """Async SQLite database wrapper.

    Usage::

        db = await init_db("state/morpheus.db")
        await db.log_trade({...})
        await db.close()

    The ``connection`` attribute is an ``aiosqlite.Connection`` that can also
    be used directly for advanced queries.
    """

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self.connection = connection
        # Enable row factory so fetchall returns dict-like rows.
        self.connection.row_factory = aiosqlite.Row

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    async def log_trade(self, trade_data: dict) -> None:
        """Upsert a trade record.

        Performs INSERT OR REPLACE so that re-logging the same order_id is
        idempotent (useful on restart when fills are re-detected).
        """
        trade_data.setdefault("timestamp", _utc_now_iso())
        trade_data.setdefault("is_paper", 0)

        metadata = trade_data.get("metadata")
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)

        await self.connection.execute(
            """
            INSERT INTO trades
                (timestamp, ticker, direction, quantity, price_cents,
                 fill_price_cents, strategy, confidence, edge_cents,
                 pnl_cents, is_paper, order_id, metadata)
            VALUES
                (:timestamp, :ticker, :direction, :quantity, :price_cents,
                 :fill_price_cents, :strategy, :confidence, :edge_cents,
                 :pnl_cents, :is_paper, :order_id, :metadata)
            ON CONFLICT(order_id) DO UPDATE SET
                fill_price_cents = excluded.fill_price_cents,
                pnl_cents        = excluded.pnl_cents,
                metadata         = excluded.metadata
            """,
            {
                "timestamp":        trade_data.get("timestamp"),
                "ticker":           trade_data["ticker"],
                "direction":        trade_data["direction"],
                "quantity":         trade_data["quantity"],
                "price_cents":      trade_data["price_cents"],
                "fill_price_cents": trade_data.get("fill_price_cents"),
                "strategy":         trade_data.get("strategy", "unknown"),
                "confidence":       trade_data.get("confidence"),
                "edge_cents":       trade_data.get("edge_cents"),
                "pnl_cents":        trade_data.get("pnl_cents"),
                "is_paper":         trade_data.get("is_paper", 0),
                "order_id":         trade_data.get("order_id"),
                "metadata":         metadata,
            },
        )
        await self.connection.commit()

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def upsert_position(self, position_data: dict) -> None:
        """Insert or update a position.

        Uniqueness is (ticker, direction).  If the position already exists,
        quantity, prices, and unrealized P&L are updated.
        """
        position_data.setdefault("opened_at", _utc_now_iso())
        position_data.setdefault("stop_loss_pct", 0.40)
        position_data.setdefault("take_profit_pct", 0.60)

        await self.connection.execute(
            """
            INSERT INTO positions
                (ticker, direction, quantity, avg_price_cents,
                 current_price_cents, unrealized_pnl_cents, opened_at,
                 strategy, stop_loss_pct, take_profit_pct)
            VALUES
                (:ticker, :direction, :quantity, :avg_price_cents,
                 :current_price_cents, :unrealized_pnl_cents, :opened_at,
                 :strategy, :stop_loss_pct, :take_profit_pct)
            ON CONFLICT(ticker, direction) DO UPDATE SET
                quantity             = excluded.quantity,
                avg_price_cents      = excluded.avg_price_cents,
                current_price_cents  = excluded.current_price_cents,
                unrealized_pnl_cents = excluded.unrealized_pnl_cents,
                stop_loss_pct        = excluded.stop_loss_pct,
                take_profit_pct      = excluded.take_profit_pct
            """,
            {
                "ticker":               position_data["ticker"],
                "direction":            position_data["direction"],
                "quantity":             position_data["quantity"],
                "avg_price_cents":      position_data["avg_price_cents"],
                "current_price_cents":  position_data.get("current_price_cents"),
                "unrealized_pnl_cents": position_data.get("unrealized_pnl_cents"),
                "opened_at":            position_data.get("opened_at"),
                "strategy":             position_data.get("strategy", "unknown"),
                "stop_loss_pct":        position_data.get("stop_loss_pct", 0.40),
                "take_profit_pct":      position_data.get("take_profit_pct", 0.60),
            },
        )
        await self.connection.commit()

    async def get_positions(self) -> list[dict]:
        """Return all rows from the positions table as plain dicts."""
        async with self.connection.execute(
            "SELECT * FROM positions"
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def delete_position(self, ticker: str, direction: str) -> None:
        """Remove a closed position."""
        await self.connection.execute(
            "DELETE FROM positions WHERE ticker = ? AND direction = ?",
            (ticker, direction),
        )
        await self.connection.commit()

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def upsert_order(self, order_data: dict) -> None:
        """Insert a new order or update an existing one by id."""
        order_data.setdefault("placed_at", _utc_now_iso())
        order_data.setdefault("status", "RESTING")
        order_data.setdefault("filled_quantity", 0)
        order_data.setdefault("is_smart_entry", 1)

        await self.connection.execute(
            """
            INSERT INTO orders
                (id, ticker, side, quantity, price_cents, order_type,
                 status, placed_at, timeout_at, filled_quantity,
                 fill_price_cents, exchange_order_id, is_smart_entry)
            VALUES
                (:id, :ticker, :side, :quantity, :price_cents, :order_type,
                 :status, :placed_at, :timeout_at, :filled_quantity,
                 :fill_price_cents, :exchange_order_id, :is_smart_entry)
            ON CONFLICT(id) DO UPDATE SET
                status           = excluded.status,
                filled_quantity  = excluded.filled_quantity,
                fill_price_cents = excluded.fill_price_cents,
                exchange_order_id= excluded.exchange_order_id
            """,
            {
                "id":               order_data["id"],
                "ticker":           order_data["ticker"],
                "side":             order_data["side"],
                "quantity":         order_data["quantity"],
                "price_cents":      order_data["price_cents"],
                "order_type":       order_data.get("order_type", "limit"),
                "status":           order_data.get("status", "RESTING"),
                "placed_at":        order_data.get("placed_at"),
                "timeout_at":       order_data.get("timeout_at"),
                "filled_quantity":  order_data.get("filled_quantity", 0),
                "fill_price_cents": order_data.get("fill_price_cents"),
                "exchange_order_id":order_data.get("exchange_order_id"),
                "is_smart_entry":   order_data.get("is_smart_entry", 1),
            },
        )
        await self.connection.commit()

    async def update_order(self, order_id: str, **kwargs: Any) -> None:
        """Update specific fields of an order by id.

        Only the supplied keyword arguments are written; unrecognised keys are
        silently ignored so callers can pass partial dicts safely.
        """
        allowed = {
            "status", "filled_quantity", "fill_price_cents",
            "exchange_order_id", "timeout_at",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return

        set_clause = ", ".join(f"{col} = ?" for col in updates)
        values = list(updates.values()) + [order_id]
        await self.connection.execute(
            f"UPDATE orders SET {set_clause} WHERE id = ?",  # noqa: S608
            values,
        )
        await self.connection.commit()

    async def get_open_orders(self) -> list[dict]:
        """Return all orders with status RESTING or PARTIALLY_FILLED."""
        async with self.connection.execute(
            """
            SELECT * FROM orders
            WHERE status IN ('RESTING', 'PARTIALLY_FILLED')
            ORDER BY placed_at
            """
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    async def log_signal(self, signal_data: dict) -> None:
        """Log a signal (approved or rejected) from any strategy."""
        signal_data.setdefault("timestamp", _utc_now_iso())

        metadata = signal_data.get("metadata")
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)

        await self.connection.execute(
            """
            INSERT INTO signals
                (timestamp, strategy, ticker, direction, confidence,
                 edge_cents, approved, reject_reason, metadata)
            VALUES
                (:timestamp, :strategy, :ticker, :direction, :confidence,
                 :edge_cents, :approved, :reject_reason, :metadata)
            """,
            {
                "timestamp":     signal_data.get("timestamp"),
                "strategy":      signal_data.get("strategy", "unknown"),
                "ticker":        signal_data["ticker"],
                "direction":     signal_data["direction"],
                "confidence":    signal_data.get("confidence"),
                "edge_cents":    signal_data.get("edge_cents"),
                "approved":      signal_data.get("approved"),
                "reject_reason": signal_data.get("reject_reason"),
                "metadata":      metadata,
            },
        )
        await self.connection.commit()

    # ------------------------------------------------------------------
    # Daily P&L
    # ------------------------------------------------------------------

    async def get_daily_pnl(self, days: int = 7) -> list[dict]:
        """Return daily P&L rows for the last *days* calendar days."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).date().isoformat()
        async with self.connection.execute(
            """
            SELECT * FROM daily_pnl
            WHERE date >= ?
            ORDER BY date DESC
            """,
            (cutoff,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def accumulate_daily_pnl(
        self,
        date: str,
        realized_delta_cents: int = 0,
        unrealized_delta_cents: int = 0,
        trade_delta: int = 0,
        win_delta: int = 0,
    ) -> None:
        """Atomically add deltas to a daily P&L row, creating it if needed."""
        await self.connection.execute(
            """
            INSERT INTO daily_pnl (date, realized_pnl_cents, unrealized_pnl_cents,
                                   trade_count, win_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                realized_pnl_cents   = realized_pnl_cents   + excluded.realized_pnl_cents,
                unrealized_pnl_cents = unrealized_pnl_cents + excluded.unrealized_pnl_cents,
                trade_count          = trade_count          + excluded.trade_count,
                win_count            = win_count            + excluded.win_count
            """,
            (
                date,
                realized_delta_cents,
                unrealized_delta_cents,
                trade_delta,
                win_delta,
            ),
        )
        await self.connection.commit()

    # ------------------------------------------------------------------
    # Market snapshots
    # ------------------------------------------------------------------

    async def snapshot_market(
        self,
        ticker: str,
        yes_price: int,
        no_price: int,
        volume: int,
    ) -> None:
        """Save a market price snapshot with the current UTC timestamp."""
        await self.connection.execute(
            """
            INSERT INTO market_snapshots (ticker, yes_price, no_price, volume, snapshot_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ticker, yes_price, no_price, volume, _utc_now_iso()),
        )
        await self.connection.commit()

    async def get_market_history(
        self, ticker: str, hours: int = 24
    ) -> list[dict]:
        """Return market snapshot rows for *ticker* over the last *hours* hours."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        async with self.connection.execute(
            """
            SELECT * FROM market_snapshots
            WHERE ticker = ? AND snapshot_at >= ?
            ORDER BY snapshot_at ASC
            """,
            (ticker, cutoff),
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # CLV
    # ------------------------------------------------------------------

    async def log_clv(
        self,
        ticker: str,
        market_type: str,
        entry_cents: int,
        close_cents: int,
    ) -> None:
        """Log a Closing Line Value record.

        CLV = (entry_price - close_price) / close_price.
        Positive CLV means we entered at a better price than where the market
        closed (we were ahead of the line).
        """
        if close_cents and close_cents != 0:
            clv = (entry_cents - close_cents) / close_cents
        else:
            clv = None

        await self.connection.execute(
            """
            INSERT INTO clv_records
                (ticker, market_type, entry_price_cents, close_price_cents, clv, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ticker, market_type, entry_cents, close_cents, clv, _utc_now_iso()),
        )
        await self.connection.commit()

    # ------------------------------------------------------------------
    # Strategy performance
    # ------------------------------------------------------------------

    async def record_strategy_result(
        self,
        strategy: str,
        date: str,
        won: bool,
        pnl_cents: int,
    ) -> None:
        """Upsert a strategy win/loss/P&L for a given date."""
        await self.connection.execute(
            """
            INSERT INTO strategy_perf (strategy, date, wins, losses, pnl_cents)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(strategy, date) DO UPDATE SET
                wins      = wins      + excluded.wins,
                losses    = losses    + excluded.losses,
                pnl_cents = pnl_cents + excluded.pnl_cents
            """,
            (
                strategy,
                date,
                1 if won else 0,
                0 if won else 1,
                pnl_cents,
            ),
        )
        await self.connection.commit()

    async def get_strategy_perf(self, days: int = 30) -> dict[str, dict]:
        """Return per-strategy aggregate stats over the last *days* days.

        Returns a dict keyed by strategy name::

            {
                "index_fast_path": {
                    "wins": 12, "losses": 3,
                    "pnl_cents": 4200,
                    "win_rate": 0.80,
                    "trade_count": 15,
                },
                ...
            }
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).date().isoformat()
        async with self.connection.execute(
            """
            SELECT strategy,
                   SUM(wins)      AS wins,
                   SUM(losses)    AS losses,
                   SUM(pnl_cents) AS pnl_cents
            FROM strategy_perf
            WHERE date >= ?
            GROUP BY strategy
            """,
            (cutoff,),
        ) as cursor:
            rows = await cursor.fetchall()

        result: dict[str, dict] = {}
        for row in rows:
            wins = row["wins"] or 0
            losses = row["losses"] or 0
            total = wins + losses
            result[row["strategy"]] = {
                "wins":        wins,
                "losses":      losses,
                "pnl_cents":   row["pnl_cents"] or 0,
                "trade_count": total,
                "win_rate":    wins / total if total else 0.0,
            }
        return result

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def daily_maintenance(self) -> None:
        """Run WAL checkpoint and integrity check.

        Should be called once per day (e.g. from a background task at midnight
        UTC).  Logs the result via structlog.
        """
        # WAL checkpoint — flush WAL file back into the main DB file.
        async with self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)") as cur:
            row = await cur.fetchone()
            if row:
                log.info(
                    "wal_checkpoint",
                    busy_pages=row[0],
                    log_pages=row[1],
                    checkpointed=row[2],
                )

        # Integrity check — returns "ok" on a healthy database.
        async with self.connection.execute("PRAGMA integrity_check") as cur:
            result = await cur.fetchone()
            status = result[0] if result else "unknown"
            if status != "ok":
                log.error("db_integrity_check_failed", result=status)
            else:
                log.info("db_integrity_check_ok")

    def _conn(self):
        """Return a context manager yielding the underlying aiosqlite connection.

        Convenience accessor so callers can write::

            async with db._conn() as conn:
                rows = await conn.execute_fetchall(query, params)

        This is equivalent to using ``db.connection`` directly but provides a
        consistent interface when the connection may be replaced in the future.
        """
        return _ConnectionContext(self.connection)

    async def close(self) -> None:
        """Close the underlying aiosqlite connection."""
        await self.connection.close()

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    async def migrate_from_jsonl(self, state_dir: str) -> dict[str, int]:
        """Import existing JSONL trade_history.jsonl into SQLite.

        Reads ``<state_dir>/trade_history.jsonl`` and maps each JSONL event
        type to the appropriate table:

        - ``order_placed``    → orders
        - ``order_filled``    → orders (status update) + trades
        - ``position_closed`` → trades (with P&L)
        - ``market_resolved`` → clv_records

        Returns a summary dict ``{event_type: imported_count}``.
        """
        jsonl_path = Path(state_dir) / "trade_history.jsonl"
        if not jsonl_path.exists():
            log.warning("migrate_from_jsonl_no_file", path=str(jsonl_path))
            return {}

        counts: dict[str, int] = {}
        errors = 0

        with open(jsonl_path, encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning("migrate_bad_json", lineno=lineno)
                    errors += 1
                    continue

                event = record.get("event", "")
                try:
                    if event == "order_placed":
                        await self._migrate_order_placed(record)
                    elif event == "order_filled":
                        await self._migrate_order_filled(record)
                    elif event == "order_failed":
                        pass  # no persistent state needed
                    elif event == "position_closed":
                        await self._migrate_position_closed(record)
                    elif event == "market_resolved":
                        await self._migrate_market_resolved(record)
                    else:
                        log.debug("migrate_unknown_event", event=event, lineno=lineno)
                        continue

                    counts[event] = counts.get(event, 0) + 1
                except Exception as exc:
                    log.warning(
                        "migrate_record_error",
                        event=event,
                        lineno=lineno,
                        error=str(exc),
                    )
                    errors += 1

        log.info("migrate_from_jsonl_complete", counts=counts, errors=errors)
        return counts

    # -- Private migration helpers ----------------------------------------

    async def _migrate_order_placed(self, r: dict) -> None:
        order_id = r.get("order_id") or f"legacy_{r.get('ticker','?')}_{r.get('logged_at','')}"
        await self.upsert_order(
            {
                "id":            order_id,
                "ticker":        r.get("ticker", ""),
                "side":          (r.get("side") or "unknown").upper(),
                "quantity":      r.get("count", 0),
                "price_cents":   r.get("price_cents", 0),
                "order_type":    r.get("order_type", "limit"),
                "status":        "RESTING",
                "placed_at":     r.get("logged_at", _utc_now_iso()),
                "is_smart_entry":1,
            }
        )

    async def _migrate_order_filled(self, r: dict) -> None:
        order_id = r.get("order_id") or f"legacy_{r.get('ticker','?')}_{r.get('logged_at','')}"
        await self.update_order(
            order_id,
            status="FILLED",
            filled_quantity=r.get("count", 0),
            fill_price_cents=r.get("fill_price_cents"),
        )
        # Also record as a trade so P&L is captured
        side = (r.get("side") or "no").upper()
        direction = "BUY_YES" if side in ("YES", "BUY_YES") else "BUY_NO"
        await self.log_trade(
            {
                "timestamp":      r.get("logged_at", _utc_now_iso()),
                "ticker":         r.get("ticker", ""),
                "direction":      direction,
                "quantity":       r.get("count", 0),
                "price_cents":    r.get("fill_price_cents", 0),
                "fill_price_cents":r.get("fill_price_cents"),
                "strategy":       r.get("strategy") or r.get("signal_source", "unknown"),
                "is_paper":       0,
                "order_id":       order_id,
            }
        )

    async def _migrate_position_closed(self, r: dict) -> None:
        side = (r.get("side") or "no").upper()
        direction = "BUY_YES" if side in ("YES", "BUY_YES") else "BUY_NO"
        pnl_usd = r.get("pnl_usd") or 0.0
        pnl_cents = int(round(pnl_usd * 100))
        order_id = f"close_{r.get('ticker','')}_{r.get('logged_at','')}"
        await self.log_trade(
            {
                "timestamp":       r.get("logged_at", _utc_now_iso()),
                "ticker":          r.get("ticker", ""),
                "direction":       direction,
                "quantity":        r.get("count", 0),
                "price_cents":     r.get("exit_price_cents", 0),
                "fill_price_cents":r.get("exit_price_cents"),
                "strategy":        "legacy_migration",
                "pnl_cents":       pnl_cents,
                "is_paper":        0,
                "order_id":        order_id,
            }
        )

    async def _migrate_market_resolved(self, r: dict) -> None:
        ticker = r.get("ticker", "")
        clv = r.get("clv")
        entry_prob = r.get("entry_probability") or 0.0
        closing_prob = r.get("closing_probability") or 0.0
        # Convert probabilities (0–1) to cents (0–100)
        entry_cents = int(round(entry_prob * 100))
        close_cents = int(round(closing_prob * 100))
        # Use provided CLV or recompute
        if clv is None and close_cents:
            clv = (entry_cents - close_cents) / close_cents

        await self.connection.execute(
            """
            INSERT INTO clv_records
                (ticker, market_type, entry_price_cents, close_price_cents, clv, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                "legacy",
                entry_cents,
                close_cents,
                clv,
                r.get("logged_at", _utc_now_iso()),
            ),
        )
        await self.connection.commit()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

async def init_db(path: str) -> Database:
    """Create or open the SQLite database at *path*, apply schema, return a
    :class:`Database` instance.

    The parent directory is created automatically if it does not exist.
    WAL mode is enabled on every open so the pragma is always applied even
    after the first ``CREATE TABLE`` pass.
    """
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row

    # Apply schema (CREATE TABLE IF NOT EXISTS + WAL pragma)
    await conn.executescript(_SCHEMA_SQL)
    await conn.commit()

    log.info("db_opened", path=str(db_path))
    return Database(conn)
