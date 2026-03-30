"""FIFO cost basis tracker for tax lot accounting.

All persistence goes through the ``cost_basis_lots`` table in the Morpheus
SQLite database (see :mod:`src.db`).

Kalshi contracts are prediction-market binary options; each contract settles
at $1.00 (100 cents) or $0.00.  For tax purposes each purchase creates a
separate lot, and sales match against the oldest open lots first (FIFO).

Holding period:
  - Short-term: lot held < 365 days before disposal
  - Long-term:  lot held >= 365 days before disposal
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.db import Database

log = structlog.get_logger()


def _parse_date(value: str | date) -> date:
    """Accept ISO date string (YYYY-MM-DD) or a date object."""
    if isinstance(value, date):
        return value
    # Handle full ISO datetime strings as well
    return datetime.fromisoformat(value).date()


def _today() -> date:
    return datetime.now(timezone.utc).date()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def record_purchase(
    db: "Database",
    ticker: str,
    quantity: int,
    price_cents: int,
    purchase_date: str | date | None = None,
) -> int:
    """Create a cost basis lot for a new purchase.

    Parameters
    ----------
    db:             Open :class:`~src.db.Database` instance.
    ticker:         Kalshi market ticker (e.g. ``KXINXU-26MAR28-T2480``).
    quantity:       Number of contracts purchased.
    price_cents:    Purchase price per contract in cents (1–99).
    purchase_date:  Date of purchase (defaults to today UTC).

    Returns the ``id`` of the newly created lot.
    """
    if purchase_date is None:
        purchase_date = _today()
    date_str = _parse_date(purchase_date).isoformat()

    cursor = await db.connection.execute(
        """
        INSERT INTO cost_basis_lots
            (ticker, quantity, purchase_price_cents, purchase_date, disposed)
        VALUES (?, ?, ?, ?, 0)
        """,
        (ticker, quantity, price_cents, date_str),
    )
    await db.connection.commit()
    lot_id = cursor.lastrowid
    log.debug(
        "cost_basis_lot_created",
        ticker=ticker,
        quantity=quantity,
        price_cents=price_cents,
        purchase_date=date_str,
        lot_id=lot_id,
    )
    return lot_id  # type: ignore[return-value]


async def record_sale(
    db: "Database",
    ticker: str,
    quantity: int,
    sale_price_cents: int,
    sale_date: str | date | None = None,
) -> list[dict]:
    """Match a sale against open FIFO lots and record realized gains.

    Partial lot disposal is supported: if a lot has more contracts than
    needed to satisfy the sale quantity, the lot is split — the sold portion
    is marked disposed and a new residual lot is created with the remaining
    quantity.

    Parameters
    ----------
    db:               Open :class:`~src.db.Database` instance.
    ticker:           Kalshi market ticker.
    quantity:         Number of contracts sold.
    sale_price_cents: Sale price per contract in cents.
    sale_date:        Date of sale (defaults to today UTC).

    Returns a list of disposal records (one per lot consumed), each containing::

        {
            "lot_id": int,
            "ticker": str,
            "quantity_disposed": int,
            "purchase_price_cents": int,
            "sale_price_cents": int,
            "realized_gain_cents": int,
            "is_long_term": bool,
            "purchase_date": str,
            "disposal_date": str,
        }

    Raises
    ------
    ValueError
        If there are not enough open lots to cover the requested sale quantity.
    """
    if sale_date is None:
        sale_date = _today()
    disposal_date_str = _parse_date(sale_date).isoformat()

    remaining = quantity
    disposals: list[dict] = []

    # Fetch open lots ordered by purchase date (FIFO)
    async with db.connection.execute(
        """
        SELECT id, ticker, quantity, purchase_price_cents, purchase_date
        FROM cost_basis_lots
        WHERE ticker = ? AND disposed = 0
        ORDER BY purchase_date ASC, id ASC
        """,
        (ticker,),
    ) as cursor:
        lots = [dict(row) for row in await cursor.fetchall()]

    total_available = sum(lot["quantity"] for lot in lots)
    if total_available < quantity:
        raise ValueError(
            f"Insufficient open lots for {ticker}: need {quantity}, "
            f"have {total_available}"
        )

    for lot in lots:
        if remaining <= 0:
            break

        lot_qty = lot["quantity"]
        disposed_qty = min(lot_qty, remaining)
        remaining -= disposed_qty

        purchase_date_obj = _parse_date(lot["purchase_date"])
        disposal_date_obj = _parse_date(disposal_date_str)
        days_held = (disposal_date_obj - purchase_date_obj).days
        is_long_term = days_held >= 365

        gain_cents = (sale_price_cents - lot["purchase_price_cents"]) * disposed_qty

        if disposed_qty == lot_qty:
            # Fully dispose the lot
            await db.connection.execute(
                """
                UPDATE cost_basis_lots
                SET disposed            = 1,
                    disposal_date       = ?,
                    sale_price_cents    = ?,
                    realized_gain_cents = ?
                WHERE id = ?
                """,
                (disposal_date_str, sale_price_cents, gain_cents, lot["id"]),
            )
        else:
            # Partially dispose: mark this lot fully disposed with the consumed
            # quantity, then insert a residual lot for the remainder.
            residual_qty = lot_qty - disposed_qty

            await db.connection.execute(
                """
                UPDATE cost_basis_lots
                SET quantity            = ?,
                    disposed            = 1,
                    disposal_date       = ?,
                    sale_price_cents    = ?,
                    realized_gain_cents = ?
                WHERE id = ?
                """,
                (disposed_qty, disposal_date_str, sale_price_cents, gain_cents, lot["id"]),
            )
            # Residual lot inherits the original purchase date
            await db.connection.execute(
                """
                INSERT INTO cost_basis_lots
                    (ticker, quantity, purchase_price_cents, purchase_date, disposed)
                VALUES (?, ?, ?, ?, 0)
                """,
                (ticker, residual_qty, lot["purchase_price_cents"], lot["purchase_date"]),
            )

        disposals.append(
            {
                "lot_id":               lot["id"],
                "ticker":               ticker,
                "quantity_disposed":    disposed_qty,
                "purchase_price_cents": lot["purchase_price_cents"],
                "sale_price_cents":     sale_price_cents,
                "realized_gain_cents":  gain_cents,
                "is_long_term":         is_long_term,
                "purchase_date":        lot["purchase_date"],
                "disposal_date":        disposal_date_str,
            }
        )

    await db.connection.commit()
    log.debug(
        "cost_basis_sale_recorded",
        ticker=ticker,
        quantity=quantity,
        sale_price_cents=sale_price_cents,
        lots_touched=len(disposals),
    )
    return disposals


async def get_tax_report(db: "Database", year: int) -> dict:
    """Compute realized gains for a calendar year, split by holding period.

    Parameters
    ----------
    db:   Open :class:`~src.db.Database` instance.
    year: Calendar year (e.g. 2025).

    Returns a dict::

        {
            "year": 2025,
            "total_realized_cents": 12345,
            "short_term_gain_cents": 9000,
            "long_term_gain_cents": 3345,
            "by_ticker": {
                "KXINXU-26MAR28-T2480": {
                    "short_term_gain_cents": 400,
                    "long_term_gain_cents":  0,
                    "total_gain_cents":      400,
                    "lots_disposed":         3,
                },
                ...
            }
        }
    """
    year_start = f"{year}-01-01"
    year_end = f"{year}-12-31"

    async with db.connection.execute(
        """
        SELECT id, ticker, quantity, purchase_price_cents, purchase_date,
               disposal_date, sale_price_cents, realized_gain_cents
        FROM cost_basis_lots
        WHERE disposed = 1
          AND disposal_date >= ?
          AND disposal_date <= ?
        ORDER BY ticker, disposal_date
        """,
        (year_start, year_end),
    ) as cursor:
        rows = [dict(row) for row in await cursor.fetchall()]

    short_term_total = 0
    long_term_total = 0
    by_ticker: dict[str, dict] = {}

    for row in rows:
        gain = row["realized_gain_cents"] or 0
        purchase_date_obj = _parse_date(row["purchase_date"])
        disposal_date_obj = _parse_date(row["disposal_date"])
        days_held = (disposal_date_obj - purchase_date_obj).days
        is_long_term = days_held >= 365

        ticker = row["ticker"]
        if ticker not in by_ticker:
            by_ticker[ticker] = {
                "short_term_gain_cents": 0,
                "long_term_gain_cents":  0,
                "total_gain_cents":      0,
                "lots_disposed":         0,
            }

        if is_long_term:
            long_term_total += gain
            by_ticker[ticker]["long_term_gain_cents"] += gain
        else:
            short_term_total += gain
            by_ticker[ticker]["short_term_gain_cents"] += gain

        by_ticker[ticker]["total_gain_cents"] += gain
        by_ticker[ticker]["lots_disposed"] += 1

    return {
        "year":                   year,
        "total_realized_cents":   short_term_total + long_term_total,
        "short_term_gain_cents":  short_term_total,
        "long_term_gain_cents":   long_term_total,
        "by_ticker":              by_ticker,
    }


async def get_unrealized(
    db: "Database",
    ticker: str,
    current_price_cents: int,
) -> dict:
    """Compute unrealized P&L for all open lots of a ticker.

    Parameters
    ----------
    db:                   Open :class:`~src.db.Database` instance.
    ticker:               Kalshi market ticker.
    current_price_cents:  Current market price per contract in cents.

    Returns::

        {
            "ticker": "KXINXU-26MAR28-T2480",
            "current_price_cents": 65,
            "total_quantity": 10,
            "avg_cost_cents": 52,
            "unrealized_gain_cents": 130,
            "lots": [
                {
                    "lot_id": 3,
                    "quantity": 5,
                    "purchase_price_cents": 50,
                    "purchase_date": "2026-01-15",
                    "unrealized_gain_cents": 75,
                },
                ...
            ]
        }
    """
    async with db.connection.execute(
        """
        SELECT id, quantity, purchase_price_cents, purchase_date
        FROM cost_basis_lots
        WHERE ticker = ? AND disposed = 0
        ORDER BY purchase_date ASC, id ASC
        """,
        (ticker,),
    ) as cursor:
        lots = [dict(row) for row in await cursor.fetchall()]

    if not lots:
        return {
            "ticker":                ticker,
            "current_price_cents":   current_price_cents,
            "total_quantity":        0,
            "avg_cost_cents":        0,
            "unrealized_gain_cents": 0,
            "lots":                  [],
        }

    total_qty = sum(lot["quantity"] for lot in lots)
    total_cost = sum(lot["quantity"] * lot["purchase_price_cents"] for lot in lots)
    avg_cost = total_cost // total_qty if total_qty else 0
    total_unrealized = sum(
        (current_price_cents - lot["purchase_price_cents"]) * lot["quantity"]
        for lot in lots
    )

    lot_details = [
        {
            "lot_id":               lot["id"],
            "quantity":             lot["quantity"],
            "purchase_price_cents": lot["purchase_price_cents"],
            "purchase_date":        lot["purchase_date"],
            "unrealized_gain_cents": (
                (current_price_cents - lot["purchase_price_cents"]) * lot["quantity"]
            ),
        }
        for lot in lots
    ]

    return {
        "ticker":                ticker,
        "current_price_cents":   current_price_cents,
        "total_quantity":        total_qty,
        "avg_cost_cents":        avg_cost,
        "unrealized_gain_cents": total_unrealized,
        "lots":                  lot_details,
    }
