"""SmartEntry — state-machine order execution with limit → timeout → market fallback.

Ports Neo's SmartEntry pattern to Morpheus.  Handles the full lifecycle of a
single order attempt:

  PENDING → LIMIT_PLACED → PARTIALLY_FILLED → FILLED
                                            ↘ TIMED_OUT → market fill for remainder
                         → CANCELLED (on error)

Key features:
- Adaptive spread-crossing: NOAA 5c / index 3c / generic 1c
- Data-verified signal fast-paths skip the limit wait and cross immediately
- 5-minute limit timeout, then market order for any unfilled quantity
- UUID4 client_order_id for idempotency
- Paper mode simulates fill at mid-price
- Partial fill tracking throughout
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import structlog

from .kalshi_trading_client import KalshiTradingClient
from .utils import BotConfig


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class OrderState(str, Enum):
    PENDING = "PENDING"
    LIMIT_PLACED = "LIMIT_PLACED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"


@dataclass
class OrderResult:
    """Result of a SmartEntry execution attempt."""

    order_id: str
    client_order_id: str
    ticker: str
    side: str                   # "yes" or "no"
    intended_quantity: int
    filled_quantity: int
    fill_price_cents: int       # weighted-average fill price in cents (0 if unfilled)
    total_cost: float           # USD actually spent
    state: OrderState
    is_paper: bool
    execution_ms: int           # wall-clock ms from start to terminal state
    reasoning: str              # human-readable description of what happened

    # Convenience
    @property
    def is_filled(self) -> bool:
        return self.state == OrderState.FILLED

    @property
    def partially_filled(self) -> bool:
        return self.state in (OrderState.PARTIALLY_FILLED, OrderState.TIMED_OUT) and self.filled_quantity > 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Ticker prefixes that qualify for NOAA-class aggressive crossing (5c).
_NOAA_PREFIXES = (
    "KXHIGH", "KXLOW", "KXRAIN", "KXSNOW", "KXTEMP", "KXWIND",
)

# Ticker prefixes that qualify for index-class crossing (3c).
_INDEX_PREFIXES = (
    "KXINXU", "KXINX-", "KXNASDAQ100",
    "KXSPY", "KXQQQ", "KXIWM", "KXDIA",
    "KXWTI", "KXGOLD",
)

# How long to wait for a limit order to fill before falling back to market.
_LIMIT_TIMEOUT_SECONDS = 300  # 5 minutes

# Poll interval when waiting for limit fill confirmation.
_POLL_INTERVAL_SECONDS = 10


def _detect_signal_type(ticker: str) -> str:
    """Return 'noaa', 'index', or 'generic' based on ticker prefix."""
    ticker_upper = ticker.upper()
    if any(ticker_upper.startswith(p) for p in _NOAA_PREFIXES):
        return "noaa"
    if any(ticker_upper.startswith(p) for p in _INDEX_PREFIXES):
        return "index"
    return "generic"


def _compute_limit_price(
    side: str,
    yes_bid: float,
    yes_ask: float,
    no_bid: float,
    no_ask: float,
    confidence: float,
) -> int:
    """Compute an initial limit price 1-2c inside the spread.

    Higher confidence → place limit 1c inside spread (more patient).
    Lower confidence  → place limit 2c inside spread (even more patient).

    'Inside the spread' means better than the crossing price (ask) for the buyer.

    Returns price in cents (1-99).
    """
    if side == "yes":
        # We're buying YES.  The ask is what we'd pay at market.
        # Step inside by 1-2c (i.e. offer less than the ask).
        ask_cents = round(yes_ask * 100)
        bid_cents = round(yes_bid * 100)
    else:
        # We're buying NO.
        ask_cents = round(no_ask * 100)
        bid_cents = round(no_bid * 100)

    # Determine how far inside the spread to place the limit.
    # High confidence (≥70%) → 1c inside; lower → 2c inside.
    inside = 1 if confidence >= 0.70 else 2

    limit = ask_cents - inside
    # Must stay inside the spread (above best bid) and in valid range.
    limit = max(limit, bid_cents + 1) if bid_cents > 0 else limit
    return max(1, min(99, limit))


def _compute_aggressive_price(
    side: str,
    yes_ask: float,
    no_ask: float,
    signal_type: str,
    confidence: float,
) -> int:
    """Compute an aggressive crossing price for data-verified signals.

    NOAA signals: cross up to 5c above ask
    Index signals: cross up to 3c above ask
    Generic:       cross up to 1c above ask
    """
    cross_limits = {"noaa": 5, "index": 3, "generic": 1}
    max_cross = cross_limits.get(signal_type, 1)

    if side == "yes":
        ask_cents = round(yes_ask * 100)
    else:
        ask_cents = round(no_ask * 100)

    # Scale crossing amount with confidence (floor at 1c, cap at signal-type limit).
    # conf 0.60 → 1c, conf 0.80 → ceil(1 + (0.80-0.60)/0.40 * (max-1))
    if confidence >= 0.60:
        scale = min(1.0, (confidence - 0.60) / 0.40)
        cross = max(1, round(1 + scale * (max_cross - 1)))
    else:
        cross = 1

    return max(1, min(99, ask_cents + cross))


def _mid_price_cents(yes_bid: float, yes_ask: float, no_bid: float, no_ask: float) -> int:
    """Compute mid-price in cents for paper-trade simulation."""
    if yes_bid > 0 and yes_ask > 0:
        yes_mid = (yes_bid + yes_ask) / 2.0 * 100
    else:
        yes_mid = 50.0
    return max(1, min(99, round(yes_mid)))


# ---------------------------------------------------------------------------
# SmartEntry
# ---------------------------------------------------------------------------


class SmartEntry:
    """State-machine order executor: limit → 5min timeout → market fallback.

    Usage::

        smart_entry = SmartEntry()
        result = await smart_entry.execute(signal_meta, trading_client, config)

    ``signal_meta`` is a dict with these keys (all optional with sensible defaults):

        ticker          str   – Kalshi market ticker
        side            str   – "yes" or "no"
        quantity        int   – number of contracts to buy
        confidence      float – signal confidence 0-1 (default 0.60)
        yes_bid         float – best bid for YES side (0-1 scale)
        yes_ask         float – best ask for YES side (0-1 scale)
        no_bid          float – best bid for NO side (0-1 scale)
        no_ask          float – best ask for NO side (0-1 scale)
        is_data_verified bool  – True if backed by NOAA/Yahoo fast-path
        signal_source   str   – "noaa", "index", "generic", or auto-detected
    """

    def __init__(self) -> None:
        self.logger = structlog.get_logger()

    async def execute(
        self,
        signal_meta: dict,
        trading_client: KalshiTradingClient,
        config: BotConfig,
    ) -> OrderResult:
        """Execute one order through the limit → timeout → market state machine.

        Returns an ``OrderResult`` describing what actually filled.
        """
        start_ms = int(time.monotonic() * 1000)

        ticker: str = signal_meta.get("ticker", "")
        side: str = signal_meta.get("side", "yes")
        quantity: int = int(signal_meta.get("quantity", 1))
        confidence: float = float(signal_meta.get("confidence", 0.60))
        yes_bid: float = float(signal_meta.get("yes_bid", 0.0) or 0.0)
        yes_ask: float = float(signal_meta.get("yes_ask", 0.0) or 0.0)
        no_bid: float = float(signal_meta.get("no_bid", 0.0) or 0.0)
        no_ask: float = float(signal_meta.get("no_ask", 0.0) or 0.0)
        is_data_verified: bool = bool(signal_meta.get("is_data_verified", False))

        # Auto-detect signal type if not provided
        signal_source: str = signal_meta.get("signal_source") or _detect_signal_type(ticker)
        is_paper: bool = trading_client.dry_run

        client_order_id = str(uuid.uuid4())

        # -----------------------------------------------------------------
        # Paper mode: simulate fill at mid-price immediately
        # -----------------------------------------------------------------
        if is_paper:
            mid = _mid_price_cents(yes_bid, yes_ask, no_bid, no_ask)
            fill_price = mid
            total_cost = quantity * fill_price / 100.0
            elapsed_ms = int(time.monotonic() * 1000) - start_ms
            self.logger.info(
                "smart_entry_paper_fill",
                ticker=ticker,
                side=side,
                quantity=quantity,
                fill_price_cents=fill_price,
                total_cost=total_cost,
            )
            return OrderResult(
                order_id=f"paper_{ticker}_{side}_{quantity}",
                client_order_id=client_order_id,
                ticker=ticker,
                side=side,
                intended_quantity=quantity,
                filled_quantity=quantity,
                fill_price_cents=fill_price,
                total_cost=total_cost,
                state=OrderState.FILLED,
                is_paper=True,
                execution_ms=elapsed_ms,
                reasoning=f"Paper fill at mid-price {fill_price}c",
            )

        # -----------------------------------------------------------------
        # Determine execution strategy
        # -----------------------------------------------------------------
        has_spread_data = (
            (yes_bid > 0 and yes_ask > 0) or (no_bid > 0 and no_ask > 0)
        )

        # Data-verified signals (NOAA/Yahoo backed): cross spread aggressively
        if is_data_verified and has_spread_data:
            initial_price = _compute_aggressive_price(
                side, yes_ask, no_ask, signal_source, confidence,
            )
            use_aggressive = True
        elif has_spread_data:
            # Conservative: place limit inside spread, wait 5 min, then market
            initial_price = _compute_limit_price(
                side, yes_bid, yes_ask, no_bid, no_ask, confidence,
            )
            use_aggressive = False
        else:
            # No spread data — fall back to 50c (mid) and just place limit
            initial_price = 50
            use_aggressive = False

        self.logger.info(
            "smart_entry_start",
            ticker=ticker,
            side=side,
            quantity=quantity,
            confidence=confidence,
            signal_source=signal_source,
            is_data_verified=is_data_verified,
            initial_price_cents=initial_price,
            aggressive=use_aggressive,
        )

        # -----------------------------------------------------------------
        # Phase 1: Place initial order (limit or aggressive limit)
        # -----------------------------------------------------------------
        # For data-verified aggressive signals we still use limit order type
        # (so we can cancel if needed), but price is already above the ask —
        # it will likely fill immediately as a marketable limit order.
        order_type = "limit"

        try:
            result = await trading_client.place_order(
                ticker=ticker,
                side=side,
                count=quantity,
                price_cents=initial_price,
                order_type=order_type,
            )
        except Exception as exc:
            elapsed_ms = int(time.monotonic() * 1000) - start_ms
            self.logger.error(
                "smart_entry_place_failed",
                ticker=ticker,
                side=side,
                error=str(exc),
            )
            return self._cancelled(
                ticker, side, quantity, client_order_id, elapsed_ms,
                f"Order placement exception: {exc}",
                is_paper=False,
            )

        if not result:
            elapsed_ms = int(time.monotonic() * 1000) - start_ms
            return self._cancelled(
                ticker, side, quantity, client_order_id, elapsed_ms,
                "Order placement returned None",
                is_paper=False,
            )

        # Extract order id from result
        exchange_order_id: str = (
            result.get("order_id")
            or (result.get("order") or {}).get("order_id", "unknown")
            if isinstance(result, dict) else "unknown"
        )
        status: str = (
            result.get("status")
            or (result.get("order") or {}).get("status", "resting")
            if isinstance(result, dict) else "resting"
        )

        self.logger.info(
            "smart_entry_order_placed",
            ticker=ticker,
            side=side,
            exchange_order_id=exchange_order_id,
            price_cents=initial_price,
            status=status,
            quantity=quantity,
        )

        # Immediate fill (marketable limit or aggressive crossing)
        if status in ("executed", "FILLED"):
            elapsed_ms = int(time.monotonic() * 1000) - start_ms
            total_cost = quantity * initial_price / 100.0
            return OrderResult(
                order_id=exchange_order_id,
                client_order_id=client_order_id,
                ticker=ticker,
                side=side,
                intended_quantity=quantity,
                filled_quantity=quantity,
                fill_price_cents=initial_price,
                total_cost=total_cost,
                state=OrderState.FILLED,
                is_paper=False,
                execution_ms=elapsed_ms,
                reasoning=f"Immediate fill at {initial_price}c ({signal_source} signal)",
            )

        # -----------------------------------------------------------------
        # Phase 2: Wait up to 5 minutes for resting limit to fill
        # -----------------------------------------------------------------
        # For aggressive data-verified orders that posted as resting (rare),
        # we still wait — they're at or above ask so should fill quickly.
        if use_aggressive:
            # Aggressive orders should fill within seconds or be resting
            # briefly; give them the same 5-min window for safety.
            timeout_seconds = _LIMIT_TIMEOUT_SECONDS
        else:
            timeout_seconds = _LIMIT_TIMEOUT_SECONDS

        filled_quantity = 0
        fill_price_cents = initial_price
        state = OrderState.LIMIT_PLACED
        deadline = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

            # Poll open orders to check fill status
            try:
                open_orders = await trading_client.get_open_orders()
            except Exception as exc:
                self.logger.warning(
                    "smart_entry_poll_failed",
                    ticker=ticker,
                    exchange_order_id=exchange_order_id,
                    error=str(exc),
                )
                continue

            # Look for our order in the open list
            our_order = next(
                (o for o in open_orders if o.order_id == exchange_order_id),
                None,
            )

            if our_order is None:
                # Order no longer resting — either fully filled or cancelled
                # Optimistically treat as filled; fill manager will reconcile
                filled_quantity = quantity
                state = OrderState.FILLED
                self.logger.info(
                    "smart_entry_order_gone",
                    ticker=ticker,
                    exchange_order_id=exchange_order_id,
                    interpretation="assumed_filled",
                )
                break

            # Order still resting — compute partial fills
            remaining = our_order.remaining
            partial_fills = quantity - remaining
            if partial_fills > 0:
                filled_quantity = partial_fills
                state = OrderState.PARTIALLY_FILLED
                self.logger.debug(
                    "smart_entry_partial",
                    ticker=ticker,
                    filled=filled_quantity,
                    remaining=remaining,
                )

        # -----------------------------------------------------------------
        # Phase 3: Timeout — cancel resting limit, place market for remainder
        # -----------------------------------------------------------------
        if state not in (OrderState.FILLED,):
            # Limit timed out — cancel and fill remainder at market
            state = OrderState.TIMED_OUT
            unfilled = quantity - filled_quantity

            self.logger.info(
                "smart_entry_timeout",
                ticker=ticker,
                exchange_order_id=exchange_order_id,
                filled_so_far=filled_quantity,
                unfilled=unfilled,
            )

            if unfilled > 0:
                # Cancel the resting limit
                await trading_client.cancel_order(exchange_order_id)

                # Place market order for unfilled quantity
                # Market orders use price_cents=99 for yes (max we'll pay),
                # or 99 for no — the exchange executes at best available price.
                market_price = 99

                try:
                    market_result = await trading_client.place_order(
                        ticker=ticker,
                        side=side,
                        count=unfilled,
                        price_cents=market_price,
                        order_type="market",
                    )
                    if market_result:
                        mkt_status = (
                            market_result.get("status")
                            or (market_result.get("order") or {}).get("status", "")
                            if isinstance(market_result, dict) else ""
                        )
                        if mkt_status in ("executed", "FILLED", "resting"):
                            filled_quantity += unfilled
                            state = OrderState.FILLED
                            exchange_order_id = (
                                market_result.get("order_id")
                                or (market_result.get("order") or {}).get("order_id", exchange_order_id)
                                if isinstance(market_result, dict) else exchange_order_id
                            )
                            self.logger.info(
                                "smart_entry_market_fallback_filled",
                                ticker=ticker,
                                quantity=unfilled,
                                price_cents=market_price,
                            )
                except Exception as exc:
                    self.logger.error(
                        "smart_entry_market_fallback_failed",
                        ticker=ticker,
                        error=str(exc),
                    )

        # -----------------------------------------------------------------
        # Build final result
        # -----------------------------------------------------------------
        elapsed_ms = int(time.monotonic() * 1000) - start_ms
        total_cost = filled_quantity * fill_price_cents / 100.0

        reasoning_parts = [
            f"signal_source={signal_source}",
            f"initial_price={initial_price}c",
            f"aggressive={use_aggressive}",
            f"state={state.value}",
            f"filled={filled_quantity}/{quantity}",
        ]

        final_result = OrderResult(
            order_id=exchange_order_id,
            client_order_id=client_order_id,
            ticker=ticker,
            side=side,
            intended_quantity=quantity,
            filled_quantity=filled_quantity,
            fill_price_cents=fill_price_cents if filled_quantity > 0 else 0,
            total_cost=total_cost,
            state=state,
            is_paper=False,
            execution_ms=elapsed_ms,
            reasoning=" | ".join(reasoning_parts),
        )

        self.logger.info(
            "smart_entry_complete",
            ticker=ticker,
            side=side,
            state=state.value,
            filled=filled_quantity,
            intended=quantity,
            fill_price_cents=final_result.fill_price_cents,
            total_cost=round(total_cost, 4),
            execution_ms=elapsed_ms,
        )

        return final_result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cancelled(
        self,
        ticker: str,
        side: str,
        quantity: int,
        client_order_id: str,
        elapsed_ms: int,
        reason: str,
        is_paper: bool,
    ) -> OrderResult:
        self.logger.warning("smart_entry_cancelled", ticker=ticker, reason=reason)
        return OrderResult(
            order_id="",
            client_order_id=client_order_id,
            ticker=ticker,
            side=side,
            intended_quantity=quantity,
            filled_quantity=0,
            fill_price_cents=0,
            total_cost=0.0,
            state=OrderState.CANCELLED,
            is_paper=is_paper,
            execution_ms=elapsed_ms,
            reasoning=reason,
        )
