"""Authenticated Kalshi trading client.

Wraps the ``kalshi-python`` SDK for order placement, balance queries,
and position management.  Auth uses RSA key signing (API key ID + PEM).

All public methods are async — SDK calls run in a thread-pool executor
to avoid blocking the event loop (same pattern as :class:`PolymarketClient`).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .trade_logger import get_trade_logger
import structlog

try:
    from kalshi_python import Configuration, KalshiClient as _KalshiApiClient
    from kalshi_python.api.portfolio_api import PortfolioApi as _PortfolioApi
    from kalshi_python.api.exchange_api import ExchangeApi as _ExchangeApi
    _HAS_KALSHI = True
except ImportError:
    _KalshiApiClient = None  # type: ignore
    _PortfolioApi = None  # type: ignore
    _ExchangeApi = None  # type: ignore
    Configuration = None  # type: ignore
    _HAS_KALSHI = False

from .utils import BotConfig


@dataclass
class KalshiOrder:
    """Parsed order from the Kalshi API."""
    order_id: str
    ticker: str
    side: str          # "yes" or "no"
    type: str          # "limit" or "market"
    status: str
    count: int         # number of contracts
    price_cents: int   # price per contract in cents
    remaining: int     # unfilled contracts


@dataclass
class KalshiPosition:
    """Parsed position from the Kalshi API."""
    ticker: str
    count: int              # net contract count (positive = long yes, negative = long no)
    market_exposure: float  # USD exposure (count * entry cost)


class KalshiTradingClient:
    """Async-friendly wrapper for the authenticated Kalshi SDK."""

    DEFAULT_HOST = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(
        self,
        config: BotConfig,
        dry_run: bool = False,
        api_key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        label: str = "default",
    ):
        self.config = config
        self.dry_run = dry_run
        self.label = label
        self.logger = structlog.get_logger()

        kalshi_cfg = getattr(config, "kalshi", None) or {}
        if isinstance(kalshi_cfg, dict):
            self.host = kalshi_cfg.get("base_url", self.DEFAULT_HOST)
        else:
            self.host = self.DEFAULT_HOST

        # Credentials from env or explicit args
        self.api_key_id = api_key_id or os.getenv("KALSHI_API_KEY_ID", "")
        if private_key_path:
            with open(private_key_path, "r") as f:
                self._private_key_pem = f.read()
        else:
            self._private_key_pem = self._load_private_key()

        self._client: Optional[Any] = None
        self._initialized = False

        # Trading halt state — stops trading on insufficient balance
        self._trading_halted = False
        self._halt_reason: Optional[str] = None
        self._halt_state_file: Optional[str] = None

        # Load persisted halt state (avoids wasting API call on $0.01 accounts)
        self._load_halt_state()

        # Closed-market cache: ticker -> (monotonic_time, status)
        # Prevents repeated API calls to markets known to be closed
        self._closed_market_cache: Dict[str, tuple[float, str]] = {}

    # ------------------------------------------------------------------
    # Trading halt management
    # ------------------------------------------------------------------

    @property
    def is_halted(self) -> bool:
        """Return True if trading is halted (e.g., insufficient balance)."""
        return self._trading_halted

    @property
    def halt_reason(self) -> Optional[str]:
        """Return the reason trading was halted, or None."""
        return self._halt_reason

    def _halt_trading(self, reason: str) -> None:
        """Halt all trading with the given reason."""
        if not self._trading_halted:
            self._trading_halted = True
            self._halt_reason = reason
            self._save_halt_state()
            self.logger.error(
                "kalshi_trading_halted",
                label=self.label,
                reason=reason,
            )

    def resume_trading(self) -> None:
        """Resume trading after a halt (e.g., after adding funds)."""
        self._trading_halted = False
        self._halt_reason = None
        self._save_halt_state()
        self.logger.info("kalshi_trading_resumed", label=self.label)

    def _load_halt_state(self) -> None:
        """Load persisted halt state from disk."""
        import json
        from pathlib import Path
        try:
            state_dir = Path("state")
            if not state_dir.exists():
                return
            self._halt_state_file = str(state_dir / f"client_halt_{self.label}.json")
            p = Path(self._halt_state_file)
            if p.exists():
                data = json.loads(p.read_text())
                if data.get("halted"):
                    self._trading_halted = True
                    self._halt_reason = data.get("reason")
                    self.logger.info(
                        "kalshi_halt_state_loaded",
                        label=self.label,
                        reason=self._halt_reason,
                    )
        except Exception as e:
            self.logger.debug("kalshi_halt_state_load_error", label=self.label, error=str(e))

    def _save_halt_state(self) -> None:
        """Persist halt state to disk."""
        import json
        from pathlib import Path
        try:
            state_dir = Path("state")
            state_dir.mkdir(exist_ok=True)
            if self._halt_state_file is None:
                self._halt_state_file = str(state_dir / f"client_halt_{self.label}.json")
            Path(self._halt_state_file).write_text(json.dumps({
                "halted": self._trading_halted,
                "reason": self._halt_reason,
            }))
        except Exception as e:
            self.logger.debug("kalshi_halt_state_save_error", label=self.label, error=str(e))

    async def check_and_resume(self, min_balance: float = 0.25) -> bool:
        """Check balance and auto-resume trading if sufficient funds available.

        Returns True if trading was resumed, False otherwise.
        Lowered from $1.00 to $0.25 — even small balances can trade
        1-2 weather contracts, and staying halted for hours wastes
        opportunities.
        """
        if not self._trading_halted:
            return False
        try:
            balance = await self.get_balance()
            if balance >= min_balance:
                self.resume_trading()
                self.logger.info(
                    "kalshi_auto_resumed",
                    label=self.label,
                    balance=balance,
                )
                return True
        except Exception as e:
            self.logger.debug(
                "kalshi_resume_check_failed",
                label=self.label,
                error=str(e),
            )
        return False

    # ------------------------------------------------------------------
    # Private key loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_private_key() -> str:
        """Load RSA private key from env var (inline) or file path."""
        inline = os.getenv("KALSHI_PRIVATE_KEY", "")
        if inline:
            return inline

        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
        if key_path:
            with open(key_path, "r") as f:
                return f.read()

        return ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Set up the SDK client with RSA auth."""
        if self._initialized:
            return

        if self.dry_run:
            self.logger.info("kalshi_trading_client_dry_run", label=self.label)
            self._initialized = True
            return

        if not _HAS_KALSHI:
            raise ImportError(
                "kalshi-python is required for Kalshi trading; "
                "pip install kalshi-python"
            )

        if not self.api_key_id or not self._private_key_pem:
            raise ValueError(
                "KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH (or KALSHI_PRIVATE_KEY) "
                "are required for authenticated Kalshi trading"
            )

        cfg = Configuration(host=self.host)
        cfg.api_key_id = self.api_key_id
        cfg.private_key_pem = self._private_key_pem

        api_client = _KalshiApiClient(cfg)
        self._portfolio = _PortfolioApi(api_client)
        self._exchange = _ExchangeApi(api_client)

        # Smoke test: fetch balance
        balance = await self._run_in_executor(self._portfolio.get_balance)
        usd = balance.balance / 100.0 if hasattr(balance, "balance") else 0.0
        self.logger.info(
            "kalshi_trading_client_initialized",
            label=self.label,
            host=self.host,
            balance_usd=usd,
        )
        self._initialized = True

    async def close(self) -> None:
        """Clean up (SDK has no explicit close)."""
        self._initialized = False

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    async def get_balance(self) -> float:
        """Return available USD balance."""
        if self.dry_run:
            return float(
                self.config.dev.get("paper_trading_balance", 10_000.0)
            )

        await self._ensure_init()
        result = await self._run_in_executor(self._portfolio.get_balance)
        # SDK returns cents
        return result.balance / 100.0 if hasattr(result, "balance") else 0.0

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_order(
        self,
        ticker: str,
        side: str,
        count: int,
        price_cents: int,
        *,
        order_type: str = "limit",
        is_exit: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Place a limit (or market) order on Kalshi.

        Args:
            ticker: Market ticker (e.g. ``KXBTC-25FEB07-T101999``).
            side: ``"yes"`` or ``"no"``.
            count: Number of contracts.
            price_cents: Price per contract in cents (1-99 for limit).
            order_type: ``"limit"`` (default) or ``"market"``.

        Returns:
            Order response dict or None on failure.
        """
        if count <= 0:
            return None

        # Check if trading is halted (exit orders always go through)
        if self._trading_halted and not is_exit:
            self.logger.warning(
                "kalshi_order_skipped_halted",
                label=self.label,
                ticker=ticker,
                reason=self._halt_reason,
            )
            return None
        if self._trading_halted and is_exit:
            self.logger.info(
                "kalshi_exit_order_despite_halt",
                label=self.label,
                ticker=ticker,
            )

        # Pre-check: skip markets known to be closed (cached, no API call)
        import time as _time
        cached = self._closed_market_cache.get(ticker)
        if cached:
            ts, status = cached
            if _time.monotonic() - ts < 120 and status in ("closed", "settled", "finalized"):
                self.logger.debug("skip_closed_market", label=self.label, ticker=ticker, status=status)
                return None

        self.logger.info(
            "kalshi_place_order",
            label=self.label,
            ticker=ticker,
            side=side,
            count=count,
            price_cents=price_cents,
            order_type=order_type,
            dry_run=self.dry_run,
        )

        if self.dry_run:
            return {
                "order_id": f"dry_run_{ticker}_{side}_{count}",
                "status": "resting" if order_type == "limit" else "executed",
                "ticker": ticker,
                "side": side,
                "count": count,
                "price_cents": price_cents,
                "average_price": price_cents,
            }

        await self._ensure_init()

        try:
            import uuid

            order_kwargs = dict(
                ticker=ticker,
                client_order_id=str(uuid.uuid4()),
                side=side,
                action="buy",
                count=count,
                type=order_type,
            )
            if side == "yes":
                order_kwargs["yes_price"] = price_cents
            else:
                order_kwargs["no_price"] = price_cents

            result = await self._run_in_executor(
                self._portfolio.create_order, **order_kwargs
            )
            order_data = result.to_dict() if hasattr(result, "to_dict") else result
            order_id = (
                getattr(result, "order_id", None)
                or (order_data.get("order", {}).get("order_id") if isinstance(order_data, dict) else None)
                or "unknown"
            )
            self.logger.info("kalshi_order_placed", label=self.label, order_id=order_id, ticker=ticker)
            # Trade logging moved to KalshiExecutor which has full signal context
            # (edge, conviction, cost, probability). Logging here caused duplicates.
            return order_data if isinstance(order_data, dict) else {"order": order_data}
        except Exception as exc:
            error_str = str(exc).lower()

            # Re-raise market_closed — callers need to handle this differently
            # (position will be settled by Kalshi, no retry needed)
            if "market_closed" in error_str:
                self._closed_market_cache[ticker] = (_time.monotonic(), "closed")
                raise

            # Detect insufficient balance and halt trading
            if "insufficient_balance" in error_str or "insufficient balance" in error_str:
                self._halt_trading(f"Insufficient balance detected: {exc}")
                # Try to send alert (import here to avoid circular)
                try:
                    from .alerts import send_alert
                    asyncio.create_task(
                        send_alert(
                            f"🛑 [{self.label}] Trading HALTED — Insufficient balance. "
                            f"Fund your Kalshi account to resume.",
                            self.config,
                        )
                    )
                except Exception:
                    pass  # Alert failure shouldn't break the flow
            
            self.logger.error("kalshi_order_failed", label=self.label, ticker=ticker, error=str(exc))
            get_trade_logger().log_order_failed("kalshi", ticker, side, count, price_cents, str(exc), account_label=self.label)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self.dry_run:
            return True

        await self._ensure_init()
        try:
            await self._run_in_executor(self._portfolio.cancel_order, order_id)
            self.logger.info("kalshi_order_cancelled", label=self.label, order_id=order_id)
            return True
        except Exception as exc:
            self.logger.error("kalshi_cancel_failed", label=self.label, order_id=order_id, error=str(exc))
            return False

    async def get_open_orders(self) -> List[KalshiOrder]:
        """Return all open/resting orders."""
        if self.dry_run:
            return []

        await self._ensure_init()
        try:
            result = await self._run_in_executor(self._portfolio.get_orders, status="resting")
            orders_raw = result.orders if hasattr(result, "orders") else []
            return [
                KalshiOrder(
                    order_id=getattr(o, "order_id", ""),
                    ticker=getattr(o, "ticker", ""),
                    side=getattr(o, "side", ""),
                    type=getattr(o, "type", "limit"),
                    status=getattr(o, "status", ""),
                    count=getattr(o, "count", 0),
                    price_cents=getattr(o, "yes_price", 0) or getattr(o, "no_price", 0),
                    remaining=getattr(o, "remaining_count", 0),
                )
                for o in orders_raw
            ]
        except Exception as exc:
            self.logger.error("kalshi_get_orders_failed", label=self.label, error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    async def get_recent_fills(self, ticker: str = None, limit: int = 20) -> List[dict]:
        """Fetch recent fills, optionally filtered by ticker."""
        if self.dry_run:
            return []

        await self._ensure_init()
        try:
            kwargs = {"limit": limit}
            if ticker:
                kwargs["ticker"] = ticker
            result = await self._run_in_executor(self._portfolio.get_fills, **kwargs)
            fills = result.fills if hasattr(result, "fills") else []
            return [
                {
                    "order_id": getattr(f, "order_id", ""),
                    "ticker": getattr(f, "ticker", ""),
                    "count": getattr(f, "count", 0),
                    "side": getattr(f, "side", ""),
                }
                for f in fills
            ]
        except Exception as e:
            self.logger.debug("kalshi_get_fills_failed", label=self.label, error=str(e))
            return []

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_positions(self) -> List[KalshiPosition]:
        """Return current positions.

        Uses the raw HTTP response because the kalshi_python SDK model
        doesn't parse ``market_positions`` correctly (the ``positions``
        attribute is always None).
        """
        if self.dry_run:
            return []

        await self._ensure_init()
        try:
            import json as _json
            raw = await self._run_in_executor(
                self._portfolio.get_positions_without_preload_content
            )
            body = raw.read().decode() if hasattr(raw, "read") else str(raw.data)
            data = _json.loads(body)
            positions_raw = data.get("market_positions") or []
            return [
                KalshiPosition(
                    ticker=p.get("ticker", ""),
                    count=p.get("position", 0),
                    market_exposure=abs(p.get("market_exposure", 0)) / 100.0,
                )
                for p in positions_raw
                if p.get("position", 0) != 0
            ]
        except Exception as exc:
            self.logger.error("kalshi_get_positions_failed", label=self.label, error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ensure_init(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def _run_in_executor(self, func, *args, **kwargs) -> Any:
        """Run a blocking SDK call in the default thread-pool."""
        loop = asyncio.get_event_loop()
        if kwargs:
            import functools
            func = functools.partial(func, **kwargs)
        return await loop.run_in_executor(None, func, *args)
