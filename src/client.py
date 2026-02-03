"""Polymarket CLOB client wrapper."""

import asyncio
import os
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.constants import POLYGON
    from py_clob_client.order_builder.constants import BUY, SELL
    from py_clob_client.clob_types import (
        MarketOrderArgs,
        OrderArgs,
        OrderType as ClobOrderType,
        OpenOrderParams,
    )
    _HAS_CLOB = True
except Exception:  # pragma: no cover
    # Allows unit tests to run in environments missing optional CLOB deps.
    ClobClient = None  # type: ignore
    POLYGON = 137  # Polygon mainnet
    BUY = "BUY"
    SELL = "SELL"
    MarketOrderArgs = None  # type: ignore
    OrderArgs = None  # type: ignore
    ClobOrderType = None  # type: ignore
    OpenOrderParams = None  # type: ignore
    _HAS_CLOB = False

from .markets import Market
from .utils import (
    BotConfig,
    RateLimiter,
    validate_ethereum_address,
    validate_private_key,
)


class OrderSide(Enum):
    """Order side constants."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type constants."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class PolymarketClient:
    """Async-friendly wrapper for Polymarket CLOB client."""

    def __init__(self, config: BotConfig, dry_run: bool = False):
        """Initialize Polymarket client."""
        self.config = config
        self.dry_run = dry_run
        self.logger = structlog.get_logger()

        # Get configuration
        polymarket_config = config.polymarket
        self.host = polymarket_config.get("clob_url", "https://clob.polymarket.com")
        self.chain_id = polymarket_config.get("chain_id", POLYGON)

        # Get credentials from environment
        self.private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
        self.funder_address = os.getenv("POLYMARKET_FUNDER_ADDRESS")
        self.signature_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))

        # Validate credentials (skip in dry-run mode)
        if not self.dry_run:
            self._validate_credentials()
        else:
            self.logger.info("dry_run_mode", msg="Skipping credential validation")

        # Rate limiting
        timing_config = config.timing
        rate_limit = timing_config.get("rate_limit_per_minute", 30)
        self.rate_limiter = RateLimiter(rate_limit, 60.0)

        # Initialize CLOB client
        self._client: Optional[ClobClient] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the CLOB client."""
        if self._initialized:
            return

        try:
            if not self.dry_run:
                if ClobClient is None:
                    raise ImportError(
                        "py-clob-client (and its deps) are required for live trading; install dependencies"
                    )

                # Initialize the synchronous CLOB client
                # We'll run blocking calls in a thread pool
                self._client = ClobClient(
                    host=self.host,
                    key=self.private_key,
                    chain_id=self.chain_id,
                    funder=self.funder_address,
                    signature_type=self.signature_type,
                )

                # py-clob-client bug: builder_config attribute is only
                # set when explicitly passed, causing AttributeError in
                # can_builder_auth() / post_order().  Patch it.
                if not hasattr(self._client, "builder_config"):
                    self._client.builder_config = None

                # CRITICAL: derive API credentials for authenticated trading
                api_creds = await self._run_in_executor(
                    self._client.create_or_derive_api_creds
                )
                self._client.set_api_creds(api_creds)

                # Test connection
                await self._run_in_executor(
                    self._client.get_sampling_simplified_markets
                )

                self.logger.info(
                    "Polymarket client initialized",
                    host=self.host,
                    dry_run=self.dry_run,
                    funder=self.funder_address,
                    sig_type=self.signature_type,
                )
            else:
                self.logger.info("Polymarket client initialized in dry-run mode")

            self._initialized = True

        except Exception as e:
            self.logger.error("Failed to initialize Polymarket client", error=str(e))
            raise

    async def close(self) -> None:
        """Close the client."""
        # The py-clob-client doesn't require explicit cleanup
        self._initialized = False

    async def get_orderbook(self, token_id: str) -> Optional[Dict[str, Any]]:
        """Get orderbook for a token."""
        if not self._initialized:
            await self.initialize()

        try:
            await self.rate_limiter.acquire()

            if self.dry_run:
                # Return mock orderbook for dry run
                return {
                    "bids": [{"price": "0.45", "size": "100"}],
                    "asks": [{"price": "0.55", "size": "100"}],
                }

            orderbook = await self._run_in_executor(
                self._client.get_orderbook, token_id
            )

            self.logger.debug("Got orderbook", token_id=token_id, orderbook=orderbook)
            return orderbook

        except Exception as e:
            self.logger.error(
                "Failed to get orderbook", token_id=token_id, error=str(e)
            )
            return None

    async def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token."""
        orderbook = await self.get_orderbook(token_id)
        if not orderbook:
            return None

        try:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])

            if not bids or not asks:
                return None

            best_bid = float(bids[0]["price"])
            best_ask = float(asks[0]["price"])

            midpoint = (best_bid + best_ask) / 2
            return midpoint

        except (IndexError, KeyError, ValueError):
            return None

    async def place_market_order(
        self, token_id: str, amount: float, side: OrderSide,
        ref_price: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Place a market order.

        Parameters
        ----------
        ref_price : float, optional
            Reference price for dry-run simulation. When omitted in dry-run
            the mock falls back to 0.50 (the old behaviour).
        """
        if not self._initialized:
            await self.initialize()

        try:
            await self.rate_limiter.acquire()

            self.logger.info(
                "Placing market order",
                token_id=token_id,
                amount=amount,
                side=side.value,
                dry_run=self.dry_run,
            )

            if self.dry_run:
                # Return mock order result — use real ref_price when available
                sim_price = ref_price if ref_price and ref_price > 0 else 0.5
                return {
                    "order_id": f"dry_run_{token_id}_{side.value}_{amount}",
                    "status": "FILLED",
                    "filled_amount": amount,
                    "average_price": sim_price,
                }

            # Convert side to py-clob-client format
            clob_side = BUY if side == OrderSide.BUY else SELL

            # Build typed MarketOrderArgs — FOK (fill-or-kill) for taker orders
            market_args = MarketOrderArgs(
                token_id=token_id,
                amount=float(amount),
                side=clob_side,
            )
            signed_order = await self._run_in_executor(
                self._client.create_market_order, market_args,
            )
            order_result = await self._run_in_executor(
                self._client.post_order, signed_order, ClobOrderType.FOK,
            )

            self.logger.info(
                "Market order placed",
                order_id=order_result.get("orderID") or order_result.get("id"),
                result=order_result,
            )
            return order_result

        except Exception as e:
            self.logger.error(
                "Failed to place market order", token_id=token_id, error=str(e)
            )
            return None

    async def place_limit_order(
        self, token_id: str, price: float, size: float, side: OrderSide
    ) -> Optional[Dict[str, Any]]:
        """Place a limit order."""
        if not self._initialized:
            await self.initialize()

        try:
            await self.rate_limiter.acquire()

            self.logger.info(
                "Placing limit order",
                token_id=token_id,
                price=price,
                size=size,
                side=side.value,
                dry_run=self.dry_run,
            )

            if self.dry_run:
                # Return mock order result for dry run
                return {
                    "order_id": f"dry_run_{token_id}_{side.value}_{price}_{size}",
                    "status": "OPEN",
                    "remaining_size": size,
                    "price": price,
                }

            # Convert side to py-clob-client format
            clob_side = BUY if side == OrderSide.BUY else SELL

            # Build typed OrderArgs — GTC (good-til-cancelled) for maker orders
            order_args = OrderArgs(
                token_id=token_id,
                price=float(price),
                size=float(size),
                side=clob_side,
            )
            signed_order = await self._run_in_executor(
                self._client.create_order, order_args,
            )
            order_result = await self._run_in_executor(
                self._client.post_order, signed_order, ClobOrderType.GTC,
            )

            self.logger.info(
                "Limit order placed",
                order_id=order_result.get("orderID") or order_result.get("id"),
                result=order_result,
            )
            return order_result

        except Exception as e:
            self.logger.error(
                "Failed to place limit order", token_id=token_id, error=str(e)
            )
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        if not self._initialized:
            await self.initialize()

        try:
            await self.rate_limiter.acquire()

            self.logger.info(
                "Cancelling order", order_id=order_id, dry_run=self.dry_run
            )

            if self.dry_run:
                return True

            result = await self._run_in_executor(self._client.cancel, order_id)

            success = result.get("canceled", [order_id]) if isinstance(result, dict) else bool(result)
            self.logger.info(
                "Order cancellation result", order_id=order_id, success=bool(success)
            )
            return bool(success)

        except Exception as e:
            self.logger.error("Failed to cancel order", order_id=order_id, error=str(e))
            return False

    async def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        if not self._initialized:
            await self.initialize()

        try:
            await self.rate_limiter.acquire()

            self.logger.info("Cancelling all orders", dry_run=self.dry_run)

            if self.dry_run:
                return True

            result = await self._run_in_executor(self._client.cancel_all)

            self.logger.info("Cancel all orders result", result=result)
            return True

        except Exception as e:
            self.logger.error("Failed to cancel all orders", error=str(e))
            return False

    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders (best-effort).

        py-clob-client APIs vary across versions; we use get_orders and filter client-side.
        """
        if not self._initialized:
            await self.initialize()

        try:
            await self.rate_limiter.acquire()

            if self.dry_run:
                return []

            return await self._run_in_executor(
                self._client.get_orders, OpenOrderParams()
            )
        except Exception as e:
            self.logger.error("Failed to get orders", error=str(e))
            return []

    async def get_order_by_id(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single order by id (via get_orders)."""
        orders = await self.get_orders()
        for o in orders:
            if str(o.get("id") or o.get("order_id")) == str(order_id):
                return o
        return None

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders."""
        orders = await self.get_orders()
        return [order for order in orders if order.get("status") == "OPEN"]

    async def get_trades(self, token_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get trade history."""
        if not self._initialized:
            await self.initialize()

        try:
            await self.rate_limiter.acquire()

            if self.dry_run:
                # Return empty list for dry run
                return []

            if token_id:
                trades = await self._run_in_executor(self._client.get_trades, token_id)
            else:
                trades = await self._run_in_executor(self._client.get_trades)

            self.logger.debug("Got trades", count=len(trades), token_id=token_id)
            return trades

        except Exception as e:
            self.logger.error("Failed to get trades", token_id=token_id, error=str(e))
            return []

    async def get_balances(self) -> Dict[str, float]:
        """Get account balances via get_balance_allowance (CLOB API)."""
        if not self._initialized:
            await self.initialize()

        try:
            await self.rate_limiter.acquire()

            if self.dry_run:
                return {"USDC": 10000.0}

            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            params = BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL,
                signature_type=self.signature_type,
            )
            result = await self._run_in_executor(
                self._client.get_balance_allowance, params,
            )

            # result has 'balance' and 'allowance' fields
            balance = float(result.get("balance", 0)) if isinstance(result, dict) else 0.0
            self.logger.debug("Got balance", usdc=balance)
            return {"USDC": balance}

        except Exception as e:
            self.logger.error("Failed to get balances", error=str(e))
            return {}

    def _validate_credentials(self) -> None:
        """Validate required credentials."""
        if not self.private_key:
            raise ValueError("POLYMARKET_PRIVATE_KEY environment variable is required")

        if not validate_private_key(self.private_key):
            raise ValueError("Invalid private key format")

        if not self.funder_address:
            raise ValueError(
                "POLYMARKET_FUNDER_ADDRESS environment variable is required"
            )

        if not validate_ethereum_address(self.funder_address):
            raise ValueError("Invalid funder address format")

    async def _run_in_executor(self, func, *args) -> Any:
        """Run blocking function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args)
