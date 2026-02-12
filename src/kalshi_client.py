"""Kalshi prediction market API client.

Fetches public market data from Kalshi's REST API (no auth required for reads).
Uses async httpx with rate limiting and pagination.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
import structlog

from .utils import BotConfig, RateLimiter, safe_float, safe_int


@dataclass
class KalshiMarket:
    """Parsed Kalshi binary market."""

    ticker: str
    title: str
    category: str
    yes_price: float  # 0-1 (converted from cents)
    no_price: float   # 0-1
    volume: int
    open_interest: int
    close_time: Optional[datetime]
    status: str

    # Extra fields useful for matching / display
    event_ticker: str = ""
    volume_24h: int = 0
    last_price: float = 0.0
    yes_bid: float = 0.0
    yes_ask: float = 0.0
    no_bid: float = 0.0
    no_ask: float = 0.0


class KalshiClient:
    """Async client for the Kalshi public REST API."""

    DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = structlog.get_logger()

        kalshi_cfg = getattr(config, "kalshi", None) or {}
        if isinstance(kalshi_cfg, dict):
            self.base_url = kalshi_cfg.get("base_url", self.DEFAULT_BASE_URL)
            rps = kalshi_cfg.get("rate_limit_per_second", 8)
        else:
            self.base_url = self.DEFAULT_BASE_URL
            rps = 8

        # Rate limiter: N requests per second
        self.rate_limiter = RateLimiter(max_calls=rps, time_window=1.0)

        self._client: Optional[httpx.AsyncClient] = None
        # event_ticker -> category cache (populated lazily)
        self._event_categories: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers={"Accept": "application/json"},
            )
        self.logger.info("kalshi_client_started", base_url=self.base_url)

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Rate-limited GET request with retry on 429."""
        if self._client is None:
            await self.start()

        max_retries = 3
        for attempt in range(max_retries):
            await self.rate_limiter.acquire()
            resp = await self._client.get(path, params=params)  # type: ignore[union-attr]
            if resp.status_code == 429 and attempt < max_retries - 1:
                wait = 1.5 * (attempt + 1)
                self.logger.debug("kalshi_rate_limited", wait=wait, attempt=attempt)
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        # Should not reach here, but satisfy type checker
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Events (for category lookup)
    # ------------------------------------------------------------------

    async def fetch_event_categories(self, *, limit: int = 200) -> Dict[str, str]:
        """Populate event_ticker -> category mapping (paginated)."""
        cursor: Optional[str] = None
        fetched = 0

        while True:
            params: Dict[str, Any] = {"limit": min(limit, 200), "status": "open"}
            if cursor:
                params["cursor"] = cursor

            try:
                data = await self._get("/events", params=params)
            except httpx.HTTPStatusError as exc:
                self.logger.warning("kalshi_events_error", status=exc.response.status_code)
                break

            events = data.get("events", [])
            if not events:
                break

            for ev in events:
                ticker = ev.get("event_ticker", "")
                cat = ev.get("category", "Other")
                if ticker:
                    self._event_categories[ticker] = cat

            fetched += len(events)
            cursor = data.get("cursor")
            if not cursor or len(events) < params["limit"]:
                break

        self.logger.info("kalshi_events_fetched", total=fetched, categories=len(self._event_categories))
        return self._event_categories

    def _lookup_category(self, event_ticker: str) -> str:
        return self._event_categories.get(event_ticker, "Other")

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    def _parse_market(self, m: Dict[str, Any]) -> Optional[KalshiMarket]:
        """Parse a single market dict from the API response."""
        ticker = m.get("ticker", "")
        title = m.get("title", "")
        status = m.get("status", "")
        if not ticker or not title:
            return None

        # Kalshi returns prices in cents (0-100) when response_price_units == "usd_cent"
        # Some newer endpoints may return dollar strings; handle both.
        price_units = m.get("response_price_units", "usd_cent")
        divisor = 100.0 if price_units == "usd_cent" else 1.0

        yes_bid = safe_float(m.get("yes_bid", 0)) / divisor
        yes_ask = safe_float(m.get("yes_ask", 0)) / divisor
        no_bid = safe_float(m.get("no_bid", 0)) / divisor
        no_ask = safe_float(m.get("no_ask", 0)) / divisor
        last_price = safe_float(m.get("last_price", 0)) / divisor

        # Best estimate of current yes/no price: midpoint of bid/ask
        if yes_bid > 0 and yes_ask > 0:
            yes_price = (yes_bid + yes_ask) / 2.0
        elif last_price > 0:
            yes_price = last_price
        else:
            yes_price = 0.0

        if no_bid > 0 and no_ask > 0:
            no_price = (no_bid + no_ask) / 2.0
        elif yes_price > 0:
            no_price = 1.0 - yes_price
        else:
            no_price = 0.0

        # Close time
        close_str = m.get("close_time") or m.get("expiration_time")
        close_time: Optional[datetime] = None
        if close_str:
            try:
                close_time = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        event_ticker = m.get("event_ticker", "")
        category = self._lookup_category(event_ticker)

        return KalshiMarket(
            ticker=ticker,
            title=title,
            category=category,
            yes_price=yes_price,
            no_price=no_price,
            volume=safe_int(m.get("volume", 0)),
            open_interest=safe_int(m.get("open_interest", 0)),
            close_time=close_time,
            status=status,
            event_ticker=event_ticker,
            volume_24h=safe_int(m.get("volume_24h", 0)),
            last_price=last_price,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
        )

    async def fetch_markets(
        self,
        *,
        status: str = "open",
        limit: int = 1000,
        min_volume: int = 0,
    ) -> List[KalshiMarket]:
        """Fetch active Kalshi markets with pagination.

        Args:
            status: Market status filter ("open", "closed", etc.)
            limit: Max total markets to return.
            min_volume: Skip markets below this total volume.

        Returns:
            List of parsed KalshiMarket objects.
        """
        markets: List[KalshiMarket] = []
        cursor: Optional[str] = None
        page_size = min(limit, 200)  # Kalshi caps at 200 per page

        while len(markets) < limit:
            params: Dict[str, Any] = {
                "limit": page_size,
                "status": status,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                data = await self._get("/markets", params=params)
            except httpx.HTTPStatusError as exc:
                self.logger.error(
                    "kalshi_markets_fetch_error",
                    status=exc.response.status_code,
                    detail=exc.response.text[:200],
                )
                break
            except httpx.HTTPError as exc:
                self.logger.error("kalshi_markets_http_error", error=str(exc))
                break

            raw_markets = data.get("markets", [])
            if not raw_markets:
                break

            zero_vol_streak = 0
            for raw in raw_markets:
                parsed = self._parse_market(raw)
                if parsed is None:
                    continue
                if parsed.status != "active":
                    continue
                if min_volume > 0 and parsed.volume < min_volume:
                    zero_vol_streak += 1
                    continue
                zero_vol_streak = 0
                markets.append(parsed)
                if len(markets) >= limit:
                    break

            cursor = data.get("cursor")
            # Stop paginating if we're clearly past the interesting markets
            if not cursor or len(raw_markets) < page_size:
                break
            # If the entire page had zero qualifying markets, stop early
            if zero_vol_streak >= page_size:
                break

        self.logger.info("kalshi_markets_fetched", total=len(markets))
        return markets

    async def fetch_markets_by_events(
        self,
        *,
        categories: Optional[List[str]] = None,
        max_markets: int = 500,
        min_volume: int = 0,
    ) -> List[KalshiMarket]:
        """Fetch markets through events — better for politics/economics.

        The generic /markets endpoint returns sports parlays first.
        This method fetches events, filters by category, then gets
        markets per event.
        """
        if categories is None:
            categories = ["Politics", "Economics", "Elections", "World", "Financials", "Climate and Weather"]

        # Ensure event categories are loaded
        if not self._event_categories:
            await self.fetch_event_categories(limit=5000)

        # Find events in target categories (cap at 100 to avoid API spam)
        target_events = [
            ticker for ticker, cat in self._event_categories.items()
            if cat in categories
        ][:100]
        self.logger.info(
            "kalshi_target_events",
            categories=categories,
            matching_events=len(target_events),
        )

        markets: List[KalshiMarket] = []
        for i, event_ticker in enumerate(target_events):
            if len(markets) >= max_markets:
                break

            try:
                data = await self._get("/markets", params={
                    "limit": 50,
                    "status": "open",
                    "event_ticker": event_ticker,
                })
            except httpx.HTTPError:
                continue

            for raw in data.get("markets", []):
                parsed = self._parse_market(raw)
                if parsed is None:
                    continue
                if parsed.volume < min_volume:
                    continue
                markets.append(parsed)

            # Rate limit between events
            await asyncio.sleep(0.15)

        self.logger.info("kalshi_markets_by_events", total=len(markets))
        return markets


    async def fetch_markets_by_close_date(
        self,
        *,
        max_days: int = 30,
        min_volume: int = 50,
        max_pages: int = 15,
    ) -> List[KalshiMarket]:
        """Fetch open markets closing within *max_days*, filtered by volume.

        Uses the ``min_close_ts`` / ``max_close_ts`` query params on the
        ``/markets`` endpoint so we get short-term markets regardless of
        category.  Sports multi-game parlays are excluded.
        """
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        min_ts = int(now.timestamp())
        max_ts = int((now + timedelta(days=max_days)).timestamp())

        all_markets: List[KalshiMarket] = []
        cursor: Optional[str] = None

        for _ in range(max_pages):
            params: Dict[str, Any] = {
                "limit": 200,
                "status": "open",
                "min_close_ts": min_ts,
                "max_close_ts": max_ts,
            }
            if cursor:
                params["cursor"] = cursor

            data = await self._get("/markets", params=params)
            raw = data.get("markets", [])

            for m in raw:
                ticker = m.get("ticker", "")
                # Skip sports parlays
                if "KXMVESPORTS" in ticker or "MULTIGAME" in ticker:
                    continue
                vol = int(m.get("volume", 0) or 0)
                if vol < min_volume:
                    continue

                parsed = self._parse_market(m)
                if parsed and parsed.yes_price > 0 and parsed.yes_price < 1:
                    all_markets.append(parsed)

            cursor = data.get("cursor")
            if not cursor or len(raw) < 200:
                break

        self.logger.info("kalshi_markets_by_close_date", total=len(all_markets), max_days=max_days)
        return all_markets

    async def get_public_trades(
        self,
        *,
        ticker: Optional[str] = None,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Fetch recent public trades from the trade tape.

        Returns list of dicts with: trade_id, ticker, price, count,
        taker_side ("yes"/"no"), created_time.
        """
        params: Dict[str, Any] = {"limit": min(limit, 1000)}
        if ticker:
            params["ticker"] = ticker
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts

        data = await self._get("/markets/trades", params=params)
        trades = data.get("trades", [])
        return trades

    async def get_orderbook(self, ticker: str) -> Dict[str, Any]:
        """Fetch the current orderbook for a market ticker.

        Returns dict with 'yes' and 'no' arrays of [price, quantity] levels.
        """
        data = await self._get(f"/orderbook/v2/{ticker}")
        return data.get("orderbook", data)

    async def fetch_market(self, ticker: str) -> Optional[KalshiMarket]:
        """Fetch a single market by ticker."""
        try:
            data = await self._get(f"/markets/{ticker}")
            market_data = data.get("market", data)
            return self._parse_market(market_data)
        except httpx.HTTPStatusError as exc:
            self.logger.warning("kalshi_market_not_found", ticker=ticker, status=exc.response.status_code)
            return None
