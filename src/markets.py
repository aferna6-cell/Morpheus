"""Market discovery and filtering for Polymarket."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

import httpx
import structlog
from pydantic import BaseModel, Field

from .utils import BotConfig, RateLimiter, safe_float


def parse_clob_token_ids(value) -> List[str]:
    """Gamma API often returns clobTokenIds as a JSON-encoded string."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return []
        # common case: '["id1","id2"]'
        if v.startswith("["):
            import json

            try:
                arr = json.loads(v)
                return [str(x) for x in arr]
            except Exception:
                return []
        return [v]
    return []


def parse_outcomes(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return []
        if v.startswith("["):
            import json

            try:
                arr = json.loads(v)
                return [str(x) for x in arr]
            except Exception:
                return []
        return [v]
    return []


def parse_outcome_prices(value) -> List[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [safe_float(x) for x in value]
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return []
        if v.startswith("["):
            import json

            try:
                arr = json.loads(v)
                return [safe_float(x) for x in arr]
            except Exception:
                return []
        return [safe_float(v)]
    return []


@dataclass
class TokenInfo:
    """Information about a token (YES/NO)."""

    token_id: str
    outcome: str  # "Yes" or "No"
    price: float
    volume_24h: float


@dataclass
class Market:
    """Polymarket prediction market."""

    id: str
    question: str
    description: str
    category: str
    end_date: Optional[datetime]
    volume_24h: float
    liquidity: float
    tokens: Dict[str, TokenInfo] = field(default_factory=dict)  # outcome -> TokenInfo

    # Derived properties
    yes_price: Optional[float] = None
    no_price: Optional[float] = None
    spread: Optional[float] = None
    time_to_close_hours: Optional[float] = None

    def __post_init__(self) -> None:
        """Calculate derived properties."""
        # tolerate different casing for outcomes
        for k, tok in self.tokens.items():
            if k.lower() == "yes":
                self.yes_price = tok.price
            if k.lower() == "no":
                self.no_price = tok.price

        if self.yes_price is not None and self.no_price is not None:
            self.spread = abs(1.0 - (self.yes_price + self.no_price))

        now = datetime.now(timezone.utc)
        if self.end_date:
            delta = self.end_date - now
            self.time_to_close_hours = delta.total_seconds() / 3600

    def get(self, key: str, default=None):
        """Dict-like access for compatibility with dependency module."""
        return getattr(self, key, default)

    @property
    def midpoint_price(self) -> Optional[float]:
        """Approximate "market price" for the YES outcome.

        Gamma provides outcomePrices; we treat YES price as the market probability.
        """
        return self.yes_price

    @property
    def is_active(self) -> bool:
        """Check if market is still active (not closed)."""
        now = datetime.now(timezone.utc)
        return self.end_date > now if self.end_date else True


class MarketFilters(BaseModel):
    """Filters for market discovery."""

    min_volume_24h: float = Field(default=1000.0)
    min_liquidity: float = Field(default=500.0)
    exclude_closing_within_hours: int = Field(default=6)
    categories: List[str] = Field(default_factory=list)
    max_markets_per_scan: int = Field(default=50)
    max_spread: float = Field(default=0.1)  # Max spread between YES + NO prices


class MarketDiscovery:
    """Discovers and filters Polymarket markets."""

    def __init__(self, config: BotConfig):
        """Initialize market discovery."""
        self.config = config
        self.logger = structlog.get_logger()
        self.gamma_url = config.polymarket["gamma_url"]

        # Rate limiting for Gamma API
        timing_config = config.timing
        rate_limit = timing_config.get("rate_limit_per_minute", 30)
        self.rate_limiter = RateLimiter(rate_limit, 60.0)

        # HTTP client
        timeout = httpx.Timeout(30.0, connect=10.0)
        self.client = httpx.AsyncClient(timeout=timeout)

        # Cache for market data
        self._market_cache: Dict[str, Market] = {}
        self._last_update: Optional[datetime] = None
        self._update_interval = timedelta(
            minutes=timing_config.get("market_scan_interval_minutes", 15)
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def get_markets(
        self, filters: Optional[MarketFilters] = None, force_refresh: bool = False
    ) -> List[Market]:
        """Get filtered list of markets."""
        if filters is None:
            filters = MarketFilters(**self.config.market_filters)

        # Check if we need to refresh cache
        now = datetime.now(timezone.utc)
        if (
            force_refresh
            or self._last_update is None
            or now - self._last_update > self._update_interval
        ):
            await self._update_market_cache(filters)

        # Apply filters and return
        filtered_markets = []
        for market in self._market_cache.values():
            if self._passes_filters(market, filters):
                filtered_markets.append(market)

        # Sort by opportunity score (volume * (1 - spread))
        filtered_markets.sort(
            key=lambda m: m.volume_24h * (1 - (m.spread or 0.5)), reverse=True
        )

        # Limit number of markets
        return filtered_markets[: filters.max_markets_per_scan]

    async def get_market_by_id(self, market_id: str) -> Optional[Market]:
        """Get a specific market by ID."""
        if market_id in self._market_cache:
            return self._market_cache[market_id]

        # Try to fetch from API
        try:
            await self.rate_limiter.acquire()
            response = await self.client.get(f"{self.gamma_url}/markets/{market_id}")
            response.raise_for_status()

            market_data = response.json()
            market = self._parse_market(market_data)
            if market:
                self._market_cache[market_id] = market
                return market

        except Exception as e:
            self.logger.error(
                "Failed to fetch market", market_id=market_id, error=str(e)
            )

        return None

    async def _update_market_cache(self, filters: MarketFilters) -> None:
        """Update the market cache from Gamma API."""
        try:
            self.logger.info("Updating market cache")

            # Build query parameters
            params = {
                "limit": min(
                    filters.max_markets_per_scan * 2, 100
                ),  # Get more than needed for filtering
                "active": "true",
                "closed": "false",
            }

            # Add category filters
            if filters.categories:
                params["category"] = ",".join(filters.categories)

            await self.rate_limiter.acquire()
            response = await self.client.get(f"{self.gamma_url}/markets", params=params)
            response.raise_for_status()

            data = response.json()
            markets = data.get("data", []) if isinstance(data, dict) else data

            # Parse and cache markets
            new_markets = 0
            for market_data in markets:
                market = self._parse_market(market_data)
                if market and market.is_active:
                    self._market_cache[market.id] = market
                    new_markets += 1

            self._last_update = datetime.now(timezone.utc)
            self.logger.info(
                "Market cache updated",
                new_markets=new_markets,
                total_cached=len(self._market_cache),
            )

        except Exception as e:
            self.logger.error("Failed to update market cache", error=str(e))

    def _parse_market(self, market_data: Dict) -> Optional[Market]:
        """Parse market data from Gamma API response."""
        try:
            # Extract basic market info
            market_id = market_data.get("id")
            if not market_id:
                return None

            question = market_data.get("question", "")
            description = market_data.get("description", "")
            category = market_data.get("category", "Other")

            # Parse end date
            end_date_str = market_data.get("endDate") or market_data.get("end_date")
            end_date = None
            if end_date_str:
                try:
                    # Handle different timestamp formats
                    if isinstance(end_date_str, str) and end_date_str.isdigit():
                        end_date = datetime.fromtimestamp(
                            int(end_date_str), tz=timezone.utc
                        )
                    elif isinstance(end_date_str, str):
                        end_date = datetime.fromisoformat(
                            end_date_str.replace("Z", "+00:00")
                        )
                    elif isinstance(end_date_str, (int, float)):
                        end_date = datetime.fromtimestamp(end_date_str, tz=timezone.utc)
                except (ValueError, OSError):
                    self.logger.warning(
                        "Failed to parse end date",
                        date=end_date_str,
                        market_id=market_id,
                    )

            # Extract volume and liquidity
            volume_24h = safe_float(market_data.get("volume24hr", 0))
            liquidity = safe_float(market_data.get("liquidity", 0))

            # Parse tokens (YES/NO outcomes)
            tokens = {}
            clob_token_ids = market_data.get("clobTokenIds", [])
            outcomes = market_data.get("outcomes", [])

            for i, outcome_info in enumerate(outcomes):
                if i >= len(clob_token_ids):
                    break

                token_id = clob_token_ids[i]
                outcome_name = outcome_info.get("outcome", f"Outcome {i}")
                price = safe_float(outcome_info.get("price", 0))
                volume = safe_float(outcome_info.get("volume24hr", 0))

                tokens[outcome_name] = TokenInfo(
                    token_id=token_id,
                    outcome=outcome_name,
                    price=price,
                    volume_24h=volume,
                )

            return Market(
                id=market_id,
                question=question,
                description=description,
                category=category,
                end_date=end_date,
                volume_24h=volume_24h,
                liquidity=liquidity,
                tokens=tokens,
            )

        except Exception as e:
            self.logger.error(
                "Failed to parse market", error=str(e), market_data=market_data
            )
            return None

    def _passes_filters(self, market: Market, filters: MarketFilters) -> bool:
        """Check if market passes all filters."""
        # Volume filter
        if market.volume_24h < filters.min_volume_24h:
            return False

        # Liquidity filter
        if market.liquidity < filters.min_liquidity:
            return False

        # Category filter
        if filters.categories and market.category not in filters.categories:
            return False

        # Time to close filter
        if (
            market.time_to_close_hours is not None
            and market.time_to_close_hours < filters.exclude_closing_within_hours
        ):
            return False

        # Spread filter (if we have price data)
        if market.spread is not None and market.spread > filters.max_spread:
            return False

        # Must have at least YES token
        if "Yes" not in market.tokens and "YES" not in market.tokens:
            return False

        return True

    def rank_markets_by_opportunity(self, markets: List[Market]) -> List[Market]:
        """Rank markets by opportunity score."""

        def opportunity_score(market: Market) -> float:
            """Calculate opportunity score for a market."""
            # Base score from volume
            score = market.volume_24h

            # Bonus for tight spreads
            if market.spread is not None:
                score *= 1.0 + (0.1 - market.spread) * 10  # Tighter spreads get bonus

            # BONUS for markets closing soon — we want same-day, fast turnover
            if market.time_to_close_hours is not None:
                if market.time_to_close_hours < 3:
                    score *= 2.0   # closing very soon — top priority
                elif market.time_to_close_hours < 8:
                    score *= 1.5   # closing today — high priority
                elif market.time_to_close_hours < 24:
                    score *= 1.2   # closing within a day

            return max(0.0, score)

        return sorted(markets, key=opportunity_score, reverse=True)


def market_from_dict(d: Dict) -> Market:
    """Rehydrate a Market from a JSON snapshot (used in replay mode)."""

    end_date = d.get("end_date")
    if isinstance(end_date, str) and end_date:
        try:
            end_date = datetime.fromisoformat(end_date)
        except Exception:
            end_date = None

    tokens_raw = d.get("tokens") or {}
    tokens: Dict[str, TokenInfo] = {}
    for outcome, t in tokens_raw.items():
        if isinstance(t, TokenInfo):
            tokens[outcome] = t
            continue
        t = t or {}
        tokens[outcome] = TokenInfo(
            token_id=str(t.get("token_id") or t.get("tokenId") or ""),
            outcome=str(t.get("outcome") or outcome),
            price=float(t.get("price") or 0.0),
            volume_24h=float(t.get("volume_24h") or t.get("volume24hr") or 0.0),
        )

    return Market(
        id=str(d.get("id")),
        question=d.get("question", ""),
        description=d.get("description", ""),
        category=d.get("category", "Other"),
        end_date=end_date,
        volume_24h=float(d.get("volume_24h") or 0.0),
        liquidity=float(d.get("liquidity") or 0.0),
        tokens=tokens,
    )
