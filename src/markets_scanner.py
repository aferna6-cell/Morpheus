"""Market scanning helpers (Gamma API + enrichment)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Set

import httpx
import structlog

from .markets import (
    Market,
    TokenInfo,
    parse_clob_token_ids,
    parse_outcome_prices,
    parse_outcomes,
)
from .utils import BotConfig, RateLimiter, safe_float


# ---------------------------------------------------------------------------
# Focus filter — keyword-based domain matching
# ---------------------------------------------------------------------------

POLITICS_KEYWORDS: Set[str] = {
    "trump", "biden", "president", "congress", "senate", "house",
    "republican", "democrat", "gop", "election", "nomination", "nominate",
    "governor", "mayor", "impeach", "veto", "executive order",
    "supreme court", "scotus", "legislation", "bill", "law",
    "government shutdown", "shutdown", "debt ceiling",
    "cabinet", "secretary", "ambassador", "confirmation",
    "primaryelection", "midterm", "inauguration", "pardon",
    "sanctions", "tariff", "trade war", "foreign policy",
    "nato", "un ", "united nations", "war ", "strike ",
    "ceasefire", "treaty", "diplomatic", "geopolit",
}

ECONOMICS_KEYWORDS: Set[str] = {
    "fed ", "federal reserve", "interest rate", "rate cut", "rate hike",
    "inflation", "cpi", "ppi", "gdp", "recession",
    "unemployment", "jobs report", "nonfarm", "payroll",
    "treasury", "bond", "yield curve",
    "fomc", "powell", "monetary policy", "quantitative",
    "fiscal", "deficit", "debt", "stimulus",
    "trade deficit", "import", "export",
    "stock market", "s&p", "dow jones", "nasdaq",
    "housing", "mortgage", "real estate",
    "oil price", "opec", "energy",
    "banking", "bank", "svb", "fdic",
}


def _matches_focus(question: str, description: str, category: str) -> bool:
    """Return True if the market matches politics or economics focus."""
    text = f"{question} {description} {category}".lower()

    for kw in POLITICS_KEYWORDS:
        if kw in text:
            return True
    for kw in ECONOMICS_KEYWORDS:
        if kw in text:
            return True
    return False


class MarketScanner:
    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = structlog.get_logger()
        self.gamma_url = config.polymarket.get(
            "gamma_url", "https://gamma-api.polymarket.com"
        )
        self.rate_limiter = RateLimiter(
            config.timing.get("rate_limit_per_minute", 30), 60.0
        )
        self.client = httpx.AsyncClient(timeout=20.0)
        self._previous_prices: dict[str, float] = {}  # market_id -> last midpoint

    async def fetch_markets(self) -> List[Market]:
        filters = self.config.market_filters
        self.focus_enabled = bool(filters.get("focus_enabled", True))
        # Pull more from API since we'll filter many out
        api_limit = int(filters.get("max_markets_per_scan", 200))
        if self.focus_enabled:
            api_limit = max(api_limit, 200)  # cast wider net, filter locally

        # Gamma API caps at ~100 per request — paginate to get more
        page_size = min(api_limit, 100)
        markets: list = []
        offset = 0

        while len(markets) < api_limit:
            params = {
                "active": "true",
                "closed": "false",
                "limit": page_size,
                "offset": offset,
                "order": "volume24hr",
                "ascending": "false",
            }
            await self.rate_limiter.acquire()
            r = await self.client.get(f"{self.gamma_url}/markets", params=params)
            r.raise_for_status()
            raw = r.json()
            page = raw.get("data", []) if isinstance(raw, dict) else raw
            if not page:
                break
            markets.extend(page)
            offset += len(page)
            if len(page) < page_size:
                break  # no more pages

        out: List[Market] = []
        for m in markets:
            try:
                end = None
                end_str = m.get("endDate")
                if isinstance(end_str, str) and end_str:
                    end = datetime.fromisoformat(
                        end_str.replace("Z", "+00:00")
                    ).astimezone(timezone.utc)

                clob_ids = parse_clob_token_ids(m.get("clobTokenIds"))
                outcomes = parse_outcomes(m.get("outcomes"))
                prices = parse_outcome_prices(m.get("outcomePrices"))

                tokens = {}
                for i, outcome in enumerate(outcomes):
                    if i >= len(clob_ids):
                        break
                    price = prices[i] if i < len(prices) else 0.0
                    tokens[outcome] = TokenInfo(
                        token_id=str(clob_ids[i]),
                        outcome=outcome,
                        price=price,
                        volume_24h=safe_float(m.get("volume24hr", 0)),
                    )

                market = Market(
                    id=str(m.get("id")),
                    question=m.get("question", ""),
                    description=m.get("description", ""),
                    category=m.get("category", "Other") or "Other",
                    end_date=end,
                    volume_24h=safe_float(m.get("volume24hr", 0)),
                    liquidity=safe_float(m.get("liquidity", 0)),
                    tokens=tokens,
                )

                # apply basic filters
                if market.volume_24h < float(filters.get("min_volume_24h", 1000)):
                    continue
                if market.liquidity < float(filters.get("min_liquidity", 500)):
                    continue
                if (
                    market.time_to_close_hours is not None
                    and market.time_to_close_hours
                    < int(filters.get("exclude_closing_within_hours", 6))
                ):
                    continue

                # Price range filter: skip extreme markets with no edge opportunity
                mid = market.midpoint_price
                min_price = float(filters.get("min_price", 0.03))
                max_price = float(filters.get("max_price", 0.97))
                if mid is not None and (mid < min_price or mid > max_price):
                    continue

                # Focus filter: only politics + economics
                if self.focus_enabled and not _matches_focus(
                    market.question, market.description, market.category
                ):
                    continue

                out.append(market)
            except Exception:
                continue

        # Resolution timeline filter: skip markets too far out
        max_res_days = int(filters.get("max_resolution_days", 90))
        now = datetime.now(timezone.utc)
        filtered = []
        for m in out:
            if m.end_date:
                days_out = (m.end_date - now).total_seconds() / 86400.0
                if days_out > max_res_days:
                    continue
            filtered.append(m)
        out = filtered

        # Price movement detection
        price_move_pct = float(filters.get("price_move_alert_pct", 0.05))
        movers = []
        rest = []
        for m in out:
            mp = m.midpoint_price
            if mp is not None and m.id in self._previous_prices:
                prev = self._previous_prices[m.id]
                if prev > 0 and abs(mp - prev) / prev > price_move_pct:
                    self.logger.info(
                        "price_movement_detected",
                        market_id=m.id,
                        question=m.question[:80],
                        prev_price=prev,
                        new_price=mp,
                        change_pct=abs(mp - prev) / prev,
                    )
                    movers.append(m)
                else:
                    rest.append(m)
            else:
                rest.append(m)
            if mp is not None:
                self._previous_prices[m.id] = mp

        # Sort by resolution proximity (sooner = higher priority)
        def _res_sort_key(m: Market) -> float:
            if m.end_date is None:
                return 999999.0
            return max(0, (m.end_date - now).total_seconds() / 86400.0)

        movers.sort(key=_res_sort_key)
        rest.sort(key=_res_sort_key)
        out = movers + rest

        self.logger.info(
            "markets_scanned",
            total_from_api=len(markets),
            after_filters=len(out),
            movers=len(movers),
            focus_enabled=self.focus_enabled,
            focus="politics+economics" if self.focus_enabled else "all",
        )
        return out

    async def enrich_midpoints(
        self, clob_client, markets: List[Market]
    ) -> List[Market]:
        # For now, Gamma outcomePrices are used; placeholder if we later pull CLOB midpoint.
        return markets

    async def close(self) -> None:
        await self.client.aclose()
