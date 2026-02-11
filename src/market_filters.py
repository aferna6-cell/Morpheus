"""Market filtering logic — strict quality gates.

Filters:
- Volume ≥ $50k
- Bid-ask spread ≤ 5%
- Resolution ≤ 30 days
- Data sources ≥ 3
- Sports blocking (kill pre-match, allow only edge cases)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

import structlog

from .utils import BotConfig


@dataclass
class FilterResult:
    """Result of market filtering."""
    passed: bool
    reason: str
    market_id: str


# Ticker prefixes that indicate junk markets where LLMs have no edge.
# These are checked before any other filter to save LLM budget.
_JUNK_TICKER_PREFIXES = [
    # Announcer/host mention markets — no LLM edge on what people will say
    "KXNBAMENTION", "KXNCAAB", "KXNFLMENTION", "KXFOXNEWSMENTION",
    "KXMLBMENTION", "KXNHLMENTION", "KXMLSMENTION",
    # Crypto price ranges (daily + intraday + bracket) — LLMs have no edge
    "KXBTCD", "KXBTC15M", "KXBTC",
    "KXETHD", "KXETH15M", "KXETH",
    "KXDOGE", "KXDOGED", "KXSOLD", "KXSOLE", "KXSOL15M",
    "KXBNBD", "KXLTCD", "KXADAD", "KXDOTD", "KXAVAXD", "KXLINKD",
    "KXMATD", "KXUNIDD", "KXSHIB", "KXXRPD",
    # Weather markets — NOW UNBLOCKED (NOAA NWS forecast anchors added)
    # "KXHIGH", "KXLOW", "KXRAIN", "KXSNOW", "KXTEMP",
    # Trump mention markets — unpredictable speech patterns
    "KXTRUMPMENTION",
    # Word/phrase mention markets
    "KXWOMENTION", "KXWMENTION",
    # Stock intraday ranges
    "KXSPY", "KXQQQ", "KXIWM", "KXDIA",
    # Entertainment / pop culture — LLMs have no edge on celebrity/media outcomes
    "KXSUPERBOWLAD", "KXRT", "KXSPOTIFY", "KXSPOTIFYD", "KXSPOTIFYGLOBALD",
    "KXSBADAPPEARANCES", "KXTOPSONG", "KXTOPALBUM", "KXALBUMDEBUT",
    "KXFIRSTSUPERBOWLSONG", "KXAAAGASW", "KXNEXTTEAMNFL",
    # Racing — LLMs have no edge on race outcomes
    "KXNASCARRACE", "KXNASCAR", "KXF1RACE", "KXINDYRACE",
    # LLM mention markets — what will an AI chatbot say? No edge.
    "KXLLM",
    # NBA All-Stars / draft — sports-adjacent entertainment
    "KXNBAALLSTAR",
    # Executive order / government action — highly unpredictable timing
    "KXEOWEEK", "KXTRUMPACT", "KXEXECORDER",
    # Person-specific mention markets
    "KXVLADTENEV", "KXELONMENTION",
]


class MarketFilters:
    """Applies strict market quality filters."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = structlog.get_logger()

        # Market filter config
        mf = getattr(config, "market_filters", {}) or {}
        if not isinstance(mf, dict):
            mf = {}

        self.min_volume = float(mf.get("min_volume_24h", 50000))
        self.max_spread_pct = float(mf.get("max_spread_pct", 0.05))
        self.max_resolution_days = int(mf.get("max_resolution_days", 30))
        self.min_data_sources = int(mf.get("min_data_sources", 3))
        self.min_price = float(mf.get("min_price", 0.05))
        self.max_price = float(mf.get("max_price", 0.95))

        # Ticker prefix blocklist (from config or default)
        self.blocked_ticker_prefixes = list(_JUNK_TICKER_PREFIXES)
        extra_blocked = mf.get("blocked_ticker_prefixes", [])
        if extra_blocked:
            self.blocked_ticker_prefixes.extend(extra_blocked)

        # Sports filter config
        sf = getattr(config, "sports_filters", {}) or {}
        if not isinstance(sf, dict):
            sf = {}

        self.sports_enabled = bool(sf.get("enabled", True))
        self.blocked_categories = [
            c.lower() for c in sf.get("blocked_categories", [
                "Sports", "Soccer", "Football", "Basketball", "Baseball",
                "Hockey", "Tennis", "Golf", "MMA", "Boxing", "Racing"
            ])
        ]
        self.blocked_patterns = sf.get("blocked_patterns", [
            "Winner", "vs", "TIE", "DRAW", "Underdog", "to Win",
            "to Beat", "Advance", "Champion", "MVP", "Gold Medal"
        ])
        self.allowed_live_only = bool(sf.get("allowed_live_only", True))
        self.allowed_patterns = [
            p.lower() for p in sf.get("allowed_patterns", [
                "red card", "injury", "ejection", "weather delay", "postponed"
            ])
        ]

    def check_volume(
        self,
        market_id: str,
        volume: float,
    ) -> FilterResult:
        """Check if volume meets minimum threshold."""
        if volume < self.min_volume:
            return FilterResult(
                passed=False,
                reason=f"Volume ${volume:,.0f} < ${self.min_volume:,.0f} minimum",
                market_id=market_id,
            )
        return FilterResult(passed=True, reason="ok", market_id=market_id)

    def check_spread(
        self,
        market_id: str,
        bid: float,
        ask: float,
    ) -> FilterResult:
        """Check if bid-ask spread is within tolerance."""
        if bid <= 0 or ask <= 0:
            return FilterResult(
                passed=False,
                reason="Invalid bid/ask prices",
                market_id=market_id,
            )

        spread = (ask - bid) / ((ask + bid) / 2)
        if spread > self.max_spread_pct:
            return FilterResult(
                passed=False,
                reason=f"Spread {spread:.1%} > {self.max_spread_pct:.1%} max",
                market_id=market_id,
            )
        return FilterResult(passed=True, reason="ok", market_id=market_id)

    def check_resolution_time(
        self,
        market_id: str,
        close_time: Optional[datetime],
    ) -> FilterResult:
        """Check if market resolves within allowed window."""
        if close_time is None:
            return FilterResult(
                passed=False,
                reason="No close time specified",
                market_id=market_id,
            )

        now = datetime.now(timezone.utc)
        days_to_close = (close_time - now).total_seconds() / 86400

        if days_to_close > self.max_resolution_days:
            return FilterResult(
                passed=False,
                reason=f"Resolves in {days_to_close:.0f} days > {self.max_resolution_days} max",
                market_id=market_id,
            )
        if days_to_close < 0:
            return FilterResult(
                passed=False,
                reason="Market already closed",
                market_id=market_id,
            )
        return FilterResult(passed=True, reason="ok", market_id=market_id)

    def check_sports(
        self,
        market_id: str,
        title: str,
        category: str,
        is_live: bool = False,
    ) -> FilterResult:
        """Check sports market rules — kill most, allow edge cases."""
        if not self.sports_enabled:
            return FilterResult(passed=True, reason="sports_filter_disabled", market_id=market_id)

        title_lower = title.lower()
        category_lower = category.lower() if category else ""

        # Check if it's a sports category
        is_sports = any(cat in category_lower for cat in self.blocked_categories)

        # Also detect sports by title patterns (Kalshi sometimes miscategorizes)
        sports_title_patterns = [
            r"\bvs\b.*\b(win|winner)\b",
            r"\b(nba|nfl|mlb|nhl|mls|uefa|fifa|atp|wta|pga)\b",
            r"\b(game|match|bout|fight|race)\b.*\b(winner|result)\b",
            r"\b(team|club)\b.*\b(beat|defeat|win)\b",
        ]
        for pattern in sports_title_patterns:
            if re.search(pattern, title_lower):
                is_sports = True
                break

        if not is_sports:
            return FilterResult(passed=True, reason="not_sports", market_id=market_id)

        # It's sports — check if it's an allowed edge case
        for allowed in self.allowed_patterns:
            if allowed in title_lower:
                if self.allowed_live_only and not is_live:
                    return FilterResult(
                        passed=False,
                        reason=f"Sports edge case '{allowed}' but not live",
                        market_id=market_id,
                    )
                return FilterResult(
                    passed=True,
                    reason=f"Sports edge case: {allowed}",
                    market_id=market_id,
                )

        # Check blocked patterns (always skip)
        for blocked in self.blocked_patterns:
            if blocked.lower() in title_lower:
                return FilterResult(
                    passed=False,
                    reason=f"Blocked sports pattern: {blocked}",
                    market_id=market_id,
                )

        # Generic sports market without edge case — skip
        return FilterResult(
            passed=False,
            reason=f"Sports market without allowed edge case (category: {category})",
            market_id=market_id,
        )

    def check_price(
        self,
        market_id: str,
        bid: float,
        ask: float,
    ) -> FilterResult:
        """Skip markets with extreme prices where LLM has no edge."""
        if bid <= 0 or ask <= 0:
            return FilterResult(passed=True, reason="ok", market_id=market_id)

        mid = (bid + ask) / 2
        if mid < self.min_price:
            return FilterResult(
                passed=False,
                reason=f"Price {mid:.0%} below {self.min_price:.0%} min (too certain NO)",
                market_id=market_id,
            )
        if mid > self.max_price:
            return FilterResult(
                passed=False,
                reason=f"Price {mid:.0%} above {self.max_price:.0%} max (too certain YES)",
                market_id=market_id,
            )
        return FilterResult(passed=True, reason="ok", market_id=market_id)

    def check_ticker_prefix(self, market_id: str) -> FilterResult:
        """Block markets by ticker prefix (junk markets with no LLM edge)."""
        ticker_upper = market_id.upper()
        for prefix in self.blocked_ticker_prefixes:
            if ticker_upper.startswith(prefix.upper()):
                return FilterResult(
                    passed=False,
                    reason=f"Blocked ticker prefix: {prefix}",
                    market_id=market_id,
                )
        return FilterResult(passed=True, reason="ok", market_id=market_id)

    def check_all(
        self,
        market_id: str,
        title: str,
        category: str,
        volume: float,
        bid: float,
        ask: float,
        close_time: Optional[datetime],
        is_live: bool = False,
    ) -> FilterResult:
        """Run all filters and return first failure or success."""
        # Ticker prefix check first — cheapest filter, saves LLM budget
        ticker_result = self.check_ticker_prefix(market_id)
        if not ticker_result.passed:
            self.logger.debug(
                "market_filtered",
                market_id=market_id,
                reason=ticker_result.reason,
            )
            return ticker_result

        checks = [
            self.check_price(market_id, bid, ask),
            self.check_volume(market_id, volume),
            self.check_spread(market_id, bid, ask),
            self.check_resolution_time(market_id, close_time),
            self.check_sports(market_id, title, category, is_live),
        ]

        for result in checks:
            if not result.passed:
                self.logger.debug(
                    "market_filtered",
                    market_id=market_id,
                    reason=result.reason,
                )
                return result

        return FilterResult(passed=True, reason="all_checks_passed", market_id=market_id)


def count_data_sources(news_articles: List[dict]) -> int:
    """Count unique data sources from news articles.

    Groups by domain to avoid counting multiple articles from
    the same source (e.g., 5 Reuters articles = 1 source).
    """
    if not news_articles:
        return 0

    domains = set()
    for article in news_articles:
        url = article.get("url", "") or article.get("link", "")
        if url:
            # Extract domain from URL
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                # Normalize: remove www., get base domain
                domain = domain.replace("www.", "")
                # Get second-level domain (e.g., "reuters.com" not "uk.reuters.com")
                parts = domain.split(".")
                if len(parts) >= 2:
                    domain = ".".join(parts[-2:])
                domains.add(domain)
            except Exception:
                pass

    return len(domains)
