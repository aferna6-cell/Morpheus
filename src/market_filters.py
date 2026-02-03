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
        checks = [
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
