"""Cross-platform market matcher: Polymarket ↔ Kalshi.

Uses fuzzy text matching on market questions/titles to find equivalent
markets across platforms.  Matches above the confidence threshold are
cached so we don't re-compute every scan cycle.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import structlog

from ..kalshi_client import KalshiMarket
from ..markets import Market as PolyMarket

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Text normalisation (shared philosophy with correlation.py)
# ---------------------------------------------------------------------------

_NOISE_RE = re.compile(
    r"\b(will|the|a|an|by|before|after|in|on|of|or|more|than|at|least|be|to|do|does|is|are|was|were)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d+[\d,.]*")
_PAREN_RE = re.compile(r"\([^)]*\)")
_EXTRA_SPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Lowercase, strip noise, normalise numbers for fuzzy comparison."""
    text = text.lower()
    text = _PAREN_RE.sub("", text)
    text = _NUMBER_RE.sub("_NUM_", text)
    text = _NOISE_RE.sub("", text)
    return _EXTRA_SPACE.sub(" ", text).strip()


def _similarity(a: str, b: str) -> float:
    """Return 0-1 SequenceMatcher ratio on normalised strings."""
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Match dataclass
# ---------------------------------------------------------------------------

@dataclass
class MarketMatch:
    """A matched pair of equivalent markets across platforms."""

    poly_market: PolyMarket
    kalshi_market: KalshiMarket
    confidence: float  # 0-1 match confidence
    matched_at: float = field(default_factory=time.time)

    @property
    def label(self) -> str:
        return (
            f"[{self.confidence:.2f}] "
            f"Poly: {self.poly_market.question[:60]} | "
            f"Kalshi: {self.kalshi_market.title[:60]}"
        )


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

class ArbMatcher:
    """Finds equivalent markets across Polymarket and Kalshi."""

    def __init__(self, *, confidence_threshold: float = 0.80, cache_ttl: float = 600.0):
        """
        Args:
            confidence_threshold: Minimum similarity to consider a match.
            cache_ttl: Seconds before a cached match is considered stale.
        """
        self.confidence_threshold = confidence_threshold
        self.cache_ttl = cache_ttl

        # Cache: (poly_id, kalshi_ticker) -> MarketMatch
        self._cache: Dict[Tuple[str, str], MarketMatch] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def _prune_stale(self) -> None:
        now = time.time()
        stale = [k for k, v in self._cache.items() if now - v.matched_at > self.cache_ttl]
        for k in stale:
            del self._cache[k]

    def match_markets(
        self,
        poly_markets: List[PolyMarket],
        kalshi_markets: List[KalshiMarket],
    ) -> List[MarketMatch]:
        """Find matching market pairs across platforms.

        For each Polymarket question, find the best matching Kalshi title.
        Only returns matches above `confidence_threshold`.

        Complexity: O(P * K) where P = len(poly), K = len(kalshi).
        For typical sizes (50-200 each) this is fine; pre-normalisation
        keeps the inner loop fast.
        """
        self._prune_stale()

        # Pre-normalise all titles
        poly_norms = [(m, _normalise(m.question)) for m in poly_markets]
        kalshi_norms = [(m, _normalise(m.title)) for m in kalshi_markets]

        matches: List[MarketMatch] = []
        used_kalshi: set = set()  # Ensure 1-to-1 matching

        # Sort poly by volume descending so higher-volume markets get first pick
        poly_norms.sort(key=lambda x: x[0].volume_24h, reverse=True)

        for pm, pn in poly_norms:
            best_score = 0.0
            best_km: Optional[KalshiMarket] = None

            for km, kn in kalshi_norms:
                if km.ticker in used_kalshi:
                    continue

                # Check cache first
                cache_key = (pm.id, km.ticker)
                cached = self._cache.get(cache_key)
                if cached is not None and (time.time() - cached.matched_at) < self.cache_ttl:
                    score = cached.confidence
                else:
                    score = _similarity(pn, kn)

                if score > best_score:
                    best_score = score
                    best_km = km

            if best_km is not None and best_score >= self.confidence_threshold:
                match = MarketMatch(
                    poly_market=pm,
                    kalshi_market=best_km,
                    confidence=best_score,
                )
                matches.append(match)
                used_kalshi.add(best_km.ticker)

                # Update cache
                self._cache[(pm.id, best_km.ticker)] = match

        logger.info(
            "arb_matcher_results",
            poly_count=len(poly_markets),
            kalshi_count=len(kalshi_markets),
            matches_found=len(matches),
            threshold=self.confidence_threshold,
        )

        return matches
