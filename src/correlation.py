"""Correlated market detection for Morpheus.

Markets like "shutdown lasts 7+ days" and "shutdown lasts 14+ days" are
structurally correlated — if one hits, the other likely does too.  Stacking
bets on all of them is effectively one big concentrated bet disguised as
diversification.

This module groups correlated markets and limits exposure to one (the best)
per cluster.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import structlog

from .markets import Market

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

# Tokens to strip before comparing questions (noise words)
_STRIP_RE = re.compile(
    r"\b(will|the|a|an|by|before|after|in|on|of|or|more|than|at|least|be)\b",
    re.IGNORECASE,
)

# Patterns that indicate parameterised variants of the same underlying event
# e.g. "shutdown last 7 days" vs "shutdown last 14 days"
_NUMBER_RE = re.compile(r"\d+[\d,.]*")


def _normalise(text: str) -> str:
    """Lowercase, strip noise words and numbers for fuzzy comparison."""
    text = _NUMBER_RE.sub("_NUM_", text.lower())
    text = _STRIP_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _similarity(a: str, b: str) -> float:
    """Return 0-1 similarity score between two normalised question strings."""
    return SequenceMatcher(None, a, b).ratio()


def _extract_stem(question: str) -> str:
    """Extract the 'stem' of a question by replacing numbers with placeholder.

    'Will the shutdown last 7 days or more?' -> 'will the shutdown last _NUM_ days or more?'

    Markets with identical stems are almost certainly correlated.
    """
    return _NUMBER_RE.sub("_NUM_", question.lower()).strip()


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

@dataclass
class MarketCluster:
    """A group of correlated markets."""
    stem: str
    markets: List[Market] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.markets)


def cluster_markets(
    markets: List[Market],
    *,
    stem_threshold: float = 0.85,
    similarity_threshold: float = 0.80,
) -> List[MarketCluster]:
    """Group markets into correlation clusters.

    Two-pass approach:
    1. Exact stem match (same question with different numbers) — very reliable
    2. Fuzzy similarity on normalised text — catches rephrased variants

    Returns clusters with 2+ markets. Uncorrelated markets are not returned.
    """
    # Pass 1: stem-based grouping
    stem_groups: Dict[str, List[Market]] = defaultdict(list)
    for m in markets:
        stem = _extract_stem(m.question)
        stem_groups[stem].append(m)

    # Collect clusters from stem groups
    clustered_ids: set = set()
    clusters: List[MarketCluster] = []

    for stem, group in stem_groups.items():
        if len(group) >= 2:
            cluster = MarketCluster(stem=stem, markets=list(group))
            clusters.append(cluster)
            for m in group:
                clustered_ids.add(m.id)

    # Pass 2: fuzzy match on remaining unclustered markets
    unclustered = [m for m in markets if m.id not in clustered_ids]
    normalised = [(m, _normalise(m.question)) for m in unclustered]

    # Also try to merge unclustered into existing clusters
    for m, norm_q in normalised:
        best_score = 0.0
        best_cluster: Optional[MarketCluster] = None
        for cluster in clusters:
            for cm in cluster.markets:
                score = _similarity(norm_q, _normalise(cm.question))
                if score > best_score:
                    best_score = score
                    best_cluster = cluster
        if best_score >= similarity_threshold and best_cluster is not None:
            best_cluster.markets.append(m)
            clustered_ids.add(m.id)

    # Pairwise fuzzy on remaining
    still_unclustered = [
        (m, norm) for m, norm in normalised if m.id not in clustered_ids
    ]
    used = set()
    for i, (m1, n1) in enumerate(still_unclustered):
        if m1.id in used:
            continue
        group = [m1]
        for j in range(i + 1, len(still_unclustered)):
            m2, n2 = still_unclustered[j]
            if m2.id in used:
                continue
            if _similarity(n1, n2) >= similarity_threshold:
                group.append(m2)
                used.add(m2.id)
        if len(group) >= 2:
            clusters.append(MarketCluster(
                stem=_normalise(m1.question),
                markets=group,
            ))
            used.add(m1.id)

    return clusters


# ---------------------------------------------------------------------------
# Selection: pick the best market from each cluster
# ---------------------------------------------------------------------------

def pick_best_per_cluster(
    clusters: List[MarketCluster],
    max_per_cluster: int = 1,
) -> Tuple[Dict[str, bool], Dict[str, str]]:
    """For each cluster, pick the best market(s) to trade.

    Selection criteria (in priority order):
    1. Highest liquidity (more reliable pricing, easier execution)
    2. Highest 24h volume (more active, tighter spreads)
    3. Soonest resolution (shorter time horizon = more predictable)

    Returns:
        allowed: dict of market_id -> True for markets allowed to trade
        reasons: dict of market_id -> skip reason for blocked markets
    """
    allowed: Dict[str, bool] = {}
    reasons: Dict[str, str] = {}

    for cluster in clusters:
        if cluster.size <= 1:
            # Single market, always allowed
            for m in cluster.markets:
                allowed[m.id] = True
            continue

        # Sort: highest liquidity first, then volume, then soonest end_date
        ranked = sorted(
            cluster.markets,
            key=lambda m: (
                -(m.liquidity or 0),
                -(m.volume_24h or 0),
                (m.end_date or datetime_max()).timestamp(),
            ),
        )

        selected = ranked[:max_per_cluster]
        blocked = ranked[max_per_cluster:]

        for m in selected:
            allowed[m.id] = True

        for m in blocked:
            best = selected[0]
            reasons[m.id] = (
                f"Correlated with '{best.question[:60]}' (id={best.id}), "
                f"which has higher liquidity/volume"
            )

        logger.info(
            "correlation_cluster",
            cluster_size=cluster.size,
            selected=[m.id for m in selected],
            blocked=[m.id for m in blocked],
            stem=cluster.stem[:80],
        )

    return allowed, reasons


def datetime_max():
    """Far-future datetime for sorting."""
    from datetime import datetime, timezone
    return datetime(2099, 12, 31, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Main filter function for use in the trading loop
# ---------------------------------------------------------------------------

def filter_correlated(
    markets: List[Market],
    *,
    max_per_cluster: int = 1,
    similarity_threshold: float = 0.80,
) -> Tuple[List[Market], Dict[str, str]]:
    """Filter a market list, keeping only the best from each correlated group.

    Returns:
        filtered_markets: markets safe to trade (uncorrelated + best per cluster)
        skip_reasons: market_id -> reason for markets that were filtered out
    """
    clusters = cluster_markets(
        markets,
        similarity_threshold=similarity_threshold,
    )

    if not clusters:
        return markets, {}

    allowed, reasons = pick_best_per_cluster(clusters, max_per_cluster=max_per_cluster)

    # All market IDs that appear in any cluster
    clustered_ids = set()
    for c in clusters:
        for m in c.markets:
            clustered_ids.add(m.id)

    # Keep: unclustered markets + allowed from clusters
    filtered = []
    for m in markets:
        if m.id not in clustered_ids:
            filtered.append(m)  # not in any cluster, keep
        elif allowed.get(m.id):
            filtered.append(m)  # best in its cluster, keep
        # else: blocked by correlation filter

    total_blocked = len(markets) - len(filtered)
    if total_blocked > 0:
        logger.info(
            "correlation_filter_summary",
            total_markets=len(markets),
            clusters_found=len(clusters),
            markets_blocked=total_blocked,
            markets_passed=len(filtered),
        )

    return filtered, reasons
