"""Polymarket event group scanner for arbitrage detection.

Fetches markets from the Gamma API and groups them by event (negRiskMarketID).
Each event group represents mutually exclusive outcomes that must sum to 1.0.

Runnable standalone:
    python3 -m src.engines.event_scanner
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import structlog

from ..markets import Market, TokenInfo, parse_clob_token_ids, parse_outcomes, parse_outcome_prices
from ..utils import BotConfig, RateLimiter, safe_float

logger = structlog.get_logger(__name__)


@dataclass
class EventGroup:
    """A group of markets belonging to the same event (neg_risk group).

    All markets in the group are mutually exclusive outcomes — exactly one
    resolves YES, the rest resolve NO.  Their YES prices should sum to 1.0.
    """

    event_id: str
    event_slug: str
    markets: List[Market] = field(default_factory=list)

    # Pre-computed aggregates (set after markets are added)
    total_yes_sum: float = 0.0
    total_no_sum: float = 0.0
    arb_profit: float = 0.0  # 1.0 - total_yes_sum (if positive → free money)

    def recompute(self) -> None:
        """Recompute aggregate pricing from current market list."""
        self.total_yes_sum = 0.0
        for m in self.markets:
            if m.yes_price is not None:
                self.total_yes_sum += m.yes_price
        n = len(self.markets)
        # Sum of all NO prices = n - sum_yes (each NO = 1 - YES in a fair market)
        self.total_no_sum = n - self.total_yes_sum
        self.arb_profit = 1.0 - self.total_yes_sum


class EventScanner:
    """Fetch Polymarket markets grouped by event for arbitrage scanning.

    The Gamma API exposes a ``neg_risk`` flag and ``negRiskMarketID`` field
    that groups related markets under a single event.  E.g. an event
    "Trump Electoral Votes" might contain markets for each vote bracket.

    Parameters
    ----------
    config : BotConfig
        Bot configuration (used to resolve gamma_url and rate limits).
    """

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.gamma_url = config.polymarket.get(
            "gamma_url", "https://gamma-api.polymarket.com"
        )
        rate_limit = config.timing.get("rate_limit_per_minute", 30)
        self.rate_limiter = RateLimiter(rate_limit, 60.0)
        self.client = httpx.AsyncClient(timeout=30.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_event_groups(self) -> Dict[str, EventGroup]:
        """Fetch neg_risk markets from Gamma API, group by negRiskMarketID.

        Returns
        -------
        dict
            Mapping of event_id → :class:`EventGroup`.  Only groups with
            ≥ 2 markets are included (single-outcome events aren't useful
            for event-group arbitrage).
        """
        raw_markets = await self.fetch_all_markets_paginated()
        groups: Dict[str, EventGroup] = {}

        for raw in raw_markets:
            neg_risk_id = raw.get("negRiskMarketID") or raw.get("neg_risk_market_id")
            if not neg_risk_id:
                continue

            market = self._parse_raw_market(raw)
            if market is None:
                continue

            if neg_risk_id not in groups:
                slug = raw.get("groupItemTitle") or raw.get("question", "")[:60]
                groups[neg_risk_id] = EventGroup(
                    event_id=neg_risk_id,
                    event_slug=slug,
                )

            groups[neg_risk_id].markets.append(market)

        # Recompute aggregates and filter out trivial groups
        result: Dict[str, EventGroup] = {}
        for eid, grp in groups.items():
            if len(grp.markets) < 2:
                continue
            grp.recompute()
            result[eid] = grp

        logger.info(
            "event_groups_fetched",
            raw_markets=len(raw_markets),
            event_groups=len(result),
            total_grouped_markets=sum(len(g.markets) for g in result.values()),
        )
        return result

    async def fetch_all_markets_paginated(self, limit: int = 2000) -> List[Dict[str, Any]]:
        """Paginate through the Gamma API to get ALL active neg_risk markets.

        Unlike the regular scanner which caps at ~200 by volume, we need
        complete coverage for arbitrage detection across all event groups.

        Parameters
        ----------
        limit : int
            Maximum total markets to fetch (safety cap).
        """
        page_size = 100  # Gamma API max per request
        all_markets: List[Dict[str, Any]] = []
        offset = 0

        while len(all_markets) < limit:
            params = {
                "active": "true",
                "closed": "false",
                "neg_risk": "true",
                "limit": page_size,
                "offset": offset,
            }
            try:
                await self.rate_limiter.acquire()
                resp = await self.client.get(
                    f"{self.gamma_url}/markets", params=params
                )
                resp.raise_for_status()
                data = resp.json()
                page = data.get("data", []) if isinstance(data, dict) else data
            except Exception as exc:
                logger.warning(
                    "event_scanner_page_error",
                    offset=offset,
                    error=str(exc),
                )
                break

            if not page:
                break

            all_markets.extend(page)
            offset += len(page)

            if len(page) < page_size:
                break  # last page

        logger.debug("event_scanner_paginated", total_fetched=len(all_markets))
        return all_markets

    async def close(self) -> None:
        await self.client.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_raw_market(self, raw: Dict[str, Any]) -> Optional[Market]:
        """Parse a raw Gamma API dict into a Market object."""
        try:
            end_date: Optional[datetime] = None
            end_str = raw.get("endDate")
            if isinstance(end_str, str) and end_str:
                end_date = datetime.fromisoformat(
                    end_str.replace("Z", "+00:00")
                ).astimezone(timezone.utc)

            clob_ids = parse_clob_token_ids(raw.get("clobTokenIds"))
            outcomes = parse_outcomes(raw.get("outcomes"))
            prices = parse_outcome_prices(raw.get("outcomePrices"))

            tokens: Dict[str, TokenInfo] = {}
            for i, outcome in enumerate(outcomes):
                if i >= len(clob_ids):
                    break
                price = prices[i] if i < len(prices) else 0.0
                tokens[outcome] = TokenInfo(
                    token_id=str(clob_ids[i]),
                    outcome=outcome,
                    price=price,
                    volume_24h=safe_float(raw.get("volume24hr", 0)),
                )

            return Market(
                id=str(raw.get("id")),
                question=raw.get("question", ""),
                description=raw.get("description", ""),
                category=raw.get("category", "Other") or "Other",
                end_date=end_date,
                volume_24h=safe_float(raw.get("volume24hr", 0)),
                liquidity=safe_float(raw.get("liquidity", 0)),
                tokens=tokens,
            )
        except Exception as exc:
            logger.debug("event_scanner_parse_error", error=str(exc))
            return None


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

async def _run_standalone() -> None:
    from ..utils import load_config, setup_logging

    config = load_config("config.yaml")
    setup_logging(config, log_json=False)

    scanner = EventScanner(config)
    try:
        print("\n" + "=" * 70)
        print("  Polymarket Event Group Scanner")
        print("=" * 70 + "\n")

        groups = await scanner.fetch_event_groups()

        if not groups:
            print("  No event groups found.\n")
            return

        # Sort by arb profit (most profitable first)
        sorted_groups = sorted(groups.values(), key=lambda g: g.arb_profit)

        for grp in sorted_groups[:20]:
            indicator = "✅" if grp.arb_profit > 0 else "❌"
            print(f"  {indicator} Event: {grp.event_id[:16]}…  ({len(grp.markets)} outcomes)")
            print(f"     YES sum: {grp.total_yes_sum:.4f}  |  Arb profit: {grp.arb_profit:+.4f}")
            for m in grp.markets[:5]:
                y = m.yes_price or 0.0
                print(f"       • {m.question[:60]:60s}  YES={y:.3f}")
            if len(grp.markets) > 5:
                print(f"       … and {len(grp.markets) - 5} more")
            print()

        profitable = [g for g in groups.values() if g.arb_profit > 0.01]
        print(f"  Total groups: {len(groups)}")
        print(f"  Potentially profitable (sum < 0.99): {len(profitable)}\n")

    finally:
        await scanner.close()


if __name__ == "__main__":
    asyncio.run(_run_standalone())
