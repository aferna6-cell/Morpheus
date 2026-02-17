"""Cross-platform arbitrage engine — Polymarket price signals for Kalshi trades.

Reads public Polymarket CLOB API prices and compares to Kalshi orderbooks.
When the same event exists on both platforms with a price divergence > threshold,
emits a TradeSignal to buy the cheap side on Kalshi.

Polymarket has 10-100x more liquidity and faster price discovery on overlapping
events (crypto, politics, economics). Kalshi prices lag — we exploit that.

No Polymarket auth needed — uses public CLOB API (GET /markets).
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import structlog

from ..engines.base import BaseEngine
from ..engines.signals import TradeSignal
from ..kalshi_client import KalshiClient, KalshiMarket
from ..utils import BotConfig


# Known cross-platform pairs: (kalshi_series_prefix, polymarket_keyword_patterns)
# These are fuzzy-matched; the engine also does automated title matching.
_MANUAL_PAIRS: List[Tuple[str, List[str]]] = [
    ("KXBTC", ["bitcoin", "btc", "Bitcoin price"]),
    ("KXETH", ["ethereum", "eth", "Ethereum price"]),
    ("KXFED", ["fed", "federal reserve", "interest rate", "rate cut", "rate hike"]),
    ("KXGDP", ["gdp", "gross domestic product"]),
    ("KXCPI", ["cpi", "inflation", "consumer price"]),
]


class KalshiCrossArbEngine(BaseEngine):
    """Cross-platform arb: Polymarket prices as signals for Kalshi trades."""

    name = "kalshi_cross_arb"

    def __init__(
        self,
        config: BotConfig,
        kalshi_client: KalshiClient,
    ):
        self.config = config
        self.kalshi_client = kalshi_client
        self.logger = structlog.get_logger()

        cfg = getattr(config, "cross_arb", None) or {}
        if not isinstance(cfg, dict):
            cfg = {}

        self._scan_interval = float(cfg.get("scan_interval_seconds", 30))
        self._min_divergence = float(cfg.get("min_divergence_pct", 3.0))
        self._max_position_usd = float(cfg.get("max_position_usd", 10.0))
        self._poly_api_url = cfg.get(
            "polymarket_api_url", "https://clob.polymarket.com"
        )
        self._pair_cache_file = cfg.get(
            "pair_cache_file", "state/cross_arb_pairs.json"
        )
        self._pair_refresh_seconds = float(cfg.get("pair_refresh_seconds", 3600))

        # State
        self._pending_signals: List[TradeSignal] = []
        self._scan_task: Optional[asyncio.Task] = None
        self._running = False

        # Cached pair mappings: {kalshi_ticker: polymarket_condition_id}
        self._pairs: Dict[str, Dict[str, Any]] = {}
        self._pairs_last_refresh: float = 0.0

        # Track recently signaled tickers (cooldown)
        self._signaled: Dict[str, float] = {}  # ticker -> monotonic timestamp
        self._signal_cooldown = float(cfg.get("signal_cooldown_seconds", 300))

        # HTTP client for Polymarket
        self._http: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        self._running = True
        self._http = httpx.AsyncClient(timeout=15.0)
        self._load_pair_cache()
        self._scan_task = asyncio.create_task(self._scan_loop())
        self.logger.info(
            "cross_arb_engine_started",
            scan_interval=self._scan_interval,
            min_divergence=self._min_divergence,
            cached_pairs=len(self._pairs),
        )

    async def stop(self) -> None:
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        if self._http:
            await self._http.aclose()
        self.logger.info("cross_arb_engine_stopped")

    async def get_signals(self) -> List[TradeSignal]:
        signals = list(self._pending_signals)
        self._pending_signals.clear()
        return signals

    # ------------------------------------------------------------------
    # Pair cache persistence
    # ------------------------------------------------------------------

    def _load_pair_cache(self) -> None:
        """Load cached pair mappings from disk."""
        try:
            p = Path(self._pair_cache_file)
            if p.exists():
                with open(p) as f:
                    self._pairs = json.load(f)
                self.logger.info("cross_arb_pairs_loaded", count=len(self._pairs))
        except Exception as e:
            self.logger.warning("cross_arb_pairs_load_failed", error=str(e))

    def _save_pair_cache(self) -> None:
        """Persist pair mappings to disk."""
        try:
            p = Path(self._pair_cache_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(self._pairs, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning("cross_arb_pairs_save_failed", error=str(e))

    # ------------------------------------------------------------------
    # Main scan loop
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        while self._running:
            try:
                await self._scan_for_arb()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error("cross_arb_scan_error", error=str(e))
            await asyncio.sleep(self._scan_interval)

    async def _scan_for_arb(self) -> None:
        """Core scan: refresh pairs if stale, then compare prices."""
        now_mono = time.monotonic()

        # Refresh pair mappings periodically
        if now_mono - self._pairs_last_refresh > self._pair_refresh_seconds:
            await self._refresh_pairs()
            self._pairs_last_refresh = now_mono

        if not self._pairs:
            self.logger.debug("cross_arb_no_pairs")
            return

        # Prune stale signal cooldowns
        stale = [t for t, ts in self._signaled.items() if now_mono - ts > self._signal_cooldown]
        for t in stale:
            del self._signaled[t]

        # Compare prices for each pair
        signals_found = 0
        for kalshi_ticker, pair_info in list(self._pairs.items()):
            if kalshi_ticker in self._signaled:
                continue

            try:
                poly_price = await self._get_polymarket_price(pair_info)
                if poly_price is None:
                    continue

                kalshi_market = await self.kalshi_client.fetch_market(kalshi_ticker)
                if kalshi_market is None or kalshi_market.status not in ("open", "active"):
                    continue

                # Compare: buy Kalshi YES if cheaper than Polymarket YES
                kalshi_yes_ask = kalshi_market.yes_ask
                kalshi_no_ask = kalshi_market.no_ask if kalshi_market.no_ask > 0 else (1.0 - kalshi_market.yes_bid)

                if kalshi_yes_ask <= 0.02 or kalshi_yes_ask >= 0.98:
                    continue  # Not developed orderbook

                # Divergence: Polymarket YES vs Kalshi YES ask
                # If Polymarket YES > Kalshi YES ask → buy YES on Kalshi
                # If Polymarket YES < Kalshi YES bid → buy NO on Kalshi
                poly_yes = poly_price
                divergence_yes = poly_yes - kalshi_yes_ask  # positive = Kalshi cheap
                divergence_no = (1.0 - poly_yes) - kalshi_no_ask  # positive = Kalshi NO cheap

                side = None
                divergence = 0.0
                entry_price = 0.0

                if divergence_yes > self._min_divergence / 100.0:
                    side = "buy_yes"
                    divergence = divergence_yes
                    entry_price = kalshi_yes_ask
                elif divergence_no > self._min_divergence / 100.0:
                    side = "buy_no"
                    divergence = divergence_no
                    entry_price = kalshi_no_ask

                if side is None:
                    continue

                # Confidence scales with divergence (3% = 0.55, 10% = 0.75)
                confidence = min(0.80, 0.50 + divergence * 2.5)
                edge = divergence  # Edge = price gap after fees (Kalshi maker = 0)

                # Size: fixed or scaled
                size_usd = min(self._max_position_usd, max(1.0, divergence * 100))

                self.logger.info(
                    "cross_arb_signal",
                    kalshi_ticker=kalshi_ticker,
                    side=side,
                    poly_yes=round(poly_yes, 4),
                    kalshi_yes_ask=round(kalshi_yes_ask, 4),
                    kalshi_no_ask=round(kalshi_no_ask, 4),
                    divergence_pct=round(divergence * 100, 2),
                    confidence=round(confidence, 3),
                    size_usd=size_usd,
                )

                signal = TradeSignal(
                    engine=self.name,
                    market_id=kalshi_ticker,
                    token_id=kalshi_ticker,
                    side=side,
                    confidence=confidence,
                    edge=edge,
                    urgency="normal",
                    metadata={
                        "estimated_prob": poly_yes if side == "buy_yes" else (1.0 - poly_yes),
                        "market_price": (kalshi_yes_ask + kalshi_market.yes_bid) / 2,
                        "net_edge": edge,
                        "conviction": "high" if divergence > 0.06 else "medium",
                        "reasoning": (
                            f"Cross-arb: Polymarket YES={poly_yes:.2%}, "
                            f"Kalshi {'ask' if side == 'buy_yes' else 'no_ask'}={entry_price:.2%}, "
                            f"divergence={divergence:.2%}"
                        ),
                        "question": kalshi_market.title,
                        "kalshi_ticker": kalshi_ticker,
                        "kalshi_yes_ask": kalshi_yes_ask,
                        "kalshi_no_ask": kalshi_no_ask,
                        "platform": "kalshi",
                        "strategy": "cross_arb",
                        "signal_source": "polymarket_cross_arb",
                        "_force_size_usd": size_usd,
                        "polymarket_price": poly_yes,
                        "divergence_pct": round(divergence * 100, 2),
                        "volume": kalshi_market.volume,
                    },
                )
                self._pending_signals.append(signal)
                self._signaled[kalshi_ticker] = now_mono
                signals_found += 1

            except Exception as e:
                self.logger.debug(
                    "cross_arb_pair_error",
                    kalshi_ticker=kalshi_ticker,
                    error=str(e),
                )

        if signals_found:
            self.logger.info("cross_arb_scan_complete", signals=signals_found)

    # ------------------------------------------------------------------
    # Polymarket API
    # ------------------------------------------------------------------

    async def _fetch_polymarket_markets(self, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch active Polymarket markets via Gamma API (better for discovery)."""
        if not self._http:
            return []

        resp = await self._http.get(
            "https://gamma-api.polymarket.com/markets",
            params={
                "limit": limit,
                "offset": offset,
                "active": "true",
                "closed": "false",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    async def _get_polymarket_price(self, pair_info: Dict[str, Any]) -> Optional[float]:
        """Get current YES price for a Polymarket market.

        Uses Gamma API outcomePrices (cached from discovery) first,
        falls back to CLOB midpoint for real-time price.
        """
        if not self._http:
            return None

        # Fast path: use cached price from Gamma API (refreshed hourly)
        cached_price = pair_info.get("yes_price")
        if cached_price is not None:
            try:
                p = float(cached_price)
                if 0.01 < p < 0.99:
                    return p
            except (ValueError, TypeError):
                pass

        # Fallback: CLOB midpoint for real-time price
        token_id = pair_info.get("token_id", "")
        if token_id:
            try:
                resp = await self._http.get(
                    f"{self._poly_api_url}/midpoint",
                    params={"token_id": token_id},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    mid = float(data.get("mid", 0))
                    if 0.01 < mid < 0.99:
                        return mid
            except Exception as e:
                self.logger.debug("poly_midpoint_error", token_id=token_id[:20], error=str(e))

        # Last resort: Gamma API single market refresh
        condition_id = pair_info.get("condition_id", "")
        if condition_id:
            try:
                resp = await self._http.get(
                    f"https://gamma-api.polymarket.com/markets?conditionId={condition_id}",
                )
                if resp.status_code == 200:
                    data = resp.json()
                    markets = data if isinstance(data, list) else [data]
                    if markets:
                        prices = markets[0].get("outcomePrices", "")
                        if isinstance(prices, str):
                            import json as _json
                            prices = _json.loads(prices)
                        if isinstance(prices, list) and len(prices) >= 1:
                            p = float(prices[0])
                            if 0.01 < p < 0.99:
                                return p
            except Exception as e:
                self.logger.debug("poly_gamma_price_error", error=str(e))

        return None

    # ------------------------------------------------------------------
    # Pair discovery
    # ------------------------------------------------------------------

    async def _refresh_pairs(self) -> None:
        """Discover cross-platform pairs by fuzzy-matching market titles."""
        self.logger.info("cross_arb_refreshing_pairs")

        # 1. Fetch Kalshi markets we care about
        kalshi_markets: List[KalshiMarket] = []
        for prefix, _ in _MANUAL_PAIRS:
            try:
                series_markets = await self.kalshi_client.fetch_markets_by_series(prefix)
                kalshi_markets.extend(series_markets)
            except Exception as e:
                self.logger.debug("cross_arb_kalshi_fetch_error", prefix=prefix, error=str(e))

        if not kalshi_markets:
            self.logger.warning("cross_arb_no_kalshi_markets")
            return

        # Also fetch any open Kalshi markets with known cross-platform tickers
        try:
            all_open = await self.kalshi_client.fetch_markets(status="open", limit=500)
            for m in all_open:
                if any(m.ticker.startswith(prefix) for prefix, _ in _MANUAL_PAIRS):
                    if m.ticker not in {km.ticker for km in kalshi_markets}:
                        kalshi_markets.append(m)
        except Exception:
            pass

        # 2. Fetch Polymarket active markets via Gamma API (paginate up to 500)
        poly_markets: List[Dict[str, Any]] = []
        for page_idx in range(5):  # Max 5 pages = 500 markets
            try:
                page = await self._fetch_polymarket_markets(offset=page_idx * 100)
                poly_markets.extend(page)
                if len(page) < 100:
                    break
            except Exception as e:
                self.logger.warning("cross_arb_poly_fetch_error", error=str(e))
                break

        if not poly_markets:
            self.logger.warning("cross_arb_no_poly_markets")
            return

        self.logger.info(
            "cross_arb_markets_fetched",
            kalshi=len(kalshi_markets),
            polymarket=len(poly_markets),
        )

        # 3. Match: for each Kalshi market, find best Polymarket match
        new_pairs: Dict[str, Dict[str, Any]] = {}

        for km in kalshi_markets:
            if km.status not in ("open", "active"):
                continue

            best_match = self._find_best_poly_match(km, poly_markets)
            if best_match:
                new_pairs[km.ticker] = best_match
                self.logger.info(
                    "cross_arb_pair_found",
                    kalshi=km.ticker,
                    kalshi_title=km.title[:60],
                    poly_title=best_match.get("title", "")[:60],
                )

        # Merge with existing pairs (keep manual overrides)
        self._pairs.update(new_pairs)
        self._save_pair_cache()
        self.logger.info("cross_arb_pairs_refreshed", total=len(self._pairs))

    def _find_best_poly_match(
        self, kalshi_market: KalshiMarket, poly_markets: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find the best matching Polymarket market for a Kalshi market."""
        kalshi_title = kalshi_market.title.lower()
        kalshi_ticker = kalshi_market.ticker.upper()

        # Determine keywords from manual pairs
        keywords: List[str] = []
        for prefix, kws in _MANUAL_PAIRS:
            if kalshi_ticker.startswith(prefix):
                keywords = [kw.lower() for kw in kws]
                break

        best_score = 0.0
        best_market = None

        for pm in poly_markets:
            poly_title = (pm.get("question", "") or pm.get("title", "")).lower()
            poly_desc = (pm.get("description", "")).lower()

            if not poly_title:
                continue

            score = 0.0

            # Keyword matching
            for kw in keywords:
                if kw in poly_title:
                    score += 2.0
                elif kw in poly_desc:
                    score += 0.5

            # Title word overlap
            kalshi_words = set(kalshi_title.split())
            poly_words = set(poly_title.split())
            overlap = kalshi_words & poly_words
            # Remove common stop words
            stop_words = {"the", "a", "an", "is", "be", "to", "in", "on", "at", "of", "or", "and", "will", "by"}
            overlap -= stop_words
            score += len(overlap) * 0.5

            # Active market bonus
            if pm.get("active", False) or pm.get("accepting_orders", False):
                score += 1.0

            if score > best_score and score >= 2.0:  # Minimum match threshold
                best_score = score
                best_market = pm

        if best_market:
            # Extract token info — Gamma API format
            condition_id = best_market.get("conditionId", "") or best_market.get("condition_id", "")

            # clobTokenIds: first = YES, second = NO
            clob_token_ids = best_market.get("clobTokenIds", "")
            if isinstance(clob_token_ids, str):
                try:
                    clob_token_ids = json.loads(clob_token_ids)
                except (json.JSONDecodeError, TypeError):
                    clob_token_ids = []
            yes_token_id = clob_token_ids[0] if clob_token_ids else ""

            # Extract current YES price from outcomePrices
            outcome_prices = best_market.get("outcomePrices", "")
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = []
            yes_price = float(outcome_prices[0]) if outcome_prices else None

            return {
                "condition_id": condition_id,
                "token_id": yes_token_id,
                "title": best_market.get("question", "") or best_market.get("title", ""),
                "match_score": best_score,
                "yes_price": yes_price,
            }

        return None
