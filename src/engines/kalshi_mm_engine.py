"""Kalshi market-making engine with Avellaneda-Stoikov dynamic spreads.

Exploits zero maker fees on Kalshi by posting two-sided limit orders.
Every filled spread is pure profit. The engine:

1. Selects high-volume, wide-spread markets suitable for MM
2. Computes optimal bid/ask using Avellaneda-Stoikov reservation price
3. Manages inventory to avoid one-sided exposure via inventory-skewed quotes
4. Pulls/widens quotes on adverse conditions (volume spikes, one-sided fills)

The A-S framework provides mathematically optimal quoting:
- Reservation price: r = mid - q * gamma * sigma^2 * (T - t)
- Optimal spread: delta = gamma * sigma^2 * (T-t) + (2/gamma) * log(1 + gamma/kappa)
- Quote placement: bid = r - delta/2, ask = r + delta/2
"""

from __future__ import annotations

import asyncio
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import structlog

from ..engines.base import BaseEngine
from ..engines.signals import TradeSignal
from ..kalshi_client import KalshiClient, KalshiMarket
from ..market_filters import MarketFilters
from ..utils import BotConfig


@dataclass
class MMQuote:
    """A two-sided quote on a market."""

    ticker: str
    yes_bid: int   # cents
    yes_ask: int   # cents
    spread: int    # ask - bid in cents
    size: int      # contracts per side
    inventory: int = 0  # net YES contracts (positive = long YES)


@dataclass
class MMMarketState:
    """Tracked state for a market we're making."""

    ticker: str
    title: str
    last_yes_mid: float  # 0-1
    last_spread: float   # 0-1
    volume: int
    close_time: Optional[datetime]
    inventory: int = 0
    total_filled_yes: int = 0
    total_filled_no: int = 0
    last_quote_time: Optional[datetime] = None
    consecutive_one_sided: int = 0  # adverse selection counter
    consecutive_ob_errors: int = 0  # orderbook fetch failures
    recent_mids: Deque[float] = field(default_factory=lambda: deque(maxlen=20))  # for sigma estimation


class KalshiMMEngine(BaseEngine):
    """Market-making engine for Kalshi with zero maker fees."""

    name = "kalshi_mm"

    def __init__(
        self,
        config: BotConfig,
        kalshi_client: KalshiClient,
    ):
        self.config = config
        self.kalshi_client = kalshi_client
        self.logger = structlog.get_logger()
        self._filters = MarketFilters(config)

        kalshi_cfg = getattr(config, "kalshi", None) or {}
        mm_cfg = getattr(config, "market_making", None) or {}
        if not isinstance(mm_cfg, dict):
            mm_cfg = {}
        if not isinstance(kalshi_cfg, dict):
            kalshi_cfg = {}

        # MM parameters
        self._max_markets = int(mm_cfg.get("max_markets", 3))
        self._min_volume = int(mm_cfg.get("min_volume", 5000))
        self._min_spread_cents = int(mm_cfg.get("min_spread_cents", 3))
        self._max_spread_cents = int(mm_cfg.get("max_spread_cents", 15))
        self._quote_size = int(mm_cfg.get("quote_size", 5))
        self._max_inventory = int(mm_cfg.get("max_inventory", 20))
        self._inventory_skew_cents = int(mm_cfg.get("inventory_skew_cents", 2))
        self._max_inventory_usd = float(mm_cfg.get("max_inventory_usd", 50.0))
        self._quote_refresh_seconds = float(mm_cfg.get("quote_refresh_seconds", 60.0))
        self._scan_interval_seconds = float(mm_cfg.get("scan_interval_seconds", 300.0))
        self._min_price_cents = int(mm_cfg.get("min_price_cents", 15))
        self._max_price_cents = int(mm_cfg.get("max_price_cents", 85))

        # Avellaneda-Stoikov parameters
        as_cfg = mm_cfg.get("avellaneda_stoikov", {})
        if not isinstance(as_cfg, dict):
            as_cfg = {}
        self._gamma = float(as_cfg.get("gamma", 0.3))          # risk aversion
        self._default_sigma = float(as_cfg.get("default_sigma", 0.15))  # volatility fallback
        self._as_min_spread = int(as_cfg.get("min_spread_cents", 2))
        self._as_max_spread = int(as_cfg.get("max_spread_cents", 15))

        # State
        self._markets: Dict[str, MMMarketState] = {}
        self._pending_signals: List[TradeSignal] = []
        self._scan_task: Optional[asyncio.Task] = None
        self._quote_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())
        self._quote_task = asyncio.create_task(self._quote_loop())
        self.logger.info(
            "kalshi_mm_engine_started",
            max_markets=self._max_markets,
            quote_size=self._quote_size,
            max_inventory=self._max_inventory,
        )

    async def stop(self) -> None:
        self._running = False
        for task in [self._scan_task, self._quote_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.logger.info("kalshi_mm_engine_stopped")

    async def get_signals(self) -> List[TradeSignal]:
        signals = list(self._pending_signals)
        self._pending_signals.clear()
        return signals

    # ------------------------------------------------------------------
    # Market selection
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        """Periodically scan for good MM markets."""
        while self._running:
            try:
                await self._select_markets()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error("mm_scan_error", error=str(e))
            await asyncio.sleep(self._scan_interval_seconds)

    async def _select_markets(self) -> None:
        """Find markets suitable for market making."""
        try:
            all_markets = await self.kalshi_client.fetch_markets_by_close_date(
                max_days=7,
                min_volume=100,
            )
        except Exception as e:
            self.logger.error("mm_fetch_markets_error", error=str(e))
            return

        candidates: List[KalshiMarket] = []

        for m in all_markets:
            # Ticker prefix filter (junk markets — crypto ranges, mentions, etc.)
            prefix_result = self._filters.check_ticker_prefix(m.ticker)
            if not prefix_result.passed:
                continue

            # Must have decent volume
            if m.volume < self._min_volume:
                continue

            # Price must be in a range where MM makes sense (not near 0 or 100)
            yes_cents = int(m.yes_price * 100)
            if yes_cents < self._min_price_cents or yes_cents > self._max_price_cents:
                continue

            # Must have a spread we can capture
            spread_cents = int((m.yes_ask - m.yes_bid) * 100) if m.yes_ask > m.yes_bid else 0
            if spread_cents < self._min_spread_cents:
                continue
            if spread_cents > self._max_spread_cents:
                continue

            # Must have close time > 4 hours away (need time for fills)
            if m.close_time:
                hours_to_close = (m.close_time - datetime.now(timezone.utc)).total_seconds() / 3600
                if hours_to_close < 4:
                    continue
                # Also skip expired markets (close time in the past)
                if hours_to_close < 0:
                    continue

            # Skip sports markets — outcome is binary event, not suitable for MM
            # (price jumps on goals/events cause adverse selection)
            _SPORTS_PREFIXES = (
                "KXNFL", "KXNBA", "KXMLB", "KXNHL", "KXMLS", "KXEPL",
                "KXUFC", "KXSOCCER", "KXSCOTTISHPREM", "KXEFLCHAMPIONSHIP",
                "KXLALIGA", "KXBUNDESLIGA", "KXSERIEA", "KXLIGUE1",
                "KXCHAMPIONSLEAGUE", "KXNCAA",
            )
            if m.ticker.upper().startswith(_SPORTS_PREFIXES):
                continue

            candidates.append(m)

        # Rank by volume * spread (maximize expected fill rate * profit per fill)
        def _mm_score(m: KalshiMarket) -> float:
            spread = (m.yes_ask - m.yes_bid) if m.yes_ask > m.yes_bid else 0.01
            return m.volume * spread

        candidates.sort(key=_mm_score, reverse=True)

        self.logger.info(
            "mm_scan_complete",
            total_fetched=len(all_markets),
            candidates=len(candidates),
            selected=min(len(candidates), self._max_markets),
        )

        # Keep top N
        selected = candidates[: self._max_markets]

        # Update tracked markets
        new_tickers = {m.ticker for m in selected}
        old_tickers = set(self._markets.keys())

        # Remove dropped markets
        for ticker in old_tickers - new_tickers:
            self.logger.info("mm_market_dropped", ticker=ticker)
            del self._markets[ticker]

        # Add/update markets
        for m in selected:
            if m.ticker not in self._markets:
                self._markets[m.ticker] = MMMarketState(
                    ticker=m.ticker,
                    title=m.title,
                    last_yes_mid=m.yes_price,
                    last_spread=m.yes_ask - m.yes_bid if m.yes_ask > m.yes_bid else 0.05,
                    volume=m.volume,
                    close_time=m.close_time,
                )
                self.logger.info(
                    "mm_market_added",
                    ticker=m.ticker,
                    title=m.title[:60],
                    mid=m.yes_price,
                    spread=m.yes_ask - m.yes_bid,
                    volume=m.volume,
                )
            else:
                state = self._markets[m.ticker]
                state.last_yes_mid = m.yes_price
                state.last_spread = m.yes_ask - m.yes_bid if m.yes_ask > m.yes_bid else state.last_spread
                state.volume = m.volume

    # ------------------------------------------------------------------
    # Quoting
    # ------------------------------------------------------------------

    async def _quote_loop(self) -> None:
        """Periodically update quotes on tracked markets."""
        while self._running:
            try:
                await self._update_all_quotes()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error("mm_quote_error", error=str(e))
            await asyncio.sleep(self._quote_refresh_seconds)

    async def _update_all_quotes(self) -> None:
        """Generate quote signals for all tracked MM markets."""
        signals_before = len(self._pending_signals)
        for ticker, state in list(self._markets.items()):
            try:
                await self._generate_quotes(state)
            except Exception as e:
                self.logger.warning("mm_quote_gen_error", ticker=ticker, error=str(e))

        quotes_generated = len(self._pending_signals) - signals_before
        if self._markets:
            self.logger.info(
                "mm_cycle_summary",
                active_markets=len(self._markets),
                quotes_generated=quotes_generated,
                inventory={t: s.inventory for t, s in self._markets.items()},
            )

    def _estimate_sigma(self, state: MMMarketState) -> float:
        """Estimate market volatility from recent midpoint observations.

        Uses standard deviation of recent mids. Falls back to default if
        insufficient data.
        """
        if len(state.recent_mids) < 3:
            return self._default_sigma

        mids = list(state.recent_mids)
        mean = sum(mids) / len(mids)
        variance = sum((m - mean) ** 2 for m in mids) / len(mids)
        sigma = math.sqrt(variance) if variance > 0 else self._default_sigma
        # Clamp to reasonable range
        return max(0.02, min(0.40, sigma))

    def _compute_as_quotes(
        self, mid_cents: float, state: MMMarketState, sigma: float,
    ) -> tuple[int, int]:
        """Compute Avellaneda-Stoikov optimal bid/ask.

        Reservation price:  r = mid - q * gamma * sigma^2 * (T - t)
        Optimal spread:     delta = gamma * sigma^2 * (T-t) + (2/gamma) * log(1 + gamma/kappa)

        Returns: (yes_bid_cents, yes_ask_cents)
        """
        gamma = self._gamma
        q = state.inventory

        # Time remaining (in hours, normalized to 0-1 range for 24h markets)
        if state.close_time:
            hours_left = max(0.1, (state.close_time - datetime.now(timezone.utc)).total_seconds() / 3600)
            t_remaining = min(hours_left / 24.0, 1.0)  # normalize to [0, 1]
        else:
            t_remaining = 0.5  # default: assume 12h left

        # Estimate order arrival rate (kappa) from volume
        # Higher volume = faster fills = can quote tighter
        kappa = max(0.5, state.volume / 5000.0)

        # Reservation price: skewed by inventory
        mid_frac = mid_cents / 100.0
        r = mid_frac - q * gamma * (sigma ** 2) * t_remaining
        r = max(0.02, min(0.98, r))

        # Optimal spread
        vol_term = gamma * (sigma ** 2) * t_remaining
        arrival_term = (2.0 / gamma) * math.log(1.0 + gamma / kappa) if kappa > 0 else 0.04
        delta = vol_term + arrival_term

        # Convert to cents
        half_spread = max(self._as_min_spread / 2.0, delta * 100.0 / 2.0)
        half_spread = min(self._as_max_spread / 2.0, half_spread)

        r_cents = r * 100.0
        bid = int(r_cents - half_spread)
        ask = int(math.ceil(r_cents + half_spread))

        # Clamp
        bid = max(1, min(98, bid))
        ask = max(2, min(99, ask))
        if bid >= ask:
            bid = max(1, ask - self._as_min_spread)

        return bid, ask

    async def _generate_quotes(self, state: MMMarketState) -> None:
        """Generate Avellaneda-Stoikov optimal bid/ask signals for a market."""
        # Get fresh orderbook
        try:
            ob = await self.kalshi_client.get_orderbook(state.ticker)
            state.consecutive_ob_errors = 0  # reset on success
        except Exception as e:
            state.consecutive_ob_errors += 1
            if state.consecutive_ob_errors >= 3:
                self.logger.warning(
                    "mm_market_dropped_errors",
                    ticker=state.ticker,
                    errors=state.consecutive_ob_errors,
                    error=str(e)[:100],
                )
                self._markets.pop(state.ticker, None)
            return

        # Parse best bid/ask from orderbook
        yes_bids = ob.get("yes", [])
        no_bids = ob.get("no", [])

        best_yes_bid = yes_bids[0][0] if yes_bids else 0
        best_no_bid = no_bids[0][0] if no_bids else 0

        # Best yes ask = 100 - best no bid (Kalshi binary market identity)
        best_yes_ask = 100 - best_no_bid if best_no_bid > 0 else 100

        if best_yes_bid <= 0 or best_yes_ask <= 0:
            return

        spread_cents = best_yes_ask - best_yes_bid
        if spread_cents < self._min_spread_cents:
            return  # Spread too tight, nothing to capture

        mid_cents = (best_yes_bid + best_yes_ask) / 2.0

        # Track mid for sigma estimation
        state.recent_mids.append(mid_cents / 100.0)
        state.last_yes_mid = mid_cents / 100.0

        # Estimate volatility
        sigma = self._estimate_sigma(state)

        # Compute A-S optimal quotes
        our_yes_bid, our_yes_ask = self._compute_as_quotes(mid_cents, state, sigma)

        # Ensure our quotes are inside (or at) the market spread
        our_yes_bid = max(our_yes_bid, best_yes_bid)
        our_yes_ask = min(our_yes_ask, best_yes_ask)

        if our_yes_bid >= our_yes_ask:
            return  # Can't quote profitably

        # Check inventory limits
        if abs(state.inventory) >= self._max_inventory:
            # Only quote on the reducing side
            if state.inventory > 0:
                self._emit_mm_signal(state, "buy_no", 100 - our_yes_ask, state.ticker)
            else:
                self._emit_mm_signal(state, "buy_yes", our_yes_bid, state.ticker)
            return

        # Emit both sides
        self._emit_mm_signal(state, "buy_yes", our_yes_bid, state.ticker)
        self._emit_mm_signal(state, "buy_no", 100 - our_yes_ask, state.ticker)

        state.last_quote_time = datetime.now(timezone.utc)
        self.logger.debug(
            "mm_as_quotes_generated",
            ticker=state.ticker,
            yes_bid=our_yes_bid,
            yes_ask=our_yes_ask,
            spread=our_yes_ask - our_yes_bid,
            inventory=state.inventory,
            sigma=round(sigma, 4),
            market_spread=spread_cents,
        )

    def _emit_mm_signal(
        self, state: MMMarketState, side: str, price_cents: int, ticker: str,
    ) -> None:
        """Create a TradeSignal for the MM quote."""
        price_frac = price_cents / 100.0
        # Edge is half the spread we're capturing
        spread_edge = state.last_spread / 2.0 if state.last_spread > 0 else 0.01

        # Fixed size: quote_size contracts × entry cost
        entry_cost = price_cents / 100.0
        force_size = self._quote_size * entry_cost

        signal = TradeSignal(
            engine=self.name,
            market_id=ticker,
            token_id=ticker,
            side=side,
            confidence=0.6,  # MM signals are mechanical, moderate confidence
            edge=spread_edge,
            urgency="normal",
            metadata={
                "estimated_prob": state.last_yes_mid,
                "market_price": state.last_yes_mid,
                "net_edge": spread_edge,
                "conviction": "medium",
                "reasoning": f"MM spread capture: {state.last_spread*100:.0f}c spread",
                "question": state.title,
                "kalshi_ticker": ticker,
                "kalshi_yes_ask": price_frac if side == "buy_yes" else (1.0 - price_frac),
                "kalshi_no_ask": price_frac if side == "buy_no" else (1.0 - price_frac),
                "mm_inventory": state.inventory,
                "platform": "kalshi",
                "strategy": "mm",
                "_force_size_usd": force_size,
            },
        )
        self._pending_signals.append(signal)

    # ------------------------------------------------------------------
    # Inventory tracking (called externally by fill manager)
    # ------------------------------------------------------------------

    def update_inventory(self, ticker: str, side: str, count: int) -> None:
        """Update inventory after a fill."""
        state = self._markets.get(ticker)
        if not state:
            return

        if side == "yes":
            state.inventory += count
            state.total_filled_yes += count
        else:
            state.inventory -= count
            state.total_filled_no += count

        # Track adverse selection via net fill imbalance
        imbalance = state.total_filled_yes - state.total_filled_no
        state.consecutive_one_sided = abs(imbalance)

        # Drop market if adverse selection detected (3+ one-sided fills)
        if state.consecutive_one_sided >= 3:
            self.logger.warning(
                "mm_adverse_selection_drop",
                ticker=ticker,
                one_sided_fills=state.consecutive_one_sided,
                filled_yes=state.total_filled_yes,
                filled_no=state.total_filled_no,
            )
            self._markets.pop(ticker, None)
            return

        self.logger.info(
            "mm_inventory_updated",
            ticker=ticker,
            side=side,
            count=count,
            inventory=state.inventory,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return MM engine statistics."""
        return {
            "active_markets": len(self._markets),
            "markets": {
                ticker: {
                    "title": s.title[:50],
                    "inventory": s.inventory,
                    "filled_yes": s.total_filled_yes,
                    "filled_no": s.total_filled_no,
                    "spread": s.last_spread,
                }
                for ticker, s in self._markets.items()
            },
        }
