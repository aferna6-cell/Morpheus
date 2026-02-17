"""Multi-engine orchestrator — the brain of Morpheus.

Collects signals from all engines, scores and ranks them via a unified
scoring formula, detects multi-engine consensus, and dispatches trades
through the execution layer.

The old main-loop still works when ``orchestrator.enabled`` is ``false``
in config.yaml.  When enabled the orchestrator replaces it with a tight
async loop that drains every engine on each cycle.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from .capital_management import get_capital_manager, CapitalManager
from .engines.base import BaseEngine
from .engines.signals import TradeSignal
from .fill_manager import FillManager
from .kalshi_executor import KalshiExecutor, KalshiTradeExecution
from .markets import Market
from .risk import RiskManager
from .signals.base import SignalResult, TradingSide
from .alerts import send_alert
from .market_filters import MarketFilters
from .runlog import log_prediction
from .survival import SurvivalMode, SurvivalTracker
from .utils import BotConfig, utc_now


# ---------------------------------------------------------------------------
# Urgency bonuses by engine / urgency level
# ---------------------------------------------------------------------------

_URGENCY_BONUS: Dict[str, float] = {
    "immediate": 2.0,   # closing soon — highest priority
    "normal": 0.5,
    "low": 0.0,
}

_ENGINE_PRIORITY: Dict[str, float] = {
    "kalshi_cross_arb": 0.95,   # Cross-platform arb — Polymarket price signal
    "kalshi_bracket_arb": 0.9,  # Structural arb — near-certain profit
    "kalshi_bonding": 0.8,      # Data-verified — NOAA/FRED backed
    "kalshi_longshot": 0.75,    # Statistical bias — favorite-longshot
    "kalshi_crypto": 0.7,       # Latency arb — Coinbase price feed
    "kalshi_contrarian": 0.6,
    "kalshi_llm": 0.5,
    "kalshi_mm": 0.4,
}

# Engines whose signals route to Kalshi executor
_KALSHI_ENGINES = {"kalshi_llm", "kalshi_mm", "kalshi_contrarian", "kalshi_crypto",
                   "kalshi_bracket_arb", "kalshi_bonding", "kalshi_longshot",
                   "kalshi_cross_arb"}


def _extract_event_prefix(ticker: str) -> str:
    """Extract event prefix from a Kalshi ticker for correlation grouping.

    Examples:
        KXBTC-25FEB07-T65749 -> KXBTC-25FEB07
        KXETHD-26FEB0617-T1699.99 -> KXETHD-26FEB0617
        INX-25FEB07-T5999.99 -> INX-25FEB07
        PRES-2028-DEM -> PRES-2028-DEM (no threshold suffix)
    """
    import re
    # Match ticker-date-Tvalue pattern (threshold markets)
    match = re.match(r'^(.+?-\d+[A-Z]*\d*)-T[\d.]+$', ticker)
    if match:
        return match.group(1)
    return ticker


class Orchestrator:
    """Collects signals from all engines, scores them, and decides what to trade."""

    def __init__(
        self,
        config: BotConfig,
        engines: List[BaseEngine],
        risk_manager: RiskManager,
        kalshi_executors: Optional[List[KalshiExecutor]] = None,
        executor: Optional[object] = None,  # legacy compat, unused
        fill_manager: Optional[FillManager] = None,
        survival_tracker: Optional[SurvivalTracker] = None,
    ):
        self.config = config
        self.engines = engines
        self.risk_manager = risk_manager
        self.kalshi_executors = kalshi_executors or []
        self.fill_manager = fill_manager
        self.survival_tracker = survival_tracker
        self.logger = structlog.get_logger()

        orch_cfg = getattr(config, "orchestrator", None) or config.__dict__.get("orchestrator", {})
        if isinstance(orch_cfg, dict):
            self._signal_ttl = float(orch_cfg.get("signal_ttl_seconds", 300))
            self._max_per_cycle = int(orch_cfg.get("max_signals_per_cycle", 10))
            self._multi_boost = float(orch_cfg.get("multi_engine_boost", 1.5))
            self._cycle_interval = float(orch_cfg.get("cycle_interval_seconds", 1))
        else:
            self._signal_ttl = 300
            self._max_per_cycle = 10
            self._multi_boost = 1.5
            self._cycle_interval = 1

        # Track consensus performance
        self._consensus_hits: Dict[str, int] = defaultdict(int)

        # Dedup: track (market_id, side, engine) combos already dispatched
        # Prevents the same arb opportunity from being re-traded every scan cycle
        self._dispatched: set = set()

        # Event-level dedup: track (event_prefix, side) to prevent correlated trades
        # e.g., buying NO on 5 different BTC threshold tickers simultaneously
        self._event_dispatched: Dict[str, int] = {}  # event_prefix -> count
        self._max_per_event = int(
            getattr(config, "market_filters", {}).get("max_per_correlation_cluster", 2)
            if isinstance(getattr(config, "market_filters", None), dict) else 2
        )

        self._running = False

        # Sports filter (defense in depth — catches signals from any engine)
        self._market_filters = MarketFilters(config)

        # Capital management (recycling rules + CLV tracking)
        self._capital_manager: Optional[CapitalManager] = None
        try:
            self._capital_manager = get_capital_manager(config, state_dir="state")
        except Exception as exc:
            self.logger.warning("capital_manager_init_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_engines(self) -> None:
        """Start all registered engines."""
        for engine in self.engines:
            try:
                await engine.start()
                self.logger.info("engine_started", engine=engine.name)
            except Exception as exc:
                self.logger.error("engine_start_failed", engine=engine.name, error=str(exc))

    async def stop_engines(self) -> None:
        """Gracefully stop all engines."""
        for engine in self.engines:
            try:
                await engine.stop()
                self.logger.info("engine_stopped", engine=engine.name)
            except Exception as exc:
                self.logger.warning("engine_stop_error", engine=engine.name, error=str(exc))

    async def run(self) -> None:
        """Main orchestrator loop — runs until cancelled or ``stop()`` called."""
        self._running = True
        self.logger.info(
            "orchestrator_started",
            engines=[e.name for e in self.engines],
            cycle_interval=self._cycle_interval,
        )

        cycle = 0
        while self._running:
            cycle += 1
            try:
                await self._run_cycle(cycle)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error("orchestrator_cycle_error", cycle=cycle, error=str(exc))

            await asyncio.sleep(self._cycle_interval)

        self.logger.info("orchestrator_stopped")

    def stop(self) -> None:
        """Signal the orchestrator to exit on the next cycle."""
        self._running = False

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    async def _run_cycle(self, cycle: int) -> None:
        # 0. Survival mode check
        if self.survival_tracker:
            survival_mode = self.survival_tracker.evaluate()
            self.risk_manager.set_survival_multiplier(self.survival_tracker.multiplier)
            if survival_mode == SurvivalMode.HALTED:
                if cycle % 60 == 1:  # log once per ~hour
                    self.logger.warning("survival_halted_skip_cycle", mode="halted")
                return

        # 1. Gather signals from all engines
        all_signals: List[TradeSignal] = []
        for engine in self.engines:
            try:
                signals = await engine.get_signals()
                all_signals.extend(signals)
            except Exception as exc:
                self.logger.warning("engine_get_signals_error", engine=engine.name, error=str(exc))

        if not all_signals:
            return

        # 2. Filter stale signals
        now = utc_now()
        fresh = [
            s for s in all_signals
            if (now - s.timestamp).total_seconds() <= self._signal_ttl
        ]

        if not fresh:
            return

        # 2b. Sports filter (defense in depth for all engines)
        filtered = []
        for s in fresh:
            meta = s.metadata or {}
            title = meta.get("title", "") or meta.get("question", "")
            category = meta.get("category", "")
            if title or category:
                result = self._market_filters.check_sports(
                    market_id=s.market_id,
                    title=title,
                    category=category,
                    is_live=False,
                )
                if not result.passed:
                    self.logger.info(
                        "orchestrator_sports_blocked",
                        market_id=s.market_id,
                        engine=s.engine,
                        reason=result.reason,
                    )
                    continue
            filtered.append(s)
        fresh = filtered

        if not fresh:
            return

        # 2c. Multi-engine consensus detection + boosting
        fresh = self._apply_consensus(fresh)

        # 3. Score and rank
        ranked = self._score_signals(fresh)

        # 4. Build set of tickers we already hold (position dedup)
        #    Also count held positions per event prefix for correlation limiting
        held_tickers: set = set()
        held_event_counts: Dict[str, int] = {}
        position_fetch_ok = False
        for executor in self.kalshi_executors:
            try:
                positions = await executor.trading_client.get_positions()
                position_fetch_ok = True
                for pos in positions:
                    if pos.count != 0:
                        held_tickers.add(pos.ticker)
                        ep = _extract_event_prefix(pos.ticker)
                        held_event_counts[ep] = held_event_counts.get(ep, 0) + 1
            except Exception as e:
                self.logger.warning("position_fetch_failed", executor=executor.label, error=str(e))

        if not position_fetch_ok:
            self.logger.error("all_position_fetches_failed_skip_cycle")
            return

        # Prune stale _dispatched entries for markets no longer held.
        # Without this, the set grows forever and prevents re-entry on
        # resolved markets (opportunity starvation).
        stale = {key for key in self._dispatched if key[0] not in held_tickers}
        if stale:
            self._dispatched -= stale
            self.logger.debug("dispatched_pruned", count=len(stale))

        # Reset event dispatch tracker to match actual held positions.
        # Direct assignment instead of max() merge — when positions resolve,
        # the counter must go down so new entries are allowed.
        self._event_dispatched = dict(held_event_counts)

        # 5. Execute top N within risk limits
        executed = 0
        for signal in ranked[: self._max_per_cycle]:
            if self.risk_manager.trading_halted:
                self.logger.warning("trading_halted_skip")
                break

            # Check capital management rules (recycling + CLV)
            if self._capital_manager:
                can_trade, reason = self._capital_manager.can_trade()
                if not can_trade:
                    self.logger.warning("capital_management_halt", reason=reason)
                    # Don't alert on capital management - only alert on actual trades
                    break

            # Dedup: skip if we already dispatched this exact signal
            # Exception: bracket_arb — engine tracks active sets internally;
            # stale _dispatched entries block completion of partial arb sets.
            dedup_key = (signal.market_id, signal.side, signal.engine)
            if dedup_key in self._dispatched:
                if signal.metadata.get("strategy") != "bracket_arb":
                    self.logger.debug(
                        "dispatch_dedup_skip",
                        market_id=signal.market_id,
                        side=signal.side,
                        engine=signal.engine,
                    )
                    continue

            # Position dedup: skip markets where we already hold a position
            # Exceptions:
            #   - MM signals: market makers should refresh quotes on held markets
            #   - bracket_arb signals: arb needs ALL legs filled; partial sets
            #     must be completable. Engine's _active_sets handles its own dedup.
            if signal.market_id in held_tickers:
                strategy = signal.metadata.get("strategy", "")
                if strategy not in ("mm", "bracket_arb"):
                    self.logger.info(
                        "dispatch_position_dedup_skip",
                        market_id=signal.market_id,
                        engine=signal.engine,
                    )
                    continue

            # Event-level dedup: prevent correlated trades
            # (e.g., NO on 5 different BTC threshold tickers)
            # Weather markets get a higher limit (3 vs 2) because each city-date
            # can have a threshold + bracket that are independent bets.
            event_prefix = _extract_event_prefix(signal.market_id)
            event_count = self._event_dispatched.get(event_prefix, 0)
            is_weather_event = any(
                event_prefix.startswith(p)
                for p in ("KXHIGH", "KXLOW", "KXRAIN", "KXSNOW", "KXTEMP", "KXWIND")
            )
            max_event = 3 if is_weather_event else self._max_per_event
            if event_count >= max_event:
                self.logger.info(
                    "dispatch_event_dedup_skip",
                    market_id=signal.market_id,
                    event_prefix=event_prefix,
                    event_count=event_count,
                    max_per_event=max_event,
                )
                continue

            try:
                trade = await self._dispatch(signal)
                if trade and trade.was_successful:
                    executed += 1
                    self._dispatched.add(dedup_key)
                    # Track event-level for correlation limiting
                    self._event_dispatched[event_prefix] = self._event_dispatched.get(event_prefix, 0) + 1

                    # Register position with capital manager for recycling/CLV tracking
                    if self._capital_manager:
                        market_data = signal.metadata.get("_market")
                        resolution_time = getattr(market_data, "end_date", None) if market_data else None
                        entry_prob = signal.metadata.get("estimated_prob", 0.5)
                        side = "yes" if signal.side == "buy_yes" else "no"
                        platform = signal.metadata.get("platform", "polymarket")
                        if signal.engine in _KALSHI_ENGINES:
                            platform = "kalshi"

                        # Resting orders report executed_amount_usd=0; use
                        # intended cost so capital manager tracks real exposure.
                        intended_cost = trade.intended_contracts * (trade.price_cents / 100.0)
                        actual_amount = trade.executed_amount_usd if trade.executed_amount_usd > 0 else intended_cost

                        self._capital_manager.add_position(
                            market_id=signal.market_id,
                            ticker=signal.token_id or signal.market_id,
                            platform=platform,
                            amount_usd=actual_amount,
                            entry_price=trade.average_price,
                            entry_probability=entry_prob,
                            side=side,
                            resolution_time=resolution_time,
                            signal_source=signal.metadata.get("signal_source", "unknown"),
                            engine=signal.engine,
                        )

                    # Send alert for successful trades
                    question = signal.metadata.get("question", "")
                    if not question:
                        m = signal.metadata.get("_market")
                        question = getattr(m, "question", signal.market_id) if m else signal.market_id
                    conviction = signal.metadata.get("conviction", "?")

                    alert_msg = (
                        f"[{signal.engine.upper()}] {signal.side.upper()} "
                        f"${trade.executed_amount_usd:.2f} @ {trade.average_price:.3f}\n"
                        f"Edge: {signal.edge:+.3f} | Conv: {conviction}\n"
                        f"{question[:100]}"
                    )
                    await send_alert(alert_msg, self.config)

                    # Log prediction for resolution tracking + accuracy analysis
                    try:
                        log_prediction(
                            state_dir="state",
                            market_id=signal.market_id,
                            predicted_p_yes=signal.metadata.get("estimated_prob", 0.5),
                            market_price_at_entry=signal.metadata.get("market_price", trade.average_price),
                            side=signal.side,
                            edge=signal.edge,
                            conviction=str(conviction),
                            net_edge=signal.metadata.get("net_edge", 0.0),
                            signal_source=signal.metadata.get("signal_source", "llm"),
                        )
                    except Exception:
                        self.logger.warning("prediction_log_failed", market_id=signal.market_id)
            except Exception as exc:
                self.logger.error(
                    "dispatch_error",
                    engine=signal.engine,
                    market_id=signal.market_id,
                    error=str(exc),
                )

        if executed or len(all_signals) > 5:
            engine_counts = {}
            for s in all_signals:
                engine_counts[s.engine] = engine_counts.get(s.engine, 0) + 1
            self.logger.info("orchestrator_cycle_summary", cycle=cycle,
                             signals_in=len(all_signals), fresh=len(fresh),
                             executed=executed, by_engine=engine_counts)

        # Periodic capital management status check (every 10 cycles)
        if cycle % 10 == 0 and self._capital_manager:
            summary = self._capital_manager.get_summary()
            self.logger.info("capital_management_status", **summary)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_signals(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Score and sort signals descending by composite score."""

        def _score(s: TradeSignal) -> float:
            urgency_bonus = _URGENCY_BONUS.get(s.urgency, 0.0)
            engine_priority = _ENGINE_PRIORITY.get(s.engine, 0.3)

            # Multi-engine bonus stored by _apply_consensus
            multi_bonus = s.metadata.get("_multi_engine_bonus", 0.0)

            # Confidence-weighted edge = expected value accounting for uncertainty
            ev_edge = max(s.edge, 0.0) * s.confidence
            score = (
                (ev_edge * 0.45)
                + (s.confidence * 0.30)
                + (urgency_bonus * 0.15)
                + (multi_bonus * 0.10)
            )
            # Tie-break with engine priority
            score += engine_priority * 0.01

            # Contrarian boost: high-conviction contrarian signals get a bonus
            if s.metadata.get("strategy") == "contrarian":
                conv = s.metadata.get("conviction", "low")
                if conv == "high" and abs(s.edge) >= 0.10:
                    score += 0.30
                elif conv == "medium":
                    score += 0.10

            # MM boost: spread capture with zero fees deserves priority
            if s.metadata.get("strategy") == "mm":
                score += 0.20

            # Crypto boost: time-sensitive 15-min windows need fast execution
            if s.metadata.get("strategy") in ("crypto", "crypto_latency"):
                score += 0.25

            # Cross-arb boost: structural edge from price divergence
            if s.metadata.get("strategy") == "cross_arb":
                score += 0.30

            return score

        return sorted(signals, key=_score, reverse=True)

    # ------------------------------------------------------------------
    # Multi-engine consensus
    # ------------------------------------------------------------------

    def _apply_consensus(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Detect consensus (2+ engines same direction on same market) and boost.

        Wave 23: consensus boost stored as metadata for scoring priority only.
        Confidence is NOT mutated — this prevents weak signals from getting
        inflated Kelly fractions just because multiple engines agree.
        The boost only affects execution priority via _score_signals().
        """

        # Group by (market_id, side)
        groups: Dict[tuple, List[TradeSignal]] = defaultdict(list)
        for s in signals:
            groups[(s.market_id, s.side)].append(s)

        for key, group in groups.items():
            engine_names = {s.engine for s in group}
            if len(engine_names) >= 2:
                combo_key = "+".join(sorted(engine_names))
                self._consensus_hits[combo_key] += 1
                self.logger.info(
                    "multi_engine_consensus",
                    market_id=key[0],
                    side=key[1],
                    engines=list(engine_names),
                )
                for s in group:
                    # Store boost as metadata for scoring — don't inflate confidence
                    s.metadata["_multi_engine_bonus"] = 1.0

        # Detect disagreement: same market, different sides
        by_market: Dict[str, set] = defaultdict(set)
        for s in signals:
            if s.side != "hold":
                by_market[s.market_id].add(s.side)

        disagree_markets = {mid for mid, sides in by_market.items() if len(sides) > 1}
        if disagree_markets:
            # Reduce confidence of conflicting signals
            for s in signals:
                if s.market_id in disagree_markets:
                    s.confidence *= 0.5
                    s.metadata["_disagreement"] = True
                    self.logger.debug(
                        "engine_disagreement",
                        market_id=s.market_id,
                        engine=s.engine,
                        side=s.side,
                    )

        return signals

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, signal: TradeSignal) -> Optional[KalshiTradeExecution]:
        """Convert a TradeSignal to an execution via KalshiExecutor.

        Bridges the TradeSignal → SignalResult → PositionSize flow,
        gets USD balance from Kalshi, and routes to KalshiExecutor.
        """
        if not self.kalshi_executors:
            self.logger.warning("kalshi_dispatch_no_executor", market_id=signal.market_id)
            return None

        side_map = {
            "buy_yes": TradingSide.BUY_YES,
            "buy_no": TradingSide.BUY_NO,
        }
        trading_side = side_map.get(signal.side, TradingSide.HOLD)
        if trading_side == TradingSide.HOLD:
            return None

        estimated_prob = signal.metadata.get("estimated_prob", 0.5)
        market_price = signal.metadata.get("market_price", 0.5)
        net_edge = signal.metadata.get("net_edge", signal.edge)
        conviction_str = signal.metadata.get("conviction", "medium")

        # Get Market from signal metadata, or build a stub for flow/monitor engines
        market_data = signal.metadata.get("_market")
        if market_data is None:
            from .markets import Market, TokenInfo
            ticker = signal.metadata.get("kalshi_ticker", signal.market_id)
            title = signal.metadata.get("title", ticker)
            trade_price = signal.metadata.get("trade_price") or signal.metadata.get("current_price", 0.5)

            tokens = {
                "Yes": TokenInfo(
                    token_id=ticker,
                    outcome="Yes",
                    price=trade_price,
                    volume_24h=0.0,
                ),
                "No": TokenInfo(
                    token_id=f"{ticker}:no",
                    outcome="No",
                    price=1.0 - trade_price,
                    volume_24h=0.0,
                ),
            }
            market_data = Market(
                id=ticker,
                question=title,
                description="",
                category="",
                end_date=None,
                volume_24h=float(signal.metadata.get("volume", 0)),
                liquidity=10000.0,
                tokens=tokens,
            )
            # Use the trade price as market price
            if not market_price or market_price == 0.5:
                market_price = trade_price
            if estimated_prob == 0.5:
                estimated_prob = trade_price

        first_result: Optional[KalshiTradeExecution] = None
        kalshi_trade: Optional[KalshiTradeExecution] = None

        for executor in self.kalshi_executors:
            label = executor.trading_client.label
            try:
                # Try to resume halted accounts before skipping
                if executor.trading_client.is_halted:
                    await executor.trading_client.check_and_resume()
                    if executor.trading_client.is_halted:
                        self.logger.info("kalshi_dispatch_skip_halted", label=label, market_id=signal.market_id)
                        continue

                sig_result = SignalResult(
                    estimated_prob=estimated_prob,
                    confidence=signal.confidence,
                    edge=signal.edge,
                    recommended_side=trading_side,
                    reasoning=f"[{signal.engine}] {signal.metadata.get('reasoning', '')}",
                    signal_name=signal.engine,
                    market_price=market_price,
                    timestamp=signal.timestamp.isoformat(),
                )

                sig_result.conviction = conviction_str  # type: ignore[attr-defined]
                sig_result.net_edge = net_edge  # type: ignore[attr-defined]
                sig_result.signal_source = signal.metadata.get("signal_source")  # type: ignore[attr-defined]

                # Pass time_to_close_hours for maker-priority execution
                ttc_hours = getattr(market_data, "time_to_close_hours", None) or 24.0
                sig_result.metadata = {  # type: ignore[attr-defined]
                    "kalshi_yes_ask": signal.metadata.get("kalshi_yes_ask"),
                    "kalshi_no_ask": signal.metadata.get("kalshi_no_ask"),
                    "strategy": signal.metadata.get("strategy", "standard"),
                    "signal_source": signal.metadata.get("signal_source"),
                    "_force_size_usd": signal.metadata.get("_force_size_usd"),
                    "time_to_close_hours": ttc_hours,
                }

                # Each account sizes independently based on its own balance
                available_usd = await executor.trading_client.get_balance()

                exposure: Dict[str, float] = {}
                try:
                    positions = await executor.trading_client.get_positions()
                    for p in positions:
                        exposure[p.ticker] = p.market_exposure
                except Exception:
                    self.logger.warning("exposure_fetch_failed_skip_signal", label=label, market_id=signal.market_id)
                    continue  # Skip this signal — don't size with empty exposure

                # Correlation guard: block if too much exposure on same event
                if not self.risk_manager.check_event_correlation(signal.market_id, exposure):
                    self.logger.debug("kalshi_dispatch_correlation_block", label=label, market_id=signal.market_id)
                    continue

                pos = self.risk_manager.calculate_position_size(
                    signal=sig_result,
                    market=market_data,
                    available_capital=available_usd,
                    current_positions=exposure,
                )

                if not self.risk_manager.check_trade_approval(pos, market_data, sig_result):
                    self.logger.debug("kalshi_dispatch_rejected_by_risk", label=label, market_id=signal.market_id)
                    continue

                kalshi_trade = await executor.execute_signal(
                    market_data, sig_result, pos,
                )

                # Register ALL successful orders with fill manager for tracking
                if kalshi_trade and kalshi_trade.was_successful and self.fill_manager:
                    if kalshi_trade.order_id:
                        # Get market close time for dynamic max-hold
                        _close_time = getattr(market_data, "end_date", None)
                        self.fill_manager.track_order(
                            order_id=kalshi_trade.order_id,
                            ticker=kalshi_trade.ticker,
                            side=kalshi_trade.side,
                            count=kalshi_trade.intended_contracts,
                            price_cents=kalshi_trade.price_cents,
                            account_label=label,
                            strategy=signal.engine,
                            signal_source=signal.metadata.get("signal_source", ""),
                            entry_edge=float(signal.metadata.get("net_edge", signal.edge or 0.0)),
                            close_time=_close_time,
                        )

                if kalshi_trade and kalshi_trade.was_successful and first_result is None:
                    first_result = kalshi_trade
                    break  # Don't trade same signal on second account

            except Exception as exc:
                self.logger.error("kalshi_dispatch_error", label=label, market_id=signal.market_id, error=str(exc))

        if kalshi_trade and kalshi_trade.was_successful:
            return kalshi_trade

        return first_result
