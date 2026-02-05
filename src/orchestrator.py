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
from .kalshi_executor import KalshiExecutor, KalshiTradeExecution
from .markets import Market
from .risk import RiskManager
from .signals.base import SignalResult, TradingSide
from .signals.llm_signal import ConvictionLevel, classify_conviction
from .alerts import send_alert
from .utils import BotConfig, utc_now


# ---------------------------------------------------------------------------
# Urgency bonuses by engine / urgency level
# ---------------------------------------------------------------------------

_URGENCY_BONUS: Dict[str, float] = {
    "immediate": 1.0,
    "normal": 0.3,
    "low": 0.0,
}

_ENGINE_PRIORITY: Dict[str, float] = {
    "kalshi_llm": 0.5,
}

# Engines whose signals route to Kalshi executor
_KALSHI_ENGINES = {"kalshi_llm", "kalshi_flow", "kalshi_monitor"}


class Orchestrator:
    """Collects signals from all engines, scores them, and decides what to trade."""

    def __init__(
        self,
        config: BotConfig,
        engines: List[BaseEngine],
        risk_manager: RiskManager,
        kalshi_executor: Optional[KalshiExecutor] = None,
        executor: Optional[object] = None,  # legacy compat, unused
    ):
        self.config = config
        self.engines = engines
        self.risk_manager = risk_manager
        self.kalshi_executor = kalshi_executor
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

        self._running = False

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

        # 3. Score and rank
        ranked = self._score_signals(fresh)

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
                    await send_alert(f"Capital management halt: {reason}", self.config)
                    break

            # Dedup: skip if we already dispatched this exact signal
            dedup_key = (signal.market_id, signal.side, signal.engine)
            if dedup_key in self._dispatched:
                self.logger.debug(
                    "dispatch_dedup_skip",
                    market_id=signal.market_id,
                    side=signal.side,
                    engine=signal.engine,
                )
                continue

            try:
                trade = await self._dispatch(signal)
                if trade and trade.was_successful:
                    executed += 1
                    self._dispatched.add(dedup_key)

                    # Register position with capital manager for recycling/CLV tracking
                    if self._capital_manager:
                        market_data = signal.metadata.get("_market")
                        resolution_time = getattr(market_data, "end_date", None) if market_data else None
                        entry_prob = signal.metadata.get("estimated_prob", 0.5)
                        side = "yes" if signal.side == "buy_yes" else "no"
                        platform = signal.metadata.get("platform", "polymarket")
                        if signal.engine in _KALSHI_ENGINES:
                            platform = "kalshi"

                        self._capital_manager.add_position(
                            market_id=signal.market_id,
                            ticker=signal.token_id or signal.market_id,
                            platform=platform,
                            amount_usd=trade.executed_amount_usd,
                            entry_price=trade.average_price,
                            entry_probability=entry_prob,
                            side=side,
                            resolution_time=resolution_time,
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
            except Exception as exc:
                self.logger.error(
                    "dispatch_error",
                    engine=signal.engine,
                    market_id=signal.market_id,
                    error=str(exc),
                )

        if executed:
            self.logger.info("orchestrator_cycle_summary", cycle=cycle,
                             signals_in=len(all_signals), fresh=len(fresh),
                             executed=executed)

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

            score = (
                (s.confidence * 0.3)
                + (max(s.edge, 0.0) * 0.3)
                + (urgency_bonus * 0.2)
                + (multi_bonus * 0.2)
            )
            # Tie-break with engine priority
            score += engine_priority * 0.01
            return score

        return sorted(signals, key=_score, reverse=True)

    # ------------------------------------------------------------------
    # Multi-engine consensus
    # ------------------------------------------------------------------

    def _apply_consensus(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Detect consensus (2+ engines same direction on same market) and boost."""

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
                    s.confidence = min(1.0, s.confidence * self._multi_boost)
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

        first_result: Optional[TradeExecution] = None

        for executor in self.kalshi_executors:
            label = executor.trading_client.label
            try:
                # Skip halted accounts
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

                try:
                    sig_result.conviction = ConvictionLevel(conviction_str)  # type: ignore[attr-defined]
                except ValueError:
                    sig_result.conviction = classify_conviction(net_edge)  # type: ignore[attr-defined]
                sig_result.net_edge = net_edge  # type: ignore[attr-defined]

                sig_result.metadata = {  # type: ignore[attr-defined]
                    "kalshi_yes_ask": signal.metadata.get("kalshi_yes_ask"),
                    "kalshi_no_ask": signal.metadata.get("kalshi_no_ask"),
                }

                # Each account sizes independently based on its own balance
                available_usd = await executor.trading_client.get_balance()

                exposure: Dict[str, float] = {}
                try:
                    positions = await executor.trading_client.get_positions()
                    for p in positions:
                        exposure[p.ticker] = p.market_exposure
                except Exception:
                    pass

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

                if kalshi_trade and kalshi_trade.was_successful and first_result is None:
                    first_result = kalshi_trade

            except Exception as exc:
                self.logger.error("kalshi_dispatch_error", label=label, market_id=signal.market_id, error=str(exc))

        if kalshi_trade and kalshi_trade.was_successful:
            return kalshi_trade

        return first_result
