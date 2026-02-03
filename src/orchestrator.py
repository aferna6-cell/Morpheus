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

from .engines.base import BaseEngine
from .engines.signals import TradeSignal
from .execution import OrderManager, TradeExecution
from .markets import Market
from .risk import RiskManager
from .signals.base import SignalResult, TradingSide
from .signals.llm_signal import ConvictionLevel, classify_conviction
from .alerts import send_alert
from .utils import BotConfig, utc_now

# Optional Kalshi executor (imported lazily if needed)
try:
    from .kalshi_executor import KalshiExecutor
except ImportError:
    KalshiExecutor = None  # type: ignore


# ---------------------------------------------------------------------------
# Urgency bonuses by engine / urgency level
# ---------------------------------------------------------------------------

_URGENCY_BONUS: Dict[str, float] = {
    "immediate": 1.0,
    "normal": 0.3,
    "low": 0.0,
}

_ENGINE_PRIORITY: Dict[str, float] = {
    "bregman_arb": 1.0,  # mathematical arb → highest
    "arb": 0.95,          # simple arb → very high
    "spike": 0.8,         # time-sensitive
    "copy": 0.6,          # moderate-high
    "llm": 0.5,           # moderate
    "kalshi_llm": 0.5,   # same priority as Polymarket LLM
}

# Engines whose signals route to Kalshi executor
_KALSHI_ENGINES = {"kalshi_llm"}

# Engines whose signals are mathematically risk-free (skip conviction gating)
_ARB_ENGINES = {"bregman_arb", "arb"}


class Orchestrator:
    """Collects signals from all engines, scores them, and decides what to trade."""

    def __init__(
        self,
        config: BotConfig,
        engines: List[BaseEngine],
        risk_manager: RiskManager,
        executor: OrderManager,
        kalshi_executor: Optional["KalshiExecutor"] = None,
    ):
        self.config = config
        self.engines = engines
        self.risk_manager = risk_manager
        self.executor = executor
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

                    # Send alert for successful trades
                    question = signal.metadata.get("question", "")
                    if not question:
                        m = signal.metadata.get("_market")
                        question = getattr(m, "question", signal.market_id) if m else signal.market_id
                    conviction = signal.metadata.get("conviction", "?")
                    layer = signal.metadata.get("layer")
                    label = signal.metadata.get("label", "")

                    if signal.engine == "bregman_arb" and layer:
                        net_profit = signal.metadata.get("net_profit", 0)
                        alert_msg = (
                            f"🎯 [BREGMAN L{layer}] {signal.side.upper()} "
                            f"${trade.executed_amount_usd:.2f} @ {trade.average_price:.3f}\n"
                            f"Net profit: ${net_profit:.4f} | Conf: {signal.confidence:.2f}\n"
                            f"{label[:100]}"
                        )
                    else:
                        alert_msg = (
                            f"💰 [{signal.engine.upper()}] {signal.side.upper()} "
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

    async def _dispatch(self, signal: TradeSignal) -> Optional[TradeExecution]:
        """Convert a TradeSignal to an execution via OrderManager.

        We need to bridge from the engine TradeSignal format to the
        existing SignalResult + Market that OrderManager.execute_signal expects.

        Kalshi signals are routed to the Kalshi executor.

        Arb engines (bregman_arb, arb) get special treatment:
        - Bypass conviction gating (it's math, not opinion)
        - Use execution plan sizing if available
        - Force HIGH conviction for risk manager approval
        """

        # Route Kalshi signals to the Kalshi executor
        if signal.engine in _KALSHI_ENGINES:
            return await self._dispatch_kalshi(signal)


        # Build a minimal SignalResult compatible with execute_signal
        side_map = {
            "buy_yes": TradingSide.BUY_YES,
            "buy_no": TradingSide.BUY_NO,
        }
        trading_side = side_map.get(signal.side, TradingSide.HOLD)
        if trading_side == TradingSide.HOLD:
            return None

        is_arb = signal.engine in _ARB_ENGINES

        estimated_prob = signal.metadata.get("estimated_prob", 0.5)
        market_price = signal.metadata.get("market_price") or signal.metadata.get("leg_price", 0.5)
        net_edge = signal.metadata.get("net_edge", signal.edge)
        conviction_str = signal.metadata.get("conviction", "high" if is_arb else "medium")

        # For arb signals, set estimated_prob from leg pricing
        if is_arb and estimated_prob == 0.5:
            leg_price = signal.metadata.get("leg_price")
            if leg_price:
                # Arb: we know the fair value, so estimated_prob reflects our side
                estimated_prob = leg_price if signal.side == "buy_yes" else 1.0 - leg_price

        sig_result = SignalResult(
            estimated_prob=estimated_prob,
            confidence=signal.confidence,
            edge=signal.edge,
            recommended_side=trading_side,
            reasoning=f"[{signal.engine}] {signal.metadata.get('reasoning', signal.metadata.get('label', ''))}",
            signal_name=signal.engine,
            market_price=market_price,
            timestamp=signal.timestamp.isoformat(),
        )

        # Arb signals → force HIGH conviction (risk-free)
        if is_arb:
            sig_result.conviction = ConvictionLevel.HIGH  # type: ignore[attr-defined]
            sig_result.net_edge = max(net_edge, 0.10)  # type: ignore[attr-defined]
        else:
            try:
                sig_result.conviction = ConvictionLevel(conviction_str)  # type: ignore[attr-defined]
            except ValueError:
                sig_result.conviction = classify_conviction(net_edge)  # type: ignore[attr-defined]
            sig_result.net_edge = net_edge  # type: ignore[attr-defined]

        # Build a minimal Market stub from signal metadata
        market_data = signal.metadata.get("_market")
        if market_data is None:
            from .markets import Market, TokenInfo
            question = signal.metadata.get("question", signal.market_id)
            original_price = signal.metadata.get("original_price") or signal.metadata.get("leg_price", 0.5)
            copy_size = signal.metadata.get("copy_size_usd")

            # Build token info from the signal
            tokens = {}
            if signal.token_id:
                wanted = "Yes" if signal.side == "buy_yes" else "No"
                tokens[wanted] = TokenInfo(
                    token_id=signal.token_id,
                    outcome=wanted,
                    price=original_price if wanted == "Yes" else 1.0 - original_price,
                    volume_24h=0.0,
                )

            if not tokens:
                self.logger.warning("dispatch_no_token", market_id=signal.market_id)
                return None

            category = "arbitrage" if is_arb else "copy_trade"
            market_data = Market(
                id=signal.market_id,
                question=question,
                description="",
                category=category,
                end_date=None,
                volume_24h=0.0,
                liquidity=10000.0,
                tokens=tokens,
            )
            # Override position size with copy_size if available
            if copy_size:
                signal.metadata["_force_size_usd"] = copy_size

        # Calculate position size
        exposure = self.executor.portfolio.get_current_exposure()
        balances = await self.executor.client.get_balances()
        usdc = balances.get(
            "USDC",
            self.config.dev.get("paper_trading_balance", 10_000.0),
        )

        # Arb: use execution plan sizing if available
        if is_arb:
            exec_plan = signal.metadata.get("execution_plan", {})
            arb_max = signal.metadata.get("max_position_usd", 100.0)
            signal.metadata["_force_size_usd"] = min(arb_max, usdc * 0.10)

        pos = self.risk_manager.calculate_position_size(
            signal=sig_result,
            market=market_data,
            available_capital=usdc,
            current_positions=exposure,
        )

        if not self.risk_manager.check_trade_approval(pos, market_data, sig_result):
            self.logger.debug("dispatch_rejected_by_risk", market_id=signal.market_id)
            return None

        # Attach metadata for execution layer
        exec_meta: Dict[str, Any] = {}
        if signal.urgency == "immediate" or is_arb:
            exec_meta["urgent"] = True
        # Pass through forced sizing so RiskManager can consume it
        force_size = signal.metadata.get("_force_size_usd")
        if force_size is not None:
            exec_meta["_force_size_usd"] = force_size
        sig_result.metadata = exec_meta  # type: ignore[attr-defined]

        trade = await self.executor.execute_signal(market_data, sig_result, pos)
        return trade

    # ------------------------------------------------------------------
    # Kalshi dispatch
    # ------------------------------------------------------------------

    async def _dispatch_kalshi(self, signal: TradeSignal) -> Optional[TradeExecution]:
        """Route a Kalshi signal to the Kalshi executor.

        Bridges the same TradeSignal → SignalResult → PositionSize flow
        but uses USD balance from Kalshi and routes to KalshiExecutor.
        Returns a duck-typed TradeExecution-compatible object.
        """
        if self.kalshi_executor is None:
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

        # Pass through Kalshi-specific metadata for the executor
        sig_result.metadata = {  # type: ignore[attr-defined]
            "kalshi_yes_ask": signal.metadata.get("kalshi_yes_ask"),
            "kalshi_no_ask": signal.metadata.get("kalshi_no_ask"),
        }

        # Get Market from signal metadata
        market_data = signal.metadata.get("_market")
        if market_data is None:
            self.logger.warning("kalshi_dispatch_no_market", market_id=signal.market_id)
            return None

        # Get Kalshi balance for position sizing
        available_usd = await self.kalshi_executor.trading_client.get_balance()

        # Use same risk manager for position sizing
        exposure: Dict[str, float] = {}
        try:
            positions = await self.kalshi_executor.trading_client.get_positions()
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
            self.logger.debug("kalshi_dispatch_rejected_by_risk", market_id=signal.market_id)
            return None

        kalshi_trade = await self.kalshi_executor.execute_signal(
            market_data, sig_result, pos,
        )

        # Wrap as duck-typed TradeExecution for the orchestrator
        if kalshi_trade and kalshi_trade.was_successful:
            return type("_KalshiTradeProxy", (), {
                "was_successful": kalshi_trade.was_successful,
                "executed_amount_usd": kalshi_trade.executed_amount_usd,
                "average_price": kalshi_trade.average_price,
            })()  # type: ignore[return-value]

        return None
