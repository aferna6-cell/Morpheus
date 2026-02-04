"""Bot entrypoint — 24/7 aggressive accuracy mode.

Operational flags:
- --once: run a single iteration
- --max-loops: cap loop count
- --no-llm: disable LLM strategy
- --log-json: force JSON logs
- --state-dir: directory for persistent state files
- --replay: replay decisions from a runs/<run_id> snapshot directory

24/7 hardening:
- Global try/except: logs + sleeps 60s + retries (never exits)
- Exponential backoff on repeated errors
- Heartbeat log every N iterations
- Telegram alerts on trades, errors, daily P&L
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import structlog

from .ab_tracker import get_active_features
from .alerts import send_alert
from .correlation import filter_correlated
from .client import PolymarketClient
from .engines.llm_engine import LLMEngine
from .execution import OrderManager
from .markets import market_from_dict
from .markets_scanner import MarketScanner
from .notify import TelegramNotifier
from .orchestrator import Orchestrator
from .portfolio import Portfolio
from .resolution_tracker import check_resolutions
from .risk import RiskManager
from .runlog import RunRecorder, log_prediction
from .signals.arb_signal import ArbitrageSignal
from .cost_tracker import CostTracker
from .signals.llm_signal import LLMSignal, ConvictionLevel
from .state import TradeMemory, parse_iso
from .utils import BotConfig, exponential_backoff, load_config, setup_logging, utc_now


HEARTBEAT_INTERVAL = 10  # log heartbeat every N iterations


def _kill_switch_tripped(state_dir: Path) -> bool:
    return (state_dir / "STOP_TRADING").exists()


def _load_snapshot_markets(path: Path):
    raw = json.loads(path.read_text())
    markets = [market_from_dict(m) for m in raw.get("markets", [])]
    iteration = int(raw.get("iteration", 0))
    return iteration, markets


async def _run_replay(config: BotConfig, replay_dir: Path, *, no_llm: bool) -> None:
    logger = structlog.get_logger()

    signals = [ArbitrageSignal(config=config)]
    if not no_llm:
        signals.insert(0, LLMSignal(config=config))

    files = sorted(replay_dir.glob("markets_*.json"))
    if not files:
        raise FileNotFoundError(f"No market snapshots found in {replay_dir}")

    logger.info("replay_started", replay_dir=str(replay_dir), snapshots=len(files))

    for snap in files:
        it, markets = _load_snapshot_markets(snap)
        structlog.contextvars.bind_contextvars(iteration=it, mode="replay")

        for m in markets:
            if m.midpoint_price is None:
                continue

            chosen = None
            for sig in signals:
                res = await sig.evaluate(m)
                if res.recommended_side.value != "hold":
                    chosen = res
                    break

            if chosen:
                logger.info(
                    "replay_decision",
                    market_id=m.id,
                    question=m.question,
                    side=chosen.recommended_side.value,
                    edge=chosen.edge,
                    confidence=chosen.confidence,
                    signal=chosen.signal_name,
                )

    logger.info("replay_finished")


async def run(
    *,
    dry_run: bool,
    once: bool,
    no_llm: bool,
    max_loops: Optional[int],
    state_dir: str,
    runs_dir: str,
    replay: Optional[str],
) -> None:
    config = load_config("config.yaml")
    logger = structlog.get_logger()

    state_path = Path(state_dir)
    state_path.mkdir(parents=True, exist_ok=True)

    if replay:
        await _run_replay(config, Path(replay), no_llm=no_llm)
        return

    recorder = RunRecorder(runs_dir=runs_dir)
    notifier = TelegramNotifier()

    # persistent state
    trade_memory_path = state_path / "trade_memory.json"
    trade_memory = TradeMemory.load(trade_memory_path)

    client = PolymarketClient(config=config, dry_run=dry_run)
    await client.initialize()

    scanner = MarketScanner(config=config)
    portfolio = Portfolio(config=config, client=client, state_dir=str(state_path))
    risk = RiskManager(config=config, state_dir=str(state_path))
    exec_mgr = OrderManager(
        config=config, client=client, portfolio=portfolio,
        state_dir=str(state_path), dry_run=dry_run,
    )

    # Cost tracker with daily + monthly caps
    monthly_budget = float(config.llm.get("monthly_budget_usd", 100.0))
    daily_budget = float(config.llm.get("daily_budget_usd", 5.0))
    cost_tracker = CostTracker(
        monthly_budget=monthly_budget,
        daily_budget=daily_budget,
        state_path=state_path / "cost_tracker.json",
    )
    logger.info("cost_tracker_init", **cost_tracker.get_summary())

    enabled = [
        s.lower()
        for s in config.strategy.get("enabled_strategies", ["llm", "arb"]) or []
    ]
    signals = []
    if (not no_llm) and ("llm" in enabled):
        llm_signal = LLMSignal(config=config)
        llm_signal.set_cost_tracker(cost_tracker)
        signals.append(llm_signal)
    if "arb" in enabled:
        signals.append(ArbitrageSignal(config=config))

    # ------------------------------------------------------------------
    # Orchestrator path: if enabled, delegate to the multi-engine loop
    # ------------------------------------------------------------------
    orch_cfg = getattr(config, "orchestrator", None) or {}
    if not isinstance(orch_cfg, dict):
        orch_cfg = {}
    orch_enabled = bool(orch_cfg.get("enabled", False))

    if orch_enabled:
        from .engines.base import BaseEngine

        engines: List[BaseEngine] = []

        # LLM engine — calibrated predictions with NO-bias
        if (not no_llm) and ("llm" in enabled):
            llm_engine = LLMEngine(
                config=config,
                client=client,
                scanner=scanner,
                cost_tracker=cost_tracker,
            )
            engines.append(llm_engine)

        # Copy trading engine — mirror specific wallets
        if "copy" in enabled:
            try:
                from .engines.copy_engine import CopyTradingEngine
                copy_engine = CopyTradingEngine(config)
                engines.append(copy_engine)
            except Exception as exc:
                logger.warning("copy_engine_init_failed", error=str(exc))

        # Bregman arb engine — event-group mispricing detection (no LLM cost)
        if "bregman_arb" in enabled:
            try:
                from .engines.bregman_arb_engine import BregmanArbEngine
                bregman_engine = BregmanArbEngine(config)
                engines.append(bregman_engine)
            except Exception as exc:
                logger.warning("bregman_arb_engine_init_failed", error=str(exc))

        # Cross-platform arb engine — Polymarket ↔ Kalshi price discrepancies (no LLM cost)
        if "arb" in enabled:
            try:
                from .engines.arb_engine import ArbEngine
                arb_engine = ArbEngine(config)
                engines.append(arb_engine)
            except Exception as exc:
                logger.warning("arb_engine_init_failed", error=str(exc))

        # Kalshi LLM engine — prediction markets on Kalshi
        kalshi_exec = None
        kalshi_cfg = getattr(config, "kalshi", None) or {}
        if isinstance(kalshi_cfg, dict) and kalshi_cfg.get("enabled", False):
            try:
                from .kalshi_client import KalshiClient as KalshiReadClient
                from .kalshi_trading_client import KalshiTradingClient
                from .kalshi_executor import KalshiExecutor
                from .engines.kalshi_llm_engine import KalshiLLMEngine

                kalshi_read = KalshiReadClient(config)
                kalshi_trading = KalshiTradingClient(config, dry_run=dry_run)
                await kalshi_trading.initialize()

                kalshi_engine = KalshiLLMEngine(
                    config=config,
                    kalshi_client=kalshi_read,
                    cost_tracker=cost_tracker,
                )
                engines.append(kalshi_engine)

                kalshi_exec = KalshiExecutor(
                    config=config,
                    trading_client=kalshi_trading,
                    risk_manager=risk,
                )
                logger.info("kalshi_engine_initialized")
            except Exception as exc:
                logger.warning("kalshi_engine_init_failed", error=str(exc))

        if engines:
            orchestrator = Orchestrator(
                config=config,
                engines=engines,
                risk_manager=risk,
                executor=exec_mgr,
                kalshi_executor=kalshi_exec,
            )

            logger.info(
                "orchestrator_mode",
                engines=[e.name for e in engines],
                run_id=recorder.run_id,
            )
            await send_alert(
                f"🧠 Orchestrator online — engines: {[e.name for e in engines]}",
                config,
            )

            await orchestrator.start_engines()
            try:
                await orchestrator.run()
            finally:
                await orchestrator.stop_engines()
            return

        logger.info("orchestrator_enabled_but_no_engines_fallback_to_legacy")

    # ------------------------------------------------------------------
    # Legacy loop (orchestrator disabled or no engines)
    # ------------------------------------------------------------------

    interval = float(config.timing.get("loop_interval_seconds", 60))
    cooldown_min = float(config.risk.get("cooldown_minutes", 60))
    prevent_retrade = bool(config.risk.get("prevent_retrade", True))

    logger.info(
        "bot_started",
        dry_run=dry_run,
        interval_seconds=interval,
        run_id=recorder.run_id,
        state_dir=str(state_path),
        mode="aggressive_accuracy",
    )

    await send_alert(
        f"🔴 Morpheus online (run={recorder.run_id}, dry_run={dry_run}, mode=aggressive_accuracy)",
        config,
    )

    loop_i = 0
    consecutive_errors = 0

    while True:
        loop_i += 1
        structlog.contextvars.bind_contextvars(iteration=loop_i)

        # Heartbeat
        if loop_i % HEARTBEAT_INTERVAL == 0:
            logger.info("heartbeat", iteration=loop_i, halted=risk.trading_halted,
                         daily_pnl=risk.daily_pnl, **cost_tracker.get_summary())

        if _kill_switch_tripped(state_path):
            logger.error("kill_switch_tripped", state_dir=str(state_path))
            await send_alert("🛑 Kill switch tripped — bot stopping", config)
            break

        started = utc_now()

        try:
            markets = await scanner.fetch_markets()
            markets = await scanner.enrich_midpoints(client, markets)

            # Filter correlated markets — keep only the best from each cluster
            max_per_cluster = int(config.market_filters.get("max_per_correlation_cluster", 1))
            markets, corr_skips = filter_correlated(
                markets, max_per_cluster=max_per_cluster,
            )

            recorder.write_market_snapshot(iteration=loop_i, markets=markets)

            exposure = portfolio.get_current_exposure()
            balances = await client.get_balances()
            usdc = balances.get(
                "USDC", config.dev.get("paper_trading_balance", 10_000.0)
            )

            for m in markets:
                if m.midpoint_price is None:
                    continue

                # cooldown / no-retrade guards
                if prevent_retrade and trade_memory.traded_markets.get(m.id):
                    continue

                last_ts = trade_memory.last_trade_ts_by_market.get(m.id)
                if last_ts:
                    dt = parse_iso(last_ts)
                    if dt:
                        age_min = (utc_now() - dt).total_seconds() / 60.0
                        if age_min < cooldown_min:
                            continue

                chosen = None
                for sig in signals:
                    res = await sig.evaluate(m)
                    if res.recommended_side.value != "hold":
                        chosen = res
                        break

                if not chosen:
                    continue

                pos = risk.calculate_position_size(
                    signal=chosen,
                    market=m,
                    available_capital=usdc,
                    current_positions=exposure,
                )

                conviction = getattr(chosen, "conviction", ConvictionLevel.NONE)
                net_edge = getattr(chosen, "net_edge", 0.0)

                decision_event = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "iteration": loop_i,
                    "market_id": m.id,
                    "question": m.question,
                    "signal": chosen.signal_name,
                    "side": chosen.recommended_side.value,
                    "edge": chosen.edge,
                    "net_edge": net_edge,
                    "conviction": conviction.value if hasattr(conviction, "value") else str(conviction),
                    "confidence": chosen.confidence,
                    "sizing_usd": pos.amount_usd,
                    "sizing_reason": pos.reasoning,
                }

                if not risk.check_trade_approval(pos, m, chosen):
                    decision_event["action"] = "skip"
                    recorder.log_decision(decision_event)
                    continue

                trade = await exec_mgr.execute_signal(m, chosen, pos)

                decision_event.update(
                    {
                        "action": "trade",
                        "order_id": trade.order_id,
                        "result": trade.result.value,
                        "executed_usd": trade.executed_amount_usd,
                        "avg_price": trade.average_price,
                        "reason": trade.reason,
                    }
                )
                recorder.log_decision(decision_event)

                # Log prediction for accuracy tracking on ALL approved trades
                # (including dry-run simulated ones)
                log_prediction(
                    str(state_path),
                    market_id=m.id,
                    predicted_p_yes=chosen.estimated_prob,
                    market_price_at_entry=chosen.market_price or m.midpoint_price or 0.5,
                    side=chosen.recommended_side.value,
                    edge=chosen.edge,
                    conviction=conviction.value if hasattr(conviction, "value") else str(conviction),
                    net_edge=net_edge,
                    features_active=get_active_features(config),
                )

                if trade.was_successful:
                    trade_memory.last_trade_ts_by_market[m.id] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    trade_memory.traded_markets[m.id] = True
                    trade_memory.save(trade_memory_path)

                    alert_msg = (
                        f"💰 Trade: {chosen.recommended_side.value.upper()} "
                        f"${trade.executed_amount_usd:.2f} @ {trade.average_price:.3f}\n"
                        f"Conviction: {conviction.value if hasattr(conviction, 'value') else conviction} "
                        f"| Edge: {chosen.edge:+.3f} | Net: {net_edge:+.3f}\n"
                        f"{m.question[:100]}"
                    )
                    await send_alert(alert_msg, config)

            # Update position prices & lifecycle exits
            await portfolio.update_position_prices()
            to_close = await portfolio.check_position_exits(risk)
            for pid in to_close:
                await exec_mgr.close_position(pid, reason="risk_exit")

            # Resolution tracker: check if past predictions have resolved
            resolution_interval = float(
                config.timing.get("resolution_check_interval_minutes", 30)
            )
            # Run resolution check every N loops (approximate minutes)
            loops_per_check = max(1, int(resolution_interval * 60 / interval))
            if loop_i % loops_per_check == 0:
                try:
                    newly = await check_resolutions(
                        config, str(state_path), dry_run=dry_run,
                    )
                    if newly > 0:
                        await send_alert(
                            f"📊 {newly} market(s) resolved — run `python3 -m src.accuracy` for updated stats",
                            config,
                        )
                except Exception as res_err:
                    logger.warning("resolution_check_error", error=str(res_err))

            # Log LLM cost optimization summary
            for sig in signals:
                if isinstance(sig, LLMSignal) and hasattr(sig, "log_periodic_summary"):
                    sig.log_periodic_summary()

            # Reset error counter on success
            consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            tb = traceback.format_exc()
            logger.error("loop_error", error=str(e), consecutive=consecutive_errors,
                         traceback=tb)

            if consecutive_errors <= 3:
                await send_alert(f"⚠️ Loop error #{consecutive_errors}: {str(e)[:200]}", config)

            # Exponential backoff on repeated errors
            await exponential_backoff(consecutive_errors, base_delay=5.0, max_delay=300.0)

        elapsed = (utc_now() - started).total_seconds()
        sleep_for = max(1.0, interval - elapsed)

        if once:
            break
        if max_loops is not None and loop_i >= max_loops:
            break

        await asyncio.sleep(sleep_for)


def main() -> None:
    # Auto-load .env so users don't need `set -a && source .env && set +a`
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--max-loops", type=int, default=None)
    parser.add_argument("--log-json", action="store_true")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--replay", default=None)
    args = parser.parse_args()

    config = load_config("config.yaml")
    setup_logging(config, log_json=args.log_json)

    # 24/7 hardening: never crash out
    while True:
        try:
            asyncio.run(
                run(
                    dry_run=args.dry_run,
                    once=args.once,
                    no_llm=args.no_llm,
                    max_loops=args.max_loops,
                    state_dir=args.state_dir,
                    runs_dir=args.runs_dir,
                    replay=args.replay,
                )
            )
            # If we get here normally (once/max_loops/kill_switch), exit
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            # Unhandled exception — sleep and retry
            import time
            print(f"FATAL ERROR (restarting in 60s): {e}", flush=True)
            traceback.print_exc()
            if args.once:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
