"""Morpheus — Kalshi trading bot entrypoint.

Operational flags:
- --once: run a single iteration
- --max-loops: cap loop count
- --no-llm: disable LLM strategy
- --log-json: force JSON logs
- --state-dir: directory for persistent state files

24/7 hardening:
- Global try/except: logs + sleeps 60s + retries (never exits)
- Exponential backoff on repeated errors
- Heartbeat log every N iterations
- Telegram alerts on trades, errors, daily P&L
"""

from __future__ import annotations

import argparse
import asyncio
import traceback
from pathlib import Path
from typing import List, Optional

import structlog

from .alerts import send_alert
from .cost_tracker import CostTracker
from .orchestrator import Orchestrator
from .resolution_tracker import check_resolutions
from .risk import RiskManager
from .trade_logger import get_trade_logger
from .utils import BotConfig, load_config, setup_logging


async def run(
    *,
    dry_run: bool,
    once: bool,
    no_llm: bool,
    max_loops: Optional[int],
    state_dir: str,
    runs_dir: str,
) -> None:
    config = load_config("config.yaml")
    logger = structlog.get_logger()

    state_path = Path(state_dir)
    state_path.mkdir(parents=True, exist_ok=True)

    risk = RiskManager(config=config, state_dir=str(state_path))

    # Cost tracker with daily + monthly caps
    monthly_budget = float(config.llm.get("monthly_budget_usd", 100.0))
    daily_budget = float(config.llm.get("daily_budget_usd", 5.0))
    cost_tracker = CostTracker(
        monthly_budget=monthly_budget,
        daily_budget=daily_budget,
        state_path=state_path / "cost_tracker.json",
    )
    logger.info("cost_tracker_init", **cost_tracker.get_summary())

    # Initialize trade logger
    get_trade_logger(log_path=str(state_path / "trade_history.jsonl"))

    # ------------------------------------------------------------------
    # Kalshi engine setup
    # ------------------------------------------------------------------
    from .engines.base import BaseEngine

    engines: List[BaseEngine] = []
    kalshi_executors: List = []
    enabled = config.strategy.get("enabled_strategies", [])

    kalshi_cfg = getattr(config, "kalshi", None) or {}
    if isinstance(kalshi_cfg, dict) and kalshi_cfg.get("enabled", False):
        try:
            import os
            from .kalshi_client import KalshiClient as KalshiReadClient
            from .kalshi_trading_client import KalshiTradingClient
            from .kalshi_executor import KalshiExecutor
            from .engines.kalshi_llm_engine import KalshiLLMEngine

            from .position_monitor import PositionMonitor
            from .fill_manager import FillManager

            kalshi_read = KalshiReadClient(config)

            # Trading clients for both accounts
            trading_clients: list[KalshiTradingClient] = []

            # Primary Kalshi account
            kalshi_trading = KalshiTradingClient(config, dry_run=dry_run, label="kalshi_primary")
            await kalshi_trading.initialize()
            trading_clients.append(kalshi_trading)
            kalshi_exec_primary = KalshiExecutor(
                config=config,
                trading_client=kalshi_trading,
                risk_manager=risk,
            )
            kalshi_executors.append(kalshi_exec_primary)

            # Secondary Kalshi account (if configured)
            key_id_2 = os.getenv("KALSHI_API_KEY_ID_2")
            key_path_2 = os.getenv("KALSHI_PRIVATE_KEY_PATH_2")
            if key_id_2 and key_path_2:
                kalshi_trading_2 = KalshiTradingClient(
                    config, dry_run=dry_run, label="kalshi_secondary",
                    api_key_id=key_id_2,
                    private_key_path=key_path_2,
                )
                await kalshi_trading_2.initialize()
                # Always add to trading_clients for position monitoring/exits
                trading_clients.append(kalshi_trading_2)
                # Always add executor — halt/resume handles low balance gracefully.
                # Disabling at startup prevents trading even after positions resolve.
                kalshi_exec_secondary = KalshiExecutor(
                    config=config,
                    trading_client=kalshi_trading_2,
                    risk_manager=risk,
                )
                kalshi_executors.append(kalshi_exec_secondary)
                try:
                    bal = await kalshi_trading_2.get_balance()
                    balance_usd = bal / 100.0 if bal > 100 else bal
                except Exception:
                    balance_usd = 0.0
                logger.info("kalshi_secondary_account_initialized", balance=balance_usd)

            # Position monitor — enforce exits
            position_monitor = PositionMonitor(
                config=config,
                trading_clients=trading_clients,
                risk_manager=risk,
                state_dir=state_dir,
            )

            # Fill manager — track real fills (persists resting orders to disk)
            fill_manager = FillManager(
                config=config,
                trading_clients=trading_clients,
                state_dir=state_dir,
            )

            # Wire fill events to position tracking
            def _on_fill(event):
                position_monitor.track_position(
                    ticker=event.ticker,
                    side=event.side,
                    count=event.filled_count,
                    entry_price_cents=event.price_cents,
                    strategy=event.strategy,
                    order_id=event.order_id,
                    account_label=event.account_label,
                    entry_edge=getattr(event, "entry_edge", 0.0),
                    close_time=getattr(event, "close_time", None),
                    signal_source=getattr(event, "signal_source", ""),
                )
            fill_manager.on_fill(_on_fill)

            # Liquidate existing positions on startup if configured
            orch_cfg = getattr(config, "orchestrator", {}) or {}
            if isinstance(orch_cfg, dict) and orch_cfg.get("liquidate_on_startup", False):
                logger.info("liquidating_all_positions_on_startup")
                await position_monitor.liquidate_all()

            if not no_llm and "kalshi_llm" in enabled:
                kalshi_engine = KalshiLLMEngine(
                    config=config,
                    kalshi_client=kalshi_read,
                    cost_tracker=cost_tracker,
                )

                # Balance gate
                _kalshi_execs = list(kalshi_executors)
                async def _total_kalshi_balance() -> float:
                    total = 0.0
                    for ex in _kalshi_execs:
                        try:
                            total += await ex.trading_client.get_balance()
                        except Exception:
                            pass
                    return total
                kalshi_engine.set_balance_checker(_total_kalshi_balance, min_balance=1.0)

                engines.append(kalshi_engine)
                logger.info("kalshi_llm_engine_initialized", accounts=len(kalshi_executors))

            # Contrarian engine (bet against overconfident crowds)
            contrarian_cfg = getattr(config, "contrarian", None) or {}
            if isinstance(contrarian_cfg, dict) and contrarian_cfg.get("enabled", False):
                if "kalshi_contrarian" in enabled:
                    from .engines.kalshi_contrarian_engine import KalshiContrarianEngine
                    contrarian_engine = KalshiContrarianEngine(
                        config=config,
                        kalshi_client=kalshi_read,
                        cost_tracker=cost_tracker,
                    )
                    contrarian_engine.set_balance_checker(_total_kalshi_balance, min_balance=1.0)
                    engines.append(contrarian_engine)
                    logger.info("kalshi_contrarian_engine_initialized")

            # Market making engine (zero maker fees)
            mm_cfg = getattr(config, "market_making", None) or {}
            if isinstance(mm_cfg, dict) and mm_cfg.get("enabled", False):
                from .engines.kalshi_mm_engine import KalshiMMEngine
                mm_engine = KalshiMMEngine(
                    config=config,
                    kalshi_client=kalshi_read,
                )
                engines.append(mm_engine)

                # Wire MM inventory tracking to fill manager
                def _on_mm_fill(event):
                    try:
                        mm_engine.update_inventory(
                            event.ticker,
                            event.side,
                            event.filled_count,
                        )
                    except Exception:
                        pass
                fill_manager.on_fill(_on_mm_fill)

                logger.info("kalshi_mm_engine_initialized")

            # Wire resume callback: when a halted account resumes, trigger engine rescans
            _engines_for_rescan = list(engines)
            def _on_account_resume():
                for eng in _engines_for_rescan:
                    if hasattr(eng, "trigger_rescan"):
                        eng.trigger_rescan()
            position_monitor.on_resume(_on_account_resume)

            logger.info("kalshi_setup_complete", executors=len(kalshi_executors), engines=[e.name for e in engines])
        except Exception as exc:
            logger.error("kalshi_engine_init_failed", error=str(exc))
            raise

    if not engines:
        logger.error("no_engines_available")
        await send_alert("No trading engines available — check config", config)
        return

    orchestrator = Orchestrator(
        config=config,
        engines=engines,
        risk_manager=risk,
        executor=None,
        kalshi_executors=kalshi_executors,
        fill_manager=fill_manager if 'fill_manager' in dir() else None,
    )

    logger.info(
        "morpheus_v2_started",
        engines=[e.name for e in engines],
        dry_run=dry_run,
        accounts=len(kalshi_executors),
    )
    await send_alert(
        f"Morpheus v2 started | Engines: {[e.name for e in engines]} | "
        f"Accounts: {len(kalshi_executors)} | Dry run: {dry_run}",
        config,
    )

    await orchestrator.start_engines()

    # Performance tracker — daily summaries and P&L alerts
    from .perf_tracker import PerfTracker
    perf_tracker = PerfTracker(config=config, state_dir=state_dir)

    # Survival tracker — self-sustainability gate
    from .survival import SurvivalTracker
    survival_tracker = SurvivalTracker(
        config=config,
        perf_tracker=perf_tracker,
        cost_tracker=cost_tracker,
        state_dir=state_dir,
    )
    perf_tracker.set_survival_tracker(survival_tracker)
    orchestrator.survival_tracker = survival_tracker

    # Resolution tracker — periodic background task
    _resolution_task: asyncio.Task | None = None

    async def _resolution_loop() -> None:
        """Poll Kalshi for resolved markets every 15 minutes."""
        while True:
            try:
                count = await check_resolutions(
                    config, state_dir, dry_run=dry_run,
                )
                if count > 0:
                    logger.info("resolution_tracker_resolved", count=count)
                    # Capital freed — re-scale risk limits to total equity
                    try:
                        total_bal = 0.0
                        for ex in kalshi_executors:
                            total_bal += await ex.trading_client.get_balance()
                            for p in await ex.trading_client.get_positions():
                                total_bal += p.market_exposure
                        risk.scale_limits_to_balance(total_bal)
                    except Exception as exc:
                        logger.debug("resolution_balance_scale_failed", error=str(exc))
                    # Trigger immediate rescan on all engines
                    for engine in engines:
                        if hasattr(engine, "trigger_rescan"):
                            engine.trigger_rescan()
                    logger.info("rescan_triggered_after_resolution", resolved=count)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("resolution_tracker_loop_error", error=str(e))
            await asyncio.sleep(900)  # 15 minutes

    # Telegram bot — /status command handler
    from .telegram_bot import TelegramBot
    telegram_bot = TelegramBot(config=config, state_dir=state_dir)
    telegram_bot.set_perf_tracker(perf_tracker)
    telegram_bot.set_cost_tracker(cost_tracker)
    telegram_bot.set_survival_tracker(survival_tracker)
    if kalshi_executors:
        telegram_bot.set_trading_clients([ex.trading_client for ex in kalshi_executors])

    # Auto-scale risk limits based on total equity (cash + positions)
    if 'kalshi_executors' in dir() and kalshi_executors:
        try:
            total_bal = 0.0
            for ex in kalshi_executors:
                total_bal += await ex.trading_client.get_balance()
                for p in await ex.trading_client.get_positions():
                    total_bal += p.market_exposure
            logger.info("equity_for_scaling", total_equity=round(total_bal, 2))
            risk.scale_limits_to_balance(total_bal)
        except Exception as exc:
            logger.warning("initial_balance_scale_failed", error=str(exc))

    # Start background services
    if 'position_monitor' in dir():
        await position_monitor.start()
    if 'fill_manager' in dir():
        await fill_manager.start()
    await perf_tracker.start()
    await telegram_bot.start()
    _resolution_task = asyncio.create_task(_resolution_loop())
    logger.info("resolution_tracker_started", interval_sec=900)

    try:
        await orchestrator.run()
    finally:
        if _resolution_task:
            _resolution_task.cancel()
            try:
                await _resolution_task
            except asyncio.CancelledError:
                pass
        await perf_tracker.stop()
        await telegram_bot.stop()
        if 'position_monitor' in dir():
            await position_monitor.stop()
        if 'fill_manager' in dir():
            await fill_manager.stop()
        await orchestrator.stop_engines()


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Morpheus — Kalshi trading bot")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--max-loops", type=int, default=None)
    parser.add_argument("--log-json", action="store_true")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--runs-dir", default="runs")
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
                )
            )
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            import time
            print(f"FATAL ERROR (restarting in 60s): {e}", flush=True)
            traceback.print_exc()
            if args.once:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
