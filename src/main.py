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

            kalshi_read = KalshiReadClient(config)

            # Primary Kalshi account
            kalshi_trading = KalshiTradingClient(config, dry_run=dry_run, label="kalshi_primary")
            await kalshi_trading.initialize()
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
                kalshi_exec_secondary = KalshiExecutor(
                    config=config,
                    trading_client=kalshi_trading_2,
                    risk_manager=risk,
                )
                kalshi_executors.append(kalshi_exec_secondary)
                logger.info("kalshi_secondary_account_initialized")

            if not no_llm and "kalshi_llm" in enabled:
                kalshi_engine = KalshiLLMEngine(
                    config=config,
                    kalshi_client=kalshi_read,
                    cost_tracker=cost_tracker,
                )

                # Balance gate — skip LLM calls when all accounts are unfunded
                _kalshi_execs = list(kalshi_executors)  # capture for closure
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

            # Kalshi flow engine — follow large trades on the tape (no LLM cost)
            if "kalshi_flow" in enabled:
                try:
                    from .engines.kalshi_flow_engine import KalshiFlowEngine
                    flow_engine = KalshiFlowEngine(config, kalshi_read)
                    engines.append(flow_engine)
                    logger.info("kalshi_flow_engine_initialized")
                except Exception as exc:
                    logger.warning("kalshi_flow_engine_init_failed", error=str(exc))

            # Kalshi monitor engine — detect price spikes (no LLM cost)
            if "kalshi_monitor" in enabled:
                try:
                    from .engines.kalshi_monitor_engine import KalshiMonitorEngine
                    monitor_engine = KalshiMonitorEngine(config, kalshi_read)
                    engines.append(monitor_engine)
                    logger.info("kalshi_monitor_engine_initialized")
                except Exception as exc:
                    logger.warning("kalshi_monitor_engine_init_failed", error=str(exc))

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
    )

    logger.info(
        "orchestrator_mode",
        engines=[e.name for e in engines],
        dry_run=dry_run,
    )
    # Only alert on actual trades, not on startup

    await orchestrator.start_engines()
    try:
        await orchestrator.run()
    finally:
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
