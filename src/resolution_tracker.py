"""Resolution tracker — polls Kalshi API for resolved markets.

Reads predictions from state_dir/predictions.jsonl, checks resolution
status via the Kalshi public API, and writes outcomes to state_dir/resolutions.jsonl.

Usage:
    python3 -m src.resolution_tracker [--state-dir state]
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
import structlog

from .trade_logger import get_trade_logger
from .utils import BotConfig, append_jsonl, load_config


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read all records from a JSONL file."""
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _resolved_market_ids(state_dir: Path) -> Set[str]:
    """Return set of market_ids already in resolutions.jsonl."""
    resolutions = _read_jsonl(state_dir / "resolutions.jsonl")
    return {r["market_id"] for r in resolutions if "market_id" in r}


async def check_resolutions(
    config: BotConfig,
    state_dir: str | Path,
    *,
    dry_run: bool = False,
) -> int:
    """Check all unresolved predictions for resolution status via Kalshi API.

    Returns the number of newly resolved markets.
    """
    logger = structlog.get_logger()
    state_path = Path(state_dir)

    predictions = _read_jsonl(state_path / "predictions.jsonl")
    if not predictions:
        logger.debug("resolution_tracker_no_predictions")
        return 0

    already_resolved = _resolved_market_ids(state_path)

    # Collect unique unresolved market ids
    unresolved: Dict[str, Dict[str, Any]] = {}
    for pred in predictions:
        mid = pred.get("market_id")
        if mid and mid not in already_resolved and mid not in unresolved:
            unresolved[mid] = pred

    if not unresolved:
        logger.debug("resolution_tracker_all_resolved")
        return 0

    # Kalshi public API base URL
    kalshi_cfg = getattr(config, "kalshi", None) or {}
    if isinstance(kalshi_cfg, dict):
        kalshi_base_url = kalshi_cfg.get(
            "base_url", "https://api.elections.kalshi.com/trade-api/v2"
        )
    else:
        kalshi_base_url = "https://api.elections.kalshi.com/trade-api/v2"

    newly_resolved = 0

    async with httpx.AsyncClient(
        base_url=kalshi_base_url,
        timeout=20.0,
        headers={"Accept": "application/json"},
    ) as client:
        for market_id, pred in unresolved.items():
            try:
                r = await client.get(f"/markets/{market_id}")
                if r.status_code != 200:
                    logger.warning(
                        "resolution_tracker_api_error",
                        market_id=market_id,
                        status=r.status_code,
                    )
                    continue

                data = r.json()
                market_data = data.get("market", data)

                # Kalshi market status: "open", "closed", "settled"
                status = market_data.get("status", "").lower()

                if status not in ("settled", "closed"):
                    continue

                # Determine outcome from Kalshi's result field
                # Kalshi uses "result": "yes" or "result": "no"
                result_str = (
                    market_data.get("result", "")
                    or market_data.get("resolution", "")
                    or ""
                ).strip().lower()

                if result_str in ("yes", "true", "1"):
                    actual_outcome = 1.0
                elif result_str in ("no", "false", "0"):
                    actual_outcome = 0.0
                else:
                    # Check if settled by looking at final prices
                    # If settled, yes_price should be ~1.0 or ~0.0
                    last_price = market_data.get("last_price", -1)
                    if isinstance(last_price, (int, float)):
                        # Kalshi prices in cents (0-100) or dollars (0-1)
                        price_val = last_price / 100.0 if last_price > 1 else last_price
                        if price_val >= 0.99:
                            actual_outcome = 1.0
                        elif price_val <= 0.01:
                            actual_outcome = 0.0
                        elif status == "settled":
                            # Settled but price unclear — skip for now
                            logger.debug(
                                "resolution_unclear",
                                market_id=market_id,
                                result=result_str,
                                last_price=last_price,
                            )
                            continue
                        else:
                            continue
                    else:
                        continue

                resolution_record = {
                    "market_id": market_id,
                    "predicted_p_yes": pred.get("predicted_p_yes", 0.5),
                    "actual_outcome": actual_outcome,
                    "market_price_at_entry": pred.get("market_price_at_entry", 0.5),
                    "side": pred.get("side", "unknown"),
                    "edge": pred.get("edge", 0.0),
                    "conviction": pred.get("conviction", "unknown"),
                    "timestamp": pred.get("timestamp", ""),
                    "resolved_at": datetime.now(timezone.utc).isoformat(),
                    "platform": "kalshi",
                }

                # Include features_active if present
                if "features_active" in pred:
                    resolution_record["features_active"] = pred["features_active"]

                append_jsonl(state_path / "resolutions.jsonl", resolution_record)
                newly_resolved += 1

                logger.info(
                    "market_resolved",
                    market_id=market_id,
                    actual_outcome=actual_outcome,
                    predicted_p_yes=pred.get("predicted_p_yes"),
                    side=pred.get("side"),
                )

                # Log to trade history
                trade_logger = get_trade_logger()
                outcome_str_log = "yes" if actual_outcome >= 0.5 else "no"
                entry_price = pred.get("market_price_at_entry", 0.5)
                closing_price = actual_outcome  # resolved = 1.0 or 0.0
                entry_prob = pred.get("predicted_p_yes", 0.5)
                side_val = pred.get("side", "unknown")

                # Calculate CLV: how much the line moved toward our prediction
                if "yes" in side_val.lower():
                    clv_val = closing_price - entry_price
                else:
                    clv_val = entry_price - closing_price

                trade_logger.log_market_resolution(
                    platform="kalshi",
                    ticker=market_id,
                    outcome=outcome_str_log,
                    held_side=side_val,
                    closing_probability=closing_price,
                    entry_probability=entry_prob,
                    clv=round(clv_val, 4),
                )

                # Log bankroll payout if not dry-run
                if not dry_run:
                    _log_resolution_payout(
                        state_path, pred, actual_outcome,
                    )

            except Exception as e:
                logger.error(
                    "resolution_tracker_error",
                    market_id=market_id,
                    error=str(e),
                )

    logger.info("resolution_check_complete", newly_resolved=newly_resolved)
    return newly_resolved


def _log_resolution_payout(
    state_path: Path,
    prediction: Dict[str, Any],
    actual_outcome: float,
) -> None:
    """Log resolution payout to bankroll."""
    try:
        from .bankroll import log_bankroll_event

        market_id = prediction.get("market_id", "unknown")
        side = prediction.get("side", "unknown")
        entry_price = prediction.get("market_price_at_entry", 0.5)

        # Determine if we won
        bet_yes = "yes" in side.lower()
        won = (bet_yes and actual_outcome >= 0.5) or (not bet_yes and actual_outcome < 0.5)

        if won:
            # Payout is 1.0 per share; profit = 1.0 - entry_price per share
            payout_per_dollar = 1.0 / entry_price if entry_price > 0 else 1.0
            log_bankroll_event(
                str(state_path),
                market_id=market_id,
                side=side,
                amount_usdc=payout_per_dollar,
                price=1.0,
                fee_estimate=0.0,
                event_type="resolution_payout",
            )
        else:
            # Loss: we lose our entry cost (already logged at entry)
            log_bankroll_event(
                str(state_path),
                market_id=market_id,
                side=side,
                amount_usdc=0.0,
                price=0.0,
                fee_estimate=0.0,
                event_type="resolution_loss",
            )
    except Exception:
        pass  # bankroll logging is best-effort


def main() -> None:
    """Standalone resolution check."""
    import argparse
    import asyncio

    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Check market resolutions")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config("config.yaml")

    from .utils import setup_logging
    setup_logging(config, log_json=False)

    count = asyncio.run(
        check_resolutions(config, args.state_dir, dry_run=args.dry_run)
    )
    print(f"\nNewly resolved: {count}")


if __name__ == "__main__":
    main()
