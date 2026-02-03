"""Resolution tracker — polls Gamma API for resolved markets.

Reads predictions from state_dir/predictions.jsonl, checks resolution
status via the Gamma API, and writes outcomes to state_dir/resolutions.jsonl.

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
    """Check all unresolved predictions for resolution status.

    Returns the number of newly resolved markets.
    """
    from .bankroll import log_bankroll_event  # local import to avoid circular

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

    gamma_url = config.polymarket.get(
        "gamma_url", "https://gamma-api.polymarket.com"
    )

    newly_resolved = 0

    async with httpx.AsyncClient(timeout=20.0) as client:
        for market_id, pred in unresolved.items():
            try:
                r = await client.get(f"{gamma_url}/markets/{market_id}")
                if r.status_code != 200:
                    logger.warning(
                        "resolution_tracker_api_error",
                        market_id=market_id,
                        status=r.status_code,
                    )
                    continue

                data = r.json()

                # Gamma API fields for resolution
                closed = data.get("closed", False)
                resolved = data.get("resolved", False)

                if not (closed or resolved):
                    continue

                # Determine outcome
                # Gamma stores resolution in various fields; try common ones
                outcome_str = (
                    data.get("resolutionOutcome")
                    or data.get("resolution")
                    or data.get("outcome")
                    or ""
                ).strip().lower()

                if outcome_str in ("yes", "true", "1"):
                    actual_outcome = 1.0
                elif outcome_str in ("no", "false", "0"):
                    actual_outcome = 0.0
                else:
                    # Try outcome prices: if YES token = 1.0, resolved YES
                    outcome_prices = data.get("outcomePrices")
                    if outcome_prices:
                        if isinstance(outcome_prices, str):
                            try:
                                outcome_prices = json.loads(outcome_prices)
                            except json.JSONDecodeError:
                                continue
                        if isinstance(outcome_prices, list) and len(outcome_prices) >= 1:
                            yes_price = float(outcome_prices[0])
                            if yes_price >= 0.99:
                                actual_outcome = 1.0
                            elif yes_price <= 0.01:
                                actual_outcome = 0.0
                            else:
                                # Not clearly resolved
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
                    platform="polymarket",
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
            payout_per_dollar = 1.0 / entry_price  # shares per dollar * $1 payout
            log_bankroll_event(
                str(state_path),
                market_id=market_id,
                side=side,
                amount_usdc=payout_per_dollar,  # normalized to $1 entry
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
