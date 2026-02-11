"""Resolution tracker — polls Kalshi API for resolved markets.

Reads predictions from state_dir/predictions.jsonl, checks resolution
status via the Kalshi public API, and writes outcomes to state_dir/resolutions.jsonl.

Usage:
    python3 -m src.resolution_tracker [--state-dir state]
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
import structlog

from .trade_logger import get_trade_logger
from .capital_management import get_capital_manager
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


def _build_order_lookup(state_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Build market_id -> list of order_placed events from trade history.

    Skips duplicate null-field entries (from old KalshiTradingClient bug
    that logged orders twice — once with nulls, once with real data).
    Deduplicates by order_id, keeping the entry with the most data.
    """
    trades = _read_jsonl(state_dir / "trade_history.jsonl")
    # Deduplicate by order_id — keep the entry with edge/cost data
    by_order_id: Dict[str, Dict[str, Any]] = {}
    for t in trades:
        if t.get("event") != "order_placed":
            continue
        oid = t.get("order_id", "")
        if oid and oid in by_order_id:
            # Keep the one with more data (non-null edge)
            if t.get("edge") is not None:
                by_order_id[oid] = t
        else:
            by_order_id[oid or id(t)] = t

    lookup: Dict[str, List[Dict[str, Any]]] = {}
    for t in by_order_id.values():
        # ticker in trade history may have ":yes"/":no" suffix — strip it
        ticker = t.get("ticker", "")
        market_id = ticker.split(":")[0] if ":" in ticker else ticker
        if market_id:
            lookup.setdefault(market_id, []).append(t)
    return lookup


def _compute_pnl(
    orders: List[Dict[str, Any]],
    actual_outcome: float,
) -> tuple[float, int]:
    """Compute dollar P&L and total contract count from orders + outcome.

    Kalshi binary: buy side at price_cents. If your side wins, payout = 100c/contract.
    If your side loses, payout = 0.

    Returns (pnl_usd, total_count).
    """
    total_pnl = 0.0
    total_count = 0
    outcome_yes = actual_outcome >= 0.5

    for order in orders:
        count = int(order.get("count") or 0)
        price_cents = float(order.get("price_cents") or 0)
        side = (order.get("side") or "").lower()

        if count <= 0 or price_cents <= 0:
            continue

        total_count += count
        cost = count * price_cents / 100.0  # dollars spent

        # Did our side win?
        won = (side == "yes" and outcome_yes) or (side == "no" and not outcome_yes)

        if won:
            payout = count * 1.00  # $1 per contract
            total_pnl += payout - cost
        else:
            total_pnl -= cost

    return round(total_pnl, 2), total_count


async def check_resolutions(
    config: BotConfig,
    state_dir: str | Path,
    *,
    dry_run: bool = False,
) -> int:
    """Check all unresolved predictions AND traded markets for resolution.

    Returns the number of newly resolved markets.
    """
    logger = structlog.get_logger()
    state_path = Path(state_dir)

    predictions = _read_jsonl(state_path / "predictions.jsonl")
    already_resolved = _resolved_market_ids(state_path)
    order_lookup = _build_order_lookup(state_path)

    # Collect unique unresolved market ids from predictions
    unresolved: Dict[str, Dict[str, Any]] = {}
    for pred in predictions:
        mid = pred.get("market_id")
        if mid and mid not in already_resolved and mid not in unresolved:
            unresolved[mid] = pred

    # Also add traded markets that have no prediction entry
    # (e.g. market-making orders, or orders placed before prediction logging)
    for market_id, orders in order_lookup.items():
        if market_id not in already_resolved and market_id not in unresolved:
            # Build a synthetic prediction from order data
            first_order = orders[0]
            side = first_order.get("side", "unknown")
            price_cents = float(first_order.get("price_cents") or 50)
            unresolved[market_id] = {
                "market_id": market_id,
                "predicted_p_yes": 0.5,  # unknown — no LLM prediction
                "market_price_at_entry": price_cents / 100.0,
                "side": f"buy_{side}" if side in ("yes", "no") else side,
                "edge": 0.0,
                "conviction": "unknown",
                "timestamp": first_order.get("logged_at", ""),
            }

    if not unresolved:
        logger.debug("resolution_tracker_all_resolved")
        return 0

    logger.info(
        "resolution_tracker_checking",
        total_unresolved=len(unresolved),
        from_predictions=len([u for u in unresolved.values() if u.get("conviction") != "unknown"]),
        from_trades_only=len([u for u in unresolved.values() if u.get("conviction") == "unknown"]),
    )

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
                if r.status_code == 429:
                    logger.debug("resolution_tracker_rate_limited", market_id=market_id)
                    await asyncio.sleep(5.0)
                    r = await client.get(f"/markets/{market_id}")
                    if r.status_code == 429:
                        logger.warning("resolution_tracker_rate_limited_backoff", checked_so_far=newly_resolved)
                        break  # Stop checking, resume next cycle
                if r.status_code != 200:
                    logger.warning(
                        "resolution_tracker_api_error",
                        market_id=market_id,
                        status=r.status_code,
                    )
                    continue

                data = r.json()
                market_data = data.get("market", data)

                # Kalshi market status: "open", "closed", "finalized"
                status = market_data.get("status", "").lower()

                if status not in ("settled", "finalized", "closed"):
                    continue

                # Determine outcome from Kalshi's result field
                # Kalshi uses "result": "yes" or "result": "no"
                # "closed" markets have empty result — skip those (not yet settled)
                result_str = (
                    market_data.get("result", "")
                    or market_data.get("resolution", "")
                    or ""
                ).strip().lower()

                if not result_str and status == "closed":
                    # Market closed but not yet settled — skip for now
                    continue

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

                # Compute dollar P&L from actual orders
                orders = order_lookup.get(market_id, [])
                pnl_usd, total_count = _compute_pnl(orders, actual_outcome)

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
                    "pnl_usd": pnl_usd,
                    "total_count": total_count,
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
                    pnl_usd=pnl_usd,
                    contracts=total_count,
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
                    held_count=total_count,
                    pnl_usd=pnl_usd,
                    closing_probability=closing_price,
                    entry_probability=entry_prob,
                    clv=round(clv_val, 4),
                )

                # Remove from capital manager's open positions
                try:
                    cm = get_capital_manager()
                    cm.remove_position(market_id, closing_price=actual_outcome)
                except Exception:
                    pass  # Capital manager may not be initialized

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

            # Rate-limit: ~1 request per 0.5s to stay under Kalshi limits
            await asyncio.sleep(0.5)

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
