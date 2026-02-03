"""Bankroll tracking — USDC flow logging and P&L summary.

Tracks deposits, withdrawals, trade costs, and resolution payouts
in state_dir/bankroll.jsonl.

Usage:
    python3 -m src.bankroll [--state-dir state]
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import structlog

from .utils import BotConfig, append_jsonl, load_config


def log_bankroll_event(
    state_dir: str,
    *,
    market_id: str,
    side: str,
    amount_usdc: float,
    price: float,
    fee_estimate: float,
    event_type: str,  # entry | exit | resolution_payout | resolution_loss | deposit | withdrawal
) -> None:
    """Append a bankroll event to state_dir/bankroll.jsonl."""
    record = {
        "market_id": market_id,
        "side": side,
        "amount_usdc": round(amount_usdc, 4),
        "price": round(price, 6),
        "fee_estimate": round(fee_estimate, 4),
        "type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    bankroll_path = Path(state_dir) / "bankroll.jsonl"
    append_jsonl(bankroll_path, record)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def bankroll_summary(state_dir: str, starting_bankroll: float = 10_000.0) -> Dict[str, Any]:
    """Compute bankroll summary from bankroll.jsonl."""
    state_path = Path(state_dir)
    events = _read_jsonl(state_path / "bankroll.jsonl")

    total_invested = 0.0
    total_returned = 0.0
    total_fees = 0.0
    entries = 0
    payouts = 0
    losses = 0

    for ev in events:
        t = ev.get("type", "")
        amount = float(ev.get("amount_usdc", 0))
        fee = float(ev.get("fee_estimate", 0))

        if t == "entry":
            total_invested += amount
            total_fees += fee
            entries += 1
        elif t == "exit":
            total_returned += amount
            total_fees += fee
        elif t == "resolution_payout":
            total_returned += amount
            payouts += 1
        elif t == "resolution_loss":
            losses += 1
        elif t == "deposit":
            starting_bankroll += amount
        elif t == "withdrawal":
            starting_bankroll -= amount

    net_pnl = total_returned - total_invested - total_fees
    current_value = starting_bankroll + net_pnl
    roi_pct = (net_pnl / starting_bankroll * 100) if starting_bankroll > 0 else 0.0

    return {
        "starting_bankroll": round(starting_bankroll, 2),
        "current_value": round(current_value, 2),
        "total_invested": round(total_invested, 2),
        "total_returned": round(total_returned, 2),
        "total_fees": round(total_fees, 2),
        "net_pnl": round(net_pnl, 2),
        "roi_pct": round(roi_pct, 2),
        "entries": entries,
        "payouts": payouts,
        "losses": losses,
        "total_events": len(events),
    }


def print_summary(state_dir: str, starting_bankroll: float = 10_000.0) -> None:
    """Print a formatted bankroll summary."""
    s = bankroll_summary(state_dir, starting_bankroll)

    print("\n" + "=" * 60)
    print("  💰 BANKROLL SUMMARY")
    print("=" * 60)
    print(f"  Starting bankroll:  ${s['starting_bankroll']:>12,.2f}")
    print(f"  Current value:      ${s['current_value']:>12,.2f}")
    print(f"  Total invested:     ${s['total_invested']:>12,.2f}")
    print(f"  Total returned:     ${s['total_returned']:>12,.2f}")
    print(f"  Total fees:         ${s['total_fees']:>12,.2f}")
    print(f"  Net P&L:            ${s['net_pnl']:>+12,.2f}")
    print(f"  ROI:                 {s['roi_pct']:>+11.2f}%")
    print("-" * 60)
    print(f"  Entries: {s['entries']}  |  Payouts: {s['payouts']}  |  Losses: {s['losses']}")
    print(f"  Total events: {s['total_events']}")
    print("=" * 60 + "\n")


def main() -> None:
    """Standalone bankroll summary."""
    import argparse

    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Bankroll summary")
    parser.add_argument("--state-dir", default="state")
    args = parser.parse_args()

    config = load_config("config.yaml")
    starting = float(config.dev.get("paper_trading_balance", 10_000.0))
    print_summary(args.state_dir, starting_bankroll=starting)


if __name__ == "__main__":
    main()
