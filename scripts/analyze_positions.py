"""Analyze actual CPI positions and estimate Feb 13 P&L."""
import json
from collections import defaultdict
from pathlib import Path


def main():
    trades = []
    with open("state/trade_history.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trades.append(json.loads(line))
                except Exception:
                    pass

    # CPI orders placed (real, not dry_run)
    cpi_orders = [
        t
        for t in trades
        if t.get("event") == "order_placed"
        and "CPI" in t.get("ticker", "").upper()
        and "dry_run" not in str(t.get("order_id", ""))
    ]

    print(f"Real CPI orders placed: {len(cpi_orders)}")

    # Group by ticker
    positions = defaultdict(
        lambda: {"contracts": 0, "cost": 0.0, "orders": 0, "side": "", "prices": []}
    )
    for t in cpi_orders:
        ticker = t.get("ticker", "")
        side = t.get("side", "")
        count = t.get("count", 0) or 0
        price = t.get("price_cents", 0) or 0
        cost_usd = t.get("cost_usd", 0) or 0

        positions[ticker]["contracts"] += count
        if cost_usd:
            positions[ticker]["cost"] += cost_usd
        else:
            if side == "no":
                positions[ticker]["cost"] += count * (100 - price) / 100
            else:
                positions[ticker]["cost"] += count * price / 100
        positions[ticker]["orders"] += 1
        positions[ticker]["side"] = side
        positions[ticker]["prices"].append(price)

    total_cost = 0
    print()
    for ticker in sorted(positions):
        p = positions[ticker]
        avg_price = sum(p["prices"]) / len(p["prices"]) if p["prices"] else 0
        total_cost += p["cost"]
        print(
            f"  {ticker}: {p['side']} x{p['contracts']} "
            f"avg@{avg_price:.0f}c cost=${p['cost']:.2f} "
            f"({p['orders']} orders)"
        )

    print(f"\nTotal CPI position cost: ${total_cost:.2f}")

    # Resolved markets
    resolved = [t for t in trades if t.get("event") == "market_resolved"]
    print(f"\nResolved markets: {len(resolved)}")
    total_pnl = sum(t.get("pnl_usd", 0) or 0 for t in resolved)
    print(f"Total resolved P&L: ${total_pnl:.2f}")
    for t in resolved:
        ticker = t.get("ticker", "?")
        outcome = t.get("outcome", "?")
        held = t.get("held_side", "?")
        pnl = t.get("pnl_usd", 0) or 0
        print(f"  {ticker}: outcome={outcome} held={held} pnl=${pnl:.2f}")

    # All orders (not just CPI)
    all_orders = [
        t
        for t in trades
        if t.get("event") == "order_placed"
        and "dry_run" not in str(t.get("order_id", ""))
    ]
    print(f"\nAll real orders: {len(all_orders)}")

    # Check fills — see if orders actually filled
    print("\n--- Checking Kalshi positions via API ---")
    # Can't call API from script without auth, but show order IDs
    sample_ids = [
        t.get("order_id", "?")[:20]
        for t in cpi_orders[:5]
    ]
    print(f"Sample order IDs: {sample_ids}")


if __name__ == "__main__":
    main()
