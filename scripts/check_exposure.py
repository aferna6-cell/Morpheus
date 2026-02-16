#!/usr/bin/env python3
"""Quick diagnostic: show positions and exposure per account."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.utils import load_config
from src.kalshi_trading_client import KalshiTradingClient


async def main():
    config = load_config()
    accounts = [
        (None, None, "primary"),  # uses default env vars
        (os.getenv("KALSHI_API_KEY_ID_2"), os.getenv("KALSHI_PRIVATE_KEY_PATH_2"), "secondary"),
    ]
    grand_total_exp = 0.0
    grand_total_cash = 0.0

    for key_id, pk, label in accounts:
        kwargs = {"config": config, "label": f"kalshi_{label}"}
        if key_id and pk:
            kwargs["api_key_id"] = key_id
            kwargs["private_key_path"] = pk
        client = KalshiTradingClient(**kwargs)
        try:
            await client.initialize()
            bal = await client.get_balance()
        except Exception as e:
            bal = 0.0
            print(f"  [balance error: {e}]")

        grand_total_cash += bal

        try:
            positions = await client.get_positions()
        except Exception as e:
            positions = []
            print(f"  [positions error: {e}]")

        total_exp = sum(p.market_exposure for p in positions)
        grand_total_exp += total_exp

        print(f"\n=== {label} | cash=${bal:.2f} | halted={client.is_halted} | positions={len(positions)} | exposure=${total_exp:.2f} ===")
        for p in sorted(positions, key=lambda x: -x.market_exposure):
            side = "yes" if p.count > 0 else "no"
            print(f"  {p.ticker:45s} {side:3s} x{abs(p.count):>5d}  exp=${p.market_exposure:>8.2f}")

    print(f"\n--- TOTALS: cash=${grand_total_cash:.2f}, exposure=${grand_total_exp:.2f}, equity=${grand_total_cash + grand_total_exp:.2f} ---")

    # Show what the risk manager would compute
    max_total_exposure_config = config.strategy.get("max_total_exposure", 25.0)
    total_equity = grand_total_cash + grand_total_exp
    baseline = 6.56
    if total_equity > baseline:
        scale = total_equity / baseline
        scaled_exp = min(max_total_exposure_config * scale, total_equity * 0.65)
    else:
        scaled_exp = max_total_exposure_config
    print(f"--- Risk: max_total_exposure=${scaled_exp:.2f}, current={grand_total_exp:.2f}, remaining=${scaled_exp - grand_total_exp:.2f} ---")


asyncio.run(main())
