"""Check actual Kalshi positions via API."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.utils import load_config
from src.kalshi_trading_client import KalshiTradingClient


async def check_account(config, key_id_env, key_path_env, label):
    client = KalshiTradingClient(
        config=config,
        api_key_id=os.getenv(key_id_env),
        private_key_path=os.getenv(key_path_env),
        label=label,
    )
    await client.initialize()

    balance = await client.get_balance()
    positions = await client.get_positions()

    print(f"\n{label.upper()} account:")
    print(f"  Cash balance: ${balance:.2f}")
    print(f"  Open positions: {len(positions)}")

    total_exposure = 0
    cpi_positions = []

    for p in positions:
        ticker = p.ticker
        qty = abs(p.count)
        exposure = p.market_exposure
        side = "yes" if p.count > 0 else "no"
        total_exposure += exposure

        is_cpi = "CPI" in ticker.upper()
        marker = " [CPI]" if is_cpi else ""
        print(f"  {ticker}: {side} x{qty} exposure=${exposure:.2f}{marker}")

        if is_cpi:
            cpi_positions.append({
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "exposure": exposure,
            })

    print(f"  Total exposure: ${total_exposure:.2f}")
    print(f"  Total value: ${balance + total_exposure:.2f}")

    return balance, total_exposure, cpi_positions


async def main():
    config = load_config("config.yaml")

    bal1, exp1, cpi1 = await check_account(
        config, "KALSHI_KEY_ID", "KALSHI_PRIVATE_KEY_PATH", "primary"
    )
    bal2, exp2, cpi2 = await check_account(
        config, "KALSHI_KEY_ID_2", "KALSHI_PRIVATE_KEY_PATH_2", "secondary"
    )

    all_cpi = cpi1 + cpi2
    print(f"\n{'='*60}")
    print(f"TOTALS:")
    print(f"  Cash: ${bal1 + bal2:.2f}")
    print(f"  Positions: ${exp1 + exp2:.2f}")
    print(f"  Total value: ${bal1 + bal2 + exp1 + exp2:.2f}")

    if all_cpi:
        cpi_exp = sum(p["exposure"] for p in all_cpi)
        print(f"\nCPI positions (resolving Feb 13):")
        for p in all_cpi:
            print(f"  {p['ticker']}: {p['side']} x{p['qty']} ${p['exposure']:.2f}")
        print(f"  Total CPI exposure: ${cpi_exp:.2f}")
    else:
        print("\nNo CPI positions found.")


asyncio.run(main())
