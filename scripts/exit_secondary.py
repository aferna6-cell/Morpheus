"""Check and exit losing CPI positions on the SECONDARY Kalshi account."""
import asyncio
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import httpx
from src.utils import load_config
from src.kalshi_trading_client import KalshiTradingClient


async def main():
    config = load_config("config.yaml")

    # Initialize secondary trading client
    client = KalshiTradingClient(
        config=config,
        api_key_id=os.getenv("KALSHI_API_KEY_ID_2"),
        private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH_2"),
        label="secondary",
    )
    await client.initialize()

    balance = await client.get_balance()
    positions = await client.get_positions()

    print(f"SECONDARY account:")
    print(f"  Cash balance: ${balance:.2f}")
    print(f"  Open positions: {len(positions)}")
    print()

    # Show all positions
    cpi_positions = []
    for p in positions:
        ticker = p.ticker
        qty = abs(p.count)
        exposure = p.market_exposure
        side = "yes" if p.count > 0 else "no"
        is_cpi = "CPI" in ticker.upper()
        marker = " [CPI]" if is_cpi else ""
        print(f"  {ticker}: {side} x{qty} exposure=${exposure:.2f}{marker}")
        if is_cpi:
            cpi_positions.append((ticker, side, qty, exposure))

    total_exposure = sum(p.market_exposure for p in positions)
    print(f"\n  Total exposure: ${total_exposure:.2f}")
    print(f"  Total value: ${balance + total_exposure:.2f}")

    if not cpi_positions:
        print("\nNo CPI positions to exit.")
        return

    # Get current market prices for CPI positions
    base = "https://api.elections.kalshi.com/trade-api/v2"
    print(f"\n{'='*70}")
    print("CPI positions to exit:")

    async with httpx.AsyncClient(base_url=base, timeout=10.0) as http:
        for ticker, side, qty, exposure in cpi_positions:
            r = await http.get(f"/markets/{ticker}")
            if r.status_code != 200:
                print(f"  SKIP {ticker}: HTTP {r.status_code}")
                continue

            data = r.json().get("market", r.json())
            no_bid = data.get("no_bid", 0) or 0
            yes_bid = data.get("yes_bid", 0) or 0

            # Determine if this is a losing position
            # NO positions on near-certain YES markets should be exited
            # YES positions on near-certain YES markets should be KEPT
            if side == "no":
                sell_price = no_bid
                if sell_price <= 0:
                    print(f"  SKIP {ticker}: NO x{qty} exp=${exposure:.2f} — bid is 0c")
                    continue

                proceeds = qty * sell_price / 100
                print(f"  SELL {ticker}: NO x{qty} @ {sell_price}c = ${proceeds:.2f} (exp=${exposure:.2f})")

                try:
                    result = await client._run_in_executor(
                        client._portfolio.create_order,
                        ticker=ticker,
                        client_order_id=str(uuid.uuid4()),
                        side="no",
                        action="sell",
                        count=qty,
                        type="limit",
                        no_price=sell_price,
                    )
                    order_data = result.to_dict() if hasattr(result, "to_dict") else result
                    order_id = (
                        getattr(result, "order_id", None)
                        or (order_data.get("order", {}).get("order_id")
                            if isinstance(order_data, dict) else None)
                        or "unknown"
                    )
                    print(f"    ORDER PLACED: {order_id}")
                except Exception as e:
                    print(f"    ERROR: {str(e)[:200]}")

                await asyncio.sleep(0.5)

            else:
                # YES position — check if it's worth keeping
                print(f"  KEEP {ticker}: YES x{qty} exp=${exposure:.2f} (yes_bid={yes_bid}c)")

    # Check final balance
    await asyncio.sleep(2)
    new_balance = await client.get_balance()
    new_positions = await client.get_positions()
    new_exposure = sum(p.market_exposure for p in new_positions)
    print(f"\n{'='*70}")
    print(f"New balance: ${new_balance:.2f} (was ${balance:.2f})")
    print(f"Remaining positions: {len(new_positions)}, exposure: ${new_exposure:.2f}")
    print(f"Total value: ${new_balance + new_exposure:.2f}")


asyncio.run(main())
