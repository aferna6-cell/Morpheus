"""Exit losing CPI positions before Feb 13 resolution.

Sells NO positions on markets almost certain to resolve YES.
Uses limit orders at or near the current bid to minimize slippage.
"""
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

# Positions to exit: (ticker, side_held, qty, reason)
POSITIONS_TO_EXIT = [
    ("KXCPI-26JAN-T0.1", "no", 33, "CPI > 0.1% near-certain YES"),
    ("KXCPICORE-26JAN-T0.1", "no", 236, "Core CPI > 0.1% near-certain YES"),
    ("KXCPICORE-26JAN-T0.2", "no", 4, "Core CPI > 0.2% likely YES"),
    ("KXCPICOMBO-26JAN-0024", "no", 2, "CPI>0% AND YoY>2.4% likely YES"),
    ("KXCPICOMBO-26JAN-0123", "no", 2, "CPI>0.1% AND YoY>2.3% likely YES"),
    ("KXCPIYOY-26JAN-T2.3", "no", 48, "YoY CPI > 2.3% near-certain YES (current 3.03%)"),
    ("KXCPIYOY-26JAN-T2.4", "no", 10, "YoY CPI > 2.4% near-certain YES (current 3.03%)"),
    # NOT exiting:
    # KXCPI-26JAN-T-0.1: NO bid at 0c, can't sell
    # KXCPI-26JAN-T0.3: YES position on 50/50 market, keep it
]


async def get_current_bids():
    """Fetch current NO bids from Kalshi for each position."""
    base = "https://api.elections.kalshi.com/trade-api/v2"
    bids = {}
    async with httpx.AsyncClient(base_url=base, timeout=10.0) as client:
        for ticker, side, qty, reason in POSITIONS_TO_EXIT:
            r = await client.get(f"/markets/{ticker}")
            if r.status_code != 200:
                print(f"  SKIP {ticker}: HTTP {r.status_code}")
                continue
            data = r.json().get("market", r.json())
            no_bid = data.get("no_bid", 0) or 0
            no_ask = data.get("no_ask", 0) or 0
            yes_bid = data.get("yes_bid", 0) or 0
            yes_ask = data.get("yes_ask", 0) or 0
            bids[ticker] = {
                "no_bid": no_bid,
                "no_ask": no_ask,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
            }
    return bids


async def main():
    config = load_config("config.yaml")
    bids = await get_current_bids()

    # Initialize trading client
    client = KalshiTradingClient(
        config=config,
        api_key_id=os.getenv("KALSHI_KEY_ID"),
        private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH"),
        label="primary_exit",
    )
    await client.initialize()

    balance = await client.get_balance()
    print(f"Account balance: ${balance:.2f}")
    print()

    print("Executing exit orders:")
    print("=" * 70)

    total_proceeds = 0
    total_exposure_saved = 0

    for ticker, side, qty, reason in POSITIONS_TO_EXIT:
        if ticker not in bids:
            print(f"  SKIP {ticker}: no market data")
            continue

        bid_data = bids[ticker]
        no_bid = bid_data["no_bid"]

        if no_bid <= 0:
            print(f"  SKIP {ticker}: NO bid is {no_bid}c (no buyers)")
            continue

        # Sell NO at the bid price
        sell_price = no_bid
        proceeds = qty * sell_price / 100
        total_proceeds += proceeds

        print(f"  SELL {ticker}: {qty} NO contracts @ {sell_price}c = ${proceeds:.2f}")
        print(f"    Reason: {reason}")

        try:
            # Use SDK directly for sell action
            order_kwargs = dict(
                ticker=ticker,
                client_order_id=str(uuid.uuid4()),
                side="no",
                action="sell",
                count=qty,
                type="limit",
                no_price=sell_price,
            )
            result = await client._run_in_executor(
                client._portfolio.create_order, **order_kwargs
            )
            order_data = result.to_dict() if hasattr(result, "to_dict") else result
            order_id = (
                getattr(result, "order_id", None)
                or (order_data.get("order", {}).get("order_id")
                    if isinstance(order_data, dict) else None)
                or "unknown"
            )
            print(f"    ORDER PLACED: {order_id}")
            total_exposure_saved += proceeds
        except Exception as e:
            error_msg = str(e)
            print(f"    ERROR: {error_msg[:200]}")

        await asyncio.sleep(0.5)  # rate limit

    print()
    print("=" * 70)
    print(f"Total exit proceeds (if filled): ${total_proceeds:.2f}")
    print(f"Remaining exposure saved from resolution loss")

    # Check final balance
    await asyncio.sleep(2)
    new_balance = await client.get_balance()
    print(f"New balance: ${new_balance:.2f} (was ${balance:.2f})")


asyncio.run(main())
