"""Emergency sell script: liquidate all KXFED positions.

Wave 33: Cross-arb engine accidentally bought year-long FED rate bets.
This script sells them all at the current bid to free up capital.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.kalshi_trading_client import KalshiTradingClient
from src.kalshi_client import KalshiClient
from src.utils import load_config


async def main():
    config = load_config()

    # Initialize the public client for price data
    public = KalshiClient(
        os.environ.get("KALSHI_API_URL", "https://api.elections.kalshi.com/trade-api/v2")
    )

    # Build list of trading clients (same as main.py)
    clients = []

    # Primary account
    primary = KalshiTradingClient(config, dry_run=False, label="kalshi_primary")
    await primary.initialize()
    clients.append(primary)

    # Secondary account
    key_id_2 = os.getenv("KALSHI_API_KEY_ID_2")
    key_path_2 = os.getenv("KALSHI_PRIVATE_KEY_PATH_2")
    if key_id_2 and key_path_2:
        secondary = KalshiTradingClient(
            config, dry_run=False, label="kalshi_secondary",
            api_key_id=key_id_2, private_key_path=key_path_2,
        )
        await secondary.initialize()
        clients.append(secondary)

    for client in clients:
        print(f"\n=== {client.label} ===")

        try:
            positions = await client.get_positions()
        except Exception as e:
            print(f"  Error getting positions: {e}")
            continue

        fed_positions = [p for p in positions if p.ticker.startswith("KXFED")]

        if not fed_positions:
            print("  No KXFED positions found")
            continue

        for pos in fed_positions:
            ticker = pos.ticker
            count = pos.count  # positive = long YES
            side = "yes" if count > 0 else "no"
            abs_count = abs(count)

            if abs_count == 0:
                continue

            # Get current market price
            try:
                market = await public.fetch_market(ticker)
                if market:
                    if side == "yes":
                        bid_price = int(market.yes_bid * 100)
                    else:
                        no_bid = 1.0 - market.yes_ask
                        bid_price = int(no_bid * 100)

                    sell_price = max(1, bid_price)
                    recovery = abs_count * sell_price / 100.0

                    print(f"  {ticker}: {abs_count}x {side}")
                    print(f"    yes_bid={market.yes_bid}, yes_ask={market.yes_ask}")
                    print(f"    Selling at {sell_price}c → recovery ${recovery:.2f}")
                else:
                    print(f"  {ticker}: market not found, selling at 1c")
                    sell_price = 1
            except Exception as e:
                print(f"  {ticker}: price error ({e}), selling at 1c")
                sell_price = 1

            # Place the sell order
            try:
                result = await client.place_order(
                    ticker=ticker,
                    side=side,
                    count=abs_count,
                    price_cents=sell_price,
                    order_type="limit",
                    is_exit=True,
                )
                if result:
                    order_info = result.get("order", {})
                    order_id = order_info.get("order_id", "unknown")
                    status = order_info.get("status", "unknown")
                    print(f"    SOLD: order_id={order_id}, status={status}")
                else:
                    print(f"    FAILED: place_order returned None")
            except Exception as e:
                print(f"    FAILED: {e}")

    print("\nDone. Check Kalshi dashboard to confirm fills.")


if __name__ == "__main__":
    asyncio.run(main())
