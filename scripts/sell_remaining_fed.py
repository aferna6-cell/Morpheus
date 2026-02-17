"""Sell remaining KXFED positions."""
import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from src.kalshi_trading_client import KalshiTradingClient
from src.kalshi_client import KalshiClient
from src.utils import load_config

async def main():
    config = load_config()
    public = KalshiClient("https://api.elections.kalshi.com/trade-api/v2")
    key_id_2 = os.getenv("KALSHI_API_KEY_ID_2")
    key_path_2 = os.getenv("KALSHI_PRIVATE_KEY_PATH_2")
    client = KalshiTradingClient(
        config, dry_run=False, label="kalshi_secondary",
        api_key_id=key_id_2, private_key_path=key_path_2,
    )
    await client.initialize()
    client._trading_halted = False

    # Get all remaining KXFED positions
    positions = await client.get_positions()
    fed_pos = [p for p in positions if p.ticker.startswith("KXFED")]
    print(f"Found {len(fed_pos)} KXFED positions")

    for pos in fed_pos:
        ticker = pos.ticker
        count = abs(pos.count)
        side = "yes" if pos.count > 0 else "no"
        if count == 0:
            continue

        m = await public.fetch_market(ticker)
        if not m:
            print(f"  {ticker}: market not found, skipping")
            continue

        if side == "yes":
            bid = int(m.yes_bid * 100)
        else:
            bid = int((1.0 - m.yes_ask) * 100)
        sell_price = max(1, bid)

        print(f"  {ticker}: {count}x {side} → sell at {sell_price}c")
        try:
            result = await client.place_order(
                ticker=ticker, side=side, count=count,
                price_cents=sell_price, order_type="limit", is_exit=True,
            )
            if result:
                order = result.get("order", {})
                print(f"    OK: {order.get('order_id', '?')} status={order.get('status', '?')}")
            else:
                print(f"    FAILED: returned None")
        except Exception as e:
            print(f"    ERROR: {e}")

    bal = await client.get_balance()
    print(f"\nBalance: ${bal:.2f}")

asyncio.run(main())
