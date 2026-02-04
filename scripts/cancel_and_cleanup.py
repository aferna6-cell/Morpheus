import asyncio, os
from dotenv import load_dotenv
load_dotenv()

from src.kalshi_trading_client import KalshiTradingClient
from src.utils import load_config

config = load_config('config.yaml')
client = KalshiTradingClient(config)

async def cancel_all():
    bal_before = await client.get_balance()
    print(f'Balance before: ${bal_before:.2f}')

    orders = await client.get_open_orders()
    print(f'\nCancelling {len(orders)} orders...')
    for o in orders:
        try:
            await client.cancel_order(o.order_id)
            print(f'  Cancelled: {o.ticker} {o.side} @ {o.price_cents}c (remaining={o.remaining})')
        except Exception as e:
            print(f'  Failed to cancel {o.order_id}: {e}')

    await asyncio.sleep(1)
    bal_after = await client.get_balance()
    print(f'\nBalance after: ${bal_after:.2f}')
    print(f'Freed up: ${bal_after - bal_before:.2f}')

asyncio.run(cancel_all())
