import asyncio, os
from dotenv import load_dotenv
load_dotenv()

from src.kalshi_trading_client import KalshiTradingClient
from src.utils import load_config

config = load_config('config.yaml')
client = KalshiTradingClient(config)

async def check():
    bal = await client.get_balance()
    print(f'Kalshi balance: ${bal:.2f}')

    positions = await client.get_positions()
    print(f'\nOpen positions: {len(positions)}')
    for p in positions:
        print(f'  {p}')

    # Check fills via the _portfolio API
    try:
        from src.kalshi_trading_client import _run_sync
        fills = await client._run_in_executor(client._portfolio.get_fills, limit=10)
        print(f'\nRecent fills:')
        if hasattr(fills, 'fills'):
            for f in (fills.fills or []):
                print(f'  ticker={f.ticker} side={f.side} count={f.count} yes_price={f.yes_price} no_price={f.no_price} created={f.created_time}')
        else:
            print(f'  Raw: {fills}')
    except Exception as e:
        print(f'Fills error: {e}')

    # Check resting orders
    try:
        orders = await client.get_open_orders()
        print(f'\nOpen orders: {len(orders)}')
        for o in orders:
            print(f'  {o}')
    except Exception as e:
        print(f'Orders error: {e}')

asyncio.run(check())
