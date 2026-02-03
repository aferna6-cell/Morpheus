import asyncio, os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timezone
from src.kalshi_client import KalshiClient
from src.utils import load_config

config = load_config('config.yaml')
client = KalshiClient(config)

async def check():
    await client.start()
    categories = ["Politics", "Economics", "Elections", "World", "Climate and Weather",
                  "Financials", "Science and Technology", "Entertainment", "Companies", "Social"]
    markets = await client.fetch_markets_by_events(categories=categories, min_volume=500)
    
    now = datetime.now(timezone.utc)
    passing = []
    for m in markets:
        if not m.close_time:
            continue
        days = (m.close_time - now).total_seconds() / 86400.0
        if days <= 30:
            passing.append((days, m))
    
    passing.sort(key=lambda x: x[0])
    print(f"Total markets fetched: {len(markets)}")
    print(f"Markets within 30 days: {len(passing)}")
    print()
    for days, m in passing:
        print(f"  {days:.1f}d | vol={m.volume_24h} | yes={m.yes_price:.2f} | {m.ticker}")
        print(f"       {m.title[:80]}")
    
    await client.stop()

asyncio.run(check())
