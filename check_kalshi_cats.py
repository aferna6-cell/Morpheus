from src.kalshi_client import KalshiClient
from src.utils import load_config
import asyncio
from datetime import datetime, timezone, timedelta

config = load_config('config.yaml')
client = KalshiClient(config)

async def check():
    await client.start()
    now = datetime.now(timezone.utc)
    week_from_now = now + timedelta(days=7)
    
    # Try filtering by max close time
    # Kalshi API supports: min_close_ts, max_close_ts as epoch seconds
    max_ts = int(week_from_now.timestamp())
    min_ts = int(now.timestamp())
    
    data = await client._get("/markets", params={
        "limit": 200,
        "status": "open",
        "min_close_ts": min_ts,
        "max_close_ts": max_ts,
    })
    markets = data.get("markets", [])
    
    print(f"Markets closing within 7 days: {len(markets)}")
    
    # Group by event_ticker prefix
    events = {}
    for m in markets:
        event = m.get("event_ticker", "unknown")
        if event not in events:
            events[event] = []
        ticker = m.get("ticker", "")
        title = m.get("title", "")[:60]
        vol = m.get("volume", 0)
        close_str = m.get("close_time", "")
        events[event].append((ticker, vol, title, close_str))
    
    for event, items in sorted(events.items(), key=lambda x: -sum(i[1] for i in x[1])):
        total_vol = sum(i[1] for i in items)
        print(f"\n{event} ({len(items)} markets, total vol={total_vol}):")
        for ticker, vol, title, close in items[:3]:
            print(f"  {ticker} | vol={vol} | {title}")
            print(f"    closes: {close}")
    
    await client.stop()

asyncio.run(check())
