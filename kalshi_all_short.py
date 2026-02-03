import asyncio, os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timezone, timedelta
from src.kalshi_client import KalshiClient
from src.utils import load_config

config = load_config('config.yaml')
client = KalshiClient(config)

async def check():
    await client.start()
    now = datetime.now(timezone.utc)
    max_ts = int((now + timedelta(days=30)).timestamp())
    min_ts = int(now.timestamp())
    
    # Fetch directly by close time - bypass category filter
    all_markets = []
    cursor = None
    for page in range(5):  # up to 1000 markets
        params = {
            "limit": 200,
            "status": "open",
            "min_close_ts": min_ts,
            "max_close_ts": max_ts,
        }
        if cursor:
            params["cursor"] = cursor
        data = await client._get("/markets", params=params)
        markets = data.get("markets", [])
        all_markets.extend(markets)
        cursor = data.get("cursor")
        if not cursor or len(markets) < 200:
            break
    
    # Filter: skip sports parlays, need some volume
    good = []
    for m in all_markets:
        ticker = m.get("ticker", "")
        title = m.get("title", "")
        vol = m.get("volume", 0)
        vol24 = m.get("volume_24h", 0)
        yes_bid = m.get("yes_bid", 0)
        yes_ask = m.get("yes_ask", 0)
        close = m.get("close_time", "")
        event = m.get("event_ticker", "")
        
        # Skip sports parlays
        if "KXMVESPORTS" in ticker or "MULTIGAME" in ticker:
            continue
        # Need SOME volume
        if vol < 50:
            continue
        # Need a real price
        if yes_bid == 0 and yes_ask == 0:
            continue
            
        days = 0
        if close:
            try:
                ct = datetime.fromisoformat(close.replace("Z", "+00:00"))
                days = (ct - now).total_seconds() / 86400.0
            except:
                pass
        
        good.append((days, vol, ticker, event, title[:70], yes_bid, yes_ask))
    
    good.sort(key=lambda x: -x[1])  # sort by volume
    
    print(f"Total markets within 30d: {len(all_markets)}")
    print(f"After filtering (no sports, vol>=50): {len(good)}")
    print()
    for days, vol, ticker, event, title, yb, ya in good[:30]:
        print(f"  {days:.1f}d | vol={vol:>6} | bid={yb} ask={ya} | {ticker}")
        print(f"       {title}")

    await client.stop()

asyncio.run(check())
