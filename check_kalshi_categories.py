from src.kalshi_client import KalshiClient
from src.utils import load_config
import asyncio
from collections import Counter

config = load_config('config.yaml')
client = KalshiClient(config)

async def check():
    await client.start()
    # Get events and count categories
    data = await client._get("/events", params={"limit": 200, "status": "open"})
    events = data.get("events", [])
    cats = Counter()
    for e in events:
        cat = e.get("category", "unknown")
        cats[cat] += 1
    
    print("All Kalshi event categories:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")
    
    await client.stop()

asyncio.run(check())
