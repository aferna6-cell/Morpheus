"""Scan available markets and show what's tradeable."""
import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import httpx

# Junk ticker prefixes (from market_filters.py)
JUNK_PREFIXES = [
    "KXBTC", "KXETH", "KXSOL", "KXDOGE", "KXPEPE", "KXFART", "KXTRUMP",
    "KXMELANIA", "KXHYPE", "KXBON", "KXXRP", "KXSUI", "KXAVAX", "KXBNB",
    "KXADA", "KXDOT", "KXLINK", "KXMKR", "KXUNI", "KXAAVE", "KXPOL",
    "KXINX", "KXNAS", "KXSP5", "KXDOW", "KXRUS", "KXGOLD", "KXSILVER",
    "KXOIL", "KXWTI", "KXNGAS", "KXTSY",
    "KXHIGH", "KXLOW", "KXRAIN", "KXSNOW", "KXTEMP",
    "KXTRUMPMENTION",
    "KXSUPERBOWL", "KXSB", "KXFIRSTSUPERBOWL", "KXSBAD",
    "KXTOPSONG", "KXTOPALBUM", "KXALBUMDEBUT", "KXSPOTIFY", "KXSTREAM",
    "KXNFL", "KXNBA", "KXMLB", "KXNHL", "KXMLS",
    "KXDOG", "KXWORD",
]


async def main():
    base = "https://api.elections.kalshi.com/trade-api/v2"
    now = datetime.now(timezone.utc)

    async with httpx.AsyncClient(base_url=base, timeout=15.0) as client:
        # Fetch same-day markets
        min_close = int((now + timedelta(hours=1)).timestamp())
        max_close = int((now + timedelta(days=1)).timestamp())

        all_markets = []
        cursor = None
        for _ in range(5):
            params = {"limit": 200, "status": "open",
                      "min_close_ts": min_close, "max_close_ts": max_close}
            if cursor:
                params["cursor"] = cursor
            r = await client.get("/markets", params=params)
            data = r.json()
            markets = data.get("markets", [])
            all_markets.extend(markets)
            cursor = data.get("cursor")
            if not cursor or not markets:
                break

        print(f"Same-day markets (closing within 24h): {len(all_markets)}")

        # Categorize
        junk = []
        low_vol = []
        candidates = []

        for m in all_markets:
            ticker = m.get("ticker", "")
            vol = m.get("volume_24h", 0) or m.get("volume", 0) or 0
            yes_bid = m.get("yes_bid", 0) or 0
            yes_ask = m.get("yes_ask", 0) or 0
            spread = (yes_ask - yes_bid) / 100.0 if yes_ask and yes_bid else 1.0
            oi = m.get("open_interest", 0) or 0
            cat = m.get("category", "")
            q = m.get("title", "")[:55]

            # Check junk prefix
            is_junk = any(ticker.startswith(p) for p in JUNK_PREFIXES)
            if is_junk:
                junk.append(ticker)
                continue

            if vol < 5000:
                low_vol.append((ticker, vol, q))
                continue

            candidates.append({
                "ticker": ticker,
                "vol": vol,
                "oi": oi,
                "spread": spread,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "cat": cat,
                "q": q,
            })

        print(f"Junk filtered: {len(junk)}")
        print(f"Low volume (<5k): {len(low_vol)}")
        print(f"CANDIDATES: {len(candidates)}")

        # Volume distribution of non-junk markets
        if low_vol:
            vols = sorted([v for _, v, _ in low_vol], reverse=True)
            print(f"\nVolume distribution of {len(low_vol)} non-junk markets:")
            brackets = [(1000, 5000), (500, 1000), (100, 500), (1, 100), (0, 1)]
            for lo, hi in brackets:
                cnt = sum(1 for v in vols if lo <= v < hi)
                if cnt:
                    print(f"  {lo}-{hi}: {cnt} markets")

            print(f"\nTop 20 non-junk markets by volume:")
            for ticker, vol, q in sorted(low_vol, key=lambda x: x[1], reverse=True)[:20]:
                print(f"  {ticker:<35} vol={vol:<6} {q}")

        if candidates:
            print(f"\nTradeable same-day markets (vol>=5k):")
            for c in sorted(candidates, key=lambda x: x["vol"], reverse=True):
                mid = (c["yes_bid"] + c["yes_ask"]) / 200.0 if c["yes_bid"] and c["yes_ask"] else 0
                print(f"  {c['ticker']:<35} vol={c['vol']:<8} spread={c['spread']:.2%} "
                      f"mid={mid:.2f} cat={c['cat']:<15} {c['q']}")

        # Also check 1-7 day for contrarian
        print(f"\n{'='*70}")
        max_close_7 = int((now + timedelta(days=7)).timestamp())
        all_7d = []
        cursor = None
        for _ in range(5):
            params = {"limit": 200, "status": "open",
                      "min_close_ts": min_close, "max_close_ts": max_close_7}
            if cursor:
                params["cursor"] = cursor
            r = await client.get("/markets", params=params)
            data = r.json()
            mkts = data.get("markets", [])
            all_7d.extend(mkts)
            cursor = data.get("cursor")
            if not cursor or not mkts:
                break

        # Find contrarian candidates (80-95% one side, vol > 5k, not junk)
        contrarian = []
        for m in all_7d:
            ticker = m.get("ticker", "")
            if any(ticker.startswith(p) for p in JUNK_PREFIXES):
                continue
            vol = m.get("volume_24h", 0) or m.get("volume", 0) or 0
            if vol < 5000:
                continue
            yes_bid = m.get("yes_bid", 0) or 0
            yes_ask = m.get("yes_ask", 0) or 0
            if not yes_bid or not yes_ask:
                continue
            mid_price = (yes_bid + yes_ask) / 200.0
            if 0.80 <= mid_price <= 0.95 or 0.05 <= mid_price <= 0.20:
                contrarian.append({
                    "ticker": ticker,
                    "vol": vol,
                    "mid": mid_price,
                    "q": m.get("title", "")[:55],
                    "cat": m.get("category", ""),
                })

        print(f"1-7 day markets: {len(all_7d)}")
        print(f"Contrarian candidates (80-95% one side, vol>5k): {len(contrarian)}")
        for c in sorted(contrarian, key=lambda x: x["vol"], reverse=True)[:15]:
            direction = "HIGH" if c["mid"] >= 0.80 else "LOW"
            print(f"  {c['ticker']:<35} vol={c['vol']:<8} mid={c['mid']:.2f} "
                  f"[{direction}] {c['q']}")

        # Also show contrarian with lower volume threshold
        contrarian_low = []
        for m in all_7d:
            ticker = m.get("ticker", "")
            if any(ticker.startswith(p) for p in JUNK_PREFIXES):
                continue
            vol = m.get("volume_24h", 0) or m.get("volume", 0) or 0
            if vol < 100:
                continue
            yes_bid = m.get("yes_bid", 0) or 0
            yes_ask = m.get("yes_ask", 0) or 0
            if not yes_bid or not yes_ask:
                continue
            mid_price = (yes_bid + yes_ask) / 200.0
            if 0.80 <= mid_price <= 0.95 or 0.05 <= mid_price <= 0.20:
                contrarian_low.append({
                    "ticker": ticker,
                    "vol": vol,
                    "mid": mid_price,
                    "q": m.get("title", "")[:55],
                    "cat": m.get("category", ""),
                })

        print(f"\nContrarian with vol>100: {len(contrarian_low)}")
        for c in sorted(contrarian_low, key=lambda x: x["vol"], reverse=True)[:15]:
            direction = "HIGH" if c["mid"] >= 0.80 else "LOW"
            print(f"  {c['ticker']:<35} vol={c['vol']:<6} mid={c['mid']:.2f} "
                  f"[{direction}] cat={c['cat']:<12} {c['q']}")


asyncio.run(main())
