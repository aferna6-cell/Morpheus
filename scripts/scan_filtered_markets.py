#!/usr/bin/env python3
"""Scan Kalshi open markets and check which pass the ticker prefix + volume filters.

Standalone script — uses httpx directly, no project imports required.
"""

import time
import httpx
from collections import Counter, defaultdict

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Mirror of _JUNK_TICKER_PREFIXES from src/market_filters.py
JUNK_TICKER_PREFIXES = [
    # NBA/NCAA announcer mentions
    "KXNBAMENTION", "KXNCAAB", "KXNFLMENTION",
    # Crypto daily price ranges
    "KXBTCD", "KXETHD", "KXDOGE", "KXSOLD", "KXBNBD",
    "KXLTCD", "KXADAD", "KXDOTD", "KXAVAXD", "KXLINKD",
    "KXMATD", "KXUNIDD", "KXSHIB", "KXXRPD",
    # Word/phrase mention markets
    "KXWOMENTION", "KXWMENTION",
    # Stock intraday ranges
    "KXSPY", "KXQQQ", "KXIWM", "KXDIA",
    # Entertainment / pop culture
    "KXSUPERBOWLAD", "KXRT", "KXSPOTIFY", "KXSPOTIFYD", "KXSPOTIFYGLOBALD",
    "KXSBADAPPEARANCES", "KXTOPSONG", "KXTOPALBUM", "KXALBUMDEBUT",
    "KXFIRSTSUPERBOWLSONG", "KXAAAGASW", "KXNEXTTEAMNFL",
]

MIN_VOLUME = 5000


def is_blocked_prefix(ticker: str) -> str | None:
    """Return the matching blocked prefix, or None if clean."""
    t = ticker.upper()
    for prefix in JUNK_TICKER_PREFIXES:
        if t.startswith(prefix.upper()):
            return prefix
    return None


def main():
    all_markets = []
    cursor = None

    print("Fetching open markets from Kalshi API...")
    with httpx.Client(timeout=30) as client:
        page = 0
        while True:
            params = {"limit": 200, "status": "open"}
            if cursor:
                params["cursor"] = cursor
            resp = client.get(f"{API_BASE}/markets", params=params)
            if resp.status_code == 429:
                print("  Rate limited, waiting 3s...")
                time.sleep(3)
                resp = client.get(f"{API_BASE}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()
            markets = data.get("markets", [])
            if not markets:
                break
            all_markets.extend(markets)
            page += 1
            print(f"  Page {page}: fetched {len(markets)} markets ({len(all_markets)} total)")
            cursor = data.get("cursor")
            if not cursor:
                break
            time.sleep(1)  # Rate limit: 1s between pages

    print(f"Fetched {len(all_markets)} open markets total.\n")

    # Classify each market
    passed = []
    blocked_by_prefix = []
    blocked_by_volume = []
    prefix_counter = Counter()
    category_pass = Counter()
    category_all = Counter()

    for m in all_markets:
        ticker = m.get("ticker", "")
        volume = m.get("volume", 0) or 0
        category = m.get("category", "unknown") or "unknown"
        title = m.get("title", "")

        category_all[category] += 1

        # Check prefix first
        blocked = is_blocked_prefix(ticker)
        if blocked:
            blocked_by_prefix.append(m)
            prefix_counter[blocked] += 1
            continue

        # Check volume
        if volume < MIN_VOLUME:
            blocked_by_volume.append(m)
            continue

        # Passed both filters
        passed.append(m)
        category_pass[category] += 1

    # ---------- Summary ----------
    total = len(all_markets)
    print("=" * 70)
    print(f"FILTER RESULTS  (volume >= ${MIN_VOLUME:,})")
    print("=" * 70)
    print(f"  Total open markets:       {total}")
    print(f"  Blocked by ticker prefix: {len(blocked_by_prefix)}")
    print(f"  Blocked by low volume:    {len(blocked_by_volume)}")
    print(f"  PASSED both filters:      {len(passed)}")
    print()

    # Prefix breakdown
    print("-" * 70)
    print("BLOCKED PREFIX BREAKDOWN:")
    print("-" * 70)
    for prefix, count in prefix_counter.most_common():
        print(f"  {prefix:<30s} {count:>5d} markets blocked")
    print()

    # Categories of passing markets
    print("-" * 70)
    print("PASSING MARKETS BY CATEGORY:")
    print("-" * 70)
    for cat, count in category_pass.most_common():
        total_in_cat = category_all[cat]
        print(f"  {cat:<30s} {count:>5d} / {total_in_cat} pass")
    print()

    # Show the actual passing markets
    print("-" * 70)
    print(f"PASSING MARKETS ({len(passed)}):")
    print("-" * 70)
    # Sort by volume descending
    passed.sort(key=lambda m: m.get("volume", 0) or 0, reverse=True)
    for m in passed:
        ticker = m.get("ticker", "")
        volume = m.get("volume", 0) or 0
        title = m.get("title", "")[:60]
        cat = m.get("category", "?")
        yes_price = m.get("yes_ask", 0) or m.get("last_price", 0) or 0
        print(f"  {ticker:<30s} vol=${volume:>10,}  yes={yes_price:>3}c  [{cat}]  {title}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
