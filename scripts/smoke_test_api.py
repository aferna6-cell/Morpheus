"""Smoke test: verify the Kalshi API field name fix.

Fetches live markets, confirms new _dollars/_fp fields are present,
and prints sample values to show prices look sane (0-1 scale, not 0-100).
"""

from __future__ import annotations

import asyncio
import os
import sys

# Load .env before importing anything that might need API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; keys must be set in environment already

# Add repo root to path so we can import src.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kalshi_client import KalshiClient
from src.utils import BotConfig


async def main() -> None:
    config = BotConfig(
        strategy={},
        dev={"dry_run": True, "paper_trading_balance": 200.0},
    )

    client = KalshiClient(config)
    await client.start()

    try:
        print("Fetching markets closing within 3 days...")
        markets = await client.fetch_markets_by_close_date(max_days=3, min_volume=0)

        print(f"\nTotal markets returned: {len(markets)}")
        assert len(markets) > 0, "FAIL: No markets returned — filter is still broken!"

        first = markets[0]
        print(f"\nFirst market:")
        print(f"  ticker:    {first.ticker}")
        print(f"  title:     {first.title[:80]}")
        print(f"  yes_bid:   {first.yes_bid:.4f}  (expect 0-1, e.g. 0.62 not 62)")
        print(f"  yes_ask:   {first.yes_ask:.4f}")
        print(f"  no_bid:    {first.no_bid:.4f}")
        print(f"  no_ask:    {first.no_ask:.4f}")
        print(f"  yes_price: {first.yes_price:.4f}")
        print(f"  volume:    {first.volume}")
        print(f"  volume_24h: {first.volume_24h}")

        # Sanity checks
        assert 0 <= first.yes_bid <= 1, f"yes_bid out of range: {first.yes_bid}"
        assert 0 <= first.yes_ask <= 1, f"yes_ask out of range: {first.yes_ask}"
        assert first.yes_bid <= first.yes_ask, "yes_bid > yes_ask — prices inverted"

        print("\nSample of first 5 markets:")
        for m in markets[:5]:
            print(
                f"  {m.ticker:<45} "
                f"yes_bid={m.yes_bid:.3f} yes_ask={m.yes_ask:.3f} "
                f"vol={m.volume}"
            )

        print("\nSMOKE TEST PASSED")
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
