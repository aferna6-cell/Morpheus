"""Test the full signal pipeline on live CPI markets.

Bypasses balance gate. Shows: structured data, LLM prediction,
calibration, threshold analysis, final signal — all with new fixes.
"""
import asyncio
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config
from src.cost_tracker import CostTracker
from src.signals.ensemble_signal import EnsembleSignal
from src.structured_data import get_structured_anchor

import httpx


async def main():
    config = load_config("config.yaml")

    # Cost tracker (won't actually charge — just tracking)
    ct = CostTracker(
        monthly_budget=50.0, daily_budget=10.0,
        state_path="state/cost_tracker.json",
    )

    # Build ensemble signal
    signal = EnsembleSignal(config)
    signal.set_cost_tracker(ct)

    print(f"Ensemble mode: {signal.ensemble_mode}")
    print(f"Calibration shrink: {signal.calibration_shrink}")
    print()

    # Fetch CPI markets from Kalshi
    base = "https://api.elections.kalshi.com/trade-api/v2"
    test_markets = [
        "KXCPIYOY-26JAN-T2.3",
        "KXCPIYOY-26JAN-T2.5",
        "KXCPI-26JAN-T0.3",
        "KXCPICORE-26JAN-T0.2",
        "KXCPICOMBO-26JAN-0024",
    ]

    async with httpx.AsyncClient(base_url=base, timeout=20.0) as client:
        for mid in test_markets:
            r = await client.get(f"/markets/{mid}")
            if r.status_code != 200:
                print(f"SKIP {mid}: HTTP {r.status_code}")
                continue

            data = r.json().get("market", r.json())
            question = data.get("title", "")
            yes_bid = data.get("yes_bid", 0)
            yes_ask = data.get("yes_ask", 0)
            volume = data.get("volume_24h", 0)
            liquidity = data.get("open_interest", 0)

            if yes_bid and yes_ask:
                mid_price = (yes_bid + yes_ask) / 200.0
            else:
                last = data.get("last_price", 50)
                mid_price = last / 100.0 if last > 1 else last

            print(f"{'='*70}")
            print(f"Market: {mid}")
            print(f"Question: {question}")
            print(f"Market price: {mid_price:.2f}")
            print(f"Volume: {volume}, Liquidity: {liquidity}")

            # Show structured data that would be injected
            anchor = await get_structured_anchor(question, "economics")
            if anchor:
                # Just show threshold analysis lines
                for line in anchor.split("\n"):
                    if any(k in line for k in ["THRESHOLD", "Assessment", "Gap", "WARNING", "Latest", "COMBO", "Question threshold"]):
                        print(f"  {line.strip()}")
            print()

            # Build a Market object for the signal
            from src.markets import Market, TokenInfo
            tokens = {}
            if yes_bid and yes_ask:
                tokens["Yes"] = TokenInfo(token_id=f"{mid}-yes", outcome="Yes", price=mid_price, volume_24h=volume)
                tokens["No"] = TokenInfo(token_id=f"{mid}-no", outcome="No", price=1.0 - mid_price, volume_24h=volume)
            market = Market(
                id=mid,
                question=question,
                description=data.get("description", ""),
                category=data.get("category", "economics"),
                end_date=None,
                volume_24h=volume,
                liquidity=liquidity,
                tokens=tokens,
            )

            result = await signal.evaluate(market)

            side = result.recommended_side
            p_yes = result.estimated_prob
            edge = result.edge
            net_edge = getattr(result, "net_edge", edge)
            confidence = result.confidence

            print(f"  Signal: side={side} p_yes={p_yes:.3f} edge={edge:+.3f} net_edge={net_edge:+.3f} conf={confidence:.2f}")
            print(f"  Reasoning: {result.reasoning[:120]}...")
            print()

            await asyncio.sleep(1)  # rate limit


asyncio.run(main())
