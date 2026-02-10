"""Check CLV (Closing Line Value) on pending CPI predictions."""
import json
import asyncio
import httpx

# Load predictions
preds = []
with open("state/predictions.jsonl") as f:
    for line in f:
        if line.strip():
            try:
                preds.append(json.loads(line.strip()))
            except Exception:
                pass

# Load resolved
resolved_ids = set()
try:
    with open("state/resolutions.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line.strip())
                resolved_ids.add(r.get("market_id", ""))
except Exception:
    pass

# Get pending CPI predictions
cpi_prefixes = ("KXCPI", "KXCPIYOY", "KXCPICORE", "KXCPICOMBO")
pending_cpi = [
    p for p in preds
    if p.get("market_id", "").startswith(cpi_prefixes)
    and p.get("market_id", "") not in resolved_ids
]

print(f"Pending CPI predictions: {len(pending_cpi)}")


async def check_clv():
    base = "https://api.elections.kalshi.com/trade-api/v2"
    results = []
    async with httpx.AsyncClient(base_url=base, timeout=20.0) as client:
        for p in pending_cpi:
            mid = p.get("market_id", "")
            try:
                r = await client.get(f"/markets/{mid}")
                if r.status_code != 200:
                    continue
                data = r.json()
                mkt = data.get("market", data)
                yes_bid = mkt.get("yes_bid", 0)
                yes_ask = mkt.get("yes_ask", 0)
                last = mkt.get("last_price", 0)
                if yes_bid and yes_ask:
                    current = (yes_bid + yes_ask) / 2 / 100.0
                elif last:
                    current = last / 100.0 if last > 1 else last
                else:
                    continue

                entry = p.get("market_price_at_entry", 0)
                pred_p = p.get("predicted_p_yes", 0.5)
                side = p.get("side", "")

                if "yes" in side.lower():
                    clv = current - entry
                else:
                    clv = entry - current

                results.append({
                    "mid": mid, "side": side, "entry": entry,
                    "current": current, "pred": pred_p, "clv": clv,
                })
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f"  Error {mid}: {e}")

    if not results:
        print("No results")
        return

    positive_clv = [r for r in results if r["clv"] > 0]
    negative_clv = [r for r in results if r["clv"] < 0]
    flat = [r for r in results if r["clv"] == 0]
    avg_clv = sum(r["clv"] for r in results) / len(results)

    print(f"\nChecked: {len(results)} markets")
    print(f"CLV positive (market moved our way): {len(positive_clv)}")
    print(f"CLV negative (market moved against us): {len(negative_clv)}")
    print(f"CLV flat: {len(flat)}")
    print(f"Average CLV: {avg_clv:+.4f} ({avg_clv*100:+.1f} cents)")
    print()

    by_clv = sorted(results, key=lambda x: x["clv"], reverse=True)
    print("=== TOP 5 BEST CLV ===")
    for r in by_clv[:5]:
        mid = r["mid"]
        print(f"  {mid}: {r['side']} entry={r['entry']:.2f} now={r['current']:.2f} CLV={r['clv']:+.3f}")
    print()
    print("=== TOP 5 WORST CLV ===")
    for r in by_clv[-5:]:
        mid = r["mid"]
        print(f"  {mid}: {r['side']} entry={r['entry']:.2f} now={r['current']:.2f} CLV={r['clv']:+.3f}")

    print()
    print("=== CLV BY CATEGORY ===")
    cats: dict[str, list] = {}
    for r in results:
        mid = r["mid"]
        for pfx in ["KXCPICOMBO", "KXCPIYOY", "KXCPICORE", "KXCPI"]:
            if mid.startswith(pfx):
                cats.setdefault(pfx, []).append(r["clv"])
                break
    for cat in sorted(cats.keys()):
        clvs = cats[cat]
        avg = sum(clvs) / len(clvs)
        pos = sum(1 for c in clvs if c > 0)
        print(f"  {cat}: {len(clvs)} bets, {pos}/{len(clvs)} positive, avg CLV={avg:+.4f}")

    # Overall assessment
    print()
    if avg_clv > 0.01:
        print("VERDICT: Markets are moving TOWARD our predictions. Positive signal.")
    elif avg_clv < -0.01:
        print("VERDICT: Markets are moving AGAINST our predictions. Negative signal.")
    else:
        print("VERDICT: Markets are roughly flat. Inconclusive.")


asyncio.run(check_clv())
