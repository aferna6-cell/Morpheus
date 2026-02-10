"""Score historical predictions against actual market outcomes."""
import json
import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


async def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Load predictions
    predictions = []
    with open("state/predictions.jsonl") as f:
        for line in f:
            if line.strip():
                try:
                    predictions.append(json.loads(line.strip()))
                except Exception:
                    pass

    # Unique market IDs
    market_ids = list(set(p["market_id"] for p in predictions))
    print(f"Checking {len(market_ids)} unique markets from {len(predictions)} predictions...")

    # Fetch market status for each (public API, no auth needed)
    client = httpx.AsyncClient(timeout=30)
    results = {}
    for mid in market_ids:
        try:
            resp = await client.get(f"{BASE_URL}/markets/{mid}")
            if resp.status_code == 200:
                data = resp.json().get("market", {})
                status = data.get("status", "unknown")
                result_val = data.get("result", "")
                # result is empty string for unresolved, "yes"/"no" for resolved
                if result_val in ("yes", "no"):
                    actual = result_val
                else:
                    actual = None
                results[mid] = {
                    "status": status,
                    "result": actual,
                    "title": data.get("title", "")[:60],
                }
            else:
                results[mid] = {"status": f"http_{resp.status_code}", "result": None}
            await asyncio.sleep(0.1)
        except Exception as e:
            results[mid] = {"status": f"error: {e}", "result": None}
    await client.aclose()

    # Count statuses
    status_counts = {}
    for r in results.values():
        s = r["status"]
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"Market statuses: {status_counts}")

    # Score predictions
    correct = 0
    wrong = 0
    pending = 0
    total_brier = 0.0
    scored_count = 0
    by_category = {}
    wrong_list = []
    correct_list = []

    for p in predictions:
        mid = p["market_id"]
        r = results.get(mid, {})
        actual = r.get("result", None)
        p_yes = p["predicted_p_yes"]
        side = p["side"]
        prefix = mid.split("-")[0]

        if actual is not None:
            outcome = 1.0 if actual == "yes" else 0.0
            brier = (p_yes - outcome) ** 2
            total_brier += brier
            scored_count += 1

            is_correct = (
                (side == "buy_yes" and outcome == 1.0) or
                (side == "buy_no" and outcome == 0.0)
            )
            is_wrong = (
                (side == "buy_yes" and outcome == 0.0) or
                (side == "buy_no" and outcome == 1.0)
            )

            entry = {
                "market": mid,
                "title": r.get("title", ""),
                "predicted": p_yes,
                "market_price": p.get("market_price_at_entry", 0),
                "actual": actual,
                "side": side,
                "edge": p.get("net_edge", 0),
            }

            if is_correct:
                correct += 1
                correct_list.append(entry)
            elif is_wrong:
                wrong += 1
                wrong_list.append(entry)

            by_category.setdefault(prefix, {"correct": 0, "wrong": 0, "brier": 0.0, "count": 0})
            by_category[prefix]["count"] += 1
            by_category[prefix]["brier"] += brier
            if is_correct:
                by_category[prefix]["correct"] += 1
            elif is_wrong:
                by_category[prefix]["wrong"] += 1
        else:
            pending += 1

    print(f"\n{'='*50}")
    print(f"  PREDICTION SCORECARD")
    print(f"{'='*50}")
    print(f"Resolved: {correct + wrong} / {len(predictions)} predictions")
    print(f"Correct:  {correct}")
    print(f"Wrong:    {wrong}")
    if correct + wrong > 0:
        print(f"Win rate: {correct / (correct + wrong):.1%}")
    if scored_count > 0:
        avg_brier = total_brier / scored_count
        print(f"Avg Brier score: {avg_brier:.4f} (lower=better, 0.25=random)")
    print(f"Still pending: {pending}")

    print(f"\n{'='*50}")
    print(f"  BY CATEGORY")
    print(f"{'='*50}")
    for prefix, stats in sorted(by_category.items(), key=lambda x: -x[1]["count"]):
        n = stats["count"]
        c = stats["correct"]
        w = stats["wrong"]
        avg_b = stats["brier"] / n if n > 0 else 0
        wr = c / (c + w) if (c + w) > 0 else 0
        print(f"  {prefix:35s} {c}/{c+w} correct ({wr:>5.0%}) | Brier={avg_b:.4f} | n={n}")

    if correct_list:
        print(f"\n{'='*50}")
        print(f"  CORRECT PREDICTIONS")
        print(f"{'='*50}")
        for c in correct_list[:10]:
            print(f"  {c['market']}")
            print(f"    {c['title']}")
            print(f"    predicted p_yes={c['predicted']:.3f}, mkt_price={c['market_price']:.3f}, actual={c['actual']}, side={c['side']}")

    if wrong_list:
        print(f"\n{'='*50}")
        print(f"  WRONG PREDICTIONS")
        print(f"{'='*50}")
        for w in wrong_list[:10]:
            print(f"  {w['market']}")
            print(f"    {w['title']}")
            print(f"    predicted p_yes={w['predicted']:.3f}, mkt_price={w['market_price']:.3f}, actual={w['actual']}, side={w['side']}")


if __name__ == "__main__":
    asyncio.run(main())
