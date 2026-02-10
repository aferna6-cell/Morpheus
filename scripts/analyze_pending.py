"""Analyze pending predictions by category."""
import json

# Load predictions
preds = []
with open("state/predictions.jsonl") as f:
    for line in f:
        if line.strip():
            try:
                preds.append(json.loads(line.strip()))
            except:
                pass

# Load resolved
resolved_ids = set()
try:
    with open("state/resolutions.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line.strip())
                resolved_ids.add(r.get("market_id", ""))
except:
    pass

pending = [p for p in preds if p.get("market_id", "") not in resolved_ids]
resolved = [p for p in preds if p.get("market_id", "") in resolved_ids]

# CPI predictions
cpi = [p for p in pending if p.get("market_id", "").startswith(("KXCPI", "KXCPIYOY", "KXCPICORE", "KXCPICOMBO"))]
junk = [p for p in pending if p.get("market_id", "").startswith(("KXRT", "KXSPOTIFY", "KXNEXTTEAM", "KXTOPALBUM"))]

print("=== CPI/ECONOMICS (where we SHOULD have edge) ===")
print(f"Count: {len(cpi)} pending")
edges = [p.get("net_edge", p.get("edge", 0)) for p in cpi]
if edges:
    avg_edge = sum(edges) / len(edges)
    print(f"Avg edge claimed: {avg_edge:.1%}")

for p in cpi[:8]:
    mid = p.get("market_id", "")
    side = p.get("side", "")
    pred = p.get("predicted_p_yes", 0)
    mkt = p.get("market_price_at_entry", 0)
    edge = p.get("net_edge", p.get("edge", 0))
    print(f"  {mid}: side={side} pred={pred:.2f} mkt={mkt:.2f} edge={edge:.1%}")

print()
print("=== JUNK (still pending, now BLOCKED from new trades) ===")
print(f"Count: {len(junk)} pending (these are legacy bets)")

# Resolved P&L breakdown
print()
print("=== RESOLVED P&L ===")
res_data = []
try:
    with open("state/resolutions.jsonl") as f:
        for line in f:
            if line.strip():
                res_data.append(json.loads(line.strip()))
except:
    pass

total_pnl = sum(r.get("pnl_usd", 0) for r in res_data)
wins = [r for r in res_data if r.get("pnl_usd", 0) > 0]
losses = [r for r in res_data if r.get("pnl_usd", 0) < 0]
print(f"Total resolved: {len(res_data)}")
print(f"Wins: {len(wins)}, Losses: {len(losses)}")
print(f"Win rate: {len(wins)/(len(wins)+len(losses)):.0%}" if (wins or losses) else "")
print(f"Total P&L: ${total_pnl:.2f}")
print(f"Avg win: ${sum(r['pnl_usd'] for r in wins)/len(wins):.2f}" if wins else "")
print(f"Avg loss: ${sum(r['pnl_usd'] for r in losses)/len(losses):.2f}" if losses else "")
