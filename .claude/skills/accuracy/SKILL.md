---
name: accuracy
description: Analyze Morpheus prediction accuracy and trading performance
disable-model-invocation: true
allowed-tools: Bash, Read
---

Analyze the bot's prediction accuracy and trading performance from state files.

## Data sources

All on the production droplet at `/opt/morpheus/state/`:
- `trade_history.jsonl` — every trade placed
- `predictions.jsonl` — prediction log with p_yes, market_price, side, edge
- `model_predictions.jsonl` — per-model (GPT-4o vs Claude) predictions
- `resolutions.jsonl` — resolved market outcomes
- `cost_tracker.json` — LLM spend tracking

## Steps

1. Copy relevant state files locally:
   ```
   scp morpheus:/opt/morpheus/state/trade_history.jsonl /tmp/
   scp morpheus:/opt/morpheus/state/predictions.jsonl /tmp/
   scp morpheus:/opt/morpheus/state/model_predictions.jsonl /tmp/
   ```

2. Analyze and report:
   - **Total trades**: count, buy_yes vs buy_no split
   - **Win/loss rate**: from resolved trades (if resolutions exist)
   - **Average edge**: mean net_edge at entry
   - **P&L**: if available from trade history
   - **Model comparison**: GPT-4o vs Claude average p_yes, divergence
   - **Market types traded**: breakdown by category
   - **LLM costs**: total spend, cost per trade

3. If `$ARGUMENTS` contains "brier", compute Brier scores per model from model_predictions + resolutions.

## Notes
- If files don't exist yet (bot just deployed), say so clearly.
- Use python3 one-liners or small scripts for analysis.
