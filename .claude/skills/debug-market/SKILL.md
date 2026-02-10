---
name: debug-market
description: Debug why the bot did or didn't trade a specific Kalshi market
disable-model-invocation: true
allowed-tools: Bash, Read, Grep
argument-hint: "[market ticker or keyword]"
---

Trace why Morpheus traded (or didn't trade) a specific market.

## Usage

`/debug-market KXTICKER-26FEB07` or `/debug-market tariff`

## Steps

1. **Search production logs** for the market ticker or keyword:
   ```
   ssh morpheus "journalctl -u morpheus --since '24 hours ago' --no-pager | grep -i '$ARGUMENTS'"
   ```

2. **Check trade history** for this market:
   ```
   ssh morpheus "grep -i '$ARGUMENTS' /opt/morpheus/state/trade_history.jsonl 2>/dev/null"
   ```

3. **Check predictions** for this market:
   ```
   ssh morpheus "grep -i '$ARGUMENTS' /opt/morpheus/state/predictions.jsonl 2>/dev/null"
   ```

4. **Trace the decision path** from logs. Look for:
   - `ensemble_evaluate_start` — did the bot evaluate this market?
   - `market_type_skip` or `junk_ticker_prefix` — was it filtered as junk?
   - `filtered_by_volume`, `filtered_by_spread` — failed market filters?
   - `cost_daily_budget_exceeded` — budget blocked the LLM call?
   - `ensemble_probabilities` — what did the models predict?
   - `net_edge` / `min_edge` — was edge too small?
   - `payout_ratio_reject` — payout ratio filter?
   - `signal_dispatched` / `trade_executed` — did it actually trade?

5. **Report**: Explain clearly why the market was traded, skipped, or filtered, with the specific values (edge, p_yes, market_price, etc.).
