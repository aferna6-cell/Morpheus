---
name: logs
description: Check Morpheus production logs on the droplet
disable-model-invocation: true
allowed-tools: Bash
---

Fetch and analyze recent production logs from the Morpheus droplet.

## Default behavior (no arguments)

Run: `ssh root@45.55.85.173 "journalctl -u morpheus -n 50 --no-pager"`

Summarize:
- Any errors or warnings
- Recent trades placed (look for `trade_executed`, `order_placed`)
- Markets being evaluated (look for `ensemble_evaluate_start`)
- Budget status (look for `cost_daily_budget_exceeded` or `cost_tracker_init`)
- Whether the bot is actively scanning or stuck

## With arguments: /logs $ARGUMENTS

If the user passes arguments like "errors", "trades", "100", etc.:
- **A number** (e.g., `/logs 200`): Show that many lines instead of 50
- **"errors"**: Filter for error/warning lines: `journalctl -u morpheus -n 200 --no-pager | grep -i 'error\|warning\|failed\|exception'`
- **"trades"**: Filter for trade activity: `journalctl -u morpheus -n 200 --no-pager | grep -i 'trade_executed\|order_placed\|fill\|signal_dispatched'`
- **"since X"**: Use `--since` flag, e.g., `/logs since 1 hour ago` → `journalctl -u morpheus --since "1 hour ago" --no-pager`
