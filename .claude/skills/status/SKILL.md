---
name: status
description: Check Morpheus bot status - balances, positions, service health
disable-model-invocation: true
allowed-tools: Bash
---

Check the full operational status of Morpheus on the production droplet.

## Steps (run in parallel where possible)

1. **Service health**: `ssh morpheus "systemctl is-active morpheus && systemctl show morpheus --property=ActiveEnterTimestamp"`
2. **Recent logs** (last 15 lines): `ssh morpheus "journalctl -u morpheus -n 15 --no-pager"`
3. **Account balances + positions**: `ssh morpheus "journalctl -u morpheus --no-pager | grep -E 'balance_usd|position_tracked|cost_tracker_init' | tail -20"`

## Report format

Summarize as a quick status report:
- **Service**: running/stopped, uptime
- **Balances**: primary + secondary account USD
- **LLM budget**: daily spend vs daily limit, monthly spend
- **Active positions**: count per account
- **Last activity**: when was the last trade or market scan
- **Errors**: any recent errors (last 15 min)
