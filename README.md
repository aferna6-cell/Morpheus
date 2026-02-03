# Morpheus

Autonomous Polymarket trading bot (async Python) — **aggressive accuracy mode**.

*"I'm trying to free your mind, Neo. But I can only show you the door. You're the one that has to walk through it."*

Swing hard on high-conviction plays, stay flat when edge isn't there.

## Architecture

- **Market discovery** via Polymarket Gamma API
- **Execution** via Polymarket CLOB (py-clob-client)
- **Strategies**: LLM probability edge + simple arbitrage checks
- **Risk**: fractional Kelly sizing + conviction gating + exposure limits + daily loss halt
- **24/7 hardening**: never crashes out, exponential backoff, heartbeat logging

## Aggressive Accuracy Mode

The bot only trades when it has genuine edge:

1. **LLM outputs strict probability** (`p_yes`) — not vague sentiment
2. **Net edge** = `|p_yes - market_price| - fees - slippage` — real edge after costs
3. **Conviction levels**: LOW (2-5%), MEDIUM (5-10%), HIGH (10%+)
4. **Only trades on MEDIUM or HIGH conviction** — no spray-and-pray
5. **HIGH conviction → 2x position size** — swing hard when you're right
6. **Market quality filters**: skip thin markets (<$10k liquidity), ambiguous questions

### Position Sizing

- **Half Kelly** (`kelly_fraction: 0.5`) — aggressive but not reckless
- **Max position**: $5,000 (2x = $10,000 for HIGH conviction)
- **Max exposure**: $25,000 total
- **Linear confidence scaling** (removed conservative quadratic damping)

### Prediction Tracking

Every trade logs to `state/predictions.jsonl`:
```json
{"market_id": "...", "predicted_p_yes": 0.72, "market_price_at_entry": 0.55, "side": "buy_yes", "edge": 0.17, "net_edge": 0.145, "conviction": "high", "timestamp": "..."}
```

Use this to measure actual calibration over time.

## Cost Optimization

The bot includes two cost optimizations to reduce LLM API spend:

### Two-Tier LLM Filtering

Instead of running the expensive main model on every market, the bot uses a two-tier system:

1. **Tier 1 — Screening (GPT-4o-mini):** A cheap, fast model receives the market question, current price, resolution date, and liquidity. It makes a quick YES/NO decision: "Is this market likely mispriced by >5%?" Markets that aren't worth evaluating are skipped immediately.

2. **Tier 2 — Full Analysis (GPT-4o / configured model):** Only markets that pass tier-1 screening get the full probability analysis with news context, key facts extraction, uncertainty estimation, etc.

This typically filters out 50-70% of markets at ~1/30th the cost per market.

Config keys:
- `llm.screening_enabled` — Enable/disable screening (default: `true`)
- `llm.screening_model` — Model for tier 1 (default: `gpt-4o-mini`)

### Result Caching

LLM results are cached in-memory with a configurable TTL. The cache key includes the market ID, question, and price (rounded to nearest 0.02 so small price fluctuations don't bust the cache).

Config keys:
- `llm.cache_enabled` — Enable/disable caching (default: `true`)
- `llm.cache_ttl_minutes` — Cache lifetime in minutes (default: `30`)

Both optimizations log debug-level metrics (cache hits/misses, screening pass/reject) and a periodic summary each iteration.

## Quickstart

### 1) Install

```bash
cd polymarket-bot
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Configure secrets

Create `.env`:

```bash
# Required
POLYMARKET_PRIVATE_KEY=0x...
POLYMARKET_FUNDER_ADDRESS=0x...
POLYMARKET_SIGNATURE_TYPE=0
OPENAI_API_KEY=sk-...

# Optional: Telegram alerts
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=-100...
```

### 3) Run

Single iteration (dry run):
```bash
python -m src.main --dry-run --once
```

24/7 continuous (dry run):
```bash
python -m src.main --dry-run
```

24/7 live trading:
```bash
python -m src.main
```

## Config Reference (`config.yaml`)

### Strategy (aggressive defaults)

| Key | Default | Description |
|-----|---------|-------------|
| `kelly_fraction` | `0.5` | Fraction of Kelly criterion (half Kelly) |
| `max_position_size` | `5000` | Max USD per position |
| `max_total_exposure` | `25000` | Max total portfolio exposure |
| `min_conviction` | `medium` | Minimum conviction to trade: none/low/medium/high |
| `high_conviction_multiplier` | `2.0` | Position size multiplier for HIGH conviction |
| `fee_pct` | `0.02` | Fee assumption for net edge calc (2%) |
| `slippage_pct` | `0.005` | Slippage assumption (0.5%) |

### Risk

| Key | Default | Description |
|-----|---------|-------------|
| `max_loss_per_trade` | `500` | Max loss per trade |
| `max_daily_loss` | `2000` | Daily loss halt threshold |
| `stop_loss_pct` | `0.15` | Stop loss percentage |
| `take_profit_pct` | `0.30` | Take profit percentage |

### Market Filters

| Key | Default | Description |
|-----|---------|-------------|
| `min_liquidity_aggressive` | `10000` | Reject markets with < $10k liquidity |
| `min_volume_24h` | `5000` | Minimum 24h volume |

## Run/ops flags

| Flag | Description |
|------|-------------|
| `--once` | Single iteration then exit |
| `--max-loops N` | Exit after N iterations |
| `--no-llm` | Disable LLM strategy |
| `--dry-run` | Simulate (no real orders) |
| `--log-json` | Force JSON log output |
| `--state-dir DIR` | Persistent state directory (default: `state/`) |
| `--runs-dir DIR` | Run artifacts directory (default: `runs/`) |
| `--replay DIR` | Replay from snapshot directory |

## Alerts

If `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set, the bot sends:
- 💰 Trade execution alerts (side, amount, conviction, edge)
- ⚠️ Error alerts (first 3 consecutive errors)
- 🛑 Kill switch notification

## Safety

### Kill switch
```bash
touch state/STOP_TRADING
```

### 24/7 Hardening
- Main loop catches all exceptions, logs, and continues
- Exponential backoff on repeated errors (5s → 300s)
- Fatal errors: sleep 60s and restart the entire async loop
- Heartbeat log every 10 iterations

## Artifacts

Each run creates:
- `runs/<run_id>/markets_*.json` — market snapshots
- `runs/<run_id>/decisions.jsonl` — trade/skip decisions
- `state/predictions.jsonl` — prediction log for accuracy tracking

## Deployment (systemd)

Template at `deploy/systemd/polymarket-bot.service`.

```bash
sudo cp deploy/systemd/polymarket-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now polymarket-bot
```

## Notes

- Execution prefers limit orders with configurable slippage
- Orders are reconciled (polled) — bot does not assume fills
- Token mapping for BUY_YES/BUY_NO requires explicit Yes/No outcomes
- Ensure token allowances are set before live trading
- You are responsible for compliance with Polymarket terms and local laws
