# Morpheus

Autonomous **Kalshi** prediction market trading bot — async Python, production-hardened.

*"I'm trying to free your mind, Neo. But I can only show you the door. You're the one that has to walk through it."*

Trades binary YES/NO contracts on Kalshi using a multi-engine ensemble: LLM probability estimation, bracket arbitrage, bonding curve capture, and orderflow signals.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in API keys
cp .env.example .env
# Edit .env: KALSHI_KEY_PATH, OPENAI_API_KEY, ANTHROPIC_API_KEY, ...

# Dry run (paper trading, no real orders)
python -m src.main --dry-run

# Single scan then exit
python -m src.main --dry-run --once

# Live trading
python -m src.main
```

### Kill switch

```bash
touch state/STOP_TRADING    # halt all trading immediately
rm state/STOP_TRADING       # resume
```

---

## Architecture

```
Kalshi REST API
  → Engines (LLM, bracket_arb, bonding, orderbook, rules, orderflow)
    → Orchestrator (score, rank, deduplicate, multi-engine consensus)
      → RiskPipeline (8-layer sequential validation + Kelly sizing)
        → SmartEntry (limit → timeout → market fallback execution)
          → SQLite DB (WAL mode, trades/positions/signals/CLV)
            → REST API (FastAPI monitoring + admin control)
```

### Engines

| Engine | Strategy |
|--------|----------|
| `kalshi_llm` | LLM ensemble (Claude Sonnet + GPT-4o) on same-day markets. Yahoo Finance fast-path for index — **zero LLM cost on index signals**. |
| `kalshi_bracket_arb` | Buys the cheapest bracket leg when the sum of all YES asks is < 100c. |
| `kalshi_bonding` | Captures near-certain contracts (≥ 93c YES or ≤ 7c NO) at premium. |
| `kalshi_orderbook` | Orderbook imbalance (OBI > 0.3) as directional signal. |
| `kalshi_rules` | Time-decay and price-consistency rules (no ML required). |
| `kalshi_orderflow` | VPIN (Volume-Synchronized Probability of Informed Trading) for toxic flow detection. |

### Risk Pipeline (8 layers)

Each signal passes through all 8 layers sequentially. **Hard-coded ceilings cannot be overridden by config.**

| Layer | Check | Hard Ceiling |
|-------|-------|-------------|
| 1 | Spread cost — net edge after Kalshi maker fee > 0 | — |
| 2 | Volume/liquidity — min 1k 24h volume | — |
| 3 | Confidence floor — per market type (politics = 70%) | 30% absolute minimum |
| 4 | Net edge cents — configurable minimum | 1c absolute minimum |
| 5 | Daily loss limit — halt when daily P&L < -limit | $500 absolute max daily loss |
| 6 | Consecutive loss breaker — halve Kelly after N losses | — |
| 7 | Drawdown / survival mode — reduce sizing in drawdown | — |
| 8 | Exposure cap — total portfolio exposure check | $10,000 absolute max |

### Execution (SmartEntry)

1. **Limit order** placed 1–2c inside the spread (inside = better price than crossing)
2. **5-minute timeout** — polls for fills every 10s
3. **Market fallback** — cancel resting limit, fill remainder at market price
4. **Paper mode** — simulates fill at mid-price, no real orders

### Configuration

All parameters are in `config.yaml`. Key sections:

```yaml
strategy:
  kelly_fraction: 0.25        # Third-Kelly sizing
  max_position_size: 10.0     # USD per trade
  max_total_exposure: 50.0    # Total portfolio exposure
  min_confidence: 0.60        # Minimum confidence to trade

risk:
  max_daily_loss: 10.0        # Halt if daily P&L < -$10
  consecutive_loss_limit: 5   # Halve Kelly after 5 straight losses

calibration:
  index_platt_alpha: 0.90     # Platt scaling per market type
  weather_platt_alpha: 0.83
  default_platt_alpha: 0.68
```

---

## Monitoring API

The bot exposes a FastAPI REST API on port 8000. Requires `X-API-Key` header (set `MORPHEUS_API_KEY` in `.env`).

```bash
# Health (no auth required)
curl http://localhost:8000/api/health

# Current positions
curl -H "X-API-Key: $KEY" http://localhost:8000/api/positions

# P&L (last 7 days)
curl -H "X-API-Key: $KEY" http://localhost:8000/api/pnl?days=7

# Strategy performance
curl -H "X-API-Key: $KEY" http://localhost:8000/api/strategies

# Emergency halt
curl -X POST -H "X-API-Key: $KEY" http://localhost:8000/api/admin/halt
```

---

## Deployment

```bash
# Deploy to production server
ssh morpheus "cd /opt/morpheus && git pull && systemctl restart morpheus"

# View logs
ssh morpheus "journalctl -u morpheus -n 50 --no-pager"

# Audit resolved trades
ssh morpheus "cd /opt/morpheus && .venv/bin/python3 scripts/audit_resolutions.py --state-dir state --since 2026-01-01"
```

---

## Testing

```bash
pytest tests/                       # run all tests
pytest tests/ --timeout=30          # with timeout guard
pytest tests/ --cov=src             # with coverage report
```

Key test modules:
- `test_risk_pipeline.py` — 8-layer risk pipeline, Kelly sizing, hard-coded ceilings
- `test_db.py` — SQLite WAL database CRUD
- `test_api.py` — REST API endpoints and auth
- `test_smart_entry.py` — SmartEntry execution state machine
- `test_strategies.py` — bracket arb math, bonding thresholds, VPIN, calibration

---

## Project Structure

```
src/
  main.py                  # Entry point, orchestration loop
  orchestrator.py          # Signal collection, scoring, deduplication
  risk_pipeline.py         # 8-layer risk validation + Kelly sizing (NEW)
  smart_entry.py           # Limit→timeout→market execution state machine (NEW)
  db.py                    # SQLite WAL async database layer (NEW)
  api.py                   # FastAPI REST monitoring API (NEW)
  market_filters.py        # Ticker blocklist, spread/volume gates
  engines/                 # Trading strategy engines
    kalshi_llm_engine.py
    kalshi_bracket_arb_engine.py
    kalshi_bonding_engine.py
    kalshi_orderbook_engine.py
    kalshi_rules_engine.py
    kalshi_orderflow_engine.py
  signals/                 # Signal definitions and LLM ensemble
  cost_basis.py            # FIFO cost basis / tax lot tracking (NEW)
tests/
  conftest.py              # Shared fixtures
  test_risk_pipeline.py
  test_db.py
  test_api.py
  test_smart_entry.py
  test_strategies.py
config.yaml                # All bot parameters
.env                       # API keys (never committed)
state/                     # Runtime state (gitignored)
```

---

## Kalshi Authentication

The bot uses RSA-PSS key signing (not username/password).

```bash
# Generate RSA key pair
openssl genrsa -out kalshi_key.pem 2048
openssl rsa -in kalshi_key.pem -pubout -out kalshi_key_pub.pem
# Upload kalshi_key_pub.pem to Kalshi dashboard → API Keys
```

Set `KALSHI_KEY_PATH=kalshi_key.pem` and `KALSHI_KEY_ID=<your-key-id>` in `.env`.

---

## Known Gotchas

- Kalshi SDK bug: use `get_positions_without_preload_content` for raw JSON
- `get_balance()` returns cash only — equity in open positions is separate
- Kelly formula: `edge*(1+odds)/odds` not `edge/odds`
- Platt scaling: alpha < 1.0 compresses toward 0.5, alpha > 1.0 amplifies
- structlog reserved kwarg: never pass `event=` as a keyword argument
- New BotConfig sections require an explicit Pydantic field declaration
