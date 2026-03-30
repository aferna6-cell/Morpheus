# ARCHITECTURE.md — Morpheus V2 (Merged)

> Design document for the merged Morpheus + Neo trading bot.
> This is the target architecture — it may not reflect current code state during the merge.
> Last updated: 2026-03-30

---

## 1. System Overview

Morpheus V2 is a 24/7 autonomous trading bot for Kalshi binary prediction markets. It combines:
- **Battle-tested signal generation** from Morpheus (36 waves, proven calibration)
- **Production-grade infrastructure** from Neo (SQLite, risk pipeline, SmartEntry, testing)

### Design Principles
1. **Safety first** — paper mode by default, hard-coded risk ceilings, kill switch
2. **Testability** — every module has unit tests; no trading logic in untestable code paths
3. **Simplicity** — one process, one SQLite database, one config file
4. **Observability** — structured logs, Telegram alerts, REST API for monitoring
5. **Survivability** — crash recovery, state persists across restarts, survival mode gate

---

## 2. High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MORPHEUS V2                                  │
│                                                                       │
│  config.yaml + .env                                                   │
│       │                                                               │
│  ┌────▼────────────────────────────────────────────────────────────┐ │
│  │                        main.py                                   │ │
│  │  • Startup / shutdown lifecycle                                  │ │
│  │  • Initializes all components                                    │ │
│  │  • Kill switch check (state/STOP_TRADING)                       │ │
│  │  • Paper / live double-gate                                      │ │
│  └────┬────────────────────────────────────────────────────────────┘ │
│       │                                                               │
│  ┌────▼───────────────────────────────────────────────────────┐      │
│  │                     Orchestrator                            │      │
│  │  • Cycle: scan → signal → risk → execute → monitor         │      │
│  │  • Signal dedup (ticker+side+engine, event prefix)         │      │
│  │  • Multi-engine consensus boost                             │      │
│  │  • Stop-loss cooldown (30min after SL exit)                │      │
│  │  • Signal scoring: edge×conf×0.45 + conf×0.30 + ...       │      │
│  └────┬──────────────────────────────────────────────────────┘      │
│       │                                                               │
│  ┌────▼──────────────────────────────────────────────────────────┐  │
│  │                    Strategy Layer                              │  │
│  │  (all async, 30s timeout, circuit-breaker isolated)           │  │
│  │                                                                │  │
│  │  Tier 1 (Proven)        Tier 2 (Active)     Tier 3 (Low wt)  │  │
│  │  ┌──────────────┐       ┌─────────────┐     ┌─────────────┐  │  │
│  │  │ IndexFastPath│       │LLM Ensemble │     │ RulesBased  │  │  │
│  │  │ BracketArb   │       │ OrderFlow   │     │ DataEnhanced│  │  │
│  │  │ Bonding      │       │             │     │             │  │  │
│  │  └──────────────┘       └─────────────┘     └─────────────┘  │  │
│  └────┬──────────────────────────────────────────────────────────┘  │
│       │  TradeSignal(market, side, confidence, edge, urgency, ...)   │
│  ┌────▼──────────────────────────────────────────────────────────┐  │
│  │                  Risk Pipeline (8 layers)                      │  │
│  │  1. Spread cost      5. Daily loss limit                       │  │
│  │  2. Volume/liquidity 6. Consecutive loss breaker               │  │
│  │  3. Confidence floor 7. Drawdown check (15% peak)             │  │
│  │  4. Net edge (fees)  8. Total exposure cap                     │  │
│  │  [Hard-coded ceilings: $200/position, $10K exposure, $500/day]│  │
│  └────┬──────────────────────────────────────────────────────────┘  │
│       │  Approved signal + position_size                             │
│  ┌────▼──────────────────────────────────────────────────────────┐  │
│  │                   Execution Engine                             │  │
│  │  SmartEntry: limit (1-2c inside spread)                       │  │
│  │    └─[5min timeout]→ market fallback                          │  │
│  │  Adaptive spread crossing (NOAA 5c, index 3c, generic 1c)    │  │
│  │  Idempotent: client_order_id = UUID4                          │  │
│  │  Multi-account: primary + optional secondary                  │  │
│  └────┬──────────────────────────────────────────────────────────┘  │
│       │                                                               │
│  ┌────▼──────────────────────────────────────────────────────────┐  │
│  │               Background Services                              │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │  │
│  │  │ FillManager  │  │PositionMonitor│  │ ResolutionTracker  │  │  │
│  │  │ (poll 30s)   │  │ (SL/TP 5min) │  │ (settled markets)  │  │  │
│  │  └──────────────┘  └──────────────┘  └────────────────────┘  │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │  │
│  │  │ TelegramBot  │  │ PerfTracker  │  │ SurvivalMonitor    │  │  │
│  │  │ (alerts)     │  │ (daily P&L)  │  │ (self-sustain gate)│  │  │
│  │  └──────────────┘  └──────────────┘  └────────────────────┘  │  │
│  └────┬──────────────────────────────────────────────────────────┘  │
│       │                                                               │
│  ┌────▼──────────────────────────────────────────────────────────┐  │
│  │                  State Layer                                   │  │
│  │  SQLite WAL (state/morpheus.db)                                │  │
│  │  Tables: trades, positions, orders, signals, daily_pnl,       │  │
│  │          cost_basis_lots, market_snapshots, strategy_perf,    │  │
│  │          clv_records, config_snapshots                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  REST API (FastAPI, port 8000)                                │   │
│  │  /api/status  /api/positions  /api/trades  /api/signals      │   │
│  │  /api/pnl     /api/risk       /api/strategies                │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

External Dependencies:
  Kalshi API (RSA auth) ←→ KalshiClient
  Kalshi WebSocket      ←→ WSClient (real-time prices)
  OpenAI GPT-4o         ←→ LLM ensemble
  Anthropic Claude      ←→ LLM ensemble
  NOAA NWS API          ←→ WeatherDataProvider
  Yahoo Finance         ←→ IndexDataProvider
  FRED API              ←→ EconDataProvider
  Brave Search API      ←→ NewsProvider
  Telegram Bot API      ←→ TelegramAlerter
```

---

## 3. Module Breakdown

```
src/
├── main.py                    # Entry point, CLI args, startup/shutdown
│
├── core/                      # Kalshi API client (no business logic)
│   ├── kalshi_client.py       # REST API wrapper (auth, rate limit, retry)
│   ├── kalshi_trading.py      # Order placement, fill polling, multi-account
│   ├── ws_client.py           # WebSocket real-time prices
│   └── rate_limiter.py        # Adaptive rate limiter (X-RateLimit tracking)
│
├── strategies/                # One module per strategy, common interface
│   ├── base.py                # Strategy ABC: analyze(market) → Signal | None
│   ├── index_fast_path.py     # Yahoo Finance → index threshold signals (Tier 1)
│   ├── bracket_arb.py         # Bracket market sum < $1.00 (Tier 1)
│   ├── bonding.py             # Near-certainty harvesting (Tier 1)
│   ├── llm_ensemble.py        # 5-model LLM ensemble (Tier 2)
│   ├── orderflow.py           # VPIN + orderbook imbalance (Tier 2)
│   ├── rules_based.py         # Time decay, price consistency (Tier 3)
│   └── data_enhanced.py       # NOAA + FRED signals (Tier 3)
│
├── signals/                   # Signal aggregation and calibration
│   ├── ensemble.py            # Bayesian aggregation, momentum adjustment
│   ├── calibration.py         # Platt scaling, per-type shrinkage
│   └── base_rates.py          # Historical base rates for anchoring
│
├── risk/                      # Risk pipeline (8-layer sequential gate)
│   ├── pipeline.py            # Orchestrates all 8 layers
│   ├── kelly.py               # Kelly sizing with regime adjustment
│   ├── circuit_breaker.py     # Per-strategy isolation
│   ├── regime.py              # Market regime detection
│   └── survival.py            # Self-sustainability gate (3-state)
│
├── execution/                 # Order placement and fill management
│   ├── smart_entry.py         # Limit → timeout → market state machine
│   ├── fill_manager.py        # Poll resting orders, detect fills
│   └── position_monitor.py    # SL/TP enforcement, max hold time
│
├── orchestrator/              # Signal orchestration
│   ├── orchestrator.py        # Cycle: gather → score → dedup → dispatch
│   ├── scorer.py              # Signal scoring formula
│   └── dedup.py               # Deduplication rules + cooldowns
│
├── data/                      # External data providers
│   ├── weather.py             # NOAA NWS (5-source blend)
│   ├── index.py               # Yahoo Finance real-time
│   ├── economics.py           # FRED indicators + FedWatch
│   └── news.py                # Brave Search + RSS feeds
│
├── state/                     # Persistence
│   ├── database.py            # SQLite WAL setup, schema, migrations
│   ├── models.py              # SQLite row ↔ dataclass mapping
│   └── cost_basis.py          # FIFO tax lot tracking
│
├── monitoring/                # Observability
│   ├── telegram.py            # Telegram bot (alerts, /status, /stats)
│   ├── perf_tracker.py        # Daily P&L, win rate, CLV tracking
│   └── rest_api.py            # FastAPI server (35+ endpoints)
│
├── config/                    # Configuration loading and validation
│   ├── settings.py            # Pydantic BaseSettings (env + YAML)
│   └── defaults.py            # Hard-coded risk ceilings (cannot override)
│
├── utils/                     # Shared utilities
│   ├── logging.py             # structlog setup with JSON output
│   ├── types.py               # Shared dataclasses (Market, Signal, Position, Order)
│   └── helpers.py             # Time utilities, retry decorators, etc.
│
└── tests/                     # Test suite (target: 90%+ coverage)
    ├── unit/                  # Fast, isolated unit tests
    │   ├── test_strategies/   # One file per strategy
    │   ├── test_risk/         # Risk pipeline layer by layer
    │   ├── test_execution/    # SmartEntry state machine
    │   └── test_config/       # Config loading + validation
    ├── integration/           # End-to-end with mocked Kalshi API
    │   ├── test_trade_flow.py # Signal → risk → execution → fill
    │   └── test_state_db.py   # SQLite operations
    └── conftest.py            # Shared fixtures, mock clients
```

---

## 4. Data Flow

```
Market Discovery (every 10 min)
  → KalshiClient.get_markets()
  → MarketFilter (blocklist, volume, spread, price range)
  → Store to market_snapshots table
  → Fan out to all enabled strategies

Strategy Execution (per market, async parallel, 30s timeout)
  → Each strategy: analyze(market) → Signal | None
  → Signal = {ticker, side, confidence, edge_cents, urgency, metadata}

Signal Aggregation (Orchestrator)
  → Collect signals from all strategies
  → Score each: edge*conf*0.45 + conf*0.30 + urgency*0.15 + consensus*0.10
  → Multi-engine consensus check (2+ same side → 1.5× confidence boost)
  → Deduplication (ticker+side+engine, event prefix, stop-loss cooldown)
  → Sort by score, take top N

Risk Pipeline (per signal, sequential)
  → Layer 1: Spread cost check (can we profit after fees?)
  → Layer 2: Volume/liquidity minimum
  → Layer 3: Confidence floor (category-specific)
  → Layer 4: Net edge after fees
  → Layer 5: Daily loss limit check
  → Layer 6: Consecutive loss circuit breaker
  → Layer 7: Portfolio drawdown check
  → Layer 8: Total exposure cap
  → Kelly sizing → position_size in dollars
  → Returns: (approved: bool, size: float, reject_reason: str)

Order Execution (SmartEntry state machine)
  → Place limit order 1-2c inside spread
  → Data-verified signals (NOAA/Yahoo): aggressive crossing
  → Wait up to 5 minutes for fill
  → Timeout → place market order for remainder
  → Generate client_order_id (UUID4) for idempotency
  → Save to orders table

Fill Management (background, 30s poll)
  → Poll open orders via KalshiClient
  → Detect partial/complete fills
  → Transition order state: RESTING → PARTIALLY_FILLED → FILLED
  → On fill: create/update position in positions table
  → Cancel orders after 10 min (stale) or 3c mid-market drift

Position Monitoring (background, 5min)
  → Fetch current prices for all open positions
  → Check stop-loss (default 40%)
  → Check take-profit (default 60%)
  → Check max hold time (80% of time-to-close)
  → On exit signal: SmartEntry exit order (aggressive market crossing)
  → Save realized P&L to trades and daily_pnl tables
  → Trigger 30min cooldown in orchestrator for the market/side

Resolution Tracking (background, 15min)
  → Fetch recent Kalshi settlements
  → Match against open positions and closed orders
  → Record final outcome in cost_basis_lots (for tax)
  → Update CLV records (entry price vs final resolution)
```

---

## 5. Configuration Schema

```yaml
# config.yaml — operator-readable configuration
# Secrets go in .env, not here

bot:
  name: "morpheus"
  dry_run: false        # Overridden by --dry-run CLI flag
  log_level: "INFO"
  loop_interval_seconds: 60

kalshi:
  base_url: "https://api.elections.kalshi.com/trade-api/v2"
  scan_interval_seconds: 600
  categories: ["Politics", "Economics", "Climate and Weather", "Financials", ...]
  max_resolution_days: 1

strategy:
  enabled_strategies: ["index_fast_path", "bracket_arb", "bonding", "llm_ensemble", "orderflow"]
  min_edge: 0.04
  kelly_fraction: 0.25          # Configurable, capped by hard ceiling (0.50)
  max_position_size: 25.0       # Configurable, capped at $200
  max_total_exposure: 200.0     # Configurable, capped at $10,000
  min_confidence: 0.60

market_filters:
  min_volume_24h: 1000
  min_price: 0.12
  max_price: 0.90
  max_spread_pct: 0.10
  blocked_tickers: ["KXBTC", "KXETH", ...]

risk:
  max_daily_loss: 20.0          # Configurable, capped at $500
  stop_loss_pct: 0.40
  take_profit_pct: 0.60
  max_position_hold_hours: 24
  consecutive_loss_limit: 5
  drawdown_halt_pct: 0.15

llm:
  ensemble_mode: "full"
  models:
    - {provider: "openai",     model: "gpt-4o",                  enabled: true}
    - {provider: "anthropic",  model: "claude-sonnet-4-20250514", enabled: true}
    - {provider: "mistral",    model: "mistral-small-latest",     enabled: true}
  monthly_budget_usd: 5.0
  daily_budget_usd: 0.50
  cache_ttl_minutes: 60

calibration:
  default_platt_alpha: 0.68
  index_platt_alpha: 0.90
  weather_platt_alpha: 0.83
  per_type:
    index:    {platt_alpha: 0.90, shrink: -0.05, kelly_boost: 1.0}
    weather:  {platt_alpha: 0.83, shrink: 0.0,   kelly_boost: 1.0}
    politics: {platt_alpha: 0.68, shrink: 0.12,  kelly_boost: 0.5}
    economics:{platt_alpha: 0.75, shrink: 0.08,  kelly_boost: 0.7}

survival:
  enabled: true
  reduced_window_days: 7
  halted_window_days: 14
  reduced_multiplier: 0.5

alerts:
  telegram_enabled: true
  alert_on_trade: true
  daily_summary_hour: 23

api:
  enabled: true
  port: 8000
  host: "0.0.0.0"
```

### Hard-Coded Risk Ceilings (cannot be overridden by config)

```python
# config/defaults.py — these are enforced in code, not YAML
ABSOLUTE_MAX_POSITION_SIZE = 200.0       # Never exceed $200 per market
ABSOLUTE_MAX_TOTAL_EXPOSURE = 10_000.0   # Never exceed $10K total open
ABSOLUTE_MAX_DAILY_LOSS = 500.0          # Never lose more than $500/day
ABSOLUTE_MAX_KELLY_FRACTION = 0.50       # Never use more than half-Kelly
ABSOLUTE_MIN_CONFIDENCE = 30             # Never trade below 30% confidence
ABSOLUTE_MIN_EDGE_CENTS = 1.0            # Never trade below 1c edge
```

---

## 6. Database Schema

```sql
-- state/morpheus.db (SQLite WAL)

-- Every executed trade (entry and exit events)
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,   -- 'BUY_YES' | 'BUY_NO'
    quantity INTEGER NOT NULL,
    price_cents INTEGER NOT NULL,
    fill_price_cents INTEGER,
    strategy TEXT NOT NULL,
    confidence REAL,
    edge_cents REAL,
    pnl_cents REAL,
    is_paper BOOLEAN DEFAULT 0,
    order_id TEXT,
    metadata TEXT,             -- JSON blob for extra context
    UNIQUE(order_id)
);

-- Currently open positions
CREATE TABLE positions (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    avg_price_cents INTEGER NOT NULL,
    current_price_cents INTEGER,
    unrealized_pnl_cents INTEGER,
    opened_at TEXT NOT NULL,
    strategy TEXT NOT NULL,
    stop_loss_pct REAL DEFAULT 0.40,
    take_profit_pct REAL DEFAULT 0.60,
    UNIQUE(ticker, direction)
);

-- All orders (pending, filled, cancelled)
CREATE TABLE orders (
    id TEXT PRIMARY KEY,          -- client_order_id (UUID4)
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price_cents INTEGER NOT NULL,
    order_type TEXT NOT NULL,     -- 'limit' | 'market'
    status TEXT NOT NULL,         -- 'RESTING'|'PARTIALLY_FILLED'|'FILLED'|'CANCELLED'
    placed_at TEXT NOT NULL,
    timeout_at TEXT,
    filled_quantity INTEGER DEFAULT 0,
    fill_price_cents INTEGER,
    exchange_order_id TEXT,
    is_smart_entry BOOLEAN DEFAULT 1
);

-- Raw signal log (every signal from every strategy)
CREATE TABLE signals (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    strategy TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL,
    edge_cents REAL,
    approved BOOLEAN,
    reject_reason TEXT,
    metadata TEXT
);

-- Daily P&L summary
CREATE TABLE daily_pnl (
    date TEXT PRIMARY KEY,
    realized_pnl_cents INTEGER DEFAULT 0,
    unrealized_pnl_cents INTEGER DEFAULT 0,
    trade_count INTEGER DEFAULT 0,
    win_count INTEGER DEFAULT 0
);

-- Market snapshots for price history
CREATE TABLE market_snapshots (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    yes_price INTEGER,
    no_price INTEGER,
    volume INTEGER,
    snapshot_at TEXT NOT NULL,
    INDEX(ticker, snapshot_at)
);

-- FIFO cost basis for tax tracking
CREATE TABLE cost_basis_lots (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    purchase_price_cents INTEGER NOT NULL,
    purchase_date TEXT NOT NULL,
    disposed BOOLEAN DEFAULT 0,
    disposal_date TEXT,
    sale_price_cents INTEGER,
    realized_gain_cents INTEGER
);

-- CLV (Closing Line Value) records
CREATE TABLE clv_records (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    market_type TEXT,
    entry_price_cents INTEGER,
    close_price_cents INTEGER,
    clv REAL,           -- (entry_price - close_price) / close_price
    timestamp TEXT NOT NULL
);

-- Per-strategy performance tracking
CREATE TABLE strategy_perf (
    strategy TEXT NOT NULL,
    date TEXT NOT NULL,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    pnl_cents INTEGER DEFAULT 0,
    PRIMARY KEY (strategy, date)
);
```

---

## 7. Error Handling Strategy

### Principle: Fail closed, log everything, never crash silently

```
Level 1 — Transient API errors (timeout, 429, 5xx):
  → Retry with exponential backoff (max 3 attempts: 1s, 2s, 4s)
  → Log warning with attempt count
  → Circuit breaker: disable strategy after 5 consecutive failures

Level 2 — Strategy errors (bad data, parse failure, LLM timeout):
  → Catch in strategy.analyze(), return None
  → Increment strategy error counter
  → Circuit breaker trips at threshold

Level 3 — Execution errors (order rejection, invalid ticker, halted market):
  → Log error with full context
  → Do not retry immediately (likely permanent)
  → Alert via Telegram for unusual errors

Level 4 — Critical errors (DB corruption, auth failure, config invalid):
  → Log critical
  → Telegram alert
  → Graceful shutdown (allow background tasks to finish)
  → Systemd will restart after 30s

Level 5 — Kill switch (state/STOP_TRADING file exists):
  → Check at start of every orchestrator cycle
  → Log info, halt all new trading
  → Background tasks (fill manager, position monitor) continue
```

### Circuit Breaker (per strategy)
```
State: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
  CLOSED → OPEN:      5 consecutive failures OR 30% error rate in 10-minute window
  OPEN:               Skip strategy entirely, log warning
  OPEN → HALF_OPEN:   After 5 minute cooldown
  HALF_OPEN → CLOSED: Single success
  HALF_OPEN → OPEN:   Any failure
```

---

## 8. Logging and Monitoring Approach

### Logging (structlog)

```python
# Every log event is structured JSON
log.info(
    "trade_executed",
    ticker="KXINXU-26MAR28-T2480",
    side="BUY_YES",
    strategy="index_fast_path",
    confidence=0.82,
    edge_cents=17,
    quantity=3,
    price_cents=65,
    account="primary",
)
```

**Log levels:**
- `DEBUG`: Internal state (signal scores, cache hits, skipped markets)
- `INFO`: Trade lifecycle (signal generated, order placed, position opened/closed)
- `WARNING`: Non-fatal issues (retried API call, circuit breaker trip, budget warning)
- `ERROR`: Failed operations (order rejected, position exit failed, DB write error)
- `CRITICAL`: System integrity (auth failure, DB corruption, unexpected halt)

**Log rotation:** Daily files, 7-day retention, max 100MB per file.

### Telegram Alerts

Sent for:
- Every trade execution (ticker, side, size, confidence, edge)
- Every position exit (P&L, hold time, exit reason)
- Daily summary (total trades, wins, P&L, open positions, survival mode status)
- Error alerts (first 3 consecutive errors, then batch every hour)
- Kill switch activated / deactivated

### REST API (FastAPI, port 8000)

Endpoints:
```
GET /api/health           → uptime, DB size, last cycle time
GET /api/status           → full bot status, exposure, daily P&L
GET /api/positions        → open positions with unrealized P&L
GET /api/trades?days=7    → recent trades
GET /api/signals?hours=1  → recent signals (approved + rejected)
GET /api/pnl              → daily P&L history
GET /api/strategies       → per-strategy win rate and P&L
GET /api/risk             → exposure, circuit breaker states
GET /api/clv              → CLV by market type
POST /api/admin/halt      → activate kill switch (requires auth)
POST /api/admin/resume    → deactivate kill switch (requires auth)
```

Authentication: API key in `X-API-Key` header (configured in `.env`).

---

## 9. How Strategies Coexist (Strategy Interface)

Every strategy implements the same interface:

```python
# src/strategies/base.py
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod

@dataclass
class TradeSignal:
    ticker: str
    side: str                    # 'BUY_YES' | 'BUY_NO'
    confidence: float            # 0.0–1.0
    edge_cents: float            # Expected P&L per contract in cents
    urgency: str                 # 'immediate' | 'normal' | 'low'
    strategy: str                # Strategy name (for dedup + attribution)
    market_type: str             # 'index' | 'weather' | 'economics' | ...
    metadata: dict               # Strategy-specific context (LLM reasoning, etc.)

class BaseStrategy(ABC):
    name: str                    # Strategy identifier
    weight: float                # Default weight in ensemble (0.0–1.0)

    @abstractmethod
    async def analyze(self, market: Market) -> Optional[TradeSignal]:
        """
        Analyze a market and return a trade signal, or None to skip.
        Must not raise exceptions — catch internally and return None.
        Must complete in < 30 seconds.
        """
        ...
```

The orchestrator runs all enabled strategies concurrently:
```python
tasks = [strategy.analyze(market) for strategy in enabled_strategies]
signals = [s for s in await asyncio.gather(*tasks, return_exceptions=True) if s]
```

### Adding a New Strategy

1. Create `src/strategies/my_strategy.py` extending `BaseStrategy`
2. Implement `async def analyze(self, market: Market) -> Optional[TradeSignal]`
3. Add to `src/strategies/__init__.py`
4. Add to `config.yaml` under `strategy.enabled_strategies`
5. Write unit tests in `tests/unit/test_strategies/test_my_strategy.py`

No changes needed to orchestrator, risk pipeline, or execution engine.

---

## 10. Deployment Architecture

### Development (local)
```bash
python -m src.main --dry-run          # Paper trading, all signals logged
python -m src.main --dry-run --once   # Single cycle, useful for testing
python -m src.main --api              # Start REST API only
```

### Production (DigitalOcean / VPS)
```bash
# Systemd service: /etc/systemd/system/morpheus.service
[Unit]
Description=Morpheus Trading Bot
After=network.target

[Service]
User=morpheus
WorkingDirectory=/opt/morpheus
ExecStart=/opt/morpheus/.venv/bin/python -m src.main
Restart=always
RestartSec=30
EnvironmentFile=/opt/morpheus/.env
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Deploy process
```bash
ssh morpheus "cd /opt/morpheus && git pull origin main && systemctl restart morpheus"
```

### Environment variables (`.env`)
```bash
# Kalshi (required)
KALSHI_API_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_PATH=/opt/morpheus/kalshi_key.pem
KALSHI_USE_DEMO=false        # Must be 'false' to enable live trading
KALSHI_API_KEY_ID_2=...      # Optional secondary account
KALSHI_PRIVATE_KEY_PATH_2=... # Optional secondary account

# AI APIs (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional
FRED_API_KEY=...
BRAVE_SEARCH_API_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
MORPHEUS_API_KEY=...          # For REST API authentication
```

---

## 11. Testing Strategy

### Test Structure (mirroring src/)
```
tests/
├── conftest.py               # Shared fixtures: mock KalshiClient, test DB, sample markets
├── unit/
│   ├── strategies/           # One file per strategy, mock all external calls
│   ├── risk/                 # Test each of the 8 risk layers independently
│   ├── execution/            # SmartEntry state machine tests
│   ├── signals/              # Ensemble aggregation, Platt calibration
│   └── config/               # Config loading, hard ceiling enforcement
└── integration/
    ├── test_trade_flow.py    # Full signal → execute → fill pipeline (mocked API)
    ├── test_state_db.py      # SQLite CRUD, concurrent access, WAL
    └── test_api.py           # REST API endpoints
```

### Key Test Principles
1. **Mock Kalshi API** — never hit real API in tests
2. **Test risk layers independently** — each layer should be testable in isolation
3. **Test hard ceilings** — verify config cannot override ABSOLUTE_MAX_* constants
4. **Test circuit breaker transitions** — CLOSED → OPEN → HALF_OPEN → CLOSED
5. **Test SmartEntry state machine** — every state transition
6. **Property-based testing** — use `hypothesis` for Kelly sizing edge cases

### Running Tests
```bash
pytest tests/ -v                     # All tests
pytest tests/unit/ -v               # Unit tests only (fast)
pytest tests/ --cov=src --cov-report=term-missing  # With coverage
ruff check src/ tests/              # Linting
mypy src/                           # Type checking
```
