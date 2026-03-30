# Changelog

## [Unreleased] — merge/neo-integration

### Summary

This branch merges Neo (private experimental bot) into Morpheus (production Kalshi bot), adopting Neo's superior infrastructure while keeping Morpheus's 35+ waves of calibration tuning and Kalshi-specific execution patterns.

### New modules (from Neo, ported to Morpheus)

#### `src/db.py` — SQLite WAL database layer
- Replaces scattered JSONL files with a proper relational schema
- 9 tables: trades, positions, orders, signals, daily_pnl, market_snapshots, cost_basis_lots, clv_records, strategy_perf
- Async via `aiosqlite`, WAL mode for concurrent reads + writes
- `init_db(path)` factory function, idempotent schema creation
- `log_clv()` tracks Closing Line Value for edge quality measurement

#### `src/risk_pipeline.py` — 8-layer sequential risk pipeline
- Replaces `src/risk.py` (single-pass ad-hoc checks) with structured pipeline
- Hard-coded ceilings that config cannot override:
  - `ABSOLUTE_MAX_POSITION_SIZE = 200.0`
  - `ABSOLUTE_MAX_TOTAL_EXPOSURE = 10_000.0`
  - `ABSOLUTE_MAX_DAILY_LOSS = 500.0`
  - `ABSOLUTE_MAX_KELLY_FRACTION = 0.50`
  - `ABSOLUTE_MIN_CONFIDENCE = 0.30`
  - `ABSOLUTE_MIN_EDGE_CENTS = 1.0`
- Per-market-type category floors (politics = 70% min confidence)
- Consecutive loss breaker (halves Kelly after N losses, returns `"_REDUCED_SIZING_"` sentinel)
- `_kelly_size()` returns `(position_size_usd, kelly_fraction_used)` tuple

#### `src/smart_entry.py` — execution state machine
- Replaces direct `place_order` calls with proper limit → timeout → market fallback
- `_compute_limit_price()`: places limit 1–2c inside spread (confidence-dependent)
- `_compute_aggressive_price()`: NOAA 5c / index 3c / generic 1c above ask
- `_detect_signal_type(ticker)`: auto-classifies NOAA/index/generic tickers
- 5-minute timeout, 10-second poll interval
- `OrderResult` dataclass with full fill metadata and `OrderState` enum
- Paper mode simulates fill at mid-price with `is_paper=True`

#### `src/api.py` — FastAPI REST monitoring API
- `GET /api/health` — public, includes kill-switch state
- `GET /api/status`, `/positions`, `/trades`, `/signals`, `/pnl`, `/strategies`, `/risk`, `/clv` — require `X-API-Key`
- `POST /api/admin/halt`, `/api/admin/resume` — admin endpoints
- `GET /api/config` — read current config (keys redacted)
- `create_app(db, config)` factory for testing

#### `src/cost_basis.py` — FIFO cost basis tracker
- Tax lot tracking for realized gains/losses
- `record_purchase()`, `record_sale()`, `get_tax_report()`, `get_unrealized()`

### New engines (from Neo)

#### `src/engines/kalshi_orderbook_engine.py`
- Orderbook Imbalance (OBI) = (yes_bid_depth − no_bid_depth) / total
- Signals when |OBI| > 0.3 with confidence proportional to imbalance magnitude

#### `src/engines/kalshi_rules_engine.py`
- Time-decay rule: markets < 24h from expiry at extreme prices (>85c or <15c)
- Price-consistency rule: YES + NO < 92c (potential free money)

### Modified modules

#### `src/market_filters.py`
- Added `is_ticker_blocked(ticker)` standalone function (wraps `_JUNK_TICKER_PREFIXES`)
- `KXFED` added to blocklist after Wave 35 cross-arb incident ($49 loss on year-long positions)

### Test infrastructure (new)

- `tests/conftest.py` — shared fixtures: `minimal_config`, `mock_trading_client`, `mock_risk_state`, sample markets
- `tests/test_risk_pipeline.py` — 30 tests: all 8 layers, Kelly sizing, hard-coded ceilings, full pipeline
- `tests/test_db.py` — 17 tests: WAL mode, table creation, trade/position/signal CRUD, CLV, strategy perf
- `tests/test_api.py` — 20 tests: health, auth, all endpoints, halt/resume admin
- `tests/test_smart_entry.py` — 7 tests: limit pricing, signal type detection, paper/live execution
- `tests/test_strategies.py` — 26 tests: bracket arb, bonding, VPIN, OBI, Platt calibration, market filters
- `pytest.ini` — `asyncio_mode = auto` for seamless async test support

### Documentation

- `README.md` — fully rewritten for Kalshi (replaced Polymarket references)
- `MERGE_ANALYSIS.md` — side-by-side feature comparison of Morpheus vs Neo
- `ARCHITECTURE.md` — full system diagram, module breakdown, data flow, SQLite schema

### Key decisions

| Decision | Rationale |
|----------|-----------|
| structlog (not loguru) | Morpheus already uses structlog; 35 waves of production log patterns |
| SQLite WAL (not JSONL) | Concurrent reads + atomic writes; queryable; JSONL fragile under high volume |
| Pydantic + YAML config (not env-only) | Type-safe config with nested sections; env overrides still work |
| Morpheus Platt alphas kept | 36 waves of calibration data; Neo had no per-market-type tuning |
| Hard-coded risk ceilings | Wave 35 cross-arb incident proved config-only gates are insufficient |
| SmartEntry from Neo | More sophisticated than Morpheus executor; handles partial fills properly |
