# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Morpheus is an autonomous Kalshi prediction market trading bot. It runs 24/7 on a DigitalOcean droplet, trading binary contracts using LLM-powered probability estimation. Two Kalshi accounts (primary + secondary), focused on same-day politics/economics/policy markets where LLMs have genuine informational edge.

## Commands

```bash
# Run locally (live trading)
python -m src.main

# Dry run (no real trades)
python -m src.main --dry-run

# Single iteration then exit
python -m src.main --dry-run --once

# Deploy to production
ssh morpheus "cd /opt/morpheus && git pull && systemctl restart morpheus"

# Production logs
ssh morpheus "journalctl -u morpheus -n 50 --no-pager"

# Tests
pytest tests/

# Formatting
black src/ tests/
isort src/ tests/
```

## Architecture

### Signal Flow

```
Kalshi API (market data)
  → Engines (generate TradeSignals)
    → Orchestrator (score, rank, deduplicate, risk-check)
      → KalshiExecutor (place orders via KalshiTradingClient)
        → Background: PositionMonitor, FillManager, PerfTracker
```

### Engines (`src/engines/`)

Three engine types, all subclass `BaseEngine` and emit `TradeSignal` via `get_signals()`:

- **kalshi_llm**: Same-day markets. Uses `EnsembleSignal` (parallel Claude Sonnet + GPT-4o). Applies market type detection to skip junk markets (announcer mentions, crypto ranges, weather, word mentions). Cost-tracked with daily budget.
- **kalshi_contrarian**: 1-7 day markets at 80-95% crowd consensus. Bets against overconfident crowds when LLM disagrees by 10%+. Requires HIGH conviction.
- **kalshi_mm**: Market-making with zero maker fees. Posts two-sided limit orders, manages inventory skew.

### Ensemble Signal (`src/signals/ensemble_signal.py`)

The core prediction engine. This is the most complex file (~600 LOC):

1. **Market type detection** (imported from `llm_signal.py`): Classifies markets and skips types with no LLM edge
2. **News fetch**: Brave Search primary, Google RSS fallback (`src/news.py`)
3. **Structured data anchors**: FRED API for Fed/economic data, polling headlines (`src/structured_data.py`)
4. **Parallel LLM calls**: Claude Sonnet + GPT-4o with structured JSON output
5. **Calibration**: Shrinkage toward 0.5, asymmetric YES-dampening (LLMs overpredict YES), per-market-type adjustments
6. **Adversarial challenge**: When p_yes is 35-75% and side is BUY_YES, GPT-4o-mini argues it's too high
7. **Per-model weighting**: After 30+ resolved samples, weights by inverse Brier score (`src/model_tracker.py`)

### Orchestrator (`src/orchestrator.py`)

Collects signals from all engines each cycle, then:
- Filters stale signals (>300s) and sports markets
- Scores: `urgency × 0.40 + edge × 0.25 + confidence × 0.20 + multi_engine_bonus × 0.15`
- Multi-engine consensus: 2+ engines same direction → 1.5× confidence boost
- Deduplicates at signal level and event/correlation level
- Dispatches top N to executor, logs predictions

### Key Modules

| Module | Role |
|--------|------|
| `risk.py` | Third-Kelly sizing, daily loss halt, position caps |
| `position_monitor.py` | Stop-loss/take-profit enforcement, max 3 exit retries with backoff |
| `cost_tracker.py` | LLM budget enforcement ($10/day, $50/month) |
| `market_filters.py` | Volume, spread, liquidity gates + ticker prefix blocklist for junk markets |
| `capital_management.py` | Capital recycling + CLV tracking |
| `resolution_tracker.py` | Matches predictions to Kalshi settled outcomes |
| `base_rates.json` | 51 empirical reference classes for anchoring LLM predictions |

## Configuration

- **`config.yaml`**: All bot parameters (strategy, filters, LLM, risk, timing, alerts)
- **`.env`**: API keys (Kalshi RSA keys, OpenAI, Anthropic, Telegram, Brave Search)
- **`state/`**: Runtime persistence (cost tracking, trade history, predictions, perf stats)
- **`state/STOP_TRADING`**: Kill switch — create this file to halt all trading

## Kalshi SDK Notes

- Use `get_positions_without_preload_content` for raw JSON (SDK bug with preloaded content)
- Auth uses RSA key signing (`kalshi_key.pem`, `kalshi_key_2.pem`)
- Maker orders have zero fees; the bot defaults to limit orders
- Two separate clients: `KalshiClient` (public/read-only) and `KalshiTradingClient` (authenticated)

## Style

- Python 3.11+, async throughout (httpx, asyncio)
- Pydantic models for config (`BotConfig` in `src/utils.py`)
- structlog for all logging
- black (line-length 88) + isort (profile "black")
- JSONL for all persistent event logs (trades, predictions, model tracking)
