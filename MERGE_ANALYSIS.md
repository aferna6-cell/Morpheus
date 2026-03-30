# MERGE_ANALYSIS.md — Morpheus + Neo Codebase Merger

> Phase 1 analysis document. Written before any code changes.
> Last updated: 2026-03-30

---

## 1. Executive Summary

Both bots trade Kalshi binary prediction markets. Morpheus is the battle-hardened production bot (36 waves of live trading, ~$77 total PnL tracked). Neo is a more architecturally sophisticated system with comprehensive test coverage, SQLite state, and a richer risk pipeline, but no live trading history.

**Merge decision:** Morpheus is the base (preserve git history, deploy to existing server). Upgrade infrastructure with Neo's best patterns. Merge all viable strategies. Target: one bot that is production-proven AND well-tested.

---

## 2. Side-by-Side Feature Matrix

| Feature | Morpheus | Neo | Winner | Notes |
|---|---|---|---|---|
| **Live trading history** | 36 waves, $77+ PnL | None | Morpheus | Morpheus wins on proven track record |
| **Test coverage** | ~5% (3 files) | 99.5% (105 files, 3,234 tests) | Neo | Critical gap in Morpheus |
| **State persistence** | JSONL flat files | SQLite WAL (15 tables) | Neo | JSONL causes concurrency bugs, hard to query |
| **Config system** | YAML + Pydantic | .env + Pydantic BaseSettings | Hybrid | YAML for humans, env-var overrides for prod |
| **Logging** | structlog (JSON) | loguru + rich | Morpheus | structlog is better for structured log parsing |
| **Risk pipeline** | Ad-hoc checks in executors | 8-layer sequential gate | Neo | Neo's pipeline is formal, auditable |
| **Hard-coded risk ceilings** | No (config-only) | Yes (Pydantic validators) | Neo | Config-only limits got bypassed in practice |
| **Execution (fill rates)** | 4% fill rate (limit-only) | SmartEntry (limit → market fallback) | Neo | 4% is unacceptable for a trading bot |
| **Multi-account support** | Yes (primary + secondary) | No (single account) | Morpheus | Dual account diversification is valuable |
| **Kalshi RSA auth** | Yes (production-proven) | Yes (same approach) | Tie | Both work correctly |
| **Rate limiting** | Basic (8 req/s) | Adaptive (tracks X-RateLimit headers) | Neo | Adaptive is more robust |
| **LLM calibration** | Platt scaling + per-type shrinkage (tuned 36 waves) | Bayesian refinement + Brier score weighting | Morpheus | 36 waves of calibration data beats theoretical Bayesian |
| **Data fast-paths** | NOAA, Yahoo Finance, FRED, Brave Search | NOAA, FRED (basic), RSS feeds | Morpheus | Morpheus fast-paths are richer and zero-cost |
| **Structured data** | 5-source weather blend, CME FedWatch | NOAA basic, FRED trend detection | Morpheus | Morpheus's data pipeline is more sophisticated |
| **Telegram alerts** | Yes (trades, daily summary, errors) | No (uses REST API instead) | Morpheus | Telegram is essential for live monitoring |
| **REST API** | No | FastAPI (35+ endpoints, JWT auth) | Neo | Valuable for monitoring without SSH |
| **Kill switch** | Yes (`state/STOP_TRADING` file) | No | Morpheus | Critical safety feature |
| **Survival mode** | Yes (3-state: normal/reduced/halted) | No (only daily loss halt) | Morpheus | Prevents slow-bleed to zero |
| **CLV tracking** | Yes (Closing Line Value per trade) | No | Morpheus | Measures edge quality, not just P&L |
| **Cost basis / tax** | No | Yes (FIFO lots, tax report) | Neo | Must-have for real money |
| **Circuit breakers** | Consecutive loss gate | Per-strategy isolation + drawdown halt | Neo | Neo's circuit breakers are more granular |
| **Regime detection** | No | Yes (5 regimes: normal/trending/mean-reverting/volatile/low) | Neo | Adjusts Kelly and thresholds to market conditions |
| **Backtesting** | No native backtester | Yes (synthetic data, multi-seed) | Neo | Valuable for pre-deploy validation |
| **Paper trading** | Yes (dry_run flag) | Yes (double-gate: CLI + env) | Neo | Neo's double-gate is safer |
| **Category-specific risk** | Per-type Kelly and calibration | Per-category confidence floors and exposure limits | Both | Complement each other |
| **Ensemble LLM** | 5-model trimmed mean (Claude+GPT-4o+Mistral+DeepSeek+Gemini) | 3-tier routing (Haiku/Sonnet/Opus) | Both | Morpheus's 5-model trimmed mean has more breadth; Neo's cost tiering is smarter |
| **Order deduplication** | Orchestrator-level (ticker+side+engine, event prefix) | Position-level duplicate check | Morpheus | Orchestrator dedup prevents event-correlated overexposure |
| **Stop-loss cooldown** | Yes (30min after SL exit, Wave 35) | No explicit cooldown | Morpheus | Critical fix for re-entry death loops |
| **Websocket data** | Partial (crypto engine, disabled) | Full (real-time orderbook, graceful fallback) | Neo | WebSocket enables better fill prices |
| **Dashboard** | No | Terminal dashboard (--dashboard mode) | Neo | Useful for monitoring |

---

## 3. Strategy Comparison

### Strategies in Morpheus (active)

| Engine | Status | Edge Source | Live Track Record |
|---|---|---|---|
| **kalshi_llm** | Active | 5-model ensemble LLM + structured fast-paths | Index: 9W/0L +$107.67; Overall: 25% WR |
| **kalshi_bracket_arb** | Active | Mathematical: sum YES asks < $1.00 | Risk-free, ~1.5-7% margins |
| **kalshi_bonding** | Active | NOAA/Yahoo verified near-certainties (90-97c) | High WR on weather/index |
| **kalshi_orderflow** | Active (Wave 36) | VPIN-based informed money detection | New, no history |
| **kalshi_contrarian** | Disabled Wave 28 | Crowd overconfidence (80-95% consensus) | Poor track record |
| **kalshi_crypto** | Disabled Wave 34 | BTC latency arb on KXBTC15M | 0% edge, blocked market |
| **kalshi_mm** | Disabled Wave 34 | Avellaneda-Stoikov spread capture | Needs >$500 capital |
| **kalshi_theta** | Disabled Wave 22 | Time decay near close | Never filled |
| **kalshi_longshot** | Disabled Wave 29 | Sell longshots <12c | Never filled |
| **kalshi_cross_arb** | Disabled Wave 33 | Polymarket cross-venue arb | Lost $49 of $50 deposit |

### Strategies in Neo (all 10 strategies)

| Strategy | Weight | Edge Source | Notes |
|---|---|---|---|
| **ai_forecaster** | 35% | Multi-LLM Claude ensemble (tiered routing) | No live trading history |
| **arbitrage** | 25% | Price consistency + historical base rates | Same concept as bracket_arb but simpler |
| **sentiment** | 20% | RSS news + TextBlob NLP | Weak signal; needs Claude integration |
| **market_maker** | 10% | Avellaneda-Stoikov (same as Morpheus) | Disabled needs capital |
| **rules_based** | 10% | Time decay, price consistency | Conservative supplement |
| **orderbook_imbalance** | 5% | Bid/ask depth imbalance ratio | Similar to Morpheus's VPIN but simpler |
| **bonding** | Near-resolution | Near-expiry certainty harvesting | Similar to Morpheus bonding |
| **fade** | Optional | Mean reversion after sharp moves | No live history |
| **cross_platform_arb** | Optional | Kalshi vs Polymarket NLP matching | Disabled, same problems as Morpheus |
| **data_enhanced** | Optional | NOAA weather + FRED economics | Subset of Morpheus's structured data |

### Strategy Merge Recommendations

| Priority | Strategy | Source | Action |
|---|---|---|---|
| **P0** | Index fast-path (NOAA/Yahoo) | Morpheus LLM engine | Keep as-is — 9W/0L proven |
| **P0** | Bracket arb | Morpheus | Keep as-is — mathematically risk-free |
| **P0** | Bonding | Morpheus | Keep as-is — high WR |
| **P1** | LLM ensemble | Merge both | Morpheus's 5-model + Neo's cost tiering |
| **P1** | OrderFlow/VPIN | Morpheus (Wave 36) | Keep + complement with Neo's orderbook OBI |
| **P2** | Data-enhanced | Neo | Port NOAA/FRED as standalone signals |
| **P2** | Rules-based | Neo | Add as low-weight supplement |
| **P3** | Market making | Both have same impl | Re-enable at $500+ capital |
| **P3** | Fade/sentiment | Neo | Add with low weight, validate before trusting |
| **Deferred** | Cross-platform arb | Both disabled | Do not port (both failed) |
| **Deferred** | Contrarian | Morpheus (disabled) | Do not port (poor track record) |
| **Deferred** | Crypto latency arb | Morpheus (disabled) | Do not port (market blocked) |

---

## 4. Architecture Comparison

### Morpheus Architecture

```
config.yaml + .env
      │
main.py (BotConfig → engines → orchestrator)
      │
┌─────┴───────────────────────────┐
│         Orchestrator             │
│  (score/dedup/rank/dispatch)    │
└────┬────┬────┬────┬─────────────┘
     │    │    │    │
  LLM  BrktArb Bonding OrderFlow    ← Engines (parallel async)
     │    │    │    │
     └────┴────┴────┘
              │
         Executor(s)
         (primary + secondary accounts)
              │
         FillManager (poll resting orders)
              │
         PositionMonitor (SL/TP background task)
              │
         State: JSONL files (state/)
```

**Strengths:** Production-hardened, multi-account, proven calibration, kill switch, survival gate, CLV tracking.
**Weaknesses:** No tests, JSONL state is fragile, 4% fill rate, no formal risk pipeline, config-only limits.

### Neo Architecture

```
.env / Pydantic BaseSettings
         │
    main.py (Neo orchestrator)
         │
┌────────┴──────────────────────────────────────┐
│  ProbabilityEstimator (ensemble aggregator)    │
│  ├── Bayesian updating per ticker              │
│  ├── Momentum analysis (price history)         │
│  ├── Regime detection (volatility/activity)    │
│  └── Category-specific weights                 │
└────────┬──────────────────────────────────────┘
         │
    10 Strategies (async, 30s timeout each)
         │
    RiskManager (8-layer sequential gate)
         │
    SmartEntry (limit → timeout → market)
         │
    OrderManager (pending order lifecycle)
         │
    PositionManager (SL/TP/expiry/manual exits)
         │
    SQLite WAL (15 tables)
         │
    FastAPI REST (35+ endpoints)
```

**Strengths:** 3,234 tests, SQLite state, formal risk pipeline, SmartEntry execution, REST API, circuit breakers, regime detection, cost basis tracking.
**Weaknesses:** No live trading history, no multi-account, no Telegram, no kill switch, no CLV tracking.

### Winner by Component

| Component | Winner | Rationale |
|---|---|---|
| Signal generation | Merge both | Morpheus's fast-paths + Neo's ensemble sophistication |
| Risk pipeline | Neo | 8-layer formal gate > ad-hoc checks |
| Execution | Neo's SmartEntry | 4% fill rate is unacceptable |
| State | Neo's SQLite | JSONL has race conditions and no querying |
| Orchestration | Morpheus | Dedup logic, multi-engine consensus, cooldowns proven in live |
| Calibration | Morpheus | Platt parameters tuned over 36 waves |
| Config | Hybrid | Pydantic BaseSettings + optional YAML |
| Logging | Morpheus | structlog is better than loguru for structured output |
| Testing | Neo | Neo's test suite is the gold standard |
| Monitoring | Merge | Morpheus's Telegram + Neo's REST API |

---

## 5. API Usage Differences

Both bots use the same Kalshi API v2 and the same `kalshi-python>=2.1.0` SDK. Key differences:

| Concern | Morpheus | Neo |
|---|---|---|
| Auth | RSA-PSS, thread-pool for async | RSA-PSS, same |
| Rate limiting | 8 req/s fixed | Adaptive (X-RateLimit header tracking) |
| Retries | Exponential backoff (max 3) | Max 3 attempts, exponential backoff |
| WebSocket | Partial (disabled crypto engine) | Full real-time orderbook with fallback |
| Order placement | Limit-first, manual spread crossing | SmartEntry (limit → timeout → market) |
| Position fetching | `get_positions_without_preload_content` (Kalshi SDK bug workaround) | Standard SDK (may hit same bug) |
| Multi-account | Yes (primary + secondary) | No |

**Action:** Adopt Neo's adaptive rate limiting and SmartEntry execution. Preserve Morpheus's multi-account support and SDK bug workaround.

---

## 6. Dependency Conflicts and Incompatibilities

### Morpheus requirements.txt (key packages)
```
kalshi-python>=2.1.0
httpx>=0.24.0
structlog>=23.0.0
openai>=1.0.0
anthropic>=0.30.0
pydantic>=2.0.0
PyYAML>=6.0
scipy>=1.10.0
```

### Neo requirements.txt (key packages, pinned)
```
kalshi-python==2.1.4
httpx==0.28.1
loguru==0.7.3
openai (not present — Neo uses Anthropic only)
anthropic==0.84.0
pydantic==2.12.5
pydantic-settings==2.13.1
textblob==0.19.0
fastapi==0.135.1
loguru (replaces structlog)
```

### Conflicts
| Package | Morpheus | Neo | Resolution |
|---|---|---|---|
| Logging | structlog | loguru | **Use structlog** — better structured output |
| OpenAI | openai>=1.0.0 | None | **Keep openai** — needed for GPT-4o ensemble |
| Kalshi SDK | >=2.1.0 | ==2.1.4 | **Pin to 2.1.4** — Neo's pinned version |
| Pydantic | >=2.0.0 | ==2.12.5 | **Pin to 2.12.5** — latest stable |
| Config | PyYAML | pydantic-settings | **Use both** — pydantic-settings as base, YAML for overrides |
| TextBlob | None | 0.19.0 | **Add for sentiment strategy** |

No irreconcilable conflicts. All packages can coexist.

---

## 7. Code Quality Comparison

| Metric | Morpheus | Neo |
|---|---|---|
| Test coverage | ~5% (3 test files) | ~99.5% (105 test files) |
| Type hints | Partial (Pydantic models) | Comprehensive throughout |
| Docstrings | Minimal | Docstrings on all public APIs |
| Error handling | Global try/except + per-engine backoff | Per-layer structured error handling |
| Code size | ~80 Python files, ~8,000 LOC | ~100 files, ~14,500 LOC |
| Linting | No CI/linting config | No CI (but clean code) |
| Dead code | Some (disabled engines have their configs) | Minimal |
| Security | API keys in .env, no hardcoded secrets | Same, plus Stripe webhook validation |
| SQL injection | N/A (JSONL) | All queries parameterized |

---

## 8. Recommended Merge Strategy

### Principle: Morpheus is the foundation, Neo is the upgrade path

The merged bot lives in the Morpheus repo (preserving production history, deployment scripts, and git log). We adopt Neo's infrastructure improvements without abandoning what's proven in Morpheus's live trading record.

### Infrastructure (adopt from Neo)
1. **SQLite WAL** — replace JSONL state files. 15-table schema with proper indexes.
2. **8-layer risk pipeline** — formal sequential gate with hard-coded ceilings.
3. **SmartEntry execution** — limit 1-2c inside spread → 5min timeout → market fallback.
4. **Adaptive rate limiting** — track `X-RateLimit-*` headers, back off before hitting limits.
5. **Comprehensive tests** — adopt Neo's test structure (105 files, pytest + asyncio).
6. **FastAPI REST server** — port Neo's 35+ endpoints for monitoring.
7. **FIFO cost basis** — port Neo's tax lot tracking.
8. **Circuit breakers** — per-strategy isolation (Neo's `CircuitBreaker` class).
9. **Regime detection** — port Neo's `RegimeDetector` for Kelly adjustment.

### Infrastructure (keep from Morpheus)
1. **structlog** — better structured output than loguru.
2. **Pydantic + YAML config** — YAML for human-readable config, Pydantic for validation.
3. **Multi-account support** — primary + secondary Kalshi accounts.
4. **Telegram integration** — critical for live monitoring without SSH.
5. **Kill switch** — `state/STOP_TRADING` file.
6. **Survival mode** — 3-state self-sustainability gate.
7. **CLV tracking** — Closing Line Value per trade (edge quality metric).
8. **Per-type Platt calibration** — 36 waves of tuning beats Neo's theoretical Bayesian.
9. **Orchestrator dedup + cooldown** — event-level correlation limits, 30min SL cooldown.

### Strategies (merge best of both)
See Section 3 for full priority list. Short version:
- **Tier 1 (must have):** Index fast-path, bracket_arb, bonding
- **Tier 2 (high value):** LLM ensemble, orderflow/VPIN
- **Tier 3 (add with validation):** Rules-based, data-enhanced
- **Deferred:** MM (needs capital), fade/sentiment (unproven), cross-arb/contrarian/theta (failed in live)

### Configuration (hybrid approach)
- `config.yaml` — operator-readable config (keep Morpheus's YAML structure)
- `.env` — secrets and deployment-specific overrides
- `Pydantic BaseSettings` — validates both sources, provides defaults
- Hard-coded ceilings — max_position, max_exposure, max_daily_loss enforced in code (cannot be overridden)

### Migration path for live deployment
1. Deploy the merged bot in `--dry-run` mode first
2. Monitor that all strategies behave correctly
3. Switch to live mode with same config as current Morpheus
4. Graduate strategy weights as confidence builds

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Breaking live Morpheus bot during merge | Work on `merge/neo-integration` branch; production stays on `main` |
| SQLite migration losing state | Write migration script from JSONL → SQLite; keep JSONL as backup |
| Calibration drift after engine changes | Keep Platt alphas unchanged initially; re-tune after 100+ live trades on merged bot |
| Test coverage gaps | Focus tests on risk pipeline, execution, and signal generation — highest blast radius |
| New strategies degrading win rate | All new strategies start at 0% weight; gradually increase as live data accumulates |
| API key exposure in new code | `grep -r "sk-" --include="*.py"` pre-commit check |

---

## 10. Success Criteria for the Merge

- [ ] All Morpheus strategies (index, bracket_arb, bonding, orderflow) are functional
- [ ] Neo's risk pipeline (8-layer) replaces ad-hoc risk checks
- [ ] SQLite WAL replaces JSONL state files
- [ ] SmartEntry execution replaces pure limit-order strategy
- [ ] Test coverage ≥ 80% (measured with `pytest --cov`)
- [ ] All tests passing (`pytest -x`)
- [ ] Linting clean (`ruff check .`)
- [ ] Telegram alerts working
- [ ] REST API health endpoint responding
- [ ] Kill switch functional
- [ ] Paper mode requires double-gate (CLI + env)
- [ ] Pushed to `merge/neo-integration` branch
- [ ] Clean checkout + install + test run succeeds
