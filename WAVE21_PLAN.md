# Wave 21: Performance Analysis & Research-Backed Strategy Improvements

## Executive Summary

Analysis of 81 resolved trades (Feb 10-13) and comparison with academic research (Whelan 300K contracts, Becker 72.1M trades, IMDEA $40M arb study, Science Advances LLM ensemble) reveals:

**What's working:**
- Index trading (LLM+index): 8W/0L, +$106.05 — bot's #1 profit center
- NO-side bias: buy_no 58% WR +$93.15 (confirmed by Whelan/Becker FLB research)
- Edge 0.20-0.30 bucket: 70% WR, +$61.85 — sweet spot

**What's broken:**
- CRITICAL BUG: Weather repricing completely non-functional (`entry_price` undefined in position_monitor.py:482)
- buy_yes: 21% WR, -$17.97 (even with 0.25 dampening, still catastrophic)
- NOAA direct on weather: 16.7% WR, -$3.82 (model confidence ≠ actual accuracy)
- Edge 0.5-1.0 bucket: 0% win rate, -$5.55 (high-edge estimates are anti-correlated with winning)
- Economics: 0W/4L, -$3.08 (CPI/Core CPI all wrong)
- Crypto: 50% WR but -$23.42 (blocked but was bypassed before Wave 16)
- Calibration severely off: buy_yes predicted avg 0.50, actual 0.25

**Key research insights:**
- Becker: NO contracts outperform YES at 69/99 price levels. Weather maker-taker gap: 2.57pp
- Whelan: Contracts <10c lose 60%+ of invested money. Makers earn +1.9% post-fee
- Meister (arXiv): Overestimating edge is asymmetrically worse than underestimating
- Market Maker's Dilemma (arXiv 2025): Fill probability negatively correlated with post-fill returns
- Temporal Evolution of Mispricing: FLB intensifies near expiration (exploitable by theta engine)

---

## Phase 1: Critical Bug Fixes (Immediate)

### 1.1 Fix weather repricing NameError (CRITICAL)
**File:** `src/position_monitor.py:482`
**Bug:** `entry_price` variable is undefined in `_check_weather_repricing()`. The variable exists in `_check_position()` but is not in scope. Because the method is wrapped in try/except, the NameError is silently caught, and **weather repricing never executes an exit**.

**Fix:** Change line 482 from:
```python
entry_price=entry_price,
```
to:
```python
entry_price=tracked.entry_price_cents / 100.0,
```

**Impact:** This single-line fix re-enables weather position exits. Weather positions that should be exited when NOAA forecasts shift against the bot have been held indefinitely, contributing to weather's -$6+ losses.

**Verification:**
- `grep -n "entry_price" src/position_monitor.py` — confirm no other references to bare `entry_price` in the method
- Deploy and check logs for `weather_repricing_check` events (should now appear without errors)

### 1.2 Fix rain probability calculation (MEDIUM)
**File:** `src/structured_data.py:1586-1588`
**Bug:** Uses max hourly PoP as p_yes directly. For "any rain" markets (threshold=0), the probability of rain during the day is `1 - product(1 - pop_i/100)`, not `max(pop_i)/100`.

**Fix:** Replace:
```python
if max_pop >= 80 or max_pop <= 15:
    p_yes = max_pop / 100.0
```
with:
```python
p_no_rain = 1.0
for pop in pop_values:
    p_no_rain *= (1.0 - pop / 100.0)
p_yes_cumulative = 1.0 - p_no_rain

if p_yes_cumulative >= 0.80 or p_yes_cumulative <= 0.15:
    p_yes = p_yes_cumulative
```

**Verification:** Run `python -c "vals=[80,60,40,30]; print(1 - __import__('functools').reduce(lambda a,b: a*(1-b/100), vals, 1.0))"` — should be ~0.97, not 0.80.

### 1.3 Fix bracket arb completeness check (HIGH)
**File:** `src/engines/kalshi_bracket_arb_engine.py`
**Bug:** No verification that bracket set forms a complete partition. Missing brackets create unhedged directional exposure.

**Fix:** After grouping brackets by event_key, parse bracket ranges from tickers and verify contiguity (each bracket's upper bound = next bracket's lower bound). Skip the set if incomplete.

**Verification:** Add a unit test with a deliberately incomplete bracket set — should be rejected.

---

## Phase 2: Calibration & Edge Estimation Fixes

### 2.1 Cap maximum trusted edge at 0.30
**Evidence:** Edge 0.5-1.0 bucket has 0% win rate (-$5.55). Edge 0.3-0.5 has only 40% WR. The sweet spot is 0.20-0.30 (70% WR, +$61.85).

**Research basis:** Meister (arXiv 2412.14144) proves that overestimating edge is asymmetrically worse than underestimating. Our data confirms: claimed edges >0.30 are unreliable.

**Implementation:** In `src/signals/ensemble_signal.py` calibration, add an edge cap:
```python
# After computing net_edge
net_edge = min(net_edge, 0.30)  # Cap: edges >0.30 are unreliable (0% WR in data)
```

Also in `src/structured_data.py` for NOAA/Yahoo fast-paths:
```python
net_edge = min(abs(p_model - p_market), 0.30)
```

**Verification:** `grep -rn "net_edge" src/` — ensure all edge pathways are capped.

### 2.2 Increase YES dampening to 0.35
**Evidence:** buy_yes predicted avg 0.50, actual avg 0.25. Even with 0.25 dampening, buy_yes still has 21% WR, -$17.97.

**Research basis:** Becker (72.1M trades): YES buyers lose -1.02%, NO buyers earn +0.83%. LLMs exhibit acquiescence bias (Science Advances). Our data confirms massive YES overestimation.

**Implementation:** In `config.yaml`:
```yaml
llm:
  yes_dampen: 0.35  # was 0.25
```

And add a minimum edge gate for buy_yes signals:
```python
# In ensemble_signal.py calibration
if recommended_side == TradingSide.BUY_YES and net_edge < 0.15:
    return None  # 15% min edge for YES (was 10%)
```

**Verification:** Check that buy_yes trade count drops significantly. Monitor for 50+ resolved trades before evaluating.

### 2.3 Fix NOAA weather confidence (lower sigma for bracket markets)
**Evidence:** NOAA direct on weather: 16.7% WR, -$3.82. Specific failures:
- KXHIGHCHI-26FEB12-T38: predicted p_yes=0.977, actual=0 (edge 0.84 → total miss)
- KXLOWTMIA-26FEB12-B58.5: predicted p_yes=0.001, actual=1 (edge 0.94 → total miss)
- KXLOWTNYC-26FEB12-B27.5: predicted p_yes=0.009, actual=1 (edge 0.47 → total miss)

**Root cause:** NOAA sigma estimates are too tight (overconfident). The normal CDF returns extreme probabilities (>0.95 or <0.05) when sigma is underestimated. These then pass the z-gate as "high confidence" but are actually miscalibrated.

**Research basis:** ForecastWatch data shows Denver 2.2°F sigma hourly is correct, but other cities may need upward adjustment. The suislanchez bot uses 8% edge threshold (vs our 3% for thresholds).

**Implementation:**
- Increase all `_CITY_SIGMA_HOURLY` values by 30% (e.g., default 1.5→2.0, Denver 2.2→2.9, SF 1.0→1.3)
- Raise NOAA z-gate from 0.7 to 1.0 (more conservative)
- Raise weather threshold min_edge from 3% to 8% (matching suislanchez bot)
- Raise weather bracket min_edge from 20% to 25%

**Verification:** Run the audit script on next day's weather trades. Target: <5 NOAA direct trades per day (was ~12).

---

## Phase 3: Strategy Improvements (Research-Backed)

### 3.1 Add VIX1D-based dynamic vol for index markets
**Research:** Bloomberg intraday vol model shows first-hour vol explains 68% of daily vol. VIX1D (CBOE 1-Day Volatility Index) provides market-implied same-day vol.

**Current state:** Index uses static 1.0% daily vol for S&P, 1.3% for NASDAQ, scaled by sqrt(hours_remaining/6.5).

**Implementation:** In `src/structured_data.py`:
```python
async def _get_vix1d() -> Optional[float]:
    """Fetch VIX1D from Yahoo Finance (^VIX1D)."""
    # 60s cache TTL
    # Convert from annualized: daily_move = VIX1D / sqrt(252)

# Use VIX1D when available, fallback to static vol
daily_vol = vix1d_daily if vix1d_daily else default_vol
```

**Verification:** Compare VIX1D-derived vol vs static vol on historical data. Track fill rate and P&L separately for VIX1D-informed trades.

### 3.2 Add NBM (National Blend of Models) as weather calibration source
**Research:** NOAA's NBM is the closest to "ground truth" for temperature thresholds. It produces exceedance probabilities directly (P(temp > X)) which can be compared against our model's output.

**Current state:** We use NWS + Open-Meteo ensemble + HRRR. NBM is a free, calibrated product that could serve as an additional validation layer.

**Implementation:** In `src/structured_data.py`:
```python
async def _get_nbm_exceedance(city: str, threshold: float, variable: str) -> Optional[float]:
    """Fetch NBM exceedance probability from gribstream.com or NOAA API."""
    # Use as validation: if our model disagrees with NBM by >15%, defer to NBM
```

**Verification:** Compare NBM exceedance probabilities against our computed probabilities for 50+ weather markets.

### 3.3 Improve HRRR weighting for day-of forecasts
**Research:** HRRR (3km) outperforms all global models for 0-18h forecasts. Our current blend is 40/35/25 NWS/ensemble/HRRR in agreement.

**Implementation:** For day-of markets with <6h to close, increase HRRR weight:
```python
# In structured_data.py weather blend
if hours_to_close < 6:
    weights = {"nws": 0.25, "ensemble": 0.25, "hrrr": 0.50}  # HRRR dominant
elif hours_to_close < 12:
    weights = {"nws": 0.35, "ensemble": 0.30, "hrrr": 0.35}  # HRRR elevated
else:
    weights = {"nws": 0.40, "ensemble": 0.35, "hrrr": 0.25}  # Current default
```

**Verification:** Track weather WR by time-to-close bucket.

### 3.4 Block economics markets until FRED sniping works
**Evidence:** Economics 0W/4L, -$3.08. CPI predictions were all wrong.

**Research:** Becker data shows Finance category approaches perfect efficiency (0.17pp maker-taker gap). Whelan confirms these are the hardest markets to exploit.

**Implementation:** Add to blocklist in `config.yaml`:
```yaml
blocklist:
  - KXCPI
  - KXCPICORE
  - KXCPICOREYOY
  - KXCPIYOY
  - KXEGGS
```

Only re-enable once the FRED economic release sniping engine (Wave 18 Phase 5) is tested and validated.

**Verification:** Confirm no economics trades placed after deployment.

### 3.5 Implement simultaneous-bet Kelly adjustment
**Research:** Meister (arXiv 2412.14144): For N simultaneous bets, individual Kelly fractions should be reduced. Current bot uses independent Kelly sizing for each position.

**Implementation:** In `src/risk.py`:
```python
# Count current open positions
n_open = len(capital_manager.get_open_positions())
# Apply simultaneous bet correction
kelly_adj = kelly_raw / (1 + 0.1 * n_open)  # ~10% reduction per open position
```

**Verification:** Track average position size before/after. Should decrease by ~20-40% with typical 3-5 open positions.

### 3.6 Lower max total exposure from 80% to 65%
**Research:** 80% is at the aggressive end (institutional standard is 50-65%). With our small bankroll (~$128), a correlation event could wipe out gains.

**Implementation:** In `config.yaml`:
```yaml
strategy:
  max_total_exposure: 0.65  # was 0.80
```

**Verification:** Monitor capital utilization and opportunity cost (missed trades due to exposure cap).

---

## Phase 4: Operational Improvements

### 4.1 Fix CLV history dedup issue
**Evidence:** CLV history shows duplicate entries for same markets (KXSTARMERMENTIONB-26FEB19-NHS appearing 8+ times with identical entry_price/market_price). Each scan cycle is re-logging CLV entries for open positions.

**Fix:** In `src/capital_management.py`, add dedup check:
```python
# Before writing CLV entry
if market_id in self._logged_clv_entries:
    continue
```

**Verification:** `wc -l state/clv_history.jsonl` should grow linearly (1 per trade), not exponentially.

### 4.2 Reduce LLM cost waste
**Evidence:** $5.44 spent this month ($3.02 today alone), 2240 calls. Daily budget of $3 is being hit. GPT-4o accounts for $5.35 (98.3%).

**Improvement options:**
- Switch more evaluations to GPT-4o-mini (1/10 cost) for initial screening
- Increase eval cooldown from 60min to 90min for non-weather markets
- Skip LLM evaluation entirely when NOAA/Yahoo fast-paths return strong signals

### 4.3 Fix MM engine orderbook price propagation
**Bug:** MM signals pass `None` for one side's ask price in metadata, breaking spread computation in executor.

**Fix:** In `src/engines/kalshi_mm_engine.py:474-475`, pass both sides from the orderbook state:
```python
"kalshi_yes_ask": state.best_yes_ask or price_frac,
"kalshi_no_ask": state.best_no_ask or (1.0 - price_frac),
```

---

## Phase 5: Monitoring & Verification

### 5.1 Deploy and run for 24h
```bash
ssh morpheus "cd /opt/morpheus && git pull && systemctl restart morpheus"
ssh morpheus "journalctl -u morpheus -n 50 --no-pager"
```

### 5.2 Verify bug fixes
- Check logs for `weather_repricing_check` events (should appear without errors)
- Verify no economics tickers in trade_history.jsonl
- Confirm CLV history isn't duplicating entries
- Check that edge values in predictions.jsonl are capped at 0.30

### 5.3 Run audit after 24h
```bash
ssh morpheus "cd /opt/morpheus && .venv/bin/python3 scripts/audit_resolutions.py --state-dir state --since 2026-02-14"
```

### 5.4 Compare metrics
| Metric | Current (Feb 10-13) | Target (Post-Wave 21) |
|--------|--------------------|-----------------------|
| Overall WR | 43.2% (35W/46L) | >50% |
| buy_no WR | 58% (+$93.15) | >55% |
| buy_yes WR | 21% (-$17.97) | >35% (or <5 trades) |
| Weather WR | 22% (-$6.17) | >40% |
| Economics WR | 0% (-$3.08) | N/A (blocked) |
| Edge 0.5+ WR | 0% (-$5.55) | N/A (capped at 0.30) |
| NOAA direct WR | 16.7% (-$3.82) | >35% |
| Daily P&L | +$18.80/day | +$20/day |

---

## Anti-Pattern Guards
- DO NOT invent APIs or parameters not in the codebase
- DO NOT change Kelly formula — it was already fixed in Wave 14
- DO NOT re-enable crypto engine — blocked for good reason (-$28/day losses)
- DO NOT reduce YES dampening below 0.25 — research uniformly supports higher dampening
- DO NOT use `event=` as a structlog kwarg — conflicts with positional arg
- Verify all changes against the actual code before deploying
