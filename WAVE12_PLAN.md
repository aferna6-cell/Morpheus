# Morpheus Wave 12: Fix BTC Losses, Signal Pipeline, Risk Caps, Calibration

## Context

Wave 11 deployed (Feb 13 00:40 UTC): NO-side repricing fix, CLV halt threshold, confidence-weighted scoring, MM decay.
Balance ~$67 across 2 accounts ($10.34 primary + positions).

**Performance analysis (Feb 11-12, 26 filled resolutions):**
- Total P&L: +$6.13 (but $14.02 came from ONE NASDAQ trade)
- Without NASDAQ: **-$7.89** — the bot is bleeding without lucky outliers
- Weather: -$3.57 (2W/6L, 25% win rate)
- Crypto (BTC+XRP): -$2.82 (6W/8L) — #1 loss category by trade count
- Politics: -$1.50 (0W/3L)
- yahoo_direct: -$3.18 (1W/2L) — worst signal source
- LLM: +$9.63 (7W/11L, 39% win rate) — carried by $14.02 NASDAQ
- Biggest single loss: KXBTCD-26FEB1218 -$4.80 (BTC, LLM, 10 contracts)

**Root causes identified:** BTC volatility param too low (2.5% vs real 4%+), z-gate too weak for crypto (0.3), signal_source pipeline broken (can't track accuracy by source), CLV code default wrong (+0.01), exposure uncapped, adversarial challenge too weak (75/25), weather event limit mismatch (2 vs 3).

---

## Phase 1: Fix BTC/Crypto Losses (HIGHEST IMPACT — stops $3-7/day bleed)

### 1A. Increase BTC daily volatility: 2.5% → 4%

**File**: `src/structured_data.py` lines 1919-1920

**Current code:**
```python
"KXBTCD": {"yahoo": "BTC-USD", "name": "Bitcoin", "daily_vol": 0.025, "trading_hours": 24.0},
"KXBTC": {"yahoo": "BTC-USD", "name": "Bitcoin", "daily_vol": 0.025, "trading_hours": 24.0},
```

BTC regularly moves 3-5% daily. With 2.5% vol, the model thinks a 2% move is 0.8 sigma (very confident), when it should be 0.5 sigma (uncertain). This creates over-confident signals on marginal BTC markets.

**Fix:** Change `daily_vol` from `0.025` to `0.04` for both KXBTCD and KXBTC.

### 1B. Raise z-score gate for crypto markets: 0.3 → 0.5

**File**: `src/structured_data.py` line 2252

**Current code:**
```python
if z_score < 0.3:
    # Very close to threshold — too uncertain, let LLM handle
    ...
    return None
```

z=0.3 means P=62% — nearly a coin flip. For crypto's high volatility, this lets through garbage signals.

**Fix:** Replace the z-gate check with a crypto-aware version:
```python
# Crypto needs higher z-gate due to higher volatility and model uncertainty
is_crypto = config.get("trading_hours", 6.5) >= 24.0
min_z = 0.5 if is_crypto else 0.3
if z_score < min_z:
    logger.info(
        "stock_index_ambiguous",
        market_id=market_id,
        current=current_price,
        threshold=threshold,
        z_score=round(z_score, 2),
        hours_left=round(hours_left, 2),
    )
    return None
```

### 1C. Block XRP 15-minute markets

**File**: `src/market_filters.py` — add `"KXXRP"` to `_JUNK_TICKER_PREFIXES`

XRP 15-minute markets (-$1.20 loss) are pure noise. No structured data fast-path exists for XRP.

**Verification:**
- `grep "daily_vol.*0.04" src/structured_data.py` — confirm BTC updated
- `grep "min_z\|is_crypto" src/structured_data.py` — confirm crypto z-gate
- After deploy: check logs for `stock_index_ambiguous` on BTC — should see more filtered

---

## Phase 2: Fix signal_source Pipeline (Enables All Future Analytics)

### 2A. Add signal_source to FillEvent

**File**: `src/fill_manager.py` line 99

Add after `close_time: Optional[datetime] = None`:
```python
    signal_source: str = ""  # "noaa_direct", "llm", "yahoo_direct"
```

### 2B. Propagate signal_source when creating FillEvent

**File**: `src/fill_manager.py` line 307-317

Add `signal_source=resting.signal_source,` after `close_time=resting.close_time,` in the FillEvent constructor.

### 2C. Wire signal_source through main.py callback

**File**: `src/main.py` line 146-157

The `_on_fill` callback calls `position_monitor.track_position()`. The position monitor already takes `close_time` — but not `signal_source`. No change needed in position_monitor for now; the key fix is FillEvent → resolutions.jsonl.

Check `src/trade_logger.py` and `src/resolution_tracker.py` to confirm signal_source gets logged.

**Verification:**
- `grep "signal_source" src/fill_manager.py` — appears in FillEvent AND FillEvent creation
- After deploy: `tail -5 state/resolutions.jsonl | python -m json.tool | grep signal_source`

---

## Phase 3: Fix Risk Management Gaps

### 3A. CLV halt code default: +0.01 → -0.01

**File**: `src/capital_management.py` line 113

**Current:**
```python
self.clv_halt_threshold = float(clv_cfg.get("halt_threshold", 0.01))
```

Config has `-0.01` (fixed in Wave 11), but code DEFAULT is `+0.01`. If config fails to parse, bot halts on positive CLV.

**Fix:** Change `0.01` to `-0.01`.

### 3B. Cap max_total_exposure at 80% of bankroll

**File**: `src/risk.py` line 130

**Current:**
```python
self.max_total_exposure = base_max_exp * scale_factor
```

No upper bound. At $67 bankroll, scale_factor=10.2, so max_exposure=$102 (152%).

**Fix:**
```python
self.max_total_exposure = min(base_max_exp * scale_factor, total_balance * 0.80)
```

### 3C. Fix hardcoded 10% bankroll cap for non-NOAA

**File**: `src/risk.py` line 221

**Current:**
```python
effective_bankroll_pct = self.max_bankroll_pct if is_noaa else 0.10
```

Config says `max_position_pct: 0.05` (5%) but non-NOAA gets hardcoded 10%. This DOUBLES position sizes vs intent.

**Fix:**
```python
effective_bankroll_pct = self.max_bankroll_pct  # respect config for all signals
```

NOAA already gets more aggressive Kelly (1/2 vs 1/3) — that's the differentiation, not bankroll cap.

**Verification:**
- `grep "halt_threshold.*-0.01" src/capital_management.py` — confirm default
- `grep "max_total_exposure.*min" src/risk.py` — confirm cap
- `grep "effective_bankroll_pct" src/risk.py` — no 0.10 hardcode

---

## Phase 4: Weather Event Limit + Adversarial Blend + Defaults

### 4A. Weather event limit: 2 → 3

**File**: `src/orchestrator.py` line 326

**Current:**
```python
max_event = 2 if is_weather_event else self._max_per_event
```

Comment on line 318 says "Weather markets get a higher limit (3 vs 2)" — code contradicts comment.

**Fix:** Change `2` to `3`.

### 4B. Strengthen adversarial challenge blend: 75/25 → 60/40

**File**: `src/signals/ensemble_signal.py` line 1154

**Current:**
```python
blended = 0.75 * p_yes + 0.25 * challenge_p
```

Data shows LLMs predict 40-70% YES when actual is 18-29%. Challenger barely moves the estimate at 75/25.

**Fix:**
```python
blended = 0.60 * p_yes + 0.40 * challenge_p
```

### 4C. Fix market filter code defaults

**File**: `src/market_filters.py` — check code defaults vs config values

If any defaults are wildly different from config (50000 vs 500 for volume, etc.), align them.

**Verification:**
- `grep "max_event = " src/orchestrator.py` — confirm 3
- `grep "blended = " src/signals/ensemble_signal.py` — confirm 0.60/0.40

---

## Phase 5: Guard _norm_cdf Against Zero Sigma

**File**: `src/structured_data.py` — the `_norm_cdf` function

Check if sigma=0 guard exists. If not, add:
```python
def _norm_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / sigma
    return 0.5 * math.erfc(-z / math.sqrt(2))
```

---

## Files Modified Summary

| File | Phase | Change |
|------|-------|--------|
| `src/structured_data.py` | 1A, 1B, 5 | BTC vol 0.025→0.04, crypto z-gate 0.5, _norm_cdf guard |
| `src/market_filters.py` | 1C, 4C | Block KXXRP + fix code defaults |
| `src/fill_manager.py` | 2A, 2B | signal_source in FillEvent + propagation |
| `src/capital_management.py` | 3A | CLV halt default -0.01 |
| `src/risk.py` | 3B, 3C | Exposure cap 80% + remove hardcoded 10% bankroll |
| `src/orchestrator.py` | 4A | Weather event limit 2→3 |
| `src/signals/ensemble_signal.py` | 4B | Adversarial blend 60/40 |

---

## Expected Impact

| Change | Expected Daily Impact |
|--------|----------------------|
| BTC vol + z-gate | Stop $3-5/day crypto bleed |
| Block KXXRP | Save $1-2/day junk trades |
| signal_source pipeline | Enable per-source accuracy tracking |
| CLV halt default | Prevent future false halts |
| Exposure cap + bankroll fix | Prevent oversized positions |
| Weather event limit | +$0.50-1/day more weather trades |
| Adversarial blend | +$0.50-1/day better calibration |
| **Combined** | **+$5-8/day improvement** |

---

## Emergency Rollback

```bash
touch state/STOP_TRADING       # immediate halt
git revert HEAD                # undo changes
ssh morpheus "cd /opt/morpheus && git pull && systemctl restart morpheus"
```
