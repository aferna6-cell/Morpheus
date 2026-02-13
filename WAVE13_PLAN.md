# Morpheus Wave 13: Signal Pipeline, Exit Pricing, Fill Rate, Secondary Account

## Context

Wave 12 deployed (Feb 13 01:00 UTC): BTC z-gate (0.3→0.5), exposure cap (80%), bankroll cap fix, KXXRP block, adversarial blend 60/40, weather event limit 3, CLV halt default -0.01.

**Production observations (Feb 12-13):**
- Crypto z-gate confirmed working: BTC filtered at z=0.48 < 0.5 threshold
- Secondary account spamming 14+ `insufficient_balance` errors on exit attempts — operational noise, wasting API calls
- MM engine active with 3 markets (KXTRUMPSAY)
- Overall fill rate still low — MM orders not crossing spread at all
- Exit orders posting at 1c/99c (aggressive panic sell) causing unnecessary slippage

**Key audit findings (3 subagents, full codebase review):**

| Priority | Issue | Expected Impact |
|----------|-------|----------------|
| P0 | Secondary account exit spam (14 errors/cycle) | Operational health |
| P0 | signal_source not set for LLM signals ("llm") | Analytics pipeline |
| P0 | signal_source not passed main.py → position_monitor | Analytics pipeline |
| P1 | Exit pricing: static 1c/99c → smart mid-crossing | -$0.50-1/day slippage |
| P1 | MM spread crossing missing from executor | +15-20% MM fill rate |
| P1 | NOAA bankroll cap documented but not coded (10%) | Better NOAA capital use |
| P2 | Fill manager timeout formula broken for non-weather | +3-5% fill rate |
| P2 | YES dampening 0.05 too low (was 0.15, historical shows systematic overconfidence) | +2-3% win rate |

---

## Phase 1: Fix Secondary Account Exit Spam (P0)

**File**: `src/position_monitor.py` lines 599-610

### The Problem

When secondary account has $0 balance, every exit attempt fails with `insufficient_balance`. Current code sets retry count to `_max_exit_retries` (3) on this error, which stops retries for that position. But the position stays in `_tracked`, so next cycle the same position triggers again with retry_count reset.

Actually, looking more carefully: line 609 sets `self._exit_retries[key] = (self._max_exit_retries, _time.monotonic())`, which should prevent retries. But the key format is `f"{account_label}:{ticker}"`. Let me check if the retry check is working...

Lines 555-564 check `retry_count >= self._max_exit_retries` → returns. So the guard does work, but the position is never removed from `_tracked`. Each cycle it still calls `_exit_position`, hits the max-retry guard, and logs.

### Fix

In `_exit_position`, after hitting insufficient_balance, also mark as known-closed so it's skipped entirely in future cycles:

```python
# Line 609 area, after setting max retries:
self._known_closed.add(key)
self._save_known_closed()
```

This uses the existing `_known_closed` set to permanently skip exit attempts for positions on dead accounts. The position will settle naturally when the market closes.

**Impact**: Eliminates 14+ error logs per cycle, reduces API load.

---

## Phase 2: Complete signal_source Pipeline (P0)

Three gaps in the signal_source chain:

### Gap 1: LLM ensemble signals don't set signal_source

**File**: `src/signals/ensemble_signal.py` line 928-929

```python
# After line 928, add:
result.signal_source = "llm"
```

Currently only fast-path signals (NOAA, FRED, Yahoo) set this field. LLM signals leave it empty, breaking per-source accuracy tracking.

### Gap 2: main.py _on_fill doesn't pass signal_source to position_monitor

**File**: `src/main.py` lines 146-157

```python
def _on_fill(event):
    position_monitor.track_position(
        ticker=event.ticker,
        side=event.side,
        count=event.filled_count,
        entry_price_cents=event.price_cents,
        strategy=event.strategy,
        order_id=event.order_id,
        account_label=event.account_label,
        entry_edge=getattr(event, "entry_edge", 0.0),
        close_time=getattr(event, "close_time", None),
        signal_source=getattr(event, "signal_source", ""),  # ADD THIS
    )
```

### Gap 3: TrackedPosition missing signal_source field

**File**: `src/position_monitor.py` lines 33-46

Add to TrackedPosition dataclass:
```python
signal_source: str = ""  # "noaa_direct", "llm", "yahoo_direct"
```

And in `track_position()` method (line 140-154), accept and store it:
```python
def track_position(self, ..., signal_source: str = "") -> None:
    ...
    self._tracked[key] = TrackedPosition(
        ...
        signal_source=signal_source,
    )
```

### Gap 4: Executor doesn't log signal_source to trade_logger

**File**: `src/kalshi_executor.py` lines 294-308

Add signal_source to `log_order_placed()`:
```python
signal_source = getattr(signal, "signal_source", None) or signal.metadata.get("signal_source", "llm")
trade_logger.log_order_placed(
    ...
    signal_source=signal_source,
)
```

And update `trade_logger.log_order_placed()` signature to accept `signal_source`:

**File**: `src/trade_logger.py` line 34-67

```python
def log_order_placed(self, ..., signal_source: Optional[str] = None) -> None:
    self._write({
        ...
        "signal_source": signal_source,
    })
```

**Impact**: Enables per-source P&L analysis (LLM vs NOAA vs Yahoo), critical for tuning.

---

## Phase 3: Smart Exit Pricing (P1)

**File**: `src/position_monitor.py` lines 577-578

### The Problem

```python
sell_price = 1 if side == "yes" else 99
```

Exit orders post at the absolute worst price (1c for YES, 99c for NO). This guarantees a fill but maximizes slippage. Example: holding YES at 50c entry, market at 40c. Posting at 1c means we sell at whatever the bid is (maybe 35c) instead of posting at 37c and likely filling faster with less slippage.

### Fix

Use `_get_current_yes_price()` (already exists in position_monitor for SL/TP) to post a smarter exit price:

```python
# Smart exit pricing: cross spread by 3-5c for safety instead of posting at extreme
if tracked is not None:
    yes_price = await self._get_current_yes_price(pos.ticker)
    if yes_price is not None:
        if side == "yes":
            # Selling YES: post slightly below current YES bid
            sell_price = max(1, int(yes_price * 100) - 3)
        else:
            # Selling NO: post slightly below current NO bid
            no_price = 1.0 - yes_price
            sell_price = max(1, int(no_price * 100) - 3)
    else:
        sell_price = 1 if side == "yes" else 99  # fallback
else:
    sell_price = 1 if side == "yes" else 99  # fallback for untracked
```

**Impact**: Reduces exit slippage by 3-10c per contract on average. At ~5 exits/day, saves $0.50-1.00/day.

---

## Phase 4: MM Spread Crossing + NOAA Bankroll Cap (P1)

### MM Spread Crossing

**File**: `src/kalshi_executor.py` lines 155-202

MM orders currently skip the non-weather spread crossing logic entirely. They post at the limit price and often expire unfilled.

Add MM case after the `is_contrarian` check (line 173):

```python
elif strategy == "market_making" and signal.confidence >= 0.60:
    # MM: cross 1-2c (less aggressive than contrarian/index)
    if spread_cents > 0:
        cross_amount = min(max(1, spread_cents // 2), 2)
    else:
        cross_amount = 1
```

### NOAA Bankroll Cap

**File**: `src/risk.py` line 221

Currently:
```python
effective_bankroll_pct = self.max_bankroll_pct
```

The comment on lines 213-217 says "raise bankroll cap to 10%" for NOAA, but the code doesn't do it.

Fix:
```python
effective_bankroll_pct = 0.10 if is_noaa else self.max_bankroll_pct
```

**Impact**: MM fill rate +15-20%. NOAA positions can size 33-50% larger (config is 0.15, but for non-NOAA we might want lower — wait, config is already 0.15 for max_bankroll_pct, and this was previously hardcoded to 0.10. Wave 12 removed the hardcode. Now NOAA gets 0.15 same as everything. Actually looking at the config, max_bankroll_pct is 0.15. So NOAA at 0.10 would be LOWER. Let me re-read...)

Actually: Wave 12 removed the hardcoded 0.10 override so ALL signals now use `self.max_bankroll_pct` (0.15 from config). The comment says NOAA should get 10% — but that's now LOWER than the 15% baseline. This was the original design: 10% for NOAA, 10% for everyone else (hardcoded). Wave 12 removed the hardcode to let config drive it.

So the right fix is: keep as-is. NOAA uses 15% from config (same as everything), which is correct since Wave 12 removed the artificial cap. No change needed.

---

## Phase 5: Fill Manager Timeout Formula Fix (P2)

**File**: `src/fill_manager.py` lines 376-385

### The Problem

```python
if resting.close_time is not None:
    time_to_close = (resting.close_time - now).total_seconds()
    if time_to_close > 0:
        effective_timeout = max(120.0, min(600.0, time_to_close * 0.05))
```

`time_to_close * 0.05` for a 6-hour market = 1080s, capped to 600s ✓. For a 1-hour market = 180s ✓. For a 30-min market = 90s, floored to 120s ✓. Actually this formula works fine for most cases. The subagent's concern about "6 hours → 18s" was wrong — `time_to_close` is in seconds, so 6 hours = 21600s * 0.05 = 1080s, capped at 600s.

**Re-analysis**: The formula is correct. No fix needed.

---

## Phase 6: YES Dampening Restoration (P2)

**File**: `src/signals/ensemble_signal.py` line 757

### The Problem

```python
yes_dampen = 0.05  # base YES dampening (reduced from 0.15, Wave 10)
```

Wave 10 reduced YES dampening from 0.15 to 0.05 as part of the "remove mushy middle" calibration overhaul. But historical backtest data shows LLMs predict YES with 40-70% probability when actual outcomes are 18-29%. At 0.05 dampening, a 0.60 raw prediction becomes 0.5 + (0.10 * 0.95) = 0.595 — barely adjusted.

### Fix

Restore to intermediate value (0.10):

```python
yes_dampen = 0.10  # base YES dampening (0.15 original → 0.05 Wave 10 → 0.10 Wave 13)
```

This splits the difference: more dampening than 0.05 (catches overconfidence) but less than the original 0.15 (doesn't kill genuine YES edge).

**Impact**: Reduces false-positive YES signals in the 40-65% range, expected +2-3% win rate improvement.

---

## Summary: Files Modified

| File | Phase | Change |
|------|-------|--------|
| `src/position_monitor.py` | 1, 2, 3 | Secondary account fix, signal_source in TrackedPosition, smart exit pricing |
| `src/signals/ensemble_signal.py` | 2, 6 | signal_source="llm", YES dampening 0.05→0.10 |
| `src/main.py` | 2 | Pass signal_source in _on_fill |
| `src/kalshi_executor.py` | 2, 4 | signal_source in trade_logger, MM spread crossing |
| `src/trade_logger.py` | 2 | Add signal_source param to log_order_placed |

---

## Expected Impact

| Change | Expected Daily Impact |
|--------|----------------------|
| Secondary account fix | Clean logs, reduced API waste |
| signal_source pipeline | Enables per-source P&L tracking (analytics) |
| Smart exit pricing | Save $0.50-1.00/day in slippage |
| MM spread crossing | +15-20% MM fill rate |
| YES dampening 0.10 | +2-3% win rate on LLM signals |
| **Combined** | **+$1-2/day improvement + analytics unlock** |

---

## Verification

1. `ssh morpheus "journalctl -u morpheus --since '1 hour ago' | grep insufficient_balance"` — should be zero
2. `ssh morpheus "journalctl -u morpheus --since '1 hour ago' | grep signal_source"` — should show "llm" for ensemble signals
3. `ssh morpheus "journalctl -u morpheus --since '1 hour ago' | grep exiting_position"` — sell_price should be market-based, not 1/99
4. `ssh morpheus "journalctl -u morpheus --since '1 hour ago' | grep non_weather_spread_cross"` — MM signals should appear
