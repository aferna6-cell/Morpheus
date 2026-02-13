# Wave 17: 15-Minute Crypto Engine (KXBTC15M)

## Context

Kalshi runs 15-minute binary crypto markets (KXBTC15M): "Will BTC be up or down vs. the start of this 15-min window?" Resolved automatically via CF Benchmarks BRTI every 15 minutes, 24/7. Volume: 65-89k contracts per window, ~$185k liquidity, 2-3c spreads.

Current state: KXBTC15M is blocked in market_filters.py (along with all KXBTC*). The existing Yahoo Finance CDF approach loses money on hourly crypto (Feb 13 audit: BTC 20W/20L, -$25.40) because hourly windows have too much noise. But 15-minute windows are fundamentally different — shorter time horizon + momentum signals = exploitable edge.

### Strategy: Directional Momentum + Mean Reversion Hybrid

Core insight from successful Polymarket bots: in short windows (5-15 min), price tends to **continue** in the direction it's already moving (momentum). But near window boundaries, extreme moves tend to **revert**.

**Momentum signal**: If BTC moves >0.15% in the first 5 minutes of a 15-min window, bet on continuation.
**Mean reversion signal**: If BTC moves >0.5% in the first 10 minutes, bet on reversion (too far too fast).
**Confidence scaling**: Larger moves = higher confidence = bigger bets.

### Why this is different from the failed hourly approach:
1. **Real-time price feed** (Binance WebSocket, <100ms) vs Yahoo Finance (60s cache)
2. **Momentum-based** (directional signal from recent price action) vs CDF-based (static normal distribution)
3. **15-min windows** (less noise, momentum persists) vs hourly (too much noise for 4% daily vol asset)
4. **High frequency** (96 windows/day) vs few hourly markets — more samples, faster convergence

---

## Phase 1: Binance WebSocket Price Feed

**New file**: `src/feeds/binance_ws.py`

A lightweight WebSocket client that maintains a real-time BTC price stream. The engine and any other module can query it for:
- Current BTC price (sub-100ms latency)
- Price at any recent timestamp (rolling 20-min buffer)
- Price change over any interval (e.g., "change in last 5 minutes")

```python
class BinancePriceFeed:
    """Real-time BTC price via Binance WebSocket."""

    def __init__(self):
        self._prices: deque  # (timestamp, price) — rolling 20 min
        self._current_price: float = 0.0
        self._ws_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Connect to wss://stream.binance.com:9443/ws/btcusdt@trade"""

    async def stop(self) -> None:
        """Disconnect cleanly."""

    @property
    def price(self) -> float:
        """Current BTC/USDT price."""

    def price_at(self, ts: float) -> Optional[float]:
        """Interpolated price at a given Unix timestamp."""

    def price_change_pct(self, seconds_ago: float) -> Optional[float]:
        """Percentage change from N seconds ago to now."""

    def is_connected(self) -> bool:
        """Whether the WebSocket is alive."""
```

Key details:
- Uses `websockets` library (already in requirements or add it)
- Auto-reconnect on disconnect (5s backoff)
- Rolling deque of (timestamp, price) tuples — keep 20 minutes of ticks
- Thin wrapper, no business logic — just price data

---

## Phase 2: Crypto Engine Core

**New file**: `src/engines/kalshi_crypto_engine.py`

Modeled on MM engine dual-loop pattern:

```python
class KalshiCryptoEngine(BaseEngine):
    name = "kalshi_crypto"

    def __init__(self, config, kalshi_client, price_feed):
        # Config from config.yaml crypto_engine section
        # Store price_feed (BinancePriceFeed)

    async def start(self):
        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())    # Find active 15M markets
        self._signal_task = asyncio.create_task(self._signal_loop()) # Generate signals every 15-30s

    async def stop(self):
        self._running = False
        # Cancel tasks

    async def get_signals(self) -> List[TradeSignal]:
        # Drain and return pending signals
```

### Signal Loop (every 15-30 seconds):

```
For each active KXBTC15M market:
  1. Determine window start time from ticker (e.g., KXBTC15M-26FEB13H1430 = 14:30 UTC)
  2. Compute elapsed minutes in window
  3. Get BTC price change since window start from Binance feed

  IF elapsed 3-7 minutes AND |change| > 0.15%:
      → MOMENTUM signal: bet that direction continues
      → confidence = min(0.75, 0.50 + |change| * 50)  # 0.30% move = 0.65 conf
      → side = buy_yes if change > 0 else buy_no

  IF elapsed 8-13 minutes AND |change| > 0.50%:
      → MEAN REVERSION signal: bet against the move
      → confidence = min(0.70, 0.45 + (|change| - 0.50) * 30)
      → side = buy_no if change > 0 else buy_yes

  IF elapsed < 3 minutes OR |change| < 0.10%:
      → No signal (too early or no momentum)

  Emit TradeSignal with:
      engine="kalshi_crypto"
      strategy metadata: "momentum" or "mean_reversion"
      signal_source="binance_direct"
      _force_size_usd computed from confidence + Kelly
```

### Scan Loop (every 60 seconds):

```
Fetch markets with prefix KXBTC15M
Filter: close_time within next 15 minutes, status=open
Track in self._active_markets dict
Parse window_start_time from ticker
```

### Ticker Parsing:

KXBTC15M tickers follow the pattern: `KXBTC15M-26FEB13H1430-TUP` or similar.
Need to parse the window start time. Investigate actual ticker format from Kalshi API.

### Position Sizing:

```python
# Crypto 15M: aggressive Kelly (quick resolution, many samples)
# Base: 2/3 Kelly for momentum, 1/2 Kelly for mean reversion
# Cap: $5 per trade (start conservative), scale up after 100+ trades
kelly = (edge * confidence) / (1.0 - edge)
fraction = 0.67 if strategy == "momentum" else 0.50
size = min(kelly * fraction * bankroll, max_position_usd)
```

### Fee Awareness:

Kalshi maker fee = `0.0175 * contracts * price * (1 - price)`.
At 50c (maximum fee point): 0.0175 * 1 * 0.50 * 0.50 = $0.004375/contract.
At 70c: 0.0175 * 1 * 0.70 * 0.30 = $0.003675/contract.

Fees are small (<0.5c/contract) but nonzero. Factor into edge calculation:
```python
fee_per_contract = 0.0175 * price * (1 - price)
net_edge = raw_edge - fee_per_contract
```

---

## Phase 3: Market Filters — Unblock KXBTC15M Only

**File**: `src/market_filters.py`

Currently all KXBTC* tickers are blocked. Need to:
1. Keep KXBTC and KXBTCD blocked (hourly/daily crypto — proven losers)
2. Unblock KXBTC15M specifically (new engine handles these)

Change the blocklist entry from:
```python
"KXBTC", "KXBTCD", "KXBTC15M",
```
to:
```python
"KXBTCD",  # Daily crypto — blocked (hourly noise)
# Note: KXBTC15M is handled by kalshi_crypto engine (not blocked)
# Note: KXBTC hourly is handled below with specific prefix matching
```

Actually, the cleanest approach: the crypto engine bypasses market_filters entirely (like MM engine selects its own markets). The blocklist only affects the LLM engine scans. So we need:
- Keep `"KXBTC"` in blocklist (blocks KXBTC hourly from LLM engine)
- `"KXBTC"` prefix match also blocks `"KXBTC15M"` — that's fine because the crypto engine finds its own markets independently
- No changes needed to market_filters.py! The crypto engine, like the MM engine, uses its own market selection logic.

Wait — we need to verify: does `"KXBTC"` prefix match block `"KXBTC15M"`? Yes, it does (`KXBTC15M`.startswith(`KXBTC`) = True). This is correct — we want the LLM engine to NOT evaluate 15M markets (waste of LLM budget). The crypto engine has its own scan loop.

**No changes needed to market_filters.py.**

---

## Phase 4: Config Additions

**File**: `config.yaml`

Add new section:
```yaml
# 15-minute crypto engine — directional momentum on KXBTC15M
crypto_engine:
  enabled: true
  scan_interval_seconds: 60         # rescan for new windows every 60s
  signal_interval_seconds: 15       # check momentum every 15s
  momentum_threshold_pct: 0.15      # 0.15% move triggers momentum signal
  reversion_threshold_pct: 0.50     # 0.50% move triggers mean reversion
  min_elapsed_minutes: 3            # don't signal in first 3 min
  max_elapsed_minutes: 13           # don't signal in last 2 min (too late)
  momentum_window_minutes: 7        # momentum signals valid minutes 3-7
  reversion_window_minutes: 13      # reversion signals valid minutes 8-13
  max_position_usd: 5.0             # $5 max per trade (start conservative)
  kelly_fraction_momentum: 0.67     # 2/3 Kelly for momentum
  kelly_fraction_reversion: 0.50    # 1/2 Kelly for reversion
  min_confidence: 0.45              # minimum confidence to trade
  cooldown_per_window_seconds: 120  # only 1 signal per window per 2 min
  maker_fee_coefficient: 0.0175     # Kalshi maker fee formula coefficient
  binance_ws_url: "wss://stream.binance.com:9443/ws/btcusdt@trade"
  ticker_prefix: "KXBTC15M"
```

Also fix the stale fee_pct:
```yaml
# In strategy section:
fee_pct: 0.004  # ~0.4% average maker fee (0.0175 * P * (1-P), max at 50c)
```

---

## Phase 5: Wire Into main.py

**File**: `src/main.py`

After MM engine registration (~line 225), add:

```python
# 15-minute crypto engine (Binance real-time + directional momentum)
crypto_cfg = getattr(config, "crypto_engine", None) or {}
if isinstance(crypto_cfg, dict) and crypto_cfg.get("enabled", False):
    from .feeds.binance_ws import BinancePriceFeed
    from .engines.kalshi_crypto_engine import KalshiCryptoEngine

    binance_feed = BinancePriceFeed(
        ws_url=crypto_cfg.get("binance_ws_url", "wss://stream.binance.com:9443/ws/btcusdt@trade"),
    )
    await binance_feed.start()

    crypto_engine = KalshiCryptoEngine(
        config=config,
        kalshi_client=kalshi_read,
        price_feed=binance_feed,
    )
    engines.append(crypto_engine)
    logger.info("kalshi_crypto_engine_initialized")
```

---

## Phase 6: Orchestrator Integration

The crypto engine emits `TradeSignal` objects just like LLM/MM engines. The orchestrator already handles multi-engine signals. But we need:

1. **Skip LLM evaluation for crypto signals**: Crypto signals from the engine should go straight to execution, not through ensemble_signal.py. Add `strategy="crypto"` metadata and check in orchestrator.

2. **Separate exposure tracking**: Add `"KXBTC15M"` as a correlation prefix so correlated exposure limits apply per-window, not globally.

3. **Signal source propagation**: Set `signal_source="binance_direct"` in metadata for resolution tracking.

---

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `src/feeds/__init__.py` | CREATE | Empty init |
| `src/feeds/binance_ws.py` | CREATE | Binance WebSocket price feed (~80 lines) |
| `src/engines/kalshi_crypto_engine.py` | CREATE | 15-min crypto engine (~250 lines) |
| `config.yaml` | EDIT | Add crypto_engine section + fix fee_pct |
| `src/main.py` | EDIT | Wire crypto engine + Binance feed |
| `src/orchestrator.py` | EDIT | Skip LLM for crypto signals, exposure tracking |

---

## Expected Performance

| Metric | Estimate | Basis |
|--------|----------|-------|
| Windows per day | 96 (24h × 4/hour) | 15-min resolution |
| Tradeable windows | 20-40 (need 0.15%+ move) | ~30-40% of windows have sufficient momentum |
| Expected win rate | 54-58% (momentum), 52-55% (reversion) | Crypto momentum persistence in short windows |
| Avg position size | $2-5 | Kelly sizing with $130 bankroll |
| Maker fee per trade | ~$0.01-0.02 | 0.0175 * P * (1-P) per contract |
| Expected daily P&L | $3-8 at 56% WR, 30 trades × $3 avg | Conservative estimate |

---

## Risk Controls

1. **Max $5 per trade** (configurable, start conservative)
2. **1 signal per window** (cooldown prevents doubling down)
3. **Correlated exposure cap**: Max 2 concurrent BTC15M positions
4. **Daily loss halt**: Shares global $6 daily loss limit
5. **Kill switch**: `state/STOP_TRADING` halts all engines including crypto
6. **Gradual scale-up**: Start at $2/trade, increase after 50+ profitable trades

---

## Verification

1. `ssh morpheus "journalctl -u morpheus --since '10 min ago' | grep crypto_engine"` — engine started, scanning
2. `ssh morpheus "journalctl -u morpheus --since '10 min ago' | grep binance"` — WebSocket connected
3. `ssh morpheus "journalctl -u morpheus --since '10 min ago' | grep KXBTC15M"` — signals being generated
4. After 1 hour: check `state/trade_history.jsonl` for crypto fills
5. After 24 hours: audit via `scripts/audit_resolutions.py --since $(date -d yesterday +%Y-%m-%d)` — verify 54%+ WR

**Emergency rollback**: Block `KXBTC15M` in config `crypto_engine.enabled: false` + restart
