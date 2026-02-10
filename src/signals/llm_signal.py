"""LLM-powered probability estimation signal — aggressive accuracy mode.

Outputs strict JSON from the LLM, computes net_edge after fees/slippage,
and gates on conviction level (LOW / MEDIUM / HIGH).

Cost optimizations:
- Two-tier LLM: cheap screening model filters markets before expensive analysis
- In-memory result cache with configurable TTL
"""

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import openai
import structlog
from openai import AsyncOpenAI

try:
    import anthropic
    from anthropic import AsyncAnthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

from pathlib import Path

from ..cost_tracker import CostTracker
from ..markets import Market
from ..news import NewsAggregator
from ..utils import BotConfig, RateLimiter
from .base import Signal, SignalResult, TradingSide


# ---------------------------------------------------------------------------
# Base rate database
# ---------------------------------------------------------------------------

_BASE_RATES: List[Dict[str, Any]] = []


def _load_base_rates() -> List[Dict[str, Any]]:
    """Load empirical base rates from JSON file."""
    global _BASE_RATES
    if _BASE_RATES:
        return _BASE_RATES
    base_rates_path = Path(__file__).parent.parent / "base_rates.json"
    if base_rates_path.exists():
        try:
            with open(base_rates_path) as f:
                data = json.load(f)
            _BASE_RATES = data.get("patterns", [])
        except Exception:
            _BASE_RATES = []
    return _BASE_RATES


def lookup_base_rate(question: str, description: str = "") -> Optional[Dict[str, Any]]:
    """Find the best matching base rate for a market question."""
    rates = _load_base_rates()
    text = f"{question} {description}".lower()
    best_match = None
    best_score = 0
    for entry in rates:
        keywords = entry.get("keywords", [])
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_match = entry
    return best_match if best_score > 0 else None


class ConvictionLevel(str, Enum):
    """Conviction level derived from net edge."""
    NONE = "none"      # < 2%
    LOW = "low"        # 2-5%
    MEDIUM = "medium"  # 5-10%
    HIGH = "high"      # 10%+


# ---------------------------------------------------------------------------
# Calibration: shrink overconfident extremes toward base rate
# ---------------------------------------------------------------------------

def calibrate_probability(
    p: float,
    *,
    floor: float = 0.05,
    ceiling: float = 0.95,
    shrink_strength: float = 0.15,
    yes_boost: float = 0.0,
    no_dampen: float = 0.0,
) -> float:
    """Apply asymmetric calibration to a raw LLM probability estimate.

    The LLM has two distinct failure modes (from backtest data):
    - NO predictions are excellent (0-20% bucket → actual 0-18%)
    - YES predictions are terrible (50-70% predicted → actual 18-29%)

    This applies:
    1. Shrinkage toward 0.5 (symmetric baseline)
    2. Asymmetric adjustment: boost low predictions outward, dampen high predictions
    3. Mushy middle correction: 0.40-0.70 → remap to empirical actual values
    4. Hard floor/ceiling clamp

    The mushy middle correction is the key insight: when the LLM predicts
    40-70% YES, it's almost always wrong. Backtest data (200 markets):
      40-50% predicted → 27% actual
      50-60% predicted → 18% actual
      60-70% predicted → 29% actual
    So we remap those predictions to match reality.
    """
    # Base symmetric shrink toward 0.5
    p = p * (1.0 - shrink_strength) + 0.5 * shrink_strength

    # Asymmetric adjustment
    if p > 0.5 and yes_boost != 0:
        overshoot = p - 0.5
        p = 0.5 + overshoot * (1.0 - yes_boost)
    elif p < 0.5 and no_dampen != 0:
        undershoot = 0.5 - p
        p = 0.5 - undershoot * (1.0 + no_dampen)

    # Mushy middle correction: empirical remap for 0.40-0.70
    # LLM is systematically wrong here — predicted YES ≈ actual NO
    # Maps: 0.40→0.28, 0.55→0.22 (worst), 0.70→0.30
    if 0.40 <= p <= 0.70:
        midpoint = 0.55
        if p <= midpoint:
            t = (p - 0.40) / (midpoint - 0.40)
            p = 0.28 - t * 0.06  # 0.28 → 0.22
        else:
            t = (p - midpoint) / (0.70 - midpoint)
            p = 0.22 + t * 0.08  # 0.22 → 0.30

    # Clamp to floor/ceiling
    p = max(floor, min(ceiling, p))
    return round(p, 4)


def detect_market_type(question: str) -> str:
    """Classify a market question into a type for calibration adjustments.

    Certain market types systematically fool LLMs:
    - announcer_mention: "Will NBA announcer say X" — random speech, no data
    - word_mention: "Will X mention Y" — can't predict exact words in speeches
    - crypto_range: "Will BTC close between $X-$Y" — narrow intraday = near-random
    - weather: temperature/rain markets — pro weather models already price this
    - exact_phrase: "Will X say Y" — LLM can't predict exact words
    - price_range: "Will BTC be between $X-$Y" — narrow ranges are near-random
    - sports: athletic outcomes — LLM has no real sports analytics
    - wild_card: Trump/volatile actor doing unpredictable things
    - politics: political events — LLM systematically underestimates YES
    - economics: data-driven markets — LLM does well here
    - normal: everything else
    """
    q = question.lower()

    # Announcer/commentator mention markets — random speech patterns, zero edge
    announcer_signals = [
        any(w in q for w in ["announcer", "commentator", "broadcaster", "analyst"]) and
        any(w in q for w in ["say", "mention", "use the word", "use the phrase"]),
        "during" in q and any(w in q for w in ["broadcast", "commentary", "coverage"]) and
        any(w in q for w in ["say", "mention"]),
    ]
    if any(announcer_signals):
        return "announcer_mention"

    # Word/phrase mention markets — "will X say the word Y during Z"
    word_mention_signals = [
        "say the word" in q or "use the word" in q or "use the phrase" in q,
        "mention" in q and any(w in q for w in [
            "during", "speech", "address", "conference", "interview",
            "broadcast", "show", "program", "segment",
        ]),
        # Generic "will X say Y" without political context
        " say " in q and " during " in q and not any(w in q for w in [
            "congress", "senate", "president", "hearing", "testimony",
        ]),
    ]
    if any(word_mention_signals):
        return "word_mention"

    # Weather markets — professional models already price these efficiently
    weather_signals = [
        any(w in q for w in [
            "temperature", "high temperature", "low temperature",
            "degrees fahrenheit", "degrees celsius",
            "rainfall", "inches of rain", "snowfall", "inches of snow",
            "wind speed", "humidity",
        ]),
        "weather" in q and any(w in q for w in ["above", "below", "between", "reach"]),
        any(w in q for w in ["heat wave", "cold snap", "freeze warning"]),
    ]
    if any(weather_signals):
        return "weather"

    # Crypto daily price range markets — narrow intraday ranges are near-random
    crypto_range_signals = [
        any(w in q for w in ["bitcoin", "btc", "ethereum", "eth", "solana", "sol",
                             "dogecoin", "doge", "xrp", "crypto"]) and
        any(w in q for w in ["between", "close above", "close below", "close between",
                             "end above", "end below", "price at"]),
        "$" in q and any(w in q for w in ["crypto", "bitcoin", "ethereum", "btc", "eth"]) and
        any(w in q for w in ["-", "to $", "between", "range"]),
    ]
    if any(crypto_range_signals):
        return "crypto_range"

    # Exact phrase / specific behavior markets
    phrase_signals = [
        "will " in q and (" say " in q or " mention " in q or " use the word" in q),
        " during " in q and (" speech" in q or " address" in q or " debate" in q),
        " exact " in q,
    ]
    if any(phrase_signals):
        return "exact_phrase"

    # Narrow price range markets (non-crypto)
    price_signals = [
        "price" in q and "between" in q,
        "$" in q and ("-" in q or "to $" in q) and "price" in q,
    ]
    if any(price_signals):
        return "price_range"

    # Sports outcomes — expanded to catch soccer, MMA, racing, etc.
    sports_signals = [
        "fantasy" in q,
        " win " in q and any(w in q for w in ["nba", "nfl", "mlb", "nhl", "super bowl",
                                               "championship", "finals", "world series",
                                               "premier league", "la liga", "serie a",
                                               "bundesliga", "champions league"]),
        any(w in q for w in ["touchdown", "rushing", "quarterback", "running back",
                             "points scored", "goals scored", "hat trick", "assists"]),
        # Soccer: "X FC vs Y FC", "X vs Y end in a draw"
        " fc " in q or " cf " in q or " sc " in q,
        " vs " in q and ("draw" in q or "win" in q),
        " vs." in q,
        any(w in q for w in ["premier league", "la liga", "serie a", "bundesliga",
                             "champions league", "europa league", "mls ", "ufc ",
                             "f1 ", "formula 1", "grand prix", "nascar"]),
        # Team name patterns: "X United", "X City", "Real X"
        any(w in q for w in [" united ", " city fc", " real madrid", " barcelona",
                             " liverpool", " chelsea", " arsenal", " palace"]),
    ]
    if any(sports_signals):
        return "sports"

    # Entertainment / pop culture — celebrity appearances, streaming charts, ads
    entertainment_signals = [
        any(w in q for w in ["spotify", "top song", "top album", "album debut",
                             "streaming", "billboard", "chart"]),
        any(w in q for w in ["rotten tomatoes", "tomatometer", "audience score",
                             "box office"]),
        any(w in q for w in ["super bowl ad", "super bowl commercial",
                             "halftime show", "halftime performer"]),
        any(w in q for w in ["celebrity", "appearance", "red carpet", "award show",
                             "grammy", "oscar", "emmy", "golden globe"]),
        any(w in q for w in ["tiktok", "instagram", "follower", "subscriber count"]),
    ]
    if any(entertainment_signals):
        return "entertainment"

    # Physical/engineering outcomes the LLM can't predict — coin flips
    coin_flip_signals = [
        any(w in q for w in ["explode", "crash", "malfunction", "abort", "fail"]) and
        any(w in q for w in ["spacex", "starship", "rocket", "launch", "flight test"]),
        "coin flip" in q or "coin toss" in q,
        "roulette" in q or "lottery" in q,
    ]
    if any(coin_flip_signals):
        return "coin_flip"

    # Wild card / volatile actor markets (subset of politics — more extreme)
    wild_card_signals = [
        "trump" in q and any(w in q for w in ["tariff", "pardon", "executive order", "announce", "tweet", "save", "ban"]),
        "elon" in q and ("tweet" in q or "post" in q or "ban" in q),
    ]
    if any(wild_card_signals):
        return "wild_card"

    # Politics / government action — LLM underestimates YES outcomes here
    politics_signals = [
        any(w in q for w in ["trump", "biden", "congress", "senate", "house",
                             "republican", "democrat", "election", "vote",
                             "governor", "president", "political", "government"]),
        any(w in q for w in ["tariff", "sanction", "executive order", "legislation",
                             "confirmation", "impeach", "shutdown"]),
    ]
    if any(politics_signals):
        return "politics"

    # Economics — data-driven markets where LLM has good calibration
    economics_signals = [
        any(w in q for w in ["fed ", "federal reserve", "interest rate", "rate cut",
                             "rate hike", "fomc", "powell", "monetary policy"]),
        any(w in q for w in ["inflation", "cpi", "ppi", "gdp", "recession",
                             "unemployment", "jobs report", "nonfarm", "payroll"]),
        any(w in q for w in ["treasury", "bond", "yield", "debt ceiling",
                             "deficit", "fiscal", "stimulus"]),
    ]
    if any(economics_signals):
        return "economics"

    return "normal"


# Per-market-type calibration profiles
# Each type gets: extra_shrink (toward 0.5), yes_boost (distrust YES), no_dampen (trust NO)
@dataclass
class MarketTypeCalibration:
    extra_shrink: float = 0.0    # Additional symmetric shrink toward 0.5
    yes_boost: float = 0.0       # Pull YES-side preds back toward 0.5 (0=none, 0.3=strong)
    no_dampen: float = 0.0       # Let NO-side preds be more confident (0=none, 0.2=moderate)
    skip: bool = False           # If True, don't trade this market type at all
    min_edge: float = 0.05       # Category-specific minimum edge threshold


MARKET_TYPE_CALIBRATION: Dict[str, MarketTypeCalibration] = {
    # Announcer/commentator mentions — random speech, zero data to forecast
    "announcer_mention": MarketTypeCalibration(skip=True, min_edge=0.99),
    # Word/phrase mention markets — can't predict exact words in speeches
    "word_mention": MarketTypeCalibration(skip=True, min_edge=0.99),
    # Weather markets — now traded with NOAA anchors, conservative edge
    "weather":     MarketTypeCalibration(skip=True, min_edge=0.99),
    # Crypto daily price ranges — narrow intraday ranges are near-random
    "crypto_range": MarketTypeCalibration(skip=True, min_edge=0.99),
    # LLM can't predict exact words → SKIP (backtest: 0.62 avg Brier)
    "exact_phrase": MarketTypeCalibration(skip=True, min_edge=0.99),
    # Narrow price ranges are near-random → SKIP (backtest: 0.56 avg Brier)
    "price_range": MarketTypeCalibration(skip=True, min_edge=0.10),
    # LLM has no real sports analytics → SKIP (backtest: 0.64+ Brier)
    "sports":      MarketTypeCalibration(skip=True, min_edge=0.10),
    # Entertainment / pop culture — unpredictable celebrity/media outcomes
    "entertainment": MarketTypeCalibration(skip=True, min_edge=0.99),
    # Physical outcomes / coin flips — LLM has zero edge
    "coin_flip":   MarketTypeCalibration(skip=True, min_edge=0.10),
    # Trump/volatile actors — LLM underestimates chaos; require higher edge
    "wild_card":   MarketTypeCalibration(extra_shrink=0.05, yes_boost=-0.10, no_dampen=0.0, min_edge=0.07),
    # Politics — LLM has some edge here (decent backtest), lower threshold
    "politics":    MarketTypeCalibration(extra_shrink=0.0, yes_boost=-0.08, no_dampen=0.10, min_edge=0.04),
    # Economics — data-driven, LLM does well with structured economic data
    "economics":   MarketTypeCalibration(extra_shrink=0.20, yes_boost=0.15, no_dampen=0.05, min_edge=0.08),
    # Normal markets — asymmetric correction based on backtest data
    "normal":      MarketTypeCalibration(extra_shrink=0.0, yes_boost=0.10, no_dampen=0.10, min_edge=0.05),
}

# Backward compat: flat extra-shrink dict for any code that references it
MARKET_TYPE_EXTRA_SHRINK: Dict[str, float] = {
    k: v.extra_shrink for k, v in MARKET_TYPE_CALIBRATION.items()
}


def classify_conviction(net_edge: float) -> ConvictionLevel:
    """Classify conviction from net edge (after fees/slippage)."""
    abs_edge = abs(net_edge)
    if abs_edge >= 0.10:
        return ConvictionLevel.HIGH
    elif abs_edge >= 0.05:
        return ConvictionLevel.MEDIUM
    elif abs_edge >= 0.02:
        return ConvictionLevel.LOW
    return ConvictionLevel.NONE


# ---------------------------------------------------------------------------
# Market quality heuristics
# ---------------------------------------------------------------------------

def _market_quality_ok(market: Market, min_liquidity: float = 500.0) -> tuple[bool, str]:
    """Return (ok, reason) for market quality checks."""
    if market.liquidity < min_liquidity:
        return False, f"Liquidity ${market.liquidity:,.0f} < ${min_liquidity:,.0f} minimum"

    # Ambiguous resolution heuristic: very long question or multiple conditions
    q = market.question or ""
    if len(q) > 300:
        return False, "Question text too long — likely ambiguous resolution criteria"
    if q.count(" and ") >= 3 or q.count(" AND ") >= 3:
        return False, "Multiple AND conditions — ambiguous resolution"

    return True, "ok"


# ---------------------------------------------------------------------------
# LLM response validation
# ---------------------------------------------------------------------------

def _validate_llm_response(data: Dict[str, Any]) -> Optional[str]:
    """Return error string if response is malformed, else None."""
    if not isinstance(data, dict):
        return "response is not a dict"

    p_yes = data.get("p_yes")
    if p_yes is None:
        return "missing p_yes"
    if not isinstance(p_yes, (int, float)):
        return f"p_yes is not a number: {type(p_yes)}"
    if not (0.0 <= float(p_yes) <= 1.0):
        return f"p_yes out of range: {p_yes}"

    uncertainty = data.get("uncertainty", "medium")
    if uncertainty not in ("low", "medium", "high"):
        return f"invalid uncertainty: {uncertainty}"

    return None


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------

def _round_price(price: float, step: float = 0.02) -> float:
    """Round price to nearest step for cache key stability."""
    return round(round(price / step) * step, 4)


def _cache_key(market_id: str, question: str, price: float) -> str:
    """Build a cache key from market id, question, and rounded price."""
    return f"{market_id}:{question}:{_round_price(price)}"


class _ResultCache:
    """Simple in-memory cache with TTL eviction."""

    def __init__(self, ttl_seconds: float = 1800.0, enabled: bool = True):
        self.ttl = ttl_seconds
        self.enabled = enabled
        self._store: Dict[str, Tuple[float, SignalResult]] = {}  # key → (timestamp, result)

        # Stats for periodic summary
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[SignalResult]:
        if not self.enabled:
            return None
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        ts, result = entry
        if time.monotonic() - ts > self.ttl:
            del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return result

    def put(self, key: str, result: SignalResult) -> None:
        if not self.enabled:
            return
        self._store[key] = (time.monotonic(), result)

    def evict_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.monotonic()
        expired = [k for k, (ts, _) in self._store.items() if now - ts > self.ttl]
        for k in expired:
            del self._store[k]
        return len(expired)

    def reset_stats(self) -> Tuple[int, int]:
        """Return (hits, misses) and reset counters."""
        h, m = self.hits, self.misses
        self.hits = self.misses = 0
        return h, m


class LLMSignal(Signal):
    """LLM-powered signal for probability estimation — aggressive accuracy mode."""

    def __init__(self, config: BotConfig):
        super().__init__("LLM")
        self.config = config
        self.logger = structlog.get_logger()

        # LLM configuration
        llm_config = config.llm
        self.model = llm_config.get("model", "gpt-4-turbo-preview")
        self.temperature = llm_config.get("temperature", 0.1)
        self.max_tokens = llm_config.get("max_tokens", 1000)
        self.timeout = llm_config.get("timeout_seconds", 30)

        # Two-tier screening config
        self.screening_enabled = llm_config.get("screening_enabled", True)
        self.screening_model = llm_config.get("screening_model", "gpt-4o-mini")

        # OpenAI client
        self.client = AsyncOpenAI(timeout=self.timeout)

        # News aggregator for context
        self.news_aggregator = NewsAggregator(config)

        # Rate limiting for OpenAI API
        self.rate_limiter = RateLimiter(50, 60.0)

        # Edge / fee parameters
        strategy = config.strategy
        # Maker orders = 0% fee; only taker fallback = 2%.
        # Default to maker (0%) since that's our execution strategy.
        self.fee_pct = float(strategy.get("fee_pct", 0.0))
        self.slippage_pct = float(strategy.get("slippage_pct", 0.005))  # 0.5%

        # Edge decay compensation — add historical erosion to costs
        self.edge_decay_compensation = self._load_edge_decay()
        if self.edge_decay_compensation > 0:
            self.logger.info(
                "edge_decay_loaded",
                avg_erosion=self.edge_decay_compensation,
                msg="Adding to slippage to compensate for historical edge decay",
            )
        self.min_conviction = ConvictionLevel(
            strategy.get("min_conviction", "low")
        )

        # Market quality
        self.min_market_liquidity = float(
            config.market_filters.get("min_liquidity_aggressive", 500.0)
        )

        # Cost tracking / budget enforcement
        monthly_budget = float(llm_config.get("monthly_budget_usd", 100.0))
        self.cost_tracker: Optional[CostTracker] = None  # set via set_cost_tracker()
        self._monthly_budget = monthly_budget

        # Result cache
        cache_ttl = float(llm_config.get("cache_ttl_minutes", 30)) * 60.0
        cache_enabled = llm_config.get("cache_enabled", True)
        self._cache = _ResultCache(ttl_seconds=cache_ttl, enabled=cache_enabled)

        # Calibration config — load dynamic params from backtest if available
        self.calibration_enabled = llm_config.get("calibration_enabled", True)
        self.calibration_floor = float(llm_config.get("calibration_floor", 0.05))
        self.calibration_ceiling = float(llm_config.get("calibration_ceiling", 0.95))
        self.calibration_shrink = float(llm_config.get("calibration_shrink_strength", 0.15))

        # Try to load optimized calibration from backtest results
        dynamic_cal = self._load_dynamic_calibration()
        if dynamic_cal:
            self.calibration_shrink = float(dynamic_cal.get("shrink_strength", self.calibration_shrink))
            self.logger.info(
                "dynamic_calibration_loaded",
                shrink=self.calibration_shrink,
                yes_boost=dynamic_cal.get("yes_boost"),
                no_dampen=dynamic_cal.get("no_dampen"),
                sample_size=dynamic_cal.get("sample_size"),
            )

        # Consensus config
        self.consensus_enabled = llm_config.get("consensus_enabled", False)
        self.consensus_model = llm_config.get("consensus_model", "claude-3-5-haiku-20241022")
        self.consensus_provider = llm_config.get("consensus_provider", "anthropic")
        self.consensus_max_divergence = float(llm_config.get("consensus_max_divergence", 0.10))
        self.consensus_min_edge = float(llm_config.get("consensus_min_edge", 0.07))

        # Daily budget cap
        self._daily_budget = float(llm_config.get("daily_budget_usd", 5.0))

        # Anthropic client for cross-family consensus
        self._anthropic_client = None
        if _HAS_ANTHROPIC and self.consensus_provider == "anthropic":
            try:
                self._anthropic_client = AsyncAnthropic(timeout=self.timeout)
            except Exception:
                self.logger.warning("anthropic_client_init_failed")

        # Screening stats for periodic summary
        self._screened_total = 0
        self._screened_passed = 0
        self._cache_served = 0

    @staticmethod
    def _load_edge_decay() -> float:
        """Load average edge erosion from state/edge_decay.jsonl.

        Returns the average erosion amount so we can add it to the
        slippage/fee cost when computing net edge.
        """
        decay_path = Path("state") / "edge_decay.jsonl"
        if not decay_path.exists():
            return 0.0
        try:
            erosions = []
            with open(decay_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.load(json.loads(line)) if False else json.loads(line)
                        ed = float(r.get("edge_at_decision", 0))
                        ee = float(r.get("edge_at_execution", 0))
                        erosions.append(ed - ee)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue
            if len(erosions) >= 5:
                avg_erosion = sum(erosions) / len(erosions)
                return max(0.0, avg_erosion)  # Only compensate for positive erosion
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _load_dynamic_calibration() -> Optional[Dict[str, Any]]:
        """Load calibration params computed by backtest (state/calibration_params.json)."""
        cal_path = Path("state") / "calibration_params.json"
        if not cal_path.exists():
            return None
        try:
            with open(cal_path) as f:
                data = json.load(f)
            # Only use if we had enough samples
            if data.get("sample_size", 0) >= 20:
                return data
        except Exception:
            pass
        return None

    def set_cost_tracker(self, tracker: CostTracker) -> None:
        """Inject a shared CostTracker instance (called from main.py)."""
        self.cost_tracker = tracker

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def evaluate(self, market: Market) -> SignalResult:
        """Evaluate market using LLM probability estimation."""
        try:
            self.logger.info(
                "llm_evaluate_start",
                market_id=market.id,
                question=market.question[:120],
            )

            # Budget check
            if self.cost_tracker and not self.cost_tracker.check_budget():
                return self._hold(market, "Monthly LLM budget exceeded — halting calls")

            # Market type skip — don't waste LLM calls on unwinnable categories
            mtype = detect_market_type(market.question)
            type_cal = MARKET_TYPE_CALIBRATION.get(mtype, MarketTypeCalibration())
            if type_cal.skip:
                return self._hold(market, f"Skipped market type '{mtype}' — LLM has no edge here")

            market_price = market.midpoint_price
            if market_price is None:
                return self._hold(market, "No market price available")

            # Market quality gate
            ok, reason = _market_quality_ok(market, self.min_market_liquidity)
            if not ok:
                return self._hold(market, f"Market quality: {reason}")

            # Check cache first
            ck = _cache_key(market.id, market.question or "", market_price)
            cached = self._cache.get(ck)
            if cached is not None:
                self._cache_served += 1
                self.logger.debug("llm_cache_hit", market_id=market.id)
                return cached

            self.logger.debug("llm_cache_miss", market_id=market.id)

            # Tier 1: cheap screening
            if self.screening_enabled:
                self._screened_total += 1
                passed = await self._screen_market(market, market_price)
                if not passed:
                    result = self._hold(market, "Screened out by tier-1 model")
                    # Don't cache screening rejections — let them be re-screened
                    return result
                self._screened_passed += 1

            # Tier 2: full LLM analysis
            # Get news context
            news_articles = await self.news_aggregator.get_market_news(market)
            news_context = self.news_aggregator.format_news_context(news_articles)

            # Build prompt
            prompt = self._build_prompt(market, news_context, market_price)

            # Call LLM
            await self.rate_limiter.acquire()
            llm_data = await self._call_llm(prompt, model=self.model)

            if llm_data is None:
                return self._hold(market, "LLM call failed")

            # Validate
            err = _validate_llm_response(llm_data)
            if err:
                self.logger.warning("llm_response_invalid", error=err, data=llm_data)
                return self._hold(market, f"Invalid LLM output: {err}")

            p_yes = float(llm_data["p_yes"])
            p_yes = max(0.001, min(0.999, p_yes))

            # Calibration: asymmetric shrink + category + time-to-resolution adjustments
            if self.calibration_enabled:
                p_yes_raw = p_yes
                # Detect market type for category-specific calibration
                mtype = detect_market_type(market.question)
                type_cal = MARKET_TYPE_CALIBRATION.get(mtype, MarketTypeCalibration())
                total_shrink = min(0.45, self.calibration_shrink + type_cal.extra_shrink)

                # Time-to-resolution shrinkage scaling:
                # Longer-horizon markets have more uncertainty → more shrinkage
                # < 3 days: no extra shrink (information is fresh)
                # 3-14 days: +0.03 shrink
                # 14-30 days: +0.06 shrink
                # > 30 days: +0.10 shrink
                time_shrink = 0.0
                if market.end_date:
                    days_to_close = (market.end_date - datetime.now(timezone.utc)).total_seconds() / 86400
                    if days_to_close > 30:
                        time_shrink = 0.10
                    elif days_to_close > 14:
                        time_shrink = 0.06
                    elif days_to_close > 3:
                        time_shrink = 0.03
                total_shrink = min(0.50, total_shrink + time_shrink)

                p_yes = calibrate_probability(
                    p_yes,
                    floor=self.calibration_floor,
                    ceiling=self.calibration_ceiling,
                    shrink_strength=total_shrink,
                    yes_boost=type_cal.yes_boost,
                    no_dampen=type_cal.no_dampen,
                )
                self.logger.debug(
                    "llm_calibrated",
                    market_id=market.id,
                    market_type=mtype,
                    raw_p_yes=p_yes_raw,
                    calibrated_p_yes=p_yes,
                    extra_shrink=type_cal.extra_shrink,
                    time_shrink=time_shrink,
                    yes_boost=type_cal.yes_boost,
                    no_dampen=type_cal.no_dampen,
                )

            # Multi-model consensus — only when edge is large enough to justify cost
            preliminary_edge = abs(p_yes - market_price)
            if self.consensus_enabled and preliminary_edge >= self.consensus_min_edge:
                self.logger.info("llm_consensus_start", market_id=market.id, primary_p_yes=p_yes, preliminary_edge=preliminary_edge)
                consensus_data = await self._call_consensus(prompt)
                if consensus_data and _validate_llm_response(consensus_data) is None:
                    p_yes_2 = float(consensus_data["p_yes"])
                    p_yes_2 = max(0.001, min(0.999, p_yes_2))
                    divergence = abs(p_yes - p_yes_2)
                    self.logger.info(
                        "llm_consensus_result",
                        market_id=market.id,
                        provider=self.consensus_provider,
                        primary_p_yes=p_yes,
                        consensus_p_yes=p_yes_2,
                        divergence=divergence,
                    )
                    if divergence > self.consensus_max_divergence:
                        result = self._hold(market, f"Consensus divergence {divergence:.3f} > {self.consensus_max_divergence} ({self.consensus_provider})")
                        self._cache.put(ck, result)
                        return result
                    p_yes = (p_yes + p_yes_2) / 2.0

            uncertainty = llm_data.get("uncertainty", "medium")
            key_facts: List[str] = llm_data.get("key_facts", [])
            disqualifiers: List[str] = llm_data.get("disqualifiers", [])
            rationale: str = llm_data.get("rationale", "")

            # If disqualifiers present, skip
            if disqualifiers:
                result = self._hold(
                    market,
                    f"Disqualifiers: {'; '.join(disqualifiers)}",
                )
                self._cache.put(ck, result)
                return result

            # Compute net edge (includes historical edge decay compensation)
            raw_edge = p_yes - market_price  # positive → YES underpriced
            net_edge = abs(raw_edge) - self.fee_pct - self.slippage_pct - self.edge_decay_compensation

            # Category-specific minimum edge (uses type_cal from calibration block above)
            category_min_edge = type_cal.min_edge if self.calibration_enabled else 0.05
            if net_edge < category_min_edge:
                result = self._hold(
                    market,
                    f"Net edge {net_edge:.3f} below category min {category_min_edge:.3f} (type={mtype})",
                )
                self._cache.put(ck, result)
                return result

            conviction = classify_conviction(net_edge)

            # Gate on conviction
            conviction_rank = {
                ConvictionLevel.NONE: 0,
                ConvictionLevel.LOW: 1,
                ConvictionLevel.MEDIUM: 2,
                ConvictionLevel.HIGH: 3,
            }
            min_rank = conviction_rank[self.min_conviction]
            if conviction_rank[conviction] < min_rank:
                result = self._hold(
                    market,
                    f"Conviction {conviction.value} below minimum {self.min_conviction.value} "
                    f"(net_edge={net_edge:.3f})",
                )
                self._cache.put(ck, result)
                return result

            # Determine side
            # Strategy insight from backtesting: the LLM is excellent at predicting NO
            # (0-25% range is well-calibrated) but terrible at predicting YES
            # in the MIDDLE range (30-75% has ~0.40 gap).
            #
            # Widened abstention zone (was 0.30-0.70, now 0.25-0.75):
            # - BUY_NO: freely, whenever raw_edge < 0 and net_edge > 0
            # - BUY_YES when p_yes > 0.75: very high conviction YES only
            # - BUY_YES when p_yes < 0.25: cheap lotto — well-calibrated range
            # - HOLD when 0.25 <= p_yes <= 0.75: the "mushy middle" — LLM unreliable
            #
            # Contrarian NO: when LLM says 0.40-0.65 YES and market agrees (price > 0.45),
            # backtest data shows actual is ~22% — strong contrarian NO signal.
            if raw_edge > 0 and net_edge > 0:
                if p_yes > 0.75 or p_yes < 0.25:
                    side = TradingSide.BUY_YES
                else:
                    side = TradingSide.HOLD
            elif raw_edge < 0 and net_edge > 0:
                side = TradingSide.BUY_NO
            elif (0.40 <= p_yes <= 0.65 and market_price > 0.45
                  and (market_price - p_yes) > self.fee_pct + self.slippage_pct):
                # Contrarian: LLM says mild YES but calibration shows actual ~22%
                # Market is overpriced relative to calibrated reality → BUY_NO
                side = TradingSide.BUY_NO
                raw_edge = p_yes - market_price  # negative → NO direction
                net_edge = abs(raw_edge) - self.fee_pct - self.slippage_pct
                conviction = classify_conviction(net_edge)
                self.logger.info(
                    "contrarian_no_signal",
                    market_id=market.id,
                    p_yes=p_yes,
                    market_price=market_price,
                    net_edge=net_edge,
                )
            else:
                side = TradingSide.HOLD

            # Payout ratio filter: skip trades where risk/reward is terrible
            # e.g. buying NO at $0.95 to win $0.05 → payout ratio 0.053
            # min_payout_ratio of 0.15 means we need at least 15% return on risk
            min_payout_ratio = float(
                self.config.strategy.get("min_payout_ratio", 0.15)
            )
            if side != TradingSide.HOLD:
                if side == TradingSide.BUY_YES:
                    entry_cost = market_price
                else:  # BUY_NO
                    entry_cost = 1.0 - market_price
                payout_ratio = (1.0 - entry_cost) / entry_cost if entry_cost > 0 else 0
                if payout_ratio < min_payout_ratio:
                    side = TradingSide.HOLD
                    self.logger.debug(
                        "payout_ratio_reject",
                        market_id=market.id,
                        entry_cost=f"{entry_cost:.3f}",
                        payout_ratio=f"{payout_ratio:.3f}",
                        min_required=min_payout_ratio,
                    )

            # Map uncertainty to a confidence value for downstream sizing
            uncertainty_to_confidence = {
                "low": 0.9,
                "medium": 0.7,
                "high": 0.5,
            }
            confidence = uncertainty_to_confidence.get(uncertainty, 0.7)

            reasoning = (
                f"{rationale} | "
                f"p_yes={p_yes:.3f} mkt={market_price:.3f} "
                f"raw_edge={raw_edge:+.3f} net_edge={net_edge:+.3f} "
                f"conviction={conviction.value} uncertainty={uncertainty}"
            )
            if key_facts:
                reasoning += f" | facts: {'; '.join(key_facts[:3])}"

            result = SignalResult(
                estimated_prob=p_yes,
                confidence=confidence,
                edge=raw_edge,
                recommended_side=side,
                reasoning=reasoning,
                signal_name=self.name,
                market_price=market_price,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            # Attach extra metadata for downstream consumers
            result.conviction = conviction  # type: ignore[attr-defined]
            result.net_edge = net_edge       # type: ignore[attr-defined]

            self.logger.info(
                "llm_signal",
                market_id=market.id,
                p_yes=p_yes,
                market_price=market_price,
                raw_edge=raw_edge,
                net_edge=net_edge,
                conviction=conviction.value,
                side=side.value,
            )

            # Cache the result
            self._cache.put(ck, result)

            return result

        except Exception as e:
            self.logger.error("llm_evaluate_error", market_id=market.id, error=str(e))
            return self._hold(market, f"Error: {e}")

    def log_periodic_summary(self) -> None:
        """Log a summary of screening and cache stats, then reset counters."""
        cache_hits, cache_misses = self._cache.reset_stats()
        self.logger.info(
            "llm_cost_summary",
            screened_total=self._screened_total,
            screened_passed=self._screened_passed,
            screened_rejected=self._screened_total - self._screened_passed,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_served=self._cache_served,
        )
        self._screened_total = 0
        self._screened_passed = 0
        self._cache_served = 0

        # Evict expired cache entries
        evicted = self._cache.evict_expired()
        if evicted:
            self.logger.debug("llm_cache_evicted", count=evicted)

    # ------------------------------------------------------------------
    # Tier 1: Screening
    # ------------------------------------------------------------------

    async def _screen_market(self, market: Market, market_price: float) -> bool:
        """Tier-1 cheap screening. Returns True if market is worth full evaluation."""
        end_date_str = "Unknown"
        if market.end_date:
            end_date_str = market.end_date.strftime("%Y-%m-%d")

        prompt = (
            f"Q: {market.question}\n"
            f"Ends: {end_date_str} | Liq: ${market.liquidity:,.0f}\n\n"
            "Based on your knowledge, is this market likely to be mispriced by >5%? "
            'Reply JSON: {{"worth_evaluating": true/false, "reason": "..."}}'
        )

        try:
            await self.rate_limiter.acquire()
            response = await self.client.chat.completions.create(
                model=self.screening_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You screen prediction markets for mispricing. "
                            "Output ONLY valid JSON. Be selective — most markets are fairly priced."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=100,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            # Track screening cost
            if self.cost_tracker and response.usage:
                self.cost_tracker.record_call(
                    self.screening_model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            if not content:
                return True  # fail open

            data = json.loads(content)
            worth = data.get("worth_evaluating", True)
            reason = data.get("reason", "")

            if not worth:
                self.logger.debug(
                    "llm_screening_rejected",
                    market_id=market.id,
                    question=market.question[:80],
                    reason=reason,
                )

            return bool(worth)

        except Exception as e:
            self.logger.warning("llm_screening_error", market_id=market.id, error=str(e))
            return True  # fail open — if screening fails, proceed to full eval

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Call LLM API and parse response."""
        try:
            response = await self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            # Track main model cost
            used_model = model or self.model
            if self.cost_tracker and response.usage:
                self.cost_tracker.record_call(
                    used_model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            if not content:
                return None

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                extracted = self._extract_json_object(content)
                if extracted is None:
                    raise
                return json.loads(extracted)

        except openai.APIError as e:
            self.logger.error("openai_api_error", error=str(e))
            return None
        except json.JSONDecodeError as e:
            self.logger.error("llm_json_parse_error", error=str(e))
            return None
        except Exception as e:
            self.logger.error("llm_call_error", error=str(e))
            return None

    async def _call_consensus(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call consensus model — supports Anthropic (Claude) or OpenAI fallback."""
        if self.consensus_provider == "anthropic" and self._anthropic_client:
            return await self._call_anthropic(prompt)
        else:
            # Fallback to OpenAI consensus
            await self.rate_limiter.acquire()
            return await self._call_llm(prompt, model=self.consensus_model)

    async def _call_anthropic(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Anthropic Claude API for cross-family consensus."""
        if not self._anthropic_client:
            return None
        try:
            response = await self._anthropic_client.messages.create(
                model=self.consensus_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self._system_prompt(),
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text if response.content else None

            # Track cost
            if self.cost_tracker and response.usage:
                self.cost_tracker.record_call(
                    self.consensus_model,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )

            if not content:
                return None

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                extracted = self._extract_json_object(content)
                if extracted is None:
                    return None
                return json.loads(extracted)

        except Exception as e:
            self.logger.warning("anthropic_consensus_error", error=str(e))
            return None

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, market: Market, news_context: str, market_price: float) -> str:
        end_date_str = "Unknown"
        time_remaining = "Unknown"

        if market.end_date:
            end_date_str = market.end_date.strftime("%Y-%m-%d %H:%M UTC")
            now = datetime.now(timezone.utc)
            delta = market.end_date - now
            days = delta.days
            hours = delta.seconds // 3600
            time_remaining = f"{days}d {hours}h" if days > 0 else f"{hours}h"

        # Lookup empirical base rate from database
        base_rate_entry = lookup_base_rate(market.question, market.description)
        base_rate_context = ""
        if base_rate_entry:
            base_rate_context = (
                f"\nEMPIRICAL BASE RATE DATA (use this as your starting anchor):\n"
                f"  Base rate: {base_rate_entry['base_rate']:.0%}\n"
                f"  Reference class: {base_rate_entry['reference_class']}\n"
                f"  Source: {base_rate_entry['source']}\n"
                f"Start from this empirical base rate and adjust based on current evidence.\n"
            )

        return f"""
You are a superforecaster estimating the TRUE probability of YES for this prediction market.

Question: {market.question}
Description: {market.description}
Category: {market.category}
End date: {end_date_str} ({time_remaining} remaining)
24h volume: ${market.volume_24h:,.0f}
Liquidity: ${market.liquidity:,.0f}

NOTE: You are intentionally NOT given the current market price. Derive your
probability estimate independently from your own analysis. Do NOT guess or
infer the market price from any contextual clues.
{base_rate_context}
{news_context}

Follow this structured forecasting methodology:

STEP 1 — BASE RATE (Reference Class Forecasting):
{"Use the EMPIRICAL BASE RATE DATA provided above as your starting point." if base_rate_entry else "Identify the most relevant reference class for this event. What is the historical base rate for similar events?"} (e.g., "Incumbents win re-election ~70% of the time",
"Government shutdowns resolve within 2 weeks ~60% of the time"). Start your estimate
here.

STEP 2 — EVIDENCE ADJUSTMENT:
List specific pieces of evidence that move the probability UP or DOWN from the base
rate. Be explicit: "+5% because polling shows…", "-10% because the timeline is…".
Each adjustment should be grounded in concrete facts.

STEP 3 — DECOMPOSITION:
If the event depends on multiple independent factors, decompose it:
P(event) = P(factor_A) × P(factor_B | factor_A) × …
This prevents overconfidence from vague holistic reasoning.

STEP 4 — FINAL ESTIMATE:
Combine base rate + adjustments + decomposition into a final p_yes. Sanity-check:
does this feel right given everything you know? Adjust if needed.

Output STRICT JSON (no commentary outside JSON):
{{
  "p_yes": <float 0.0-1.0>,
  "base_rate": <float — your starting base rate>,
  "base_rate_reference_class": "description of reference class used",
  "adjustments": ["+0.05: reason", "-0.03: reason", ...],
  "decomposition": "P(A)=X × P(B|A)=Y = Z (if applicable, else null)",
  "key_facts": ["fact1", "fact2", ...],
  "uncertainty": "low" | "medium" | "high",
  "disqualifiers": [],
  "rationale": "one-paragraph final reasoning incorporating all steps"
}}

Rules:
- p_yes is your best point estimate of P(YES resolves true)
- base_rate: the starting probability from your reference class
- adjustments: explicit list of evidence-based adjustments from the base rate
- key_facts: 2-5 most important facts driving your estimate
- uncertainty: low = strong evidence, high = guessing
- disqualifiers: list reasons to NOT trade (ambiguous resolution, stale info, etc). Empty list if tradeable.

Domain guidance:
- For political markets: consider partisan dynamics, committee compositions, historical precedent, polling data, insider reporting, legislative vote counts, and historical base rates for similar events.
- For economic markets: consider recent Fed minutes, dot plot projections, economic indicators (CPI, jobs, GDP), and CME FedWatch consensus on rate moves.
- Be careful with conditional probabilities — decompose into independent factors where possible.
- Reference class matters: "How often does X happen?" is almost always the right first question.
""".strip()

    def _system_prompt(self) -> str:
        return (
            "You are an elite superforecaster for prediction markets, trained in "
            "the methodology of Philip Tetlock's Good Judgment Project. "
            "You output ONLY valid JSON matching the requested schema. "
            "You ALWAYS start with a base rate from the most relevant reference class, "
            "then adjust based on specific evidence. You decompose complex events into "
            "independent factors when possible. "
            "You are aggressive about finding mispriced markets — look for genuine edge. "
            "Do not hedge; give your best point estimate for p_yes. "
            "Avoid round numbers (0.50, 0.70) — real probabilities are rarely round. "
            "Consider base rates, recent news, time remaining, and information quality."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hold(self, market: Market, reason: str) -> SignalResult:
        mp = market.midpoint_price or 0.5
        self.logger.info(
            "llm_signal",
            market_id=market.id,
            market_price=mp,
            side="hold",
            reason=reason[:120],
            p_yes=mp,
            raw_edge=0.0,
            net_edge=0.0,
            conviction="none",
        )
        result = SignalResult(
            estimated_prob=mp,
            confidence=0.0,
            edge=0.0,
            recommended_side=TradingSide.HOLD,
            reasoning=reason,
            signal_name=self.name,
            market_price=mp,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        result.conviction = ConvictionLevel.NONE  # type: ignore[attr-defined]
        result.net_edge = 0.0                      # type: ignore[attr-defined]
        return result

    async def backtest_evaluate(self, market: Market) -> SignalResult:
        """Evaluate a market for backtesting — no screening, no cache, no price anchor.

        This tests raw LLM prediction ability by:
        1. Skipping the tier-1 screener
        2. Not telling the LLM the current market price (prevents anchoring)
        3. Skipping market quality gates and budget checks
        4. No caching
        """
        try:
            # Market type skip — don't waste LLM calls on unwinnable categories
            mtype_early = detect_market_type(market.question)
            type_cal_early = MARKET_TYPE_CALIBRATION.get(mtype_early, MarketTypeCalibration())
            if type_cal_early.skip:
                return self._hold(market, f"Skipped market type '{mtype_early}' — LLM has no edge here")

            end_date_str = "Unknown"
            time_remaining = "Unknown"
            if market.end_date:
                end_date_str = market.end_date.strftime("%Y-%m-%d %H:%M UTC")

            prompt = f"""
Analyze this prediction market. Estimate the TRUE probability of YES.

Question: {market.question}
Description: {market.description}
Category: {market.category or "General"}
End date: {end_date_str}

You do NOT have the current market price. Estimate the probability purely
from your knowledge. Be decisive — avoid defaulting to 0.50 unless you
genuinely believe it's a coin flip.

Output STRICT JSON (no commentary outside JSON):
{{
  "p_yes": <float 0.0-1.0>,
  "key_facts": ["fact1", "fact2", ...],
  "uncertainty": "low" | "medium" | "high",
  "disqualifiers": [],
  "rationale": "one-paragraph reasoning"
}}

Rules:
- p_yes is your best estimate of P(YES resolves true)
- key_facts: 2-5 most important facts driving your estimate
- uncertainty: low = strong evidence, high = guessing
- disqualifiers: list reasons to NOT trade (ambiguous resolution, stale info, etc). Empty list if tradeable.
- Consider base rates and historical precedent. Be specific, not vague.
- A p_yes of exactly 0.50 should be RARE — most events are not perfect coin flips.

Domain guidance:
- For political markets: consider partisan dynamics, committee compositions, historical precedent, polling data, insider reporting, legislative vote counts.
- For economic markets: consider Fed minutes, dot plot projections, economic indicators (CPI, jobs, GDP), CME FedWatch.
- For sports: consider team records, recent form, injuries, head-to-head history.
- For crypto/price markets: consider recent trends, volatility, support/resistance levels.
- Be careful with conditional probabilities — decompose into independent factors where possible.
""".strip()

            await self.rate_limiter.acquire()
            llm_data = await self._call_llm(prompt, model=self.model)

            if llm_data is None:
                return self._hold(market, "LLM call failed")

            err = _validate_llm_response(llm_data)
            if err:
                return self._hold(market, f"Invalid LLM output: {err}")

            p_yes = float(llm_data["p_yes"])
            p_yes = max(0.001, min(0.999, p_yes))

            # Calibration: asymmetric shrink + category adjustments
            mtype = detect_market_type(market.question)
            type_cal = MARKET_TYPE_CALIBRATION.get(mtype, MarketTypeCalibration())
            total_shrink = min(0.45, 0.15 + type_cal.extra_shrink)
            p_yes = calibrate_probability(
                p_yes,
                shrink_strength=total_shrink,
                yes_boost=type_cal.yes_boost,
                no_dampen=type_cal.no_dampen,
            )

            uncertainty = llm_data.get("uncertainty", "medium")
            key_facts = llm_data.get("key_facts", [])
            rationale = llm_data.get("rationale", "")

            # Use 0.5 as reference for edge calculation (no market price)
            ref_price = 0.5
            raw_edge = p_yes - ref_price

            # Determine conviction from uncertainty + edge magnitude
            abs_edge = abs(raw_edge)
            if uncertainty == "low" and abs_edge >= 0.15:
                conviction = ConvictionLevel.HIGH
            elif uncertainty == "low" or abs_edge >= 0.10:
                conviction = ConvictionLevel.MEDIUM
            elif abs_edge >= 0.05:
                conviction = ConvictionLevel.LOW
            else:
                conviction = ConvictionLevel.NONE

            # Same NO-bias strategy as live: BUY_YES only outside widened mushy middle
            if raw_edge > 0:
                if p_yes > 0.75 or p_yes < 0.25:
                    side = TradingSide.BUY_YES
                else:
                    side = TradingSide.HOLD
            elif raw_edge < -0.05:
                side = TradingSide.BUY_NO
            else:
                side = TradingSide.HOLD

            confidence_map = {"low": 0.9, "medium": 0.7, "high": 0.5}
            confidence = confidence_map.get(uncertainty, 0.7)

            result = SignalResult(
                estimated_prob=p_yes,
                confidence=confidence,
                edge=raw_edge,
                recommended_side=side,
                reasoning=f"{rationale} | p_yes={p_yes:.3f} uncertainty={uncertainty}",
                signal_name=self.name,
                market_price=ref_price,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            result.conviction = conviction
            result.net_edge = abs_edge

            return result

        except Exception as e:
            self.logger.error("backtest_eval_error", market_id=market.id, error=str(e))
            return self._hold(market, f"Error: {e}")

    @staticmethod
    def _extract_json_object(text: str) -> Optional[str]:
        if not text:
            return None
        cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
        start = cleaned.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start : i + 1]
        return None
