"""Structured data anchors — hard data to anchor LLM predictions.

For market types where real data exists, fetch it and inject into prompts
as a starting anchor. Transforms predictions from "LLM guessing from vibes"
to "LLM adjusting from data-driven anchor."

Sources:
- CME FedWatch (via scraping) — Fed rate probability from futures
- Economic calendar (via API/scraping) — consensus forecasts
- Polling data (via RealClearPolitics/FiveThirtyEight) — aggregate polls
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx
import structlog

_FRED_API_KEY = os.getenv("FRED_API_KEY", "")

logger = structlog.get_logger()


async def get_structured_anchor(question: str, category: str = "") -> Optional[str]:
    """Get structured data context for a market question.

    Returns a formatted string for prompt injection, or None if no data available.
    """
    q = question.lower()

    # Fed/FOMC markets — get FedWatch probabilities
    if any(w in q for w in ["fed ", "fomc", "rate cut", "rate hike", "interest rate",
                             "federal reserve", "powell", "monetary policy"]):
        return await _get_fedwatch_context()

    # Economic data markets — get consensus forecasts
    if any(w in q for w in ["cpi", "inflation", "jobs report", "nonfarm",
                             "unemployment", "gdp", "ppi", "payroll"]):
        return await _get_econ_calendar_context(question)

    # Polling/election markets
    if any(w in q for w in ["approval rating", "poll", "favorability",
                             "election", "primary", "caucus"]):
        return await _get_polling_context(question)

    # Weather markets — NOAA NWS forecast data
    if any(w in q for w in ["temperature", "high temperature", "low temperature",
                             "degrees fahrenheit", "degrees celsius",
                             "rainfall", "inches of rain", "snowfall", "inches of snow",
                             "wind speed", "heat wave", "cold snap"]):
        return await _get_weather_context(question)

    return None


async def _get_fedwatch_context() -> Optional[str]:
    """Fetch CME FedWatch-style data from public sources.

    Uses the CME website's public data or financial news for current
    Fed rate probabilities implied by futures markets.
    """
    if not _FRED_API_KEY:
        return None

    try:
        # Fetch Fed funds futures data from FRED API
        async with httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
            follow_redirects=True,
        ) as client:
            resp = await client.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": "DFEDTARU",  # Federal funds target rate upper
                    "api_key": _FRED_API_KEY,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 1,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                observations = data.get("observations", [])
                if observations:
                    current_rate = observations[0].get("value", "unknown")
                    date = observations[0].get("date", "unknown")
                    return (
                        f"\nSTRUCTURED DATA ANCHOR — Federal Reserve:\n"
                        f"  Current Fed Funds Target Rate (upper): {current_rate}%\n"
                        f"  As of: {date}\n"
                        f"  Note: Check CME FedWatch for next-meeting probability distribution.\n"
                        f"  Use this as your starting anchor for any Fed rate decision market.\n"
                    )

    except Exception as e:
        logger.debug("fedwatch_fetch_error", error=str(e))

    return None


async def _fetch_fred_series(series_id: str, limit: int = 6) -> List[Dict]:
    """Fetch recent observations from a FRED series. Returns list of {date, value}."""
    if not _FRED_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": _FRED_API_KEY,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": limit,
                },
            )
            if resp.status_code == 200:
                return resp.json().get("observations", [])
    except Exception as e:
        logger.debug("fred_fetch_error", series_id=series_id, error=str(e))
    return []


async def _get_econ_calendar_context(question: str) -> Optional[str]:
    """Get consensus forecasts for upcoming economic releases.

    For CPI markets, fetches multiple related series to give the LLM
    a comprehensive data anchor: headline CPI, core CPI, CPI YoY, and
    recent trend data.
    """
    try:
        q = question.lower()

        # CPI gets special deep treatment — it's our strongest category
        if any(w in q for w in ["cpi", "inflation"]):
            return await _get_cpi_deep_context(question)

        # Map other indicators to FRED series
        indicator = None
        if any(w in q for w in ["jobs", "nonfarm", "payroll"]):
            indicator = "Employment Situation"
        elif "unemployment" in q:
            indicator = "Unemployment"
        elif "gdp" in q:
            indicator = "GDP"
        elif "ppi" in q:
            indicator = "PPI"

        if not indicator:
            return None

        series_map = {
            "Employment Situation": [
                ("PAYEMS", "Total Nonfarm Payrolls (thousands)"),
                ("UNRATE", "Unemployment Rate (%)"),
            ],
            "Unemployment": [
                ("UNRATE", "Unemployment Rate (%)"),
                ("U6RATE", "U-6 Unemployment Rate (%)"),
            ],
            "GDP": [
                ("GDP", "Gross Domestic Product (billions $)"),
                ("A191RL1Q225SBEA", "Real GDP Growth Rate (%)"),
            ],
            "PPI": [
                ("PPIACO", "Producer Price Index"),
                ("PPIFIS", "PPI Final Demand"),
            ],
        }

        series_list = series_map.get(indicator, [])
        if not series_list:
            return None

        context = f"\nSTRUCTURED DATA ANCHOR — {indicator}:\n"
        for series_id, label in series_list:
            obs = await _fetch_fred_series(series_id, limit=3)
            if obs:
                latest = obs[0]
                prev = obs[1] if len(obs) > 1 else None
                val = latest.get("value", "?")
                dt = latest.get("date", "?")
                context += f"  {label}: {val} (as of {dt})"
                if prev:
                    prev_val = prev.get("value", "?")
                    try:
                        change = float(val) - float(prev_val)
                        context += f" | prev: {prev_val} (change: {change:+.1f})"
                    except (ValueError, TypeError):
                        context += f" | prev: {prev_val}"
                context += "\n"

        context += (
            "  Historical pattern: ~50% of releases beat consensus, ~50% miss.\n"
            "  Use this data as your starting anchor.\n"
        )
        return context

    except Exception as e:
        logger.debug("econ_calendar_fetch_error", error=str(e))

    return None


async def _get_cpi_deep_context(question: str) -> Optional[str]:
    """Deep CPI data anchor — multiple FRED series for CPI markets.

    CPI is the bot's strongest category. Fetch:
    - CPIAUCSL: Headline CPI (seasonally adjusted, index)
    - CPILFESL: Core CPI (ex food & energy, index)
    - MEDCPIM158SFRBCLE: Median CPI (Cleveland Fed)
    YoY computed from index in threshold analysis (not stale CPALTT01USM657N).
    """
    if not _FRED_API_KEY:
        return None

    series = [
        ("CPIAUCSL", "Headline CPI Index (SA)"),
        ("CPILFESL", "Core CPI Index (ex food/energy, SA)"),
        # CPALTT01USM657N removed — returns stale monthly rate, not annual YoY.
        # YoY is computed directly from CPIAUCSL index in threshold analysis.
        ("MEDCPIM158SFRBCLE", "Median CPI (Cleveland Fed, annualized %)"),
    ]

    context = "\nSTRUCTURED DATA ANCHOR — CPI Deep Dive:\n"
    has_data = False

    for series_id, label in series:
        obs = await _fetch_fred_series(series_id, limit=6)
        if not obs:
            continue
        has_data = True

        # Show latest value + trend
        latest = obs[0]
        val = latest.get("value", "?")
        dt = latest.get("date", "?")
        context += f"  {label}: {val} (as of {dt})\n"

        # Show MoM change for index series
        if len(obs) >= 2 and series_id in ("CPIAUCSL", "CPILFESL"):
            try:
                curr = float(obs[0]["value"])
                prev = float(obs[1]["value"])
                mom_pct = ((curr - prev) / prev) * 100
                context += f"    MoM change: {mom_pct:+.2f}%\n"
                # 3-month trend
                if len(obs) >= 4:
                    three_ago = float(obs[3]["value"])
                    three_mo_annualized = (((curr / three_ago) ** 4) - 1) * 100
                    context += f"    3-month annualized: {three_mo_annualized:.1f}%\n"
            except (ValueError, TypeError):
                pass

    if not has_data:
        return None

    # ---- Threshold analysis: compare question threshold to actual data ----
    threshold_context = await _get_cpi_threshold_analysis(question)
    if threshold_context:
        context += threshold_context

    # Add interpretive guidance
    q = question.lower()
    context += "\n  INTERPRETATION GUIDANCE:\n"
    if "month" in q or "rise" in q or "change" in q:
        context += (
            "  - MoM CPI typically ranges 0.1%-0.4% (rounded to 1 decimal)\n"
            "  - Core CPI is stickier (less volatile month to month)\n"
            "  - Consensus forecasts are usually within 0.1% of actual\n"
        )
    if "year" in q or "yoy" in q or "annual" in q:
        context += (
            "  - YoY CPI has been trending between 2.5%-3.5% in recent months\n"
            "  - Fed target is 2.0% — anything above suggests continued tightening\n"
        )

    context += "  Use the MoM trend and 3-month annualized rate as your primary anchors.\n"
    return context


def _safe_float(val: str) -> Optional[float]:
    """Parse FRED value, returning None for missing data (e.g., '.')."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _find_valid_pair(obs: List[Dict], offset: int = 1) -> Optional[Tuple[float, float, str]]:
    """Find a valid (current, previous) pair skipping missing values.

    Returns (current_value, prev_value, as_of_date) or None.
    """
    values = []
    for o in obs:
        v = _safe_float(o.get("value", ""))
        if v is not None:
            values.append((v, o.get("date", "?")))
        if len(values) >= offset + 1:
            break
    if len(values) >= offset + 1:
        return values[0][0], values[offset][0], values[0][1]
    return None


async def _get_cpi_threshold_analysis(question: str) -> Optional[str]:
    """Compare the question's threshold to actual FRED data.

    Parses "above X%" from the question, fetches the relevant FRED series,
    and tells the LLM exactly how the current value compares to the threshold.
    This prevents the LLM from guessing when hard data is available.
    """
    q = question.lower()

    # Parse threshold from question
    threshold = _parse_cpi_threshold(q)
    if threshold is None:
        return None

    # Determine which type of CPI question this is
    is_yoy = any(w in q for w in ["year ending", "yoy", "annual", "year over year"])
    is_core = "core" in q
    is_combo = "and" in q and "yoy" in q  # COMBO markets

    if is_combo:
        # Combo markets have two thresholds — handle the YoY part
        return await _analyze_combo_threshold(q)

    if is_yoy:
        # YoY CPI — compute from CPIAUCSL index (13 months)
        # Don't use CPALTT01USM657N — it's stale/monthly, not annual
        series_id = "CPILFESL" if is_core else "CPIAUCSL"
        obs = await _fetch_fred_series(series_id, limit=14)
        if not obs or len(obs) < 13:
            return None
        # Find current and 12-months-ago values, skipping missing data
        current_vals = [(o, _safe_float(o.get("value", ""))) for o in obs]
        valid = [(o, v) for o, v in current_vals if v is not None]
        if len(valid) < 13:
            return None
        curr_val = valid[0][1]
        year_ago_val = valid[12][1]
        as_of = valid[0][0].get("date", "?")
        current_val = round(((curr_val / year_ago_val) - 1) * 100, 2)

        gap = current_val - threshold
        series_label = "CPI YoY (computed from index)"
        volatility = 0.2  # typical month-to-month change in YoY CPI

    elif is_core:
        # Core CPI MoM — compute from CPILFESL index
        obs = await _fetch_fred_series("CPILFESL", limit=4)
        pair = _find_valid_pair(obs or [])
        if not pair:
            return None
        curr_idx, prev_idx, as_of = pair
        current_val = round(((curr_idx - prev_idx) / prev_idx) * 100, 3)

        gap = current_val - threshold
        series_label = "Core CPI MoM change"
        volatility = 0.1  # core is stickier

    else:
        # Headline CPI MoM — compute from CPIAUCSL index
        obs = await _fetch_fred_series("CPIAUCSL", limit=4)
        pair = _find_valid_pair(obs or [])
        if not pair:
            return None
        curr_idx, prev_idx, as_of = pair
        current_val = round(((curr_idx - prev_idx) / prev_idx) * 100, 3)

        gap = current_val - threshold
        series_label = "Headline CPI MoM change"
        volatility = 0.15

    # Build threshold analysis
    ctx = f"\n  THRESHOLD ANALYSIS (CRITICAL — use this as your primary anchor):\n"
    ctx += f"    Question threshold: {threshold}%\n"
    ctx += f"    Latest {series_label}: {current_val}% (as of {as_of})\n"
    ctx += f"    Gap: current value is {gap:+.2f} percentage points vs threshold\n"
    ctx += f"    Typical month-to-month volatility: ±{volatility} pp\n"

    # Classify likelihood
    gap_in_sigmas = abs(gap) / volatility if volatility > 0 else 0
    if gap > 0 and gap_in_sigmas >= 3:
        ctx += (
            f"    Assessment: Current value ({current_val}%) is FAR ABOVE threshold ({threshold}%). "
            f"This is {gap_in_sigmas:.1f}x the typical volatility. "
            f"Unless there is a massive deflationary shock, p_yes should be VERY HIGH (>0.90).\n"
            f"    WARNING: Do NOT let anti-YES bias override this hard data. "
            f"The base rate for 'above X%' when current value is {gap:.1f}pp above X is >95%.\n"
        )
    elif gap > 0 and gap_in_sigmas >= 1.5:
        ctx += (
            f"    Assessment: Current value ({current_val}%) is well above threshold ({threshold}%). "
            f"p_yes should be HIGH (0.75-0.95) unless you have specific evidence of a sharp reversal.\n"
        )
    elif gap < 0 and gap_in_sigmas >= 3:
        ctx += (
            f"    Assessment: Current value ({current_val}%) is FAR BELOW threshold ({threshold}%). "
            f"Unless there is a massive inflationary shock, p_yes should be VERY LOW (<0.10).\n"
        )
    elif gap < 0 and gap_in_sigmas >= 1.5:
        ctx += (
            f"    Assessment: Current value ({current_val}%) is well below threshold ({threshold}%). "
            f"p_yes should be LOW (0.05-0.25) unless you have specific evidence of a sharp increase.\n"
        )
    else:
        ctx += (
            f"    Assessment: Current value ({current_val}%) is CLOSE to threshold ({threshold}%). "
            f"This is a genuinely uncertain market — careful analysis needed.\n"
        )

    return ctx


async def _analyze_combo_threshold(question: str) -> Optional[str]:
    """Handle COMBO markets like 'CPI above 0.0% AND YoY above 2.4%'."""
    # Extract both thresholds
    import re as _re
    mom_match = _re.search(r'cpi\s+(?:be\s+)?above\s+(-?[\d.]+)%', question)
    yoy_match = _re.search(r'yoy(?:cpi)?\s+(?:be\s+)?above\s+(-?[\d.]+)%', question)

    if not mom_match or not yoy_match:
        return None

    mom_threshold = float(mom_match.group(1))
    yoy_threshold = float(yoy_match.group(1))

    # Fetch headline CPI index (14 months for MoM + YoY)
    headline_obs = await _fetch_fred_series("CPIAUCSL", limit=14)

    ctx = f"\n  COMBO THRESHOLD ANALYSIS:\n"

    if headline_obs:
        # MoM from index
        pair = _find_valid_pair(headline_obs)
        if pair:
            curr, prev, as_of = pair
            mom_val = round(((curr - prev) / prev) * 100, 3)
            mom_gap = mom_val - mom_threshold
            ctx += f"    MoM CPI threshold: {mom_threshold}% | Latest MoM: {mom_val}% | Gap: {mom_gap:+.3f}pp\n"

        # YoY from index (need 13+ months)
        valid = [(o, _safe_float(o.get("value", ""))) for o in headline_obs]
        valid = [(o, v) for o, v in valid if v is not None]
        if len(valid) >= 13:
            yoy_val = round(((valid[0][1] / valid[12][1]) - 1) * 100, 2)
            yoy_gap = yoy_val - yoy_threshold
            ctx += f"    YoY CPI threshold: {yoy_threshold}% | Latest YoY: {yoy_val}% | Gap: {yoy_gap:+.2f}pp\n"

    ctx += "    COMBO requires BOTH conditions to be true. Estimate each independently, then multiply.\n"
    return ctx


def _parse_cpi_threshold(question: str) -> Optional[float]:
    """Extract the numeric threshold from a CPI market question.

    Handles: "above 2.3%", "rise more than 0.3%", "above -0.1%"
    """
    patterns = [
        r'above\s+(-?[\d.]+)%',
        r'more than\s+(-?[\d.]+)%',
        r'exceed\s+(-?[\d.]+)%',
        r'greater than\s+(-?[\d.]+)%',
        r'over\s+(-?[\d.]+)%',
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


async def _get_polling_context(question: str) -> Optional[str]:
    """Get aggregate polling data for political markets.

    Uses Google News RSS for latest polling information since
    direct polling APIs require authentication.
    """
    try:
        q = question.lower()

        # Extract the subject of polling
        subjects = []
        for name in ["trump", "biden", "harris", "desantis", "newsom",
                     "approval", "favorability"]:
            if name in q:
                subjects.append(name)

        if not subjects:
            return None

        search_query = " ".join(subjects) + " poll latest"

        async with httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": "Mozilla/5.0"},
            follow_redirects=True,
        ) as client:
            from urllib.parse import quote_plus
            url = f"https://news.google.com/rss/search?q={quote_plus(search_query)}&hl=en-US&gl=US&ceid=US:en"
            resp = await client.get(url)
            if resp.status_code == 200:
                xml = resp.text
                items = re.findall(r"<title>(.*?)</title>", xml, re.DOTALL)
                # Skip the feed title (first item)
                headlines = [h.strip() for h in items[1:4] if h.strip()]

                if headlines:
                    context = (
                        f"\nSTRUCTURED DATA ANCHOR — Recent Polling Headlines:\n"
                    )
                    for i, h in enumerate(headlines, 1):
                        context += f"  {i}. {h}\n"
                    context += (
                        f"  Note: Candidates leading in polls 30 days out win ~75% of the time.\n"
                        f"  Use polling data as your starting anchor for political markets.\n"
                    )
                    return context

    except Exception as e:
        logger.debug("polling_fetch_error", error=str(e))

    return None


# ---------------------------------------------------------------------------
# NOAA Weather — NWS forecast data (free, no API key needed)
# ---------------------------------------------------------------------------

# ~35 major US cities → (lat, lon)
_WEATHER_CITIES: Dict[str, Tuple[float, float]] = {
    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "houston": (29.7604, -95.3698),
    "phoenix": (33.4484, -112.0740),
    "philadelphia": (39.9526, -75.1652),
    "san antonio": (29.4241, -98.4936),
    "san diego": (32.7157, -117.1611),
    "dallas": (32.7767, -96.7970),
    "san jose": (37.3382, -121.8863),
    "austin": (30.2672, -97.7431),
    "jacksonville": (30.3322, -81.6557),
    "fort worth": (32.7555, -97.3308),
    "columbus": (39.9612, -82.9988),
    "charlotte": (35.2271, -80.8431),
    "indianapolis": (39.7684, -86.1581),
    "san francisco": (37.7749, -122.4194),
    "seattle": (47.6062, -122.3321),
    "denver": (39.7392, -104.9903),
    "washington": (38.9072, -77.0369),
    "nashville": (36.1627, -86.7816),
    "oklahoma city": (35.4676, -97.5164),
    "el paso": (31.7619, -106.4850),
    "boston": (42.3601, -71.0589),
    "portland": (45.5051, -122.6750),
    "las vegas": (36.1699, -115.1398),
    "memphis": (35.1495, -90.0490),
    "louisville": (38.2527, -85.7585),
    "baltimore": (39.2904, -76.6122),
    "milwaukee": (43.0389, -87.9065),
    "albuquerque": (35.0844, -106.6504),
    "tucson": (32.2226, -110.9747),
    "fresno": (36.7378, -119.7871),
    "miami": (25.7617, -80.1918),
    "atlanta": (33.7490, -84.3880),
    "detroit": (42.3314, -83.0458),
    "minneapolis": (44.9778, -93.2650),
    "tampa": (27.9506, -82.4572),
    "new orleans": (29.9511, -90.0715),
    "cleveland": (41.4993, -81.6944),
    "kansas city": (39.0997, -94.5786),
    "st. louis": (38.6270, -90.1994),
    "pittsburgh": (40.4406, -79.9959),
    "cincinnati": (39.1031, -84.5120),
    "raleigh": (35.7796, -78.6382),
    "salt lake city": (40.7608, -111.8910),
}

# Cache for NWS gridpoint URLs (permanent — grid doesn't change)
_gridpoint_cache: Dict[str, str] = {}

# Cache for forecast data (1 hour TTL)
_forecast_cache: Dict[str, Tuple[float, Dict]] = {}
_FORECAST_CACHE_TTL = 3600.0  # 1 hour


def _parse_city(question: str) -> Optional[str]:
    """Extract city name from market question using longest-match."""
    q = question.lower()
    best_match = None
    best_len = 0
    for city in _WEATHER_CITIES:
        if city in q and len(city) > best_len:
            best_match = city
            best_len = len(city)
    return best_match


def _parse_target_date(question: str) -> Optional[datetime]:
    """Extract target date from market question.

    Handles: "February 15", "Feb 15", "tomorrow", "today", "2/15"
    """
    q = question.lower()
    now = datetime.now(timezone.utc)

    if "today" in q:
        return now
    if "tomorrow" in q:
        return now + timedelta(days=1)

    # "February 15" / "Feb 15" style
    months = {
        "january": 1, "jan": 1, "february": 2, "feb": 2,
        "march": 3, "mar": 3, "april": 4, "apr": 4,
        "may": 5, "june": 6, "jun": 6,
        "july": 7, "jul": 7, "august": 8, "aug": 8,
        "september": 9, "sep": 9, "october": 10, "oct": 10,
        "november": 11, "nov": 11, "december": 12, "dec": 12,
    }
    for month_name, month_num in months.items():
        pattern = rf"\b{month_name}\s+(\d{{1,2}})\b"
        match = re.search(pattern, q)
        if match:
            day = int(match.group(1))
            year = now.year
            try:
                target = datetime(year, month_num, day, tzinfo=timezone.utc)
                # If date is in the past, use next year
                if target < now - timedelta(days=1):
                    target = datetime(year + 1, month_num, day, tzinfo=timezone.utc)
                return target
            except ValueError:
                continue

    # "2/15" style
    match = re.search(r"\b(\d{1,2})/(\d{1,2})\b", q)
    if match:
        month_num = int(match.group(1))
        day = int(match.group(2))
        if 1 <= month_num <= 12 and 1 <= day <= 31:
            year = now.year
            try:
                target = datetime(year, month_num, day, tzinfo=timezone.utc)
                if target < now - timedelta(days=1):
                    target = datetime(year + 1, month_num, day, tzinfo=timezone.utc)
                return target
            except ValueError:
                pass

    return None


async def _get_nws_gridpoint_url(lat: float, lon: float) -> Optional[str]:
    """Step 1: Get NWS gridpoint forecast URL from coordinates. Cached permanently."""
    cache_key = f"{lat:.4f},{lon:.4f}"
    if cache_key in _gridpoint_cache:
        return _gridpoint_cache[cache_key]

    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": "(Morpheus Trading Bot, contact@example.com)"},
        ) as client:
            resp = await client.get(f"https://api.weather.gov/points/{lat},{lon}")
            if resp.status_code == 200:
                data = resp.json()
                forecast_url = data.get("properties", {}).get("forecast")
                if forecast_url:
                    _gridpoint_cache[cache_key] = forecast_url
                    return forecast_url
    except Exception as e:
        logger.debug("nws_gridpoint_error", error=str(e), lat=lat, lon=lon)

    return None


async def _get_nws_forecast(forecast_url: str) -> Optional[Dict]:
    """Step 2: Fetch forecast periods from NWS. Cached 1 hour."""
    now = time.monotonic()
    if forecast_url in _forecast_cache:
        cached_time, cached_data = _forecast_cache[forecast_url]
        if now - cached_time < _FORECAST_CACHE_TTL:
            return cached_data

    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": "(Morpheus Trading Bot, contact@example.com)"},
        ) as client:
            resp = await client.get(forecast_url)
            if resp.status_code == 200:
                data = resp.json()
                _forecast_cache[forecast_url] = (now, data)
                return data
    except Exception as e:
        logger.debug("nws_forecast_error", error=str(e), url=forecast_url)

    return None


def _format_weather_anchor(
    city: str,
    periods: List[Dict],
    target_date: Optional[datetime],
) -> str:
    """Format NWS forecast periods into a structured anchor string."""
    context = f"\nSTRUCTURED DATA ANCHOR — NOAA/NWS Forecast for {city.title()}:\n"

    # If we have a target date, try to find matching periods
    relevant = []
    if target_date:
        target_str = target_date.strftime("%Y-%m-%d")
        for period in periods:
            start = period.get("startTime", "")
            if target_str in start:
                relevant.append(period)

    # Fall back to first few periods if no target match
    if not relevant:
        relevant = periods[:4]

    for period in relevant[:4]:
        name = period.get("name", "?")
        temp = period.get("temperature", "?")
        temp_unit = period.get("temperatureUnit", "F")
        wind_speed = period.get("windSpeed", "?")
        wind_dir = period.get("windDirection", "")
        precip_pct = period.get("probabilityOfPrecipitation", {})
        precip_val = precip_pct.get("value") if isinstance(precip_pct, dict) else None
        short_forecast = period.get("shortForecast", "")

        line = f"  {name}: {temp}{temp_unit}"
        if precip_val is not None:
            line += f", precip {precip_val}%"
        if wind_speed:
            line += f", wind {wind_speed} {wind_dir}"
        if short_forecast:
            line += f" — {short_forecast}"
        context += line + "\n"

    context += (
        "  Source: NOAA National Weather Service (official US government forecast)\n"
        "  Use this forecast data as your starting anchor for weather markets.\n"
    )
    return context


async def _get_weather_context(question: str) -> Optional[str]:
    """Get NOAA NWS forecast data for weather market questions."""
    city = _parse_city(question)
    if not city:
        logger.debug("weather_no_city_match", question=question[:80])
        return None

    coords = _WEATHER_CITIES[city]
    target_date = _parse_target_date(question)

    # Step 1: Get gridpoint URL
    forecast_url = await _get_nws_gridpoint_url(coords[0], coords[1])
    if not forecast_url:
        return None

    # Step 2: Fetch forecast
    forecast_data = await _get_nws_forecast(forecast_url)
    if not forecast_data:
        return None

    periods = forecast_data.get("properties", {}).get("periods", [])
    if not periods:
        return None

    anchor = _format_weather_anchor(city, periods, target_date)

    logger.info(
        "noaa_weather_anchor",
        city=city,
        target_date=target_date.strftime("%Y-%m-%d") if target_date else None,
        periods_found=len(periods),
    )

    return anchor
