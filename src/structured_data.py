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

import asyncio
import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import math

import httpx
import structlog

def _get_fred_api_key() -> str:
    """Lazy FRED key lookup — load_dotenv() may not have run at import time."""
    return os.getenv("FRED_API_KEY", "")

logger = structlog.get_logger()


async def get_structured_anchor(
    question: str, category: str = "", market_id: str = ""
) -> Optional[str]:
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
    if any(w in q for w in ["temperature", "high temp", "low temp",
                             "degrees fahrenheit", "degrees celsius",
                             "rain", "rainfall", "inches of rain",
                             "precipitation", "snowfall", "inches of snow",
                             "wind speed", "heat wave", "cold snap"]):
        return await _get_weather_context(question, market_id)

    # Stock index markets — real-time price data
    if market_id:
        mid_upper = market_id.upper()
        if any(mid_upper.startswith(p) for p in ("KXINXU", "KXINX-", "KXNASDAQ100", "KXBTCD", "KXBTC")):
            return await _get_stock_index_context(question, market_id)

    return None


async def _get_fedwatch_context() -> Optional[str]:
    """Fetch CME FedWatch-style data from public sources.

    Uses the CME website's public data or financial news for current
    Fed rate probabilities implied by futures markets.
    """
    if not _get_fred_api_key():
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
                    "api_key": _get_fred_api_key(),
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
    if not _get_fred_api_key():
        return []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": _get_fred_api_key(),
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
    if not _get_fred_api_key():
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

# Aliases for city name matching (Kalshi uses abbreviations in titles)
_CITY_ALIASES: Dict[str, str] = {
    "nyc": "new york", "ny": "new york",
    "la": "los angeles", "lax": "los angeles",
    "chi": "chicago",
    "phx": "phoenix",
    "philly": "philadelphia",
    "sf": "san francisco", "sfo": "san francisco",
    "dc": "washington", "d.c.": "washington",
    "nola": "new orleans",
    "lv": "las vegas", "vegas": "las vegas",
    "atl": "atlanta",
    "slc": "salt lake city",
    "kc": "kansas city",
    "stl": "st. louis", "st louis": "st. louis",
    "okc": "oklahoma city",
    "jax": "jacksonville",
    "min": "minneapolis",
    "aus": "austin",
    "den": "denver",
    "sea": "seattle",
    "bos": "boston",
    "mia": "miami",
}

# ICAO station codes for METAR/ASOS observations per city
# Multiple stations per metro area for multi-station averaging
_CITY_STATIONS: Dict[str, List[str]] = {
    "new york": ["KJFK", "KLGA", "KEWR"],
    "los angeles": ["KLAX", "KBUR", "KSNA"],
    "chicago": ["KORD", "KMDW"],
    "houston": ["KIAH", "KHOU"],
    "phoenix": ["KPHX", "KDVT"],
    "philadelphia": ["KPHL"],
    "san antonio": ["KSAT"],
    "san diego": ["KSAN"],
    "dallas": ["KDFW", "KDAL"],
    "san jose": ["KSJC"],
    "austin": ["KAUS"],
    "jacksonville": ["KJAX"],
    "fort worth": ["KDFW"],
    "columbus": ["KCMH"],
    "charlotte": ["KCLT"],
    "indianapolis": ["KIND"],
    "san francisco": ["KSFO", "KOAK"],
    "seattle": ["KSEA"],
    "denver": ["KDEN", "KAPA"],
    "washington": ["KDCA", "KIAD"],
    "nashville": ["KBNA"],
    "oklahoma city": ["KOKC"],
    "el paso": ["KELP"],
    "boston": ["KBOS"],
    "portland": ["KPDX"],
    "las vegas": ["KLAS"],
    "memphis": ["KMEM"],
    "louisville": ["KSDF"],
    "baltimore": ["KBWI"],
    "milwaukee": ["KMKE"],
    "albuquerque": ["KABQ"],
    "tucson": ["KTUS"],
    "fresno": ["KFAT"],
    "miami": ["KMIA", "KFLL"],
    "atlanta": ["KATL"],
    "detroit": ["KDTW"],
    "minneapolis": ["KMSP"],
    "tampa": ["KTPA"],
    "new orleans": ["KMSY"],
    "cleveland": ["KCLE"],
    "kansas city": ["KMCI"],
    "st. louis": ["KSTL"],
    "pittsburgh": ["KPIT"],
    "cincinnati": ["KCVG"],
    "raleigh": ["KRDU"],
    "salt lake city": ["KSLC"],
}

# City-specific sigma (forecast error std dev in °F) for hourly forecasts.
# Coastal cities have tighter forecasts; inland/desert cities are more variable.
_CITY_SIGMA_HOURLY: Dict[str, float] = {
    # Wave 21: All values increased 30% — NOAA direct was 16.7% WR (-$3.82)
    # because overconfident sigma led to extreme CDF probs that passed z-gate
    # Coastal — marine layer stabilizes temps
    "san francisco": 1.3, "san diego": 1.3, "los angeles": 1.6,
    "miami": 1.6, "tampa": 1.6, "seattle": 1.6, "portland": 1.6,
    "boston": 1.7, "new york": 1.7,
    # Moderate — humid subtropical or maritime-influenced
    "houston": 1.7, "new orleans": 1.7, "jacksonville": 1.7,
    "atlanta": 1.8, "charlotte": 1.8, "raleigh": 1.8,
    "philadelphia": 1.8, "washington": 1.8, "baltimore": 1.8,
    "nashville": 1.8, "memphis": 1.8, "louisville": 1.8,
    "cleveland": 1.8, "pittsburgh": 1.8, "detroit": 1.8,
    "chicago": 2.0, "milwaukee": 2.0, "indianapolis": 2.0,
    "columbus": 2.0, "cincinnati": 2.0, "st. louis": 2.0,
    "kansas city": 2.0, "fort worth": 2.0,
    # High variability — inland/desert/elevation
    "dallas": 2.1, "austin": 2.1, "san antonio": 2.1,
    "oklahoma city": 2.3, "minneapolis": 2.3,
    "denver": 2.9, "salt lake city": 2.6, "albuquerque": 2.6,
    "el paso": 2.6, "las vegas": 2.6, "phoenix": 2.9,
    "tucson": 2.6, "fresno": 2.3,
    "san jose": 1.7,
}
_DEFAULT_SIGMA_HOURLY = 2.0  # fallback for unlisted cities (was 1.5)


# ---------------------------------------------------------------------------
# Wave 25: Weather bias tracking & adaptive sigma
# ---------------------------------------------------------------------------

def _weather_bias_path(state_dir: str = "state") -> str:
    return f"{state_dir}/weather_bias.json"


def _weather_sigma_path(state_dir: str = "state") -> str:
    return f"{state_dir}/weather_sigma.json"


def _load_weather_json(path: str) -> Dict:
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_weather_json(path: str, data: Dict) -> None:
    import json
    from pathlib import Path
    Path(path).write_text(json.dumps(data, indent=2))


def update_weather_bias(
    city: str, t_type: str, nws_forecast: float, actual: float,
    state_dir: str = "state",
) -> None:
    """Track rolling NWS forecast bias per city/type.

    Stores exponential moving average of (forecast - actual) errors.
    Positive bias = NWS forecasts too high; negative = too low.
    """
    key = f"{city.lower()}|{t_type.lower()}"
    path = _weather_bias_path(state_dir)
    data = _load_weather_json(path)

    entry = data.get(key, {"bias": 0.0, "count": 0})
    error = nws_forecast - actual
    count = entry["count"] + 1
    alpha = 0.3  # EMA smoothing
    if count == 1:
        new_bias = error
    else:
        new_bias = alpha * error + (1.0 - alpha) * entry["bias"]

    data[key] = {"bias": round(new_bias, 2), "count": count}
    _save_weather_json(path, data)


def get_weather_bias(city: str, t_type: str, state_dir: str = "state") -> Optional[float]:
    """Get current NWS bias correction for city/type.

    Returns bias correction (subtract from forecast) when ≥5 samples
    AND |bias| ≥ 1.0°F. Returns None otherwise.
    """
    key = f"{city.lower()}|{t_type.lower()}"
    data = _load_weather_json(_weather_bias_path(state_dir))
    entry = data.get(key)
    if entry is None:
        return None
    if entry["count"] >= 5 and abs(entry["bias"]) >= 1.0:
        return entry["bias"]
    return None


def update_weather_sigma(
    city: str, t_type: str, abs_error: float,
    state_dir: str = "state",
) -> None:
    """Track adaptive sigma per city/type using EMA of absolute forecast errors."""
    key = f"{city.lower()}|{t_type.lower()}"
    path = _weather_sigma_path(state_dir)
    data = _load_weather_json(path)

    entry = data.get(key, {"sigma": None, "count": 0})
    count = entry["count"] + 1
    alpha = 0.3
    if entry["sigma"] is None or count == 1:
        new_sigma = abs_error
    else:
        new_sigma = alpha * abs_error + (1.0 - alpha) * entry["sigma"]

    data[key] = {"sigma": round(new_sigma, 2), "count": count}
    _save_weather_json(path, data)


def get_adaptive_sigma(city: str, t_type: str, state_dir: str = "state") -> Optional[float]:
    """Get adaptive sigma for city/type when ≥10 samples available."""
    key = f"{city.lower()}|{t_type.lower()}"
    data = _load_weather_json(_weather_sigma_path(state_dir))
    entry = data.get(key)
    if entry is None:
        return None
    if entry["count"] >= 10 and entry["sigma"] is not None:
        return entry["sigma"]
    return None


# Cache for NWS gridpoint URLs (permanent — grid doesn't change)
_gridpoint_cache: Dict[str, str] = {}

# Cache for forecast data (10 min TTL — tighter for same-day accuracy)
_forecast_cache: Dict[str, Tuple[float, Dict]] = {}
_FORECAST_CACHE_TTL = 600.0  # 10 min — catch NOAA forecast updates faster

# Cache for NWS forecast updateTime (ISO string) — keyed by forecast_url
_forecast_update_times: Dict[str, str] = {}

# Cache for METAR observations (5 min TTL — observations update hourly but we want freshness)
_observation_cache: Dict[str, Tuple[float, Dict]] = {}
_OBSERVATION_CACHE_TTL = 300.0  # 5 min

# Cache for Open-Meteo GFS ensemble data (15 min TTL)
_ensemble_cache: Dict[str, Tuple[float, Dict]] = {}
_ENSEMBLE_CACHE_TTL = 900.0  # 15 min


async def _open_meteo_get_with_retry(
    client: httpx.AsyncClient, url: str, params: dict, max_retries: int = 3,
) -> Optional[httpx.Response]:
    """GET with exponential backoff on 429 (rate limit).

    Open-Meteo free tier has ~10K req/day; bursts of weather market
    evaluation can trigger per-minute limits.
    """
    for attempt in range(max_retries):
        resp = await client.get(url, params=params)
        if resp.status_code != 429:
            return resp
        wait = 2.0 * (2 ** attempt)  # 2s, 4s, 8s
        logger.info("open_meteo_rate_limited", attempt=attempt + 1, wait=wait, url=url)
        await asyncio.sleep(wait)
    return resp  # return last 429 response so caller can handle


async def _fetch_open_meteo_ensemble(
    lat: float, lon: float, target_date: str,
) -> Optional[Dict[str, List[float]]]:
    """Fetch multi-model ensemble from Open-Meteo (GFS + GEM + ICON).

    Returns dict with:
      temperature_max: List[float]  — combined ensemble high temps (°F)
      temperature_min: List[float]  — combined ensemble low temps (°F)

    GFS: 31 members, GEM: 21 members, ICON: 40 members = ~92 total.
    Free API, no key required. Rate limit: ~10K/day.
    """
    cache_key = f"{lat:.2f},{lon:.2f}:{target_date}"
    cached = _ensemble_cache.get(cache_key)
    if cached:
        ts, data = cached
        if time.monotonic() - ts < _ENSEMBLE_CACHE_TTL:
            return data

    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "models": "gfs_seamless,gem_global,icon_seamless",
        "start_date": target_date,
        "end_date": target_date,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await _open_meteo_get_with_retry(client, url, params)
            if resp is None or resp.status_code != 200:
                logger.debug("open_meteo_http_error", status=getattr(resp, "status_code", None))
                return None
            data = resp.json()

        # Parse ensemble members from response.
        # Multi-model responses use per-member keys like:
        #   temperature_2m_max_ncep_gefs_seamless (control)
        #   temperature_2m_max_member01_ncep_gefs_seamless
        #   temperature_2m_max_member01_gem_global  etc.
        # Single-model uses simple: temperature_2m_max
        daily = data.get("daily", {})

        # Collect all max/min values across all model/member keys
        max_temps: list = []
        min_temps: list = []
        for key, vals in daily.items():
            if not isinstance(vals, list):
                continue
            if "temperature_2m_max" in key:
                max_temps.extend(v for v in vals if v is not None)
            elif "temperature_2m_min" in key:
                min_temps.extend(v for v in vals if v is not None)

        if not max_temps and not min_temps:
            logger.debug("open_meteo_no_data", response_keys=list(daily.keys())[:10])
            return None

        result = {
            "temperature_max": [float(t) for t in max_temps],
            "temperature_min": [float(t) for t in min_temps],
        }

        _ensemble_cache[cache_key] = (time.monotonic(), result)
        logger.info(
            "open_meteo_ensemble_fetched",
            lat=lat, lon=lon, date=target_date,
            n_max=len(result["temperature_max"]),
            n_min=len(result["temperature_min"]),
        )
        return result

    except Exception as e:
        logger.debug("open_meteo_fetch_error", error=str(e))
        return None


# Cache for ECMWF IFS ensemble data (15 min TTL)
_ecmwf_cache: Dict[str, Tuple[float, Dict]] = {}
_ECMWF_CACHE_TTL = 900.0  # 15 min


async def _fetch_ecmwf_ensemble(
    lat: float, lon: float, target_date: str,
) -> Optional[Dict[str, List[float]]]:
    """Fetch ECMWF IFS ensemble from Open-Meteo (51 members).

    Returns dict with:
      temperature_max: List[float]  — ECMWF ensemble high temps (°F)
      temperature_min: List[float]  — ECMWF ensemble low temps (°F)

    ECMWF IFS: 51 members (1 control + 50 perturbed).
    Free API, no key required. Same endpoint as GFS ensemble.
    """
    cache_key = f"ecmwf:{lat:.2f},{lon:.2f}:{target_date}"
    cached = _ecmwf_cache.get(cache_key)
    if cached:
        ts, data = cached
        if time.monotonic() - ts < _ECMWF_CACHE_TTL:
            return data

    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "models": "ecmwf_ifs",
        "start_date": target_date,
        "end_date": target_date,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await _open_meteo_get_with_retry(client, url, params)
            if resp is None or resp.status_code != 200:
                logger.debug("ecmwf_http_error", status=getattr(resp, "status_code", None))
                return None
            data = resp.json()

        # Parse ECMWF ensemble members from response.
        # Keys like: temperature_2m_max_ecmwf_ifs (control),
        #            temperature_2m_max_member01_ecmwf_ifs, etc.
        daily = data.get("daily", {})

        max_temps: list = []
        min_temps: list = []
        for key, vals in daily.items():
            if not isinstance(vals, list):
                continue
            if "temperature_2m_max" in key:
                max_temps.extend(v for v in vals if v is not None)
            elif "temperature_2m_min" in key:
                min_temps.extend(v for v in vals if v is not None)

        if not max_temps and not min_temps:
            logger.debug("ecmwf_no_data", response_keys=list(daily.keys())[:10])
            return None

        result = {
            "temperature_max": [float(t) for t in max_temps],
            "temperature_min": [float(t) for t in min_temps],
        }

        _ecmwf_cache[cache_key] = (time.monotonic(), result)
        logger.info(
            "ecmwf_ensemble_fetched",
            lat=lat, lon=lon, date=target_date,
            n_max=len(result["temperature_max"]),
            n_min=len(result["temperature_min"]),
        )
        return result

    except Exception as e:
        logger.debug("ecmwf_fetch_error", error=str(e))
        return None


# Cache for HRRR forecasts (10 min TTL — updates hourly)
_hrrr_cache: Dict[str, Tuple[float, Dict]] = {}
_HRRR_CACHE_TTL = 600.0  # 10 min


async def _fetch_open_meteo_hrrr(
    lat: float, lon: float, target_date: str,
) -> Optional[Dict[str, float]]:
    """Fetch HRRR hourly temps from Open-Meteo forecast API.

    HRRR is 3km resolution, US-only, 48h horizon.
    Returns {"temperature_max": float, "temperature_min": float} computed
    from hourly temps for the target date.
    """
    cache_key = f"hrrr:{lat:.2f},{lon:.2f}:{target_date}"
    cached = _hrrr_cache.get(cache_key)
    if cached:
        ts, data = cached
        if time.monotonic() - ts < _HRRR_CACHE_TTL:
            return data

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "models": "gfs_hrrr",
        "start_date": target_date,
        "end_date": target_date,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await _open_meteo_get_with_retry(client, url, params)
            if resp is None or resp.status_code != 200:
                logger.debug("hrrr_http_error", status=getattr(resp, "status_code", None))
                return None
            data = resp.json()

        hourly = data.get("hourly", {})
        temps = hourly.get("temperature_2m", [])
        temps = [float(t) for t in temps if t is not None]

        if len(temps) < 12:
            logger.debug("hrrr_insufficient_data", n_hours=len(temps))
            return None

        result = {
            "temperature_max": max(temps),
            "temperature_min": min(temps),
        }

        _hrrr_cache[cache_key] = (time.monotonic(), result)
        logger.info(
            "hrrr_fetched",
            lat=lat, lon=lon, date=target_date,
            t_max=round(result["temperature_max"], 1),
            t_min=round(result["temperature_min"], 1),
            n_hours=len(temps),
        )
        return result

    except Exception as e:
        logger.debug("hrrr_fetch_error", error=str(e))
        return None


# Cache for GraphCast (AIGFS) forecasts (15 min TTL)
_graphcast_cache: Dict[str, Tuple[float, Dict]] = {}
_GRAPHCAST_CACHE_TTL = 900.0  # 15 min


async def _fetch_graphcast_forecast(
    lat: float, lon: float, target_date: str,
) -> Optional[Dict[str, float]]:
    """Fetch AIGFS (GraphCast) forecast from Open-Meteo.

    GraphCast is Google DeepMind's ML weather model, available via Open-Meteo
    as gfs_graphcast025 (0.25° resolution). Free REST API, no auth needed.

    Returns {"temperature_max": float, "temperature_min": float} computed
    from hourly temps for the target date, or None on failure.

    GraphCast adds 18-24h skill extension beyond traditional NWP models,
    making it a valuable 5th source for weather probability blending.
    """
    cache_key = f"graphcast:{lat:.2f},{lon:.2f}:{target_date}"
    cached = _graphcast_cache.get(cache_key)
    if cached:
        ts, data = cached
        if time.monotonic() - ts < _GRAPHCAST_CACHE_TTL:
            return data

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "models": "gfs_graphcast025",
        "start_date": target_date,
        "end_date": target_date,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await _open_meteo_get_with_retry(client, url, params)
            if resp is None or resp.status_code != 200:
                logger.debug("graphcast_http_error", status=getattr(resp, "status_code", None))
                return None
            data = resp.json()

        hourly = data.get("hourly", {})
        temps = hourly.get("temperature_2m", [])
        temps = [float(t) for t in temps if t is not None]

        if len(temps) < 12:
            logger.debug("graphcast_insufficient_data", n_hours=len(temps))
            return None

        result = {
            "temperature_max": max(temps),
            "temperature_min": min(temps),
        }

        _graphcast_cache[cache_key] = (time.monotonic(), result)
        logger.info(
            "graphcast_fetched",
            lat=lat, lon=lon, date=target_date,
            t_max=round(result["temperature_max"], 1),
            t_min=round(result["temperature_min"], 1),
            n_hours=len(temps),
        )
        return result

    except Exception as e:
        logger.debug("graphcast_fetch_error", error=str(e))
        return None


# ---------------------------------------------------------------------------
# NBM validation — NWS hourly forecast (NBM-calibrated) as independent check
# ---------------------------------------------------------------------------

_nbm_cache: Dict[str, Tuple[float, Dict]] = {}  # gridpoint_url -> (ts, data)
_NBM_CACHE_TTL = 600.0  # 10 minutes
_griddata_url_cache: Dict[str, str] = {}  # lat,lon -> forecastGridData URL


async def _get_nws_griddata_url(lat: float, lon: float) -> Optional[str]:
    """Get the NWS forecastGridData URL for NBM quantitative data."""
    cache_key = f"{lat:.4f},{lon:.4f}"
    if cache_key in _griddata_url_cache:
        return _griddata_url_cache[cache_key]

    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": "(Morpheus Trading Bot, contact@example.com)"},
        ) as client:
            resp = await client.get(f"https://api.weather.gov/points/{lat},{lon}")
            if resp.status_code == 200:
                data = resp.json()
                url = data.get("properties", {}).get("forecastGridData")
                if url:
                    _griddata_url_cache[cache_key] = url
                    return url
    except Exception as e:
        logger.debug("nws_griddata_url_error", error=str(e))

    return None


async def _fetch_nbm_temperature(
    lat: float, lon: float, target_date: str, t_type: str,
) -> Optional[float]:
    """Fetch NBM-calibrated temperature from NWS gridpoints quantitative data.

    The NWS gridpoints endpoint returns the raw NBM-derived quantitative forecast.
    Returns the max or min temperature for the target date, or None on failure.
    """
    griddata_url = await _get_nws_griddata_url(lat, lon)
    if not griddata_url:
        return None

    now = time.monotonic()
    cached = _nbm_cache.get(griddata_url)
    if cached:
        ts, data = cached
        if now - ts < _NBM_CACHE_TTL:
            return _extract_nbm_temp(data, target_date, t_type)

    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            headers={"User-Agent": "(Morpheus Trading Bot, contact@example.com)"},
        ) as client:
            resp = await client.get(griddata_url)
            if resp.status_code != 200:
                logger.debug("nbm_griddata_error", status=resp.status_code)
                return None
            data = resp.json()
            _nbm_cache[griddata_url] = (now, data)
            return _extract_nbm_temp(data, target_date, t_type)
    except Exception as e:
        logger.debug("nbm_fetch_error", error=str(e))
        return None


def _extract_nbm_temp(data: Dict, target_date: str, t_type: str) -> Optional[float]:
    """Extract max or min temperature from NWS gridpoints data for a target date.

    NWS gridpoints data uses ISO 8601 durations (e.g., "2026-02-15T12:00:00+00:00/PT6H").
    Temperatures are in Celsius — convert to Fahrenheit.
    """
    props = data.get("properties", {})

    # Choose the right temperature field
    if "high" in t_type or "above" in t_type or t_type.endswith("_above"):
        field = "maxTemperature"
    elif "low" in t_type or "below" in t_type:
        field = "minTemperature"
    else:
        field = "maxTemperature"

    temp_data = props.get(field, {})
    values = temp_data.get("values", [])
    uom = temp_data.get("uom", "")

    is_celsius = "degC" in uom or "celsius" in uom.lower()

    for entry in values:
        valid_time = entry.get("validTime", "")
        if target_date in valid_time:
            val = entry.get("value")
            if val is not None:
                temp_f = val * 9.0 / 5.0 + 32.0 if is_celsius else val
                return temp_f

    return None


async def compute_nbm_validation(
    p_blended: float,
    t_value: float,
    t_type: str,
    sigma: float,
    coords: Tuple[float, float],
    target_date_str: str,
    bracket_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[float, Optional[str]]:
    """Validate blended probability against NBM-calibrated forecast.

    If NBM disagrees with our blend by >15%, adjust toward NBM estimate.
    Returns (adjusted_p_yes, nbm_note) where nbm_note explains any adjustment.
    """
    nbm_temp = await _fetch_nbm_temperature(
        coords[0], coords[1], target_date_str, t_type,
    )
    if nbm_temp is None:
        return (p_blended, None)

    # Compute NBM-based probability
    if bracket_bounds:
        lo, hi = bracket_bounds
        p_nbm = _norm_cdf(hi, nbm_temp, sigma) - _norm_cdf(lo, nbm_temp, sigma)
    elif "below" in t_type:
        p_nbm = _norm_cdf(t_value, nbm_temp, sigma)
    else:
        p_nbm = 1.0 - _norm_cdf(t_value, nbm_temp, sigma)

    p_nbm = max(0.02, min(0.98, p_nbm))
    disagreement = abs(p_blended - p_nbm)

    if disagreement > 0.15:
        # Significant disagreement: blend 60% ours + 40% NBM
        adjusted = 0.60 * p_blended + 0.40 * p_nbm
        note = (
            f"NBM validation: nbm_temp={nbm_temp:.1f}°F, p_nbm={p_nbm:.3f}, "
            f"disagreement={disagreement:.3f}, adjusted {p_blended:.3f}→{adjusted:.3f}"
        )
        logger.info(
            "nbm_validation_adjustment",
            nbm_temp=round(nbm_temp, 1),
            p_nbm=round(p_nbm, 4),
            p_blended=round(p_blended, 4),
            p_adjusted=round(adjusted, 4),
            disagreement=round(disagreement, 4),
        )
        return (adjusted, note)
    else:
        logger.debug(
            "nbm_validation_ok",
            nbm_temp=round(nbm_temp, 1),
            p_nbm=round(p_nbm, 4),
            p_blended=round(p_blended, 4),
            disagreement=round(disagreement, 4),
        )
        return (p_blended, None)


def _ensemble_probability(
    members: List[float], threshold: float, direction: str,
    bracket_bounds: Optional[Tuple[float, float]] = None,
) -> Optional[float]:
    """Compute probability from ensemble members.

    Args:
        members: List of ensemble member forecast values.
        threshold: Temperature threshold.
        direction: "above", "below", or "bracket".
        bracket_bounds: (lower, upper) for bracket markets.

    Returns:
        Probability estimate (0-1) or None if insufficient data.
    """
    if not members or len(members) < 5:
        return None

    n = len(members)
    if direction == "above":
        count = sum(1 for m in members if m >= threshold)
    elif direction == "below":
        count = sum(1 for m in members if m < threshold)
    elif direction == "bracket" and bracket_bounds:
        lo, hi = bracket_bounds
        count = sum(1 for m in members if lo <= m < hi)
    else:
        return None

    return count / n


def _parse_city(question: str) -> Optional[str]:
    """Extract city name from market question using longest-match.

    Checks both canonical city names and common aliases/abbreviations.
    Uses word-boundary matching for short aliases to avoid false positives.
    """
    q = question.lower()
    best_match = None
    best_len = 0

    # Check canonical city names (longest match wins)
    for city in _WEATHER_CITIES:
        if city in q and len(city) > best_len:
            best_match = city
            best_len = len(city)

    # If no canonical match, try aliases with word-boundary check
    if best_match is None:
        for alias, canonical in _CITY_ALIASES.items():
            # Word-boundary match to avoid "la" matching "plan" etc.
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, q):
                if len(alias) > best_len:
                    best_match = canonical
                    best_len = len(alias)

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
    """Step 2: Fetch forecast periods from NWS. Cached 20 min."""
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
                # Track forecast updateTime for staleness check
                update_time = data.get("properties", {}).get("updateTime")
                if update_time:
                    _forecast_update_times[forecast_url] = update_time
                return data
    except Exception as e:
        logger.debug("nws_forecast_error", error=str(e), url=forecast_url)

    return None


# Cache for hourly forecast data (20 min TTL, separate from 12h forecast)
_hourly_forecast_cache: Dict[str, Tuple[float, List[Dict]]] = {}


async def _get_nws_hourly_forecast(forecast_url: str) -> Optional[List[Dict]]:
    """Fetch hourly forecast from NWS (/forecast/hourly endpoint).

    Hourly forecasts have ~1.5F accuracy for day-0 vs ~2.5F for 12h periods.
    Returns list of hourly period dicts, or None on failure.
    """
    # The hourly endpoint is the same URL with /hourly appended
    hourly_url = forecast_url.rstrip("/") + "/hourly"

    now = time.monotonic()
    if hourly_url in _hourly_forecast_cache:
        cached_time, cached_data = _hourly_forecast_cache[hourly_url]
        if now - cached_time < _FORECAST_CACHE_TTL:
            return cached_data

    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": "(Morpheus Trading Bot, contact@example.com)"},
        ) as client:
            resp = await client.get(hourly_url)
            if resp.status_code == 200:
                data = resp.json()
                periods = data.get("properties", {}).get("periods", [])
                if periods:
                    _hourly_forecast_cache[hourly_url] = (now, periods)
                    # Track forecast updateTime for staleness check
                    update_time = data.get("properties", {}).get("updateTime")
                    if update_time:
                        _forecast_update_times[forecast_url] = update_time
                    return periods
    except Exception as e:
        logger.debug("nws_hourly_forecast_error", error=str(e), url=hourly_url)

    return None


async def _get_metar_observation(city: str) -> Optional[Dict]:
    """Fetch latest METAR/ASOS observation for a city.

    Returns dict with 'temperature_f', 'timestamp', 'station' or None.
    Uses NWS observations API (free, no key needed).
    """
    stations = _CITY_STATIONS.get(city)
    if not stations:
        return None

    # Try each station, return first successful
    for station in stations:
        cache_key = station
        now = time.monotonic()
        if cache_key in _observation_cache:
            cached_time, cached_data = _observation_cache[cache_key]
            if now - cached_time < _OBSERVATION_CACHE_TTL:
                return cached_data

        try:
            async with httpx.AsyncClient(
                timeout=8.0,
                headers={"User-Agent": "(Morpheus Trading Bot, contact@example.com)"},
            ) as client:
                resp = await client.get(
                    f"https://api.weather.gov/stations/{station}/observations/latest"
                )
                if resp.status_code == 200:
                    data = resp.json()
                    props = data.get("properties", {})
                    temp_c = props.get("temperature", {}).get("value")
                    if temp_c is not None and isinstance(temp_c, (int, float)):
                        temp_f = temp_c * 9.0 / 5.0 + 32.0
                        obs_time = props.get("timestamp", "")
                        result = {
                            "temperature_f": round(temp_f, 1),
                            "timestamp": obs_time,
                            "station": station,
                        }
                        _observation_cache[cache_key] = (now, result)
                        return result
        except Exception as e:
            logger.debug("metar_fetch_error", station=station, error=str(e))
            continue

    return None


async def _get_multi_station_observation(city: str) -> Optional[Dict]:
    """Fetch observations from multiple stations and average them.

    Returns dict with 'temperature_f' (averaged), 'station_count', 'stations'.
    Averaging reduces noise from any single station's microclimate.
    """
    stations = _CITY_STATIONS.get(city, [])
    if not stations:
        return None

    temps: List[float] = []
    station_names: List[str] = []

    for station in stations[:3]:  # max 3 stations
        cache_key = station
        now = time.monotonic()
        cached = _observation_cache.get(cache_key)
        if cached and now - cached[0] < _OBSERVATION_CACHE_TTL:
            temps.append(cached[1]["temperature_f"])
            station_names.append(station)
            continue

        try:
            async with httpx.AsyncClient(
                timeout=8.0,
                headers={"User-Agent": "(Morpheus Trading Bot, contact@example.com)"},
            ) as client:
                resp = await client.get(
                    f"https://api.weather.gov/stations/{station}/observations/latest"
                )
                if resp.status_code == 200:
                    data = resp.json()
                    props = data.get("properties", {})
                    temp_c = props.get("temperature", {}).get("value")
                    if temp_c is not None and isinstance(temp_c, (int, float)):
                        temp_f = round(temp_c * 9.0 / 5.0 + 32.0, 1)
                        obs_data = {
                            "temperature_f": temp_f,
                            "timestamp": props.get("timestamp", ""),
                            "station": station,
                        }
                        _observation_cache[station] = (now, obs_data)
                        temps.append(temp_f)
                        station_names.append(station)
        except Exception:
            continue

    if not temps:
        return None

    avg_temp = sum(temps) / len(temps)
    return {
        "temperature_f": round(avg_temp, 1),
        "station_count": len(temps),
        "stations": station_names,
        "temps": temps,
    }


def _parse_weather_threshold(question: str) -> Optional[Tuple[str, float]]:
    """Parse weather market question for threshold type and value.

    Returns (type, threshold) where type is 'high', 'low', 'precip', etc.
    Examples:
      "Will the high temp be >65°" → ("high", 65.0)
      "Will the high temp in LA be 63-64°" → ("high_bracket", 63.5)
      "Will the minimum temperature be <54°" → ("low", 54.0)
    """
    q = question.lower()

    # High temperature threshold: ">65°", "<65°", "above 65", "below 65"
    if "high" in q or "maximum" in q:
        # Bracket: "63-64°" or "63.5-64.5"
        m = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)°", q)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            return ("high_bracket", (lo + hi) / 2)
        # Threshold: ">65°", "<65°", ">65", "<65" — capture direction
        m = re.search(r"([><])\s*(\d+\.?\d*)", q)
        if m:
            direction = "high_below" if m.group(1) == "<" else "high"
            return (direction, float(m.group(2)))
        # "above 65" / "below 65"
        m = re.search(r"(?:above|over|exceed)\s+(\d+\.?\d*)", q)
        if m:
            return ("high", float(m.group(1)))
        m = re.search(r"(?:below|under)\s+(\d+\.?\d*)", q)
        if m:
            return ("high_below", float(m.group(1)))

    if "low" in q or "minimum" in q:
        m = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)°", q)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            return ("low_bracket", (lo + hi) / 2)
        m = re.search(r"([><])\s*(\d+\.?\d*)", q)
        if m:
            direction = "low_below" if m.group(1) == "<" else "low"
            return (direction, float(m.group(2)))

    # Rain/precipitation threshold
    if "rain" in q or "precipitation" in q or "inches of precipitation" in q:
        m = re.search(r"(?:>|greater than|more than|exceed)\s*(\d+\.?\d*)", q)
        if m:
            return ("rain", float(m.group(1)))
        # Default: any rain (>0 inches)
        return ("rain", 0.0)

    # Snowfall threshold
    if "snow" in q or "snowfall" in q:
        m = re.search(r"(?:>|greater than|more than|exceed)\s*(\d+\.?\d*)", q)
        if m:
            return ("snow", float(m.group(1)))
        # Default: any snow (>0 inches)
        return ("snow", 0.0)

    # Wind speed threshold
    if "wind" in q:
        m = re.search(r"(?:>|greater than|more than|exceed)\s*(\d+\.?\d*)", q)
        if m:
            return ("wind", float(m.group(1)))
        m = re.search(r"(?:<|less than|under|below)\s*(\d+\.?\d*)", q)
        if m:
            return ("wind_below", float(m.group(1)))
        m = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*mph", q)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            return ("wind_bracket", (lo + hi) / 2)
        return None

    return None


# ---------------------------------------------------------------------------
# NWS forecast error sigma (empirical, Fahrenheit) and direct probability
# ---------------------------------------------------------------------------

# NWS forecast error standard deviation by lead time in days (12h forecast)
# Day-0 uses city-specific hourly sigma; this is for 12h fallback
_NWS_SIGMA_12H = {0: 2.5, 1: 2.5, 2: 3.5, 3: 5.5}

# Track previous forecast temps for change detection
_previous_forecasts: Dict[str, float] = {}  # "city:period" -> temp

# Track forecast changes for cache invalidation
_forecast_change_events: List[Dict] = []  # recent change events


def _get_forecast_age_hours(forecast_url: str) -> Optional[float]:
    """Compute age of NWS forecast in hours from cached updateTime.

    Returns hours since last forecast update, or None if not available.
    Parses the ISO 8601 updateTime string from the NWS API response.
    """
    update_time_str = _forecast_update_times.get(forecast_url)
    if not update_time_str:
        return None
    try:
        # NWS uses ISO format like "2025-02-15T14:30:00+00:00"
        # Handle both +00:00 and Z suffixes
        ut = update_time_str.replace("Z", "+00:00")
        update_dt = datetime.fromisoformat(ut)
        age = datetime.now(timezone.utc) - update_dt
        return max(0.0, age.total_seconds() / 3600.0)
    except (ValueError, TypeError):
        return None


def _norm_cdf(x: float, mu: float, sigma: float) -> float:
    """Normal CDF using math.erfc (no scipy needed)."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / sigma
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _parse_nws_wind_speed(wind_str: str) -> Optional[float]:
    """Parse NWS wind speed '15 mph' or '10 to 20 mph' → average mph."""
    if not wind_str or wind_str == "?":
        return None
    m = re.search(r"(\d+)\s*to\s*(\d+)", wind_str)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.search(r"(\d+)", wind_str)
    return float(m.group(1)) if m else None


def weather_forecast_changed(city: str, period_name: str, new_temp: float) -> bool:
    """Return True if forecast temp shifted >= 2F from last seen value.

    When a significant change is detected, logs it for potential cache invalidation.
    """
    key = f"{city}:{period_name}"
    prev = _previous_forecasts.get(key)
    _previous_forecasts[key] = new_temp
    if prev is None:
        return False
    delta = abs(new_temp - prev)
    if delta >= 2.0:
        _forecast_change_events.append({
            "city": city,
            "period": period_name,
            "old_temp": prev,
            "new_temp": new_temp,
            "delta": delta,
            "time": time.monotonic(),
        })
        # Keep only last 20 events
        while len(_forecast_change_events) > 20:
            _forecast_change_events.pop(0)
        logger.info(
            "weather_forecast_change_detected",
            city=city,
            period=period_name,
            old_temp=prev,
            new_temp=new_temp,
            delta=round(delta, 1),
        )
        return True
    return False


def get_recent_forecast_changes(since_seconds: float = 600.0) -> List[Dict]:
    """Return forecast change events in the last N seconds.

    Used by ensemble_signal to invalidate cache when forecasts shift.
    """
    cutoff = time.monotonic() - since_seconds
    return [e for e in _forecast_change_events if e["time"] > cutoff]


async def compute_weather_probability(
    question: str, market_id: str = "", close_time: Optional[datetime] = None,
) -> Optional[Tuple[float, float, str]]:
    """Compute weather probability from 5-source blend.

    Sources: NWS (point), GFS+GEM+ICON (ensemble), ECMWF IFS (ensemble),
    HRRR (hi-res deterministic), GraphCast/AIGFS (ML deterministic).

    Returns (p_yes, confidence, reasoning) or None if can't compute.
    Only returns a result when forecast data is unambiguous (|forecast - threshold| / sigma > 1.0).
    For ambiguous cases, returns None so the LLM handles it.
    Applies 20% confidence penalty if NWS forecast is >6h stale.
    """
    # 1. Parse city
    city = _parse_city(question)
    if not city and market_id:
        city = _parse_city_from_ticker(market_id)
    if not city:
        return None

    # 2. Parse threshold
    threshold_info = _parse_weather_threshold(question)
    if threshold_info is None:
        return None

    t_type, t_value = threshold_info

    # 3. Parse target date
    target_date = _parse_target_date(question)

    # 4. Compute lead time in days
    now = datetime.now(timezone.utc)
    if target_date:
        lead_days = max(0, (target_date - now).days)
    elif close_time:
        lead_days = max(0, (close_time - now).days)
    else:
        lead_days = 0

    # City-specific sigma for hourly forecasts; fall back to 12h sigma for longer lead
    if lead_days == 0:
        sigma = _CITY_SIGMA_HOURLY.get(city, _DEFAULT_SIGMA_HOURLY)
    else:
        base_12h = _NWS_SIGMA_12H.get(min(lead_days, 3), 5.0)
        # Scale 12h sigma by city's relative variability
        city_hourly = _CITY_SIGMA_HOURLY.get(city, _DEFAULT_SIGMA_HOURLY)
        sigma = base_12h * (city_hourly / _DEFAULT_SIGMA_HOURLY)

    # 5. Fetch NWS forecast
    coords = _WEATHER_CITIES.get(city)
    if not coords:
        return None

    forecast_url = await _get_nws_gridpoint_url(coords[0], coords[1])
    if not forecast_url:
        return None

    # 5a. METAR actual observation check (same-day only)
    # If we have a real observation that already decisively resolves the market,
    # return near-certain probability. This is the strongest possible signal.
    if lead_days == 0 and t_type not in ("rain", "snow"):
        obs = await _get_multi_station_observation(city)
        if obs and obs["station_count"] >= 1:
            observed_temp = obs["temperature_f"]
            # For HIGH temp markets: if observed temp already exceeds threshold,
            # the high for the day is AT LEAST this value (can only go higher).
            if "high" in t_type and "bracket" not in t_type:
                if "below" in t_type:
                    # "<X" market: YES means temp stays below X
                    if observed_temp >= t_value:
                        # Already at/above threshold — YES is impossible
                        p_yes = 0.02
                        logger.info(
                            "metar_decisive_signal",
                            city=city, observed=observed_temp,
                            threshold=t_value, t_type=t_type,
                            stations=obs["stations"], p_yes=p_yes,
                        )
                        return (p_yes, 0.95, f"METAR decisive: observed {observed_temp:.0f}°F >= threshold {t_value:.0f}°F, high_below impossible ({obs['stations']})")
                else:
                    # ">X" market: YES means temp exceeds X
                    if observed_temp >= t_value + 1.0:
                        # Already above threshold — YES is near-certain
                        p_yes = 0.98
                        logger.info(
                            "metar_decisive_signal",
                            city=city, observed=observed_temp,
                            threshold=t_value, t_type=t_type,
                            stations=obs["stations"], p_yes=p_yes,
                        )
                        return (p_yes, 0.95, f"METAR decisive: observed {observed_temp:.0f}°F > threshold {t_value:.0f}°F ({obs['stations']})")

            # For LOW temp markets with observations:
            # We can't be as decisive since low hasn't fully occurred yet,
            # but if current temp is already well below threshold, it helps.
            if "low" in t_type and "bracket" not in t_type:
                if "below" not in t_type:
                    # ">X" low: YES means low stays above X
                    if observed_temp < t_value - 1.0:
                        # Current temp already below threshold — low likely below too
                        p_yes = 0.05
                        logger.info(
                            "metar_decisive_signal",
                            city=city, observed=observed_temp,
                            threshold=t_value, t_type=t_type,
                            stations=obs["stations"], p_yes=p_yes,
                        )
                        return (p_yes, 0.90, f"METAR: current {observed_temp:.0f}°F already below low threshold {t_value:.0f}°F ({obs['stations']})")

            # For bracket markets: if observed temp is far outside bracket, decisive
            if "bracket" in t_type:
                q_lower = question.lower()
                m_br = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)°", q_lower)
                if m_br:
                    bracket_lo = float(m_br.group(1))
                    bracket_hi = float(m_br.group(2))
                    if "high" in t_type and observed_temp > bracket_hi + 2.0:
                        # High already exceeds bracket upper bound
                        p_yes = 0.03
                        logger.info(
                            "metar_bracket_decisive",
                            city=city, observed=observed_temp,
                            bracket=f"{bracket_lo}-{bracket_hi}",
                            stations=obs["stations"],
                        )
                        return (p_yes, 0.92, f"METAR: observed {observed_temp:.0f}°F already above bracket {bracket_lo}-{bracket_hi}°F ({obs['stations']})")

            # Even if not decisive, use observation to refine forecast
            # Blend: when we have an observation, reduce sigma (more certain)
            if obs["station_count"] >= 2:
                sigma *= 0.85  # multi-station observation tightens estimate
                logger.debug(
                    "metar_sigma_tightened",
                    city=city, observed=observed_temp,
                    station_count=obs["station_count"],
                    sigma=round(sigma, 2),
                )

    # Rain fast-path: use NWS hourly probabilityOfPrecipitation (PoP)
    if t_type == "rain":
        hourly_periods = await _get_nws_hourly_forecast(forecast_url)
        if not hourly_periods:
            return None

        # Determine target date string
        if target_date:
            target_str = target_date.strftime("%Y-%m-%d")
        else:
            target_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Extract PoP values for the target date
        pop_values: List[float] = []
        for hp in hourly_periods:
            start = hp.get("startTime", "")
            if target_str not in start:
                continue
            pop_data = hp.get("probabilityOfPrecipitation", {})
            pop_val = pop_data.get("value") if isinstance(pop_data, dict) else None
            if pop_val is not None and isinstance(pop_val, (int, float)):
                pop_values.append(float(pop_val))

        if not pop_values:
            return None

        max_pop = max(pop_values)

        # Amount threshold (e.g., >0.5 inches) — too uncertain, defer to LLM
        if t_value > 0.0:
            logger.info(
                "rain_amount_threshold_defer",
                city=city,
                threshold_inches=t_value,
                msg="Amount thresholds deferred to LLM",
            )
            return None

        # Compute cumulative daily rain probability (not just max hourly)
        p_no_rain = 1.0
        for pop in pop_values:
            p_no_rain *= (1.0 - pop / 100.0)
        p_yes_cumulative = 1.0 - p_no_rain

        # Confidence gate: only signal when clearly raining or clearly dry
        if p_yes_cumulative >= 0.80 or p_yes_cumulative <= 0.15:
            p_yes = max(0.02, min(0.98, p_yes_cumulative))
            confidence = 0.85 if (p_yes_cumulative >= 0.90 or p_yes_cumulative <= 0.05) else 0.70
            reasoning = (
                f"NOAA rain direct: cumulative p_rain={p_yes_cumulative:.3f} (max hourly PoP={max_pop:.0f}%) "
                f"across {len(pop_values)} hours, p_yes={p_yes:.3f} (city={city}, lead={lead_days}d)"
            )
            logger.info(
                "noaa_direct_signal",
                city=city,
                t_type="rain",
                max_pop=max_pop,
                pop_hours=len(pop_values),
                p_yes=round(p_yes, 4),
                confidence=round(confidence, 3),
                lead_days=lead_days,
            )
            return (p_yes, confidence, reasoning)
        else:
            logger.info(
                "rain_pop_ambiguous",
                city=city,
                max_pop=max_pop,
                msg="PoP 15-80%, deferring to LLM",
            )
            return None

    # Snow fast-path: use NWS hourly PoP + temperature + shortForecast text
    if t_type == "snow":
        # Amount thresholds (>X inches) — too uncertain for heuristic, defer to LLM
        if t_value > 0.0:
            logger.info(
                "snow_amount_threshold_defer",
                city=city,
                threshold_inches=t_value,
                msg="Snow amount thresholds deferred to LLM",
            )
            return None

        hourly_periods = await _get_nws_hourly_forecast(forecast_url)
        if not hourly_periods:
            return None

        # Determine target date string
        if target_date:
            target_str = target_date.strftime("%Y-%m-%d")
        else:
            target_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Count hours with snow potential on target date
        snow_hours = 0
        total_hours = 0
        for hp in hourly_periods:
            start = hp.get("startTime", "")
            if target_str not in start:
                continue
            total_hours += 1

            temp = hp.get("temperature")
            pop_data = hp.get("probabilityOfPrecipitation", {})
            pop_val = pop_data.get("value") if isinstance(pop_data, dict) else None
            short_fc = (hp.get("shortForecast") or "").lower()

            # Snow likely if: (PoP >= 50% AND temp <= 34°F) OR "snow" in forecast text
            temp_ok = isinstance(temp, (int, float)) and temp <= 34
            pop_ok = pop_val is not None and pop_val >= 50
            text_snow = "snow" in short_fc

            if (pop_ok and temp_ok) or text_snow:
                snow_hours += 1

        if total_hours < 6:
            # Not enough hourly data to be confident
            return None

        snow_fraction = snow_hours / total_hours

        # Signal only on clear cases
        if snow_fraction >= 0.4:
            p_yes = min(0.95, snow_fraction * 1.1)
            confidence = 0.70
            reasoning = (
                f"NOAA snow direct: {snow_hours}/{total_hours} hours with snow potential, "
                f"fraction={snow_fraction:.2f}, p_yes={p_yes:.3f} (city={city}, lead={lead_days}d)"
            )
            logger.info(
                "noaa_direct_signal",
                city=city,
                t_type="snow",
                snow_hours=snow_hours,
                total_hours=total_hours,
                snow_fraction=round(snow_fraction, 3),
                p_yes=round(p_yes, 4),
                confidence=round(confidence, 3),
                lead_days=lead_days,
            )
            return (p_yes, confidence, reasoning)
        elif snow_fraction <= 0.05:
            p_yes = 0.05
            confidence = 0.75
            reasoning = (
                f"NOAA snow direct: {snow_hours}/{total_hours} hours with snow potential, "
                f"fraction={snow_fraction:.2f}, p_yes={p_yes:.3f} (city={city}, lead={lead_days}d)"
            )
            logger.info(
                "noaa_direct_signal",
                city=city,
                t_type="snow",
                snow_hours=snow_hours,
                total_hours=total_hours,
                snow_fraction=round(snow_fraction, 3),
                p_yes=round(p_yes, 4),
                confidence=round(confidence, 3),
                lead_days=lead_days,
            )
            return (p_yes, confidence, reasoning)
        else:
            logger.info(
                "snow_ambiguous",
                city=city,
                snow_fraction=round(snow_fraction, 3),
                msg="Snow fraction 5-40%, deferring to LLM",
            )
            return None

    # Wind speed fast-path: use NWS hourly windSpeed field
    if t_type in ("wind", "wind_below", "wind_bracket"):
        hourly_periods = await _get_nws_hourly_forecast(forecast_url)
        if not hourly_periods:
            return None

        if target_date:
            target_str = target_date.strftime("%Y-%m-%d")
        else:
            target_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        wind_speeds: List[float] = []
        for hp in hourly_periods:
            start = hp.get("startTime", "")
            if target_str not in start:
                continue
            ws = _parse_nws_wind_speed(hp.get("windSpeed", ""))
            if ws is not None:
                wind_speeds.append(ws)

        if not wind_speeds:
            return None

        max_wind = max(wind_speeds)
        avg_wind = sum(wind_speeds) / len(wind_speeds)
        wind_sigma = 4.0  # empirical NWS wind forecast error (~4 mph)

        # Wave 23: inflate sigma for extreme z-scores — forecast error has fat tails
        _z_pre_wind = abs(max_wind - t_value) / wind_sigma if wind_sigma > 0 else 0
        if _z_pre_wind > 2.5:
            wind_sigma *= 1.20

        if t_type == "wind":
            # ">X mph" threshold
            p_yes = 1.0 - _norm_cdf(t_value, max_wind, wind_sigma)
        elif t_type == "wind_below":
            # "<X mph" threshold
            p_yes = _norm_cdf(t_value, max_wind, wind_sigma)
        else:
            # wind_bracket — t_value is midpoint
            bracket_half = 2.5  # assume ~5 mph bracket width
            p_yes = (_norm_cdf(t_value + bracket_half, avg_wind, wind_sigma)
                     - _norm_cdf(t_value - bracket_half, avg_wind, wind_sigma))

        p_yes = max(0.02, min(0.98, p_yes))
        z_score = abs(max_wind - t_value) / wind_sigma

        if z_score < 1.0:  # Wave 21: raised from 0.7
            logger.info(
                "wind_ambiguous",
                city=city,
                max_wind=max_wind,
                threshold=t_value,
                z_score=round(z_score, 2),
                msg="Wind near threshold, deferring to LLM",
            )
            return None

        confidence = min(0.90, 0.5 + z_score * 0.15)
        reasoning = (
            f"NOAA wind direct: NWS max wind={max_wind:.0f} mph, "
            f"avg={avg_wind:.0f} mph across {len(wind_speeds)} hours, "
            f"threshold={t_value:.0f} mph, sigma={wind_sigma:.0f}, "
            f"p_yes={p_yes:.3f} (city={city}, lead={lead_days}d)"
        )

        logger.info(
            "noaa_direct_signal",
            city=city,
            t_type=t_type,
            max_wind=max_wind,
            avg_wind=round(avg_wind, 1),
            threshold=t_value,
            p_yes=round(p_yes, 4),
            confidence=round(confidence, 3),
            lead_days=lead_days,
        )

        return (p_yes, confidence, reasoning)

    # For day-0 markets, try hourly forecast first (sigma ~1.5F vs 2.5F)
    forecast_temp = None
    period_name = None
    used_hourly = False

    if lead_days == 0:
        hourly_periods = await _get_nws_hourly_forecast(forecast_url)
        if hourly_periods:
            today_str = now.strftime("%Y-%m-%d")
            today_temps = []
            overnight_temps = []

            # For high-temp markets: use today's daytime hours
            # For low-temp markets: use tonight + tomorrow early morning (overnight window)
            tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")

            for hp in hourly_periods:
                start = hp.get("startTime", "")
                t = hp.get("temperature")
                if not isinstance(t, (int, float)):
                    continue

                if today_str in start:
                    today_temps.append(float(t))
                    # Evening hours (18:00+) also count for overnight low
                    hour_str = start[11:13] if len(start) > 12 else ""
                    if hour_str.isdigit() and int(hour_str) >= 18:
                        overnight_temps.append(float(t))
                elif tomorrow_str in start:
                    # Early morning hours (00:00-11:00) for overnight low
                    hour_str = start[11:13] if len(start) > 12 else ""
                    if hour_str.isdigit() and int(hour_str) < 12:
                        overnight_temps.append(float(t))

            hourly_sigma = _CITY_SIGMA_HOURLY.get(city, _DEFAULT_SIGMA_HOURLY)
            if "high" in t_type and today_temps:
                forecast_temp = max(today_temps)
                period_name = f"Today hourly max ({len(today_temps)} hours)"
                sigma = hourly_sigma
                used_hourly = True
            elif "low" in t_type and overnight_temps:
                forecast_temp = min(overnight_temps)
                period_name = f"Overnight hourly min ({len(overnight_temps)} hours)"
                sigma = hourly_sigma
                used_hourly = True
            elif "low" in t_type and today_temps:
                # Fallback: use today's min if no overnight data yet
                forecast_temp = min(today_temps)
                period_name = f"Today hourly min ({len(today_temps)} hours)"
                sigma = hourly_sigma
                used_hourly = True

    # Fallback to standard 12-hour forecast
    if forecast_temp is None:
        forecast_data = await _get_nws_forecast(forecast_url)
        if not forecast_data:
            return None

        periods = forecast_data.get("properties", {}).get("periods", [])
        if not periods:
            return None

        # 6. Find the relevant forecast period
        relevant = []
        if target_date:
            target_str = target_date.strftime("%Y-%m-%d")
            for period in periods:
                start = period.get("startTime", "")
                if target_str in start:
                    relevant.append(period)
        if not relevant:
            relevant = periods[:4]

        # Get the forecast temp for the right period type
        for p in relevant:
            t = p.get("temperature")
            name = p.get("name", "")
            if not isinstance(t, (int, float)):
                continue
            if "high" in t_type:
                if "night" not in name.lower():
                    forecast_temp = float(t)
                    period_name = name
                    break
            elif "low" in t_type:
                if "night" in name.lower():
                    forecast_temp = float(t)
                    period_name = name
                    break

        # Fallback: use first available temp
        if forecast_temp is None:
            for p in relevant:
                t = p.get("temperature")
                if isinstance(t, (int, float)):
                    forecast_temp = float(t)
                    period_name = p.get("name", "unknown")
                    break

    if forecast_temp is None:
        return None

    # Track forecast changes
    if period_name:
        weather_forecast_changed(city, period_name, forecast_temp)

    # Wave 25: apply NWS bias correction if available
    bias = get_weather_bias(city, t_type)
    if bias is not None:
        forecast_temp_raw = forecast_temp
        forecast_temp -= bias  # subtract positive bias (NWS too high → reduce)
        logger.info(
            "weather_bias_correction",
            city=city,
            t_type=t_type,
            raw_temp=round(forecast_temp_raw, 1),
            corrected_temp=round(forecast_temp, 1),
            bias=round(bias, 2),
        )

    # Wave 25: use adaptive sigma when enough samples available
    adaptive_sig = get_adaptive_sigma(city, t_type)
    if adaptive_sig is not None:
        sigma_before = sigma
        sigma = adaptive_sig
        logger.info(
            "weather_adaptive_sigma",
            city=city,
            t_type=t_type,
            static_sigma=round(sigma_before, 2),
            adaptive_sigma=round(sigma, 2),
        )

    # 7. Compute NWS-based probability
    bracket_bounds = None
    if "bracket" in t_type:
        # Bracket market: "X-Y°F" — t_value is midpoint, need to find bounds
        q = question.lower()
        m = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)°", q)
        if not m:
            return None
        lower = float(m.group(1))
        upper = float(m.group(2))
        bracket_bounds = (lower, upper)
        # Wave 23: inflate sigma for extreme z-scores — forecast error has fat tails
        _z_pre = abs(forecast_temp - (lower + upper) / 2) / sigma if sigma > 0 else 0
        if _z_pre > 2.5:
            sigma *= 1.20
        p_nws = _norm_cdf(upper, forecast_temp, sigma) - _norm_cdf(lower, forecast_temp, sigma)
    elif "below" in t_type:
        # "<X" market: YES means temp is below threshold
        # Wave 23: inflate sigma for extreme z-scores — forecast error has fat tails
        _z_pre = abs(forecast_temp - t_value) / sigma if sigma > 0 else 0
        if _z_pre > 2.5:
            sigma *= 1.20
        p_nws = _norm_cdf(t_value, forecast_temp, sigma)
    else:
        # ">X" market: YES means temp is above threshold
        # Wave 23: inflate sigma for extreme z-scores — forecast error has fat tails
        _z_pre = abs(forecast_temp - t_value) / sigma if sigma > 0 else 0
        if _z_pre > 2.5:
            sigma *= 1.20
        p_nws = 1.0 - _norm_cdf(t_value, forecast_temp, sigma)

    # 7b. Multi-model ensemble blend (GFS+GEM+ICON ~92 + ECMWF IFS 51 members)
    p_gfs = None  # GFS+GEM+ICON combined probability
    p_ecmwf = None  # ECMWF IFS probability
    gfs_members: List[float] = []
    ecmwf_members: List[float] = []
    graphcast_data = None  # GraphCast (AIGFS) forecast data

    if coords:
        target_str = target_date.strftime("%Y-%m-%d") if target_date else now.strftime("%Y-%m-%d")

        # Fetch GFS+GEM+ICON, ECMWF IFS, and GraphCast in parallel
        gfs_task = _fetch_open_meteo_ensemble(coords[0], coords[1], target_str)
        ecmwf_task = _fetch_ecmwf_ensemble(coords[0], coords[1], target_str)
        graphcast_task = _fetch_graphcast_forecast(coords[0], coords[1], target_str)
        ensemble_data, ecmwf_data, graphcast_data = await asyncio.gather(
            gfs_task, ecmwf_task, graphcast_task,
        )

        # Determine direction for probability computation
        if "bracket" in t_type and bracket_bounds:
            direction = "bracket"
        elif "below" in t_type:
            direction = "below"
        else:
            direction = "above"

        # Parse GFS+GEM+ICON ensemble
        if ensemble_data:
            if "high" in t_type:
                gfs_members = ensemble_data.get("temperature_max", [])
            elif "low" in t_type:
                gfs_members = ensemble_data.get("temperature_min", [])
            else:
                gfs_members = ensemble_data.get("temperature_max", [])

            if gfs_members and len(gfs_members) >= 5:
                p_gfs = _ensemble_probability(
                    gfs_members, t_value, direction,
                    bracket_bounds=bracket_bounds,
                )

        # Parse ECMWF IFS ensemble
        if ecmwf_data:
            if "high" in t_type:
                ecmwf_members = ecmwf_data.get("temperature_max", [])
            elif "low" in t_type:
                ecmwf_members = ecmwf_data.get("temperature_min", [])
            else:
                ecmwf_members = ecmwf_data.get("temperature_max", [])

            if ecmwf_members and len(ecmwf_members) >= 5:
                p_ecmwf = _ensemble_probability(
                    ecmwf_members, t_value, direction,
                    bracket_bounds=bracket_bounds,
                )

    # 7c. HRRR high-resolution forecast (3km, deterministic, same-day only)
    p_hrrr = None
    if coords:
        target_str = target_date.strftime("%Y-%m-%d") if target_date else now.strftime("%Y-%m-%d")
        hrrr_data = await _fetch_open_meteo_hrrr(coords[0], coords[1], target_str)
        if hrrr_data:
            if "high" in t_type:
                hrrr_temp = hrrr_data.get("temperature_max")
            elif "low" in t_type:
                hrrr_temp = hrrr_data.get("temperature_min")
            else:
                hrrr_temp = hrrr_data.get("temperature_max")

            if hrrr_temp is not None:
                # HRRR is deterministic — use normal CDF with city sigma
                if "bracket" in t_type and bracket_bounds:
                    lo, hi = bracket_bounds
                    p_hrrr = _norm_cdf(hi, hrrr_temp, sigma) - _norm_cdf(lo, hrrr_temp, sigma)
                elif "below" in t_type:
                    p_hrrr = _norm_cdf(t_value, hrrr_temp, sigma)
                else:
                    p_hrrr = 1.0 - _norm_cdf(t_value, hrrr_temp, sigma)

    # 7d. GraphCast (AIGFS) deterministic forecast — ML-based, 0.25° resolution
    p_graphcast = None
    graphcast_temp = None
    if coords and graphcast_data:
        if "high" in t_type:
            graphcast_temp = graphcast_data.get("temperature_max")
        elif "low" in t_type:
            graphcast_temp = graphcast_data.get("temperature_min")
        else:
            graphcast_temp = graphcast_data.get("temperature_max")

        if graphcast_temp is not None:
            # GraphCast is deterministic — use normal CDF with city sigma
            if "bracket" in t_type and bracket_bounds:
                lo, hi = bracket_bounds
                p_graphcast = (
                    _norm_cdf(hi, graphcast_temp, sigma)
                    - _norm_cdf(lo, graphcast_temp, sigma)
                )
            elif "below" in t_type:
                p_graphcast = _norm_cdf(t_value, graphcast_temp, sigma)
            else:
                p_graphcast = 1.0 - _norm_cdf(t_value, graphcast_temp, sigma)

    # Blend NWS + GFS ensemble + ECMWF IFS + HRRR (+ GraphCast post-blend):
    # NWS is calibrated point forecast, GFS+GEM+ICON captures ensemble spread,
    # ECMWF IFS is the gold-standard global model (51 members),
    # HRRR adds high-resolution spatial detail for same-day,
    # GraphCast (AIGFS) is applied as a 10% post-blend refinement (5th source).
    #
    # HRRR time-weighting: HRRR 3km outperforms global models for 0-18h forecasts.
    # Boost HRRR weight when close to market close time.
    #
    # Agreement check: compare mean forecast temps across sources.
    # Within 2°F → sources agree (trust NWS calibration).
    # >2°F disagreement → trust model ensembles over NWS.

    # Compute hours to close for HRRR time-weighting
    _hours_to_close = None
    if close_time:
        _hours_to_close = max(0, (close_time - now).total_seconds() / 3600)
    elif lead_days == 0:
        _hours_to_close = 6.0  # default: assume ~6h remaining for day-0

    # HRRR weight multiplier based on hours to close
    # <6h: HRRR at peak accuracy → 50% weight (2x boost)
    # <12h: HRRR still strong → 35% weight (~1.4x boost)
    # >12h or unknown: default weights
    if _hours_to_close is not None and _hours_to_close < 6.0:
        _hrrr_tier = "near"  # <6h
    elif _hours_to_close is not None and _hours_to_close < 12.0:
        _hrrr_tier = "mid"   # 6-12h
    else:
        _hrrr_tier = "far"   # >12h or unknown

    # Compute mean ensemble temperatures for agreement check
    gfs_mean = sum(gfs_members) / len(gfs_members) if gfs_members else None
    ecmwf_mean = sum(ecmwf_members) / len(ecmwf_members) if ecmwf_members else None

    # Determine temperature-based agreement between sources
    _AGREE_THRESHOLD = 2.0  # °F
    all_agree = False
    nws_ecmwf_agree_gfs_disagrees = False

    if gfs_mean is not None and ecmwf_mean is not None:
        nws_gfs_diff = abs(forecast_temp - gfs_mean)
        nws_ecmwf_diff = abs(forecast_temp - ecmwf_mean)
        gfs_ecmwf_diff = abs(gfs_mean - ecmwf_mean)

        all_agree = (
            nws_gfs_diff <= _AGREE_THRESHOLD
            and nws_ecmwf_diff <= _AGREE_THRESHOLD
            and gfs_ecmwf_diff <= _AGREE_THRESHOLD
        )
        # NWS + ECMWF agree but GFS is the outlier
        nws_ecmwf_agree_gfs_disagrees = (
            nws_ecmwf_diff <= _AGREE_THRESHOLD
            and gfs_ecmwf_diff > _AGREE_THRESHOLD
        )

    if p_gfs is not None and p_ecmwf is not None and p_hrrr is not None:
        # Full 4-source core blend: NWS + GFS + ECMWF + HRRR (GraphCast applied post-blend)
        # Time-weighted: HRRR gets boosted for near-close markets
        if _hrrr_tier == "near":
            # <6h: HRRR 50%, others split remaining 50%
            if nws_ecmwf_agree_gfs_disagrees:
                w_nws, w_gfs, w_ecmwf, w_hrrr = 0.20, 0.05, 0.25, 0.50
            elif all_agree:
                w_nws, w_gfs, w_ecmwf, w_hrrr = 0.20, 0.15, 0.15, 0.50
            else:
                w_nws, w_gfs, w_ecmwf, w_hrrr = 0.15, 0.18, 0.17, 0.50
        elif _hrrr_tier == "mid":
            # 6-12h: HRRR 35%, rebalance rest
            if nws_ecmwf_agree_gfs_disagrees:
                w_nws, w_gfs, w_ecmwf, w_hrrr = 0.25, 0.08, 0.32, 0.35
            elif all_agree:
                w_nws, w_gfs, w_ecmwf, w_hrrr = 0.28, 0.19, 0.18, 0.35
            else:
                w_nws, w_gfs, w_ecmwf, w_hrrr = 0.20, 0.23, 0.22, 0.35
        else:
            # >12h: original weights (HRRR less reliable at longer leads)
            if nws_ecmwf_agree_gfs_disagrees:
                w_nws, w_gfs, w_ecmwf, w_hrrr = 0.35, 0.10, 0.40, 0.15
            elif all_agree:
                w_nws, w_gfs, w_ecmwf, w_hrrr = 0.35, 0.25, 0.25, 0.15
            else:
                w_nws, w_gfs, w_ecmwf, w_hrrr = 0.25, 0.30, 0.30, 0.15

        p_yes = w_nws * p_nws + w_gfs * p_gfs + w_ecmwf * p_ecmwf + w_hrrr * p_hrrr
        logger.info(
            "weather_full_blend",
            city=city,
            p_nws=round(p_nws, 4),
            p_gfs=round(p_gfs, 4),
            p_ecmwf=round(p_ecmwf, 4),
            p_hrrr=round(p_hrrr, 4),
            p_blended=round(p_yes, 4),
            all_agree=all_agree,
            nws_ecmwf_agree=nws_ecmwf_agree_gfs_disagrees,
            hrrr_tier=_hrrr_tier,
            hours_to_close=round(_hours_to_close, 1) if _hours_to_close is not None else None,
            w_hrrr=w_hrrr,
            n_gfs=len(gfs_members),
            n_ecmwf=len(ecmwf_members),
        )
    elif p_gfs is not None and p_ecmwf is not None:
        # GFS + ECMWF available, no HRRR
        if nws_ecmwf_agree_gfs_disagrees:
            p_yes = 0.40 * p_nws + 0.10 * p_gfs + 0.50 * p_ecmwf
        elif all_agree:
            p_yes = 0.40 * p_nws + 0.30 * p_gfs + 0.30 * p_ecmwf
        else:
            p_yes = 0.30 * p_nws + 0.35 * p_gfs + 0.35 * p_ecmwf
        logger.info(
            "weather_triple_blend",
            city=city,
            p_nws=round(p_nws, 4),
            p_gfs=round(p_gfs, 4),
            p_ecmwf=round(p_ecmwf, 4),
            p_blended=round(p_yes, 4),
            all_agree=all_agree,
            nws_ecmwf_agree=nws_ecmwf_agree_gfs_disagrees,
            n_gfs=len(gfs_members),
            n_ecmwf=len(ecmwf_members),
        )
    elif p_gfs is not None and p_hrrr is not None:
        # GFS + HRRR available, no ECMWF — time-weight HRRR
        disagreement = abs(p_nws - p_gfs)
        if _hrrr_tier == "near":
            w_hrrr = 0.45
            w_nws = 0.20 if disagreement > 0.15 else 0.25
            w_gfs = 1.0 - w_nws - w_hrrr
        elif _hrrr_tier == "mid":
            w_hrrr = 0.35
            w_nws = 0.25 if disagreement > 0.15 else 0.30
            w_gfs = 1.0 - w_nws - w_hrrr
        else:
            if disagreement > 0.15:
                w_nws, w_gfs, w_hrrr = 0.30, 0.50, 0.20
            else:
                w_nws, w_gfs, w_hrrr = 0.40, 0.35, 0.25
        p_yes = w_nws * p_nws + w_gfs * p_gfs + w_hrrr * p_hrrr
        logger.info(
            "weather_gfs_hrrr_blend",
            city=city,
            p_nws=round(p_nws, 4),
            p_gfs=round(p_gfs, 4),
            p_hrrr=round(p_hrrr, 4),
            p_blended=round(p_yes, 4),
            disagreement=round(disagreement, 4),
            hrrr_tier=_hrrr_tier,
            w_hrrr=w_hrrr,
            n_gfs=len(gfs_members),
        )
    elif p_ecmwf is not None and p_hrrr is not None:
        # ECMWF + HRRR available, no GFS — time-weight HRRR
        disagreement = abs(p_nws - p_ecmwf)
        if _hrrr_tier == "near":
            w_hrrr = 0.45
            w_nws = 0.20 if disagreement > 0.15 else 0.25
            w_ecmwf = 1.0 - w_nws - w_hrrr
        elif _hrrr_tier == "mid":
            w_hrrr = 0.35
            w_nws = 0.25 if disagreement > 0.15 else 0.30
            w_ecmwf = 1.0 - w_nws - w_hrrr
        else:
            if disagreement > 0.15:
                w_nws, w_ecmwf, w_hrrr = 0.30, 0.50, 0.20
            else:
                w_nws, w_ecmwf, w_hrrr = 0.40, 0.35, 0.25
        p_yes = w_nws * p_nws + w_ecmwf * p_ecmwf + w_hrrr * p_hrrr
        logger.info(
            "weather_ecmwf_hrrr_blend",
            city=city,
            p_nws=round(p_nws, 4),
            p_ecmwf=round(p_ecmwf, 4),
            p_hrrr=round(p_hrrr, 4),
            p_blended=round(p_yes, 4),
            disagreement=round(disagreement, 4),
            hrrr_tier=_hrrr_tier,
            w_hrrr=w_hrrr,
            n_ecmwf=len(ecmwf_members),
        )
    elif p_gfs is not None:
        # GFS only (no ECMWF, no HRRR)
        disagreement = abs(p_nws - p_gfs)
        if disagreement > 0.15:
            p_yes = 0.4 * p_nws + 0.6 * p_gfs
        else:
            p_yes = 0.6 * p_nws + 0.4 * p_gfs
        logger.info(
            "weather_gfs_blend",
            city=city,
            p_nws=round(p_nws, 4),
            p_gfs=round(p_gfs, 4),
            p_blended=round(p_yes, 4),
            disagreement=round(disagreement, 4),
            n_gfs=len(gfs_members),
        )
    elif p_ecmwf is not None:
        # ECMWF only (no GFS, no HRRR)
        disagreement = abs(p_nws - p_ecmwf)
        if disagreement > 0.15:
            p_yes = 0.4 * p_nws + 0.6 * p_ecmwf
        else:
            p_yes = 0.6 * p_nws + 0.4 * p_ecmwf
        logger.info(
            "weather_ecmwf_blend",
            city=city,
            p_nws=round(p_nws, 4),
            p_ecmwf=round(p_ecmwf, 4),
            p_blended=round(p_yes, 4),
            disagreement=round(disagreement, 4),
            n_ecmwf=len(ecmwf_members),
        )
    elif p_hrrr is not None:
        # HRRR only (both ensembles failed) — time-weight with NWS
        if _hrrr_tier == "near":
            p_yes = 0.35 * p_nws + 0.65 * p_hrrr  # HRRR dominant near close
        elif _hrrr_tier == "mid":
            p_yes = 0.45 * p_nws + 0.55 * p_hrrr
        else:
            p_yes = 0.55 * p_nws + 0.45 * p_hrrr
        logger.info(
            "weather_hrrr_blend",
            city=city,
            p_nws=round(p_nws, 4),
            p_hrrr=round(p_hrrr, 4),
            p_blended=round(p_yes, 4),
            hrrr_tier=_hrrr_tier,
        )
    else:
        # NWS only — no ensemble or HRRR data
        if p_graphcast is not None:
            # GraphCast available: 60% NWS + 40% GraphCast
            p_yes = 0.60 * p_nws + 0.40 * p_graphcast
            logger.info(
                "weather_nws_graphcast_blend",
                city=city,
                p_nws=round(p_nws, 4),
                p_graphcast=round(p_graphcast, 4),
                p_blended=round(p_yes, 4),
                graphcast_temp=round(graphcast_temp, 1) if graphcast_temp else None,
            )
        else:
            p_yes = p_nws

    # GraphCast post-blend refinement — when other sources already contributed,
    # GraphCast adds 20% weight as a 5th independent ML model signal.
    # Wave 25: increased from 10% to 20% — GraphCast is a distinct ML model
    # that adds real diversification beyond NWS/GFS/ECMWF/HRRR.
    # Skipped if GraphCast was already used in the NWS-only path above.
    if p_graphcast is not None and (
        p_gfs is not None or p_ecmwf is not None or p_hrrr is not None
    ):
        p_yes_pre_gc = p_yes
        p_yes = 0.80 * p_yes + 0.20 * p_graphcast
        logger.info(
            "graphcast_post_blend",
            city=city,
            p_pre=round(p_yes_pre_gc, 4),
            p_graphcast=round(p_graphcast, 4),
            p_adjusted=round(p_yes, 4),
            graphcast_temp=round(graphcast_temp, 1) if graphcast_temp else None,
        )

    # Clamp
    p_yes = max(0.02, min(0.98, p_yes))

    # 7e. NBM validation layer — compare against NWS gridpoints quantitative data
    # If our multi-model blend disagrees with NBM by >15%, adjust toward NBM.
    if coords:
        target_str = target_date.strftime("%Y-%m-%d") if target_date else now.strftime("%Y-%m-%d")
        p_yes, nbm_note = await compute_nbm_validation(
            p_blended=p_yes,
            t_value=t_value,
            t_type=t_type,
            sigma=sigma,
            coords=coords,
            target_date_str=target_str,
            bracket_bounds=bracket_bounds,
        )
        p_yes = max(0.02, min(0.98, p_yes))

    # 8. Only return for confident cases.
    # Lowered from 1.0 to 0.7 — at z=0.7, probability is ~76%/24%.
    # "Obvious bet" strategy: take many small edges on clearly directional
    # thresholds rather than waiting for 84%+ certainty.
    if "bracket" not in t_type:
        z_score = abs(forecast_temp - t_value) / sigma
        if z_score < 1.0:  # Wave 21: raised from 0.7 (NOAA direct 16.7% WR at 0.7)
            logger.info(
                "weather_direct_ambiguous",
                city=city,
                forecast=forecast_temp,
                threshold=t_value,
                z_score=round(z_score, 2),
                msg="Near threshold, deferring to LLM",
            )
            return None
    else:
        # For brackets, check distance from nearest bracket edge (not midpoint)
        # A 2-degree bracket means +-1F from midpoint is inside the bracket
        q = question.lower()
        m_b = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)", q)
        if m_b:
            lower_b = float(m_b.group(1))
            upper_b = float(m_b.group(2))
        else:
            lower_b = t_value - 1
            upper_b = t_value + 1
        # Distance from forecast to nearest bracket edge
        if forecast_temp < lower_b:
            edge_dist = lower_b - forecast_temp
        elif forecast_temp > upper_b:
            edge_dist = forecast_temp - upper_b
        else:
            edge_dist = 0  # forecast is inside the bracket
        z_from_edge = edge_dist / sigma
        z_score = abs(forecast_temp - t_value) / sigma
        # Hard physical floor: day-0 hourly (sigma<=1.5) uses 2°F floor,
        # longer-lead (sigma>1.5) keeps 3°F floor.
        floor = 2.0 if used_hourly else 3.0
        if edge_dist < floor:
            logger.info(
                "weather_bracket_physical_floor",
                market_id=market_id,
                forecast=forecast_temp,
                bracket=f"{lower_b}-{upper_b}",
                edge_dist=round(edge_dist, 1),
                floor=floor,
                msg=f"Forecast within {floor:.0f}°F of bracket edge",
            )
            return None
        # Z-score gate: uniform 2.0 sigma (was 1.5 hourly — let marginal brackets
        # through). Wave 16: weather brackets 1W/7L, -$4.83.
        z_gate = 2.0
        if z_from_edge < z_gate:
            logger.info(
                "weather_bracket_too_close",
                market_id=market_id,
                forecast=forecast_temp,
                bracket=f"{lower_b}-{upper_b}",
                edge_dist=round(edge_dist, 1),
                z_from_edge=round(z_from_edge, 2),
                z_gate=z_gate,
                msg=f"Forecast too close to bracket edge (z_gate={z_gate}), skipping",
            )
            return None

    confidence = min(0.95, 0.5 + z_score * 0.15) if "bracket" not in t_type else min(0.90, 0.4 + abs(0.5 - p_yes) * 1.5)

    # Forecast staleness check: if NWS forecast is >6h old, reduce confidence
    forecast_age_hours = _get_forecast_age_hours(forecast_url)
    if forecast_age_hours is not None and forecast_age_hours > 6.0:
        confidence *= 0.80
        logger.info(
            "stale_forecast_penalty",
            city=city,
            age_hours=round(forecast_age_hours, 1),
            confidence_after=round(confidence, 3),
        )

    source = "hourly" if used_hourly else "12h"
    gc_note = ""
    if p_graphcast is not None and graphcast_temp is not None:
        gc_note = f", GraphCast={graphcast_temp:.0f}°F (p={p_graphcast:.3f})"
    reasoning = (
        f"NOAA direct ({source}): NWS forecast={forecast_temp:.0f}°F, "
        f"threshold={t_value:.1f}°F, sigma={sigma:.1f}°F, "
        f"p_yes={p_yes:.3f} (city={city}, lead={lead_days}d){gc_note}"
    )

    logger.info(
        "noaa_direct_signal",
        city=city,
        forecast_temp=forecast_temp,
        threshold=t_value,
        t_type=t_type,
        sigma=sigma,
        p_yes=round(p_yes, 4),
        confidence=round(confidence, 3),
        lead_days=lead_days,
        graphcast_temp=round(graphcast_temp, 1) if graphcast_temp else None,
        p_graphcast=round(p_graphcast, 4) if p_graphcast is not None else None,
    )

    return (p_yes, confidence, reasoning)


def _format_weather_anchor(
    city: str,
    periods: List[Dict],
    target_date: Optional[datetime],
    question: str = "",
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

    # Threshold analysis — compare NWS forecast to market question
    threshold_info = _parse_weather_threshold(question) if question else None
    if threshold_info and relevant:
        t_type, t_value = threshold_info
        # Get forecast high/low temperatures
        forecast_temps = []
        for p in relevant:
            t = p.get("temperature")
            if isinstance(t, (int, float)):
                forecast_temps.append((p.get("name", ""), float(t)))

        if forecast_temps:
            # For high temp markets, use daytime period; for low, use nighttime
            if "high" in t_type:
                day_temps = [(n, t) for n, t in forecast_temps if "night" not in n.lower()]
                forecast_temp = day_temps[0][1] if day_temps else forecast_temps[0][1]
            else:
                night_temps = [(n, t) for n, t in forecast_temps if "night" in n.lower()]
                forecast_temp = night_temps[0][1] if night_temps else forecast_temps[-1][1]

            diff = forecast_temp - t_value
            abs_diff = abs(diff)

            if "bracket" in t_type:
                # For bracket markets, distance from bracket center
                if abs_diff < 1.5:
                    label = "CLOSE to bracket range"
                elif abs_diff < 4:
                    label = "well outside bracket range"
                else:
                    label = "FAR OUTSIDE bracket range"
            else:
                if abs_diff >= 8:
                    label = "FAR ABOVE" if diff > 0 else "FAR BELOW"
                elif abs_diff >= 4:
                    label = "well above" if diff > 0 else "well below"
                elif abs_diff >= 2:
                    label = "slightly above" if diff > 0 else "slightly below"
                else:
                    label = "CLOSE to threshold"

            context += (
                f"\n  THRESHOLD ANALYSIS:\n"
                f"    NWS forecast: {forecast_temp:.0f}°F\n"
                f"    Market threshold: {t_value:.1f}°F\n"
                f"    Assessment: Forecast is {label} threshold ({diff:+.1f}°F)\n"
            )

    context += (
        "  Source: NOAA National Weather Service (official US government forecast)\n"
        "  Use this forecast data as your starting anchor for weather markets.\n"
    )
    return context


def _parse_city_from_ticker(ticker: str) -> Optional[str]:
    """Extract city from Kalshi weather ticker (e.g., KXHIGHTSFO → san francisco)."""
    # Kalshi weather tickers: KXHIGH{T}{CODE}-date-params or KXLOW{T}{CODE}-date
    # Extract 2-4 char city code after KXHIGH/KXHIGHT/KXLOW/KXLOWT
    t = ticker.upper()
    code = None
    for prefix in ("KXHIGHT", "KXHIGH", "KXLOWT", "KXLOW", "KXRAIN", "KXSNOW", "KXTEMP", "KXWIND"):
        if t.startswith(prefix):
            rest = t[len(prefix):]
            # Extract code before the dash (e.g., "SFO" from "SFO-26FEB10-B56.5")
            code = rest.split("-")[0] if "-" in rest else rest
            break
    if not code:
        return None

    # Map Kalshi city codes to canonical city names
    _TICKER_CODE_MAP = {
        "SFO": "san francisco", "LAX": "los angeles", "CHI": "chicago",
        "AUS": "austin", "ATL": "atlanta", "NOLA": "new orleans",
        "DEN": "denver", "MIN": "minneapolis", "LV": "las vegas",
        "SEA": "seattle", "MIA": "miami", "NYC": "new york", "NY": "new york",
        "PHX": "phoenix", "BOS": "boston", "DFW": "dallas", "DAL": "dallas",
        "HOU": "houston", "PHL": "philadelphia", "PHIL": "philadelphia", "DET": "detroit",
        "DC": "washington",
        "MSP": "minneapolis", "TPA": "tampa", "CLE": "cleveland",
        "PIT": "pittsburgh", "CIN": "cincinnati", "STL": "st. louis",
        "MEM": "memphis", "NAS": "nashville", "OKC": "oklahoma city",
        "JAX": "jacksonville", "SLC": "salt lake city", "RAL": "raleigh",
        "MIL": "milwaukee", "BAL": "baltimore", "ALB": "albuquerque",
        "TUC": "tucson", "FRE": "fresno", "POR": "portland",
        "IND": "indianapolis", "CLT": "charlotte", "COL": "columbus",
        "FTW": "fort worth", "EP": "el paso", "LOU": "louisville",
        "KC": "kansas city", "SA": "san antonio", "SD": "san diego",
        "SJ": "san jose",
    }
    return _TICKER_CODE_MAP.get(code)


async def _get_weather_context(question: str, market_id: str = "") -> Optional[str]:
    """Get NOAA NWS forecast data for weather market questions."""
    city = _parse_city(question)
    # Fallback: extract city from ticker if not in question text
    if not city and market_id:
        city = _parse_city_from_ticker(market_id)
    if not city:
        logger.debug("weather_no_city_match", question=question[:80], ticker=market_id)
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

    anchor = _format_weather_anchor(city, periods, target_date, question)

    logger.info(
        "noaa_weather_anchor",
        city=city,
        target_date=target_date.strftime("%Y-%m-%d") if target_date else None,
        periods_found=len(periods),
    )

    return anchor


# ---------------------------------------------------------------------------
# Stock index fast-path — Yahoo Finance real-time prices + normal CDF
# ---------------------------------------------------------------------------

# Yahoo Finance symbol mapping and daily volatility estimates
_INDEX_CONFIG: Dict[str, Dict[str, Any]] = {
    "KXINXU": {"yahoo": "^GSPC", "name": "S&P 500", "daily_vol": 0.010, "trading_hours": 6.5},
    "KXINX": {"yahoo": "^GSPC", "name": "S&P 500", "daily_vol": 0.010, "trading_hours": 6.5},
    "KXNASDAQ100U": {"yahoo": "^NDX", "name": "NASDAQ 100", "daily_vol": 0.013, "trading_hours": 6.5},
    "KXNASDAQ100": {"yahoo": "^NDX", "name": "NASDAQ 100", "daily_vol": 0.013, "trading_hours": 6.5},
    "KXBTCD": {"yahoo": "BTC-USD", "name": "Bitcoin", "daily_vol": 0.04, "trading_hours": 24.0},
    "KXBTC": {"yahoo": "BTC-USD", "name": "Bitcoin", "daily_vol": 0.04, "trading_hours": 24.0},
    # ETH — Ethereum (same pattern as BTC)
    "KXETHD": {"yahoo": "ETH-USD", "name": "Ethereum", "daily_vol": 0.040, "trading_hours": 24.0},
    "KXETH": {"yahoo": "ETH-USD", "name": "Ethereum", "daily_vol": 0.040, "trading_hours": 24.0},
    # SPY/QQQ/IWM/DIA — ETF mirrors of existing index configs
    "KXSPY": {"yahoo": "^GSPC", "name": "S&P 500 (SPY)", "daily_vol": 0.010, "trading_hours": 6.5},
    "KXQQQ": {"yahoo": "^NDX", "name": "NASDAQ 100 (QQQ)", "daily_vol": 0.013, "trading_hours": 6.5},
    "KXIWM": {"yahoo": "^RUT", "name": "Russell 2000 (IWM)", "daily_vol": 0.012, "trading_hours": 6.5},
    "KXDIA": {"yahoo": "^DJI", "name": "Dow Jones (DIA)", "daily_vol": 0.009, "trading_hours": 6.5},
    # WTI Crude Oil — near-continuous futures (CME Globex), ~2.5% daily vol
    "KXWTI": {"yahoo": "CL=F", "name": "WTI Crude Oil", "daily_vol": 0.025, "trading_hours": 23.0},
    "KXWTIW": {"yahoo": "CL=F", "name": "WTI Crude Oil (Weekly)", "daily_vol": 0.025, "trading_hours": 23.0},
    # Gold — safe haven, ~1.2% daily vol
    "KXGOLD": {"yahoo": "GC=F", "name": "Gold", "daily_vol": 0.012, "trading_hours": 23.0},
}

# Cache for real-time prices: {symbol: (timestamp, price, prev_close)}
_index_price_cache: Dict[str, Tuple[float, float, float]] = {}
_INDEX_PRICE_CACHE_TTL = 60.0  # 60 seconds

# Cache for realized volatility: {symbol: (timestamp, vol)}
_realized_vol_cache: Dict[str, Tuple[float, float]] = {}
_REALIZED_VOL_CACHE_TTL = 3600.0  # 1 hour

# VIX1D cache: (timestamp, vix1d_value)
_vix1d_cache: Optional[Tuple[float, float]] = None
_VIX1D_CACHE_TTL = 300.0  # 5 minutes

# First-hour realized vol cache: {yahoo_symbol: (timestamp, vol)}
_first_hour_vol_cache: Dict[str, Tuple[float, float]] = {}
_FIRST_HOUR_VOL_CACHE_TTL = 300.0  # 5 minutes


async def _fetch_vix1d() -> Optional[float]:
    """Fetch VIX1D (1-day expected S&P 500 move) from Yahoo Finance.

    Returns daily vol as a fraction (e.g., 0.012 for 1.2%) or None on failure.
    VIX1D is annualized, so: daily_vol = VIX1D / 100 / sqrt(252).
    """
    global _vix1d_cache
    now = time.monotonic()
    if _vix1d_cache is not None and now - _vix1d_cache[0] < _VIX1D_CACHE_TTL:
        return _vix1d_cache[1]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX1D",
                params={"interval": "1m", "range": "1d"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if resp.status_code != 200:
                logger.debug("vix1d_fetch_failed", status=resp.status_code)
                return None

            data = resp.json()
            meta = data["chart"]["result"][0]["meta"]
            vix1d_raw = float(meta["regularMarketPrice"])

            # VIX1D is annualized percentage → convert to daily fraction
            daily_vol = vix1d_raw / 100.0 / math.sqrt(252)

            _vix1d_cache = (now, daily_vol)
            logger.info("vix1d_fetched", vix1d_raw=round(vix1d_raw, 2), daily_vol=round(daily_vol, 4))
            return daily_vol
    except Exception as e:
        logger.debug("vix1d_fetch_error", error=str(e))
        return None


async def _fetch_first_hour_vol(yahoo_symbol: str) -> Optional[float]:
    """Compute realized vol from first trading hour (9:30-10:30 ET) intraday data.

    After 10:30 ET, the first-hour realized vol explains ~68% of daily vol
    (Bloomberg research). Returns daily vol estimate as fraction or None.
    """
    from zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("America/New_York"))
    # Only available after 10:30 ET on trading days (Mon-Fri)
    if now_et.weekday() >= 5:  # Weekend
        return None
    current_hour = now_et.hour + now_et.minute / 60.0
    if current_hour < 10.5:  # Before 10:30 ET
        return None

    now_mono = time.monotonic()
    cached = _first_hour_vol_cache.get(yahoo_symbol)
    if cached and now_mono - cached[0] < _FIRST_HOUR_VOL_CACHE_TTL:
        return cached[1]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}",
                params={"interval": "1m", "range": "1d"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            result = data["chart"]["result"][0]
            timestamps = result.get("timestamp", [])
            closes = result["indicators"]["quote"][0].get("close", [])

            if not timestamps or not closes:
                return None

            # Filter to 9:30-10:30 ET window (timestamps are Unix epoch)
            et_tz = ZoneInfo("America/New_York")
            first_hour_closes = []
            for ts, c in zip(timestamps, closes):
                if c is None:
                    continue
                dt = datetime.fromtimestamp(ts, tz=et_tz)
                hour_frac = dt.hour + dt.minute / 60.0
                if 9.5 <= hour_frac <= 10.5:
                    first_hour_closes.append(c)

            if len(first_hour_closes) < 10:  # Need at least 10 minutes of data
                return None

            # Compute 1-min log returns
            returns = [
                math.log(first_hour_closes[i] / first_hour_closes[i - 1])
                for i in range(1, len(first_hour_closes))
            ]
            if not returns:
                return None

            mean_r = sum(returns) / len(returns)
            var = sum((r - mean_r) ** 2 for r in returns) / max(len(returns) - 1, 1)
            vol_per_min = math.sqrt(var)

            # Scale to daily: 1-min vol * sqrt(390 trading minutes)
            daily_vol = vol_per_min * math.sqrt(390)

            _first_hour_vol_cache[yahoo_symbol] = (now_mono, daily_vol)
            logger.info(
                "first_hour_vol",
                symbol=yahoo_symbol,
                n_bars=len(first_hour_closes),
                vol_per_min=round(vol_per_min, 6),
                daily_vol=round(daily_vol, 4),
            )
            return daily_vol
    except Exception as e:
        logger.debug("first_hour_vol_error", symbol=yahoo_symbol, error=str(e))
        return None


async def _fetch_realized_volatility(yahoo_symbol: str) -> Optional[float]:
    """Fetch 10-day realized volatility from Yahoo Finance daily bars.

    Returns annualized daily vol (stdev of log returns) or None on failure.
    Cached for 1 hour.
    """
    now = time.monotonic()
    cached = _realized_vol_cache.get(yahoo_symbol)
    if cached and now - cached[0] < _REALIZED_VOL_CACHE_TTL:
        return cached[1]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}",
                params={"interval": "1d", "range": "10d"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if resp.status_code != 200:
                return None

            closes = resp.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            closes = [c for c in closes if c is not None]
            if len(closes) < 3:
                return None

            returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
            mean_r = sum(returns) / len(returns)
            var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
            vol = math.sqrt(var)
            _realized_vol_cache[yahoo_symbol] = (now, vol)
            return vol
    except Exception:
        return None


async def _fetch_index_price(yahoo_symbol: str) -> Optional[Tuple[float, float]]:
    """Fetch real-time price and previous close from Yahoo Finance.

    Returns (current_price, previous_close) or None on failure.
    Cached for 60 seconds.
    """
    now = time.monotonic()
    cached = _index_price_cache.get(yahoo_symbol)
    if cached is not None:
        ts, price, prev_close = cached
        if now - ts < _INDEX_PRICE_CACHE_TTL:
            return (price, prev_close)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}",
                params={"interval": "1m", "range": "1d"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if resp.status_code != 200:
                logger.debug("yahoo_fetch_failed", symbol=yahoo_symbol, status=resp.status_code)
                return None

            data = resp.json()
            result = data["chart"]["result"][0]
            meta = result["meta"]
            price = float(meta["regularMarketPrice"])
            prev_close = float(meta["chartPreviousClose"])

            _index_price_cache[yahoo_symbol] = (now, price, prev_close)
            return (price, prev_close)
    except Exception as e:
        logger.debug("yahoo_fetch_error", symbol=yahoo_symbol, error=str(e))
        return None


def _parse_index_ticker(market_id: str) -> Optional[Dict[str, Any]]:
    """Parse Kalshi stock index ticker into components.

    Examples:
        KXINXU-26FEB12H1600-T6974     → S&P threshold 6974
        KXINXU-26FEB12H1600-T6924.9999 → S&P threshold 6925
        KXNASDAQ100U-26FEB12H1000-T25279.99 → NASDAQ threshold 25280
        KXINX-26FEB12H1600-B6987      → S&P bracket 6987

    Returns dict with: prefix, threshold, is_bracket, close_hour, config
    """
    mid = market_id.upper()

    # Find matching prefix
    matched_prefix = None
    config = None
    for prefix, cfg in _INDEX_CONFIG.items():
        if mid.startswith(prefix):
            # Pick longest matching prefix (KXINXU before KXINX)
            if matched_prefix is None or len(prefix) > len(matched_prefix):
                matched_prefix = prefix
                config = cfg

    if matched_prefix is None or config is None:
        return None

    # Parse threshold/bracket value
    t_match = re.search(r"-T([\d.]+)$", mid)
    b_match = re.search(r"-B([\d.]+)$", mid)

    if t_match:
        # Round .9999 values up (Kalshi convention for "above X")
        raw_val = float(t_match.group(1))
        threshold = round(raw_val + 0.0001)  # 6924.9999 → 6925
        is_bracket = False
    elif b_match:
        threshold = float(b_match.group(1))
        is_bracket = True
    else:
        return None

    # Parse close hour from ticker (H1600 = 16:00 ET)
    h_match = re.search(r"H(\d{4})", mid)
    close_hour = int(h_match.group(1)) / 100.0 if h_match else 16.0

    return {
        "prefix": matched_prefix,
        "threshold": threshold,
        "is_bracket": is_bracket,
        "close_hour_et": close_hour,
        "config": config,
    }


def _market_hours_remaining(close_hour_et: float) -> Optional[float]:
    """Compute hours remaining in the trading session.

    Returns None if market is closed or close time has passed.
    NYSE hours: 9:30 AM - 4:00 PM ET.
    """
    from zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("America/New_York"))
    current_hour = now_et.hour + now_et.minute / 60.0

    # Market open at 9:30 ET
    market_open = 9.5
    # Close hour from ticker (usually 16.0 = 4:00 PM)
    market_close = close_hour_et

    if current_hour >= market_close:
        return 0.0  # Market closed
    if current_hour < market_open:
        return market_close - market_open  # Full day remaining

    return market_close - current_hour


def _parse_bracket_range(question: str) -> Optional[Tuple[float, float]]:
    """Parse bracket range from Kalshi market question text.

    Examples:
        "Will the S&P 500 close between 6,950 and 7,000?" → (6950, 7000)
        "Will Bitcoin be between $95,000 and $97,500?" → (95000, 97500)
        "...close between 21,000 and 21,050..." → (21000, 21050)

    Returns (lower, upper) or None if can't parse.
    """
    # Pattern: "between X and Y" where X, Y may have commas, $, decimals
    match = re.search(
        r'between\s+\$?([\d,]+(?:\.\d+)?)\s+and\s+\$?([\d,]+(?:\.\d+)?)',
        question, re.IGNORECASE,
    )
    if match:
        try:
            lower = float(match.group(1).replace(",", ""))
            upper = float(match.group(2).replace(",", ""))
            if lower < upper:
                return (lower, upper)
        except ValueError:
            pass
    return None


async def _best_daily_vol(config: Dict[str, Any]) -> Tuple[float, str]:
    """Select best daily volatility estimate from multiple sources.

    Priority (for equity indices only):
    1. First-hour realized vol (after 10:30 ET, explains ~68% of daily vol)
    2. VIX1D implied vol (market-implied same-day expectation)
    3. 10-day realized vol (historical baseline)
    4. Static config default (floor)

    For non-equity (crypto, commodities): uses realized vol or config default.

    Returns (daily_vol, source_name).
    """
    yahoo = config["yahoo"]
    floor = config["daily_vol"]
    trading_hours = config.get("trading_hours", 6.5)
    is_equity = trading_hours < 24.0 and yahoo in ("^GSPC", "^NDX", "^RUT", "^DJI")

    if is_equity:
        # Try first-hour vol (best after 10:30 ET)
        fh_vol = await _fetch_first_hour_vol(yahoo)
        if fh_vol is not None and fh_vol > floor * 0.5:
            # Blend: 60% first-hour, 40% VIX1D/realized for stability
            vix1d = await _fetch_vix1d()
            if vix1d is not None and vix1d > floor * 0.5:
                blended = 0.6 * fh_vol + 0.4 * vix1d
                return (max(blended, floor), "first_hour+vix1d")
            return (max(fh_vol, floor), "first_hour")

        # Try VIX1D (market-implied same-day vol)
        vix1d = await _fetch_vix1d()
        if vix1d is not None and vix1d > floor * 0.5:
            return (max(vix1d, floor), "vix1d")

    # Fallback: 10-day realized vol
    realized = await _fetch_realized_volatility(yahoo)
    if realized is not None and realized > floor:
        return (realized, "realized_10d")

    return (floor, "config_default")


async def compute_stock_index_probability(
    question: str, market_id: str,
    close_time: Optional[datetime] = None,
) -> Optional[Tuple[float, float, str]]:
    """Compute probability for stock index threshold markets.

    Uses real-time Yahoo Finance price + dynamic volatility model.
    Vol sources (equity): first-hour realized → VIX1D implied → 10d realized → static.
    Returns (p_yes, confidence, reasoning) or None if can't compute.
    """
    parsed = _parse_index_ticker(market_id)
    if parsed is None:
        return None

    config = parsed["config"]
    threshold = parsed["threshold"]
    is_bracket = parsed["is_bracket"]

    # Bracket markets: P(inside range) = CDF(upper) - CDF(lower)
    if is_bracket:
        bracket_range = _parse_bracket_range(question)
        if bracket_range is None:
            logger.debug("index_bracket_parse_fail", market_id=market_id, question=question[:100])
            return None

        lower, upper = bracket_range

        # Fetch real-time price for bracket
        price_data = await _fetch_index_price(config["yahoo"])
        if price_data is None:
            return None
        current_price, prev_close = price_data
        if current_price <= 0:
            return None

        # Compute remaining time
        trading_hours = config["trading_hours"]
        if trading_hours >= 24.0 and close_time is not None:
            remaining = (close_time - datetime.now(timezone.utc)).total_seconds() / 3600
            hours_left = max(remaining, 0.001)
        else:
            hours_left = _market_hours_remaining(parsed["close_hour_et"])
            if hours_left is None or hours_left <= 0:
                hours_left = 0.001

        daily_vol, vol_source = await _best_daily_vol(config)
        time_fraction = max(hours_left / trading_hours, 0.001)
        sigma_remaining = current_price * daily_vol * math.sqrt(time_fraction)
        sigma_remaining = max(sigma_remaining, current_price * 0.001)

        # P(lower < price < upper at close)
        p_inside = _norm_cdf(upper, current_price, sigma_remaining) - _norm_cdf(lower, current_price, sigma_remaining)
        p_inside = max(0.02, min(0.98, p_inside))

        # Safety gate: z-score from nearest bracket edge must be >= 1.5
        dist_lower = abs(current_price - lower)
        dist_upper = abs(current_price - upper)
        z_from_edge = min(dist_lower, dist_upper) / sigma_remaining if sigma_remaining > 0 else 0
        if z_from_edge < 1.5:
            logger.info(
                "index_bracket_ambiguous",
                market_id=market_id,
                current=current_price,
                lower=lower,
                upper=upper,
                z_from_edge=round(z_from_edge, 2),
            )
            return None

        confidence = min(0.85, 0.45 + z_from_edge * 0.08)
        daily_change_pct = ((current_price - prev_close) / prev_close) * 100

        reasoning = (
            f"Yahoo Finance direct (bracket): {config['name']} = {current_price:.2f} "
            f"(day: {daily_change_pct:+.2f}%), range = [{lower:.0f}, {upper:.0f}], "
            f"σ_remaining = {sigma_remaining:.1f} pts ({hours_left:.1f}h left), "
            f"vol_source = {vol_source}, daily_vol = {daily_vol:.4f}, "
            f"z_from_edge = {z_from_edge:.2f}, P(inside) = {p_inside:.4f}"
        )

        logger.info(
            "stock_index_bracket_fast_path",
            market_id=market_id,
            index=config["name"],
            current=current_price,
            lower=lower,
            upper=upper,
            sigma=round(sigma_remaining, 1),
            hours_left=round(hours_left, 2),
            z_from_edge=round(z_from_edge, 2),
            p_inside=round(p_inside, 4),
            vol_source=vol_source,
            daily_vol=round(daily_vol, 4),
        )

        return (p_inside, confidence, reasoning)

    # Fetch real-time price
    price_data = await _fetch_index_price(config["yahoo"])
    if price_data is None:
        return None

    current_price, prev_close = price_data
    if current_price <= 0:
        return None

    # Compute remaining time
    trading_hours = config["trading_hours"]
    if trading_hours >= 24.0 and close_time is not None:
        # 24h market (crypto): use close_time directly
        remaining = (close_time - datetime.now(timezone.utc)).total_seconds() / 3600
        hours_left = max(remaining, 0.001)
    else:
        # NYSE hours: use ticker close hour
        hours_left = _market_hours_remaining(parsed["close_hour_et"])
        if hours_left is None or hours_left <= 0:
            hours_left = 0.001  # tiny epsilon to avoid division by zero

    # Intraday volatility: daily_vol * sqrt(hours_left / trading_hours)
    daily_vol, vol_source = await _best_daily_vol(config)
    time_fraction = max(hours_left / trading_hours, 0.001)
    sigma_remaining = current_price * daily_vol * math.sqrt(time_fraction)

    # Add minimum sigma floor (prevent overconfidence when close to threshold)
    sigma_remaining = max(sigma_remaining, current_price * 0.001)  # 0.1% floor

    # P(index > threshold at close)
    p_above = 1.0 - _norm_cdf(threshold, current_price, sigma_remaining)
    p_above = max(0.02, min(0.98, p_above))

    # z-score for confidence
    z_score = abs(current_price - threshold) / sigma_remaining if sigma_remaining > 0 else 0
    # Crypto needs higher z-gate due to higher volatility and model uncertainty
    is_crypto = config.get("trading_hours", 6.5) >= 24.0
    min_z = 0.5  # Uniform z-gate: 0.3 was too permissive, took marginal index trades
    if z_score < min_z:
        logger.info(
            "stock_index_ambiguous",
            market_id=market_id,
            current=current_price,
            threshold=threshold,
            z_score=round(z_score, 2),
            hours_left=round(hours_left, 2),
            min_z=min_z,
        )
        return None

    confidence = min(0.90, 0.50 + z_score * 0.10)

    # Direction of daily change for context
    daily_change_pct = ((current_price - prev_close) / prev_close) * 100

    reasoning = (
        f"Yahoo Finance direct: {config['name']} = {current_price:.2f} "
        f"(day: {daily_change_pct:+.2f}%), threshold = {threshold:.0f}, "
        f"gap = {current_price - threshold:+.1f} pts, "
        f"σ_remaining = {sigma_remaining:.1f} pts ({hours_left:.1f}h left), "
        f"vol_source = {vol_source}, daily_vol = {daily_vol:.4f}, "
        f"z = {z_score:.2f}, P(above) = {p_above:.4f}"
    )

    logger.info(
        "stock_index_fast_path",
        market_id=market_id,
        index=config["name"],
        current=current_price,
        threshold=threshold,
        gap=round(current_price - threshold, 1),
        sigma=round(sigma_remaining, 1),
        hours_left=round(hours_left, 2),
        z_score=round(z_score, 2),
        p_above=round(p_above, 4),
        vol_source=vol_source,
        daily_vol=round(daily_vol, 4),
    )

    return (p_above, confidence, reasoning)


async def _get_stock_index_context(
    question: str, market_id: str,
) -> Optional[str]:
    """Get structured data context for stock index markets."""
    parsed = _parse_index_ticker(market_id)
    if parsed is None:
        return None

    config = parsed["config"]
    price_data = await _fetch_index_price(config["yahoo"])
    if price_data is None:
        return None

    current_price, prev_close = price_data
    daily_change_pct = ((current_price - prev_close) / prev_close) * 100
    hours_left = _market_hours_remaining(parsed["close_hour_et"])

    return (
        f"\nSTRUCTURED DATA ANCHOR — {config['name']} (Real-Time):\n"
        f"  Current Level: {current_price:.2f}\n"
        f"  Previous Close: {prev_close:.2f}\n"
        f"  Day Change: {daily_change_pct:+.2f}%\n"
        f"  Threshold: {parsed['threshold']:.0f}\n"
        f"  Gap to Threshold: {current_price - parsed['threshold']:+.1f} points\n"
        f"  Hours to Market Close: {hours_left:.1f}h\n"
        f"  Daily Volatility: ~{config['daily_vol']*100:.1f}%\n"
        f"  Use this as your primary anchor. The current price is HARD DATA.\n"
    )


# ---------------------------------------------------------------------------
# Jobless claims fast-path — FRED ICSA data + historical distribution
# ---------------------------------------------------------------------------

async def compute_jobless_claims_probability(
    question: str, market_id: str = "",
) -> Optional[Tuple[float, float, str]]:
    """Compute probability for initial jobless claims markets from FRED data.

    Uses 4-week moving average as forecast and 8-week stdev as error.
    Returns (p_yes, confidence, reasoning) or None if can't compute.
    """
    if not _get_fred_api_key():
        return None

    # Parse threshold from market question or ticker
    threshold = _parse_jobless_threshold(question, market_id)
    if threshold is None:
        return None

    # Fetch recent ICSA data from FRED
    obs = await _fetch_fred_series("ICSA", limit=12)
    if not obs or len(obs) < 4:
        return None

    vals = []
    for o in obs:
        v = _safe_float(o.get("value", ""))
        if v is not None:
            vals.append(v)

    if len(vals) < 4:
        return None

    avg_4w = sum(vals[:4]) / 4
    # Use 8-week stdev for volatility estimate (minimum 8k floor)
    import statistics
    stdev = max(8000.0, statistics.stdev(vals[:min(8, len(vals))]))

    # Compute P(claims > threshold) using normal CDF
    p_above = 1.0 - _norm_cdf(threshold, avg_4w, stdev)
    p_above = max(0.02, min(0.98, p_above))

    # Confidence based on z-score (how far from threshold)
    z_score = abs(avg_4w - threshold) / stdev
    if z_score < 0.5:
        # Too close to call — let LLM handle it
        return None

    confidence = min(0.90, 0.5 + z_score * 0.15)

    reasoning = (
        f"FRED ICSA direct: 4wk avg={avg_4w:.0f}, stdev={stdev:.0f}, "
        f"threshold={threshold:.0f}, p_above={p_above:.3f}, z={z_score:.2f}"
    )

    logger.info(
        "jobless_claims_fast_path",
        avg_4w=avg_4w,
        stdev=round(stdev),
        threshold=threshold,
        p_above=round(p_above, 4),
        z_score=round(z_score, 2),
    )

    return (p_above, confidence, reasoning)


def _parse_jobless_threshold(question: str, market_id: str = "") -> Optional[float]:
    """Extract jobless claims threshold from question or ticker.

    Kalshi tickers: KXJOBLESSCLAIMS-26FEB12-215000 → threshold 215000
    Question: "Will initial jobless claims be above 215,000?" → 215000
    """
    # Try ticker first (most reliable)
    if market_id:
        m = re.search(r"KXJOBLESSCLAIMS-\w+-(\d+)", market_id.upper())
        if m:
            return float(m.group(1))

    # Try question text
    q = question.lower()
    patterns = [
        r"(?:above|over|exceed|more than)\s+([\d,]+)",
        r"([\d,]+)\s+(?:or more|or higher)",
    ]
    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue

    return None


# ---------------------------------------------------------------------------
# Economic Data Release Sniping
# ---------------------------------------------------------------------------

# FRED series + typical release schedule (hour in ET, day pattern)
_ECON_RELEASES = {
    "cpi": {
        "series_id": "CPIAUCSL",
        "label": "CPI Index (SA)",
        "release_hour_et": 8,   # 8:30 AM ET
        "release_minute_et": 30,
        "typical_day_of_month": (10, 14),  # usually 10th-14th
        "sigma_pct": 0.2,       # typical MoM surprise magnitude
    },
    "jobs": {
        "series_id": "PAYEMS",
        "label": "Nonfarm Payrolls (thousands)",
        "release_hour_et": 8,
        "release_minute_et": 30,
        "typical_day_of_month": (1, 7),  # first Friday
        "sigma_pct": 0.5,       # jobs often surprise by 50K+
    },
    "unemployment": {
        "series_id": "UNRATE",
        "label": "Unemployment Rate (%)",
        "release_hour_et": 8,
        "release_minute_et": 30,
        "typical_day_of_month": (1, 7),
        "sigma_pct": 0.1,       # typically ±0.1%
    },
    "gdp": {
        "series_id": "A191RL1Q225SBEA",
        "label": "Real GDP Growth Rate (%)",
        "release_hour_et": 8,
        "release_minute_et": 30,
        "typical_day_of_month": (25, 30),  # end of month
        "sigma_pct": 0.3,
    },
    "ppi": {
        "series_id": "PPIFIS",
        "label": "PPI Final Demand",
        "release_hour_et": 8,
        "release_minute_et": 30,
        "typical_day_of_month": (11, 16),
        "sigma_pct": 0.3,
    },
    "jobless_claims": {
        "series_id": "ICSA",
        "label": "Initial Jobless Claims",
        "release_hour_et": 8,
        "release_minute_et": 30,
        "typical_day_of_month": (1, 31),  # every Thursday
        "sigma_pct": 2.0,  # claims can swing 10-20K
    },
}

# Cache for last-seen FRED values (to detect new releases)
_econ_last_seen: Dict[str, Tuple[str, float]] = {}  # series_id → (date, value)


async def check_economic_release(
    question: str, market_id: str = "",
) -> Optional[Tuple[float, float, str]]:
    """Check if a relevant economic release just happened.

    Compares latest FRED data against previous value + trend.
    Returns (p_yes, confidence, reasoning) if a significant release is detected,
    None otherwise. This is a fast-path that bypasses LLM.

    Only fires near scheduled release times (±2 hours) to avoid false positives.
    """
    q = question.lower()

    # Match question to economic indicator
    matched_release = None
    if any(w in q for w in ["cpi", "inflation", "consumer price"]):
        matched_release = "cpi"
    elif any(w in q for w in ["nonfarm", "payroll", "jobs report", "jobs added"]):
        matched_release = "jobs"
    elif any(w in q for w in ["unemployment rate", "jobless rate"]):
        matched_release = "unemployment"
    elif "gdp" in q:
        matched_release = "gdp"
    elif "ppi" in q:
        matched_release = "ppi"
    elif any(w in q for w in ["jobless claims", "initial claims", "weekly claims"]):
        matched_release = "jobless_claims"

    if not matched_release:
        return None

    release_info = _ECON_RELEASES[matched_release]
    series_id = release_info["series_id"]

    # Check if we're near a release window (within 2 hours after scheduled time)
    now_utc = datetime.now(timezone.utc)
    # ET is UTC-5 (EST) or UTC-4 (EDT). Use UTC-5 as conservative estimate.
    now_et_hour = (now_utc.hour - 5) % 24
    release_hour = release_info["release_hour_et"]
    hours_since_release = now_et_hour - release_hour
    if hours_since_release < 0:
        hours_since_release += 24

    # Only check within 2 hours after scheduled release time
    if hours_since_release > 2:
        return None

    # Fetch latest FRED data
    observations = await _fetch_fred_series(series_id, limit=3)
    if len(observations) < 2:
        return None

    try:
        latest_val = float(observations[0]["value"])
        latest_date = observations[0]["date"]
        prev_val = float(observations[1]["value"])
    except (ValueError, TypeError, KeyError):
        return None

    # Check if this is a NEW release (not already seen)
    last_seen = _econ_last_seen.get(series_id)
    if last_seen and last_seen[0] == latest_date:
        return None  # Already processed this release

    _econ_last_seen[series_id] = (latest_date, latest_val)

    # Compute surprise magnitude
    if prev_val == 0:
        return None
    change_pct = ((latest_val - prev_val) / abs(prev_val)) * 100
    sigma = release_info["sigma_pct"]
    z_score = abs(change_pct) / sigma if sigma > 0 else 0

    # Only signal on significant surprises (> 1 sigma)
    if z_score < 1.0:
        logger.info(
            "econ_release_insignificant",
            indicator=matched_release,
            latest=latest_val,
            previous=prev_val,
            change_pct=round(change_pct, 3),
            z_score=round(z_score, 2),
        )
        return None

    # Parse threshold from question to compute probability
    threshold = _parse_econ_threshold(question, matched_release)
    if threshold is None:
        logger.info(
            "econ_release_no_threshold",
            indicator=matched_release,
            latest=latest_val,
            msg="Could not parse threshold from question",
        )
        return None

    # Compute probability: is the actual value above or below the threshold?
    if latest_val >= threshold:
        p_yes = min(0.95, 0.70 + z_score * 0.05)  # data already above → likely YES
    else:
        p_yes = max(0.05, 0.30 - z_score * 0.05)  # data below → likely NO

    confidence = min(0.90, 0.60 + z_score * 0.10)

    reasoning = (
        f"Econ release snipe ({matched_release}): {release_info['label']} = {latest_val} "
        f"(prev: {prev_val}, change: {change_pct:+.2f}%, z={z_score:.1f}σ). "
        f"Threshold: {threshold}, p_yes={p_yes:.3f}"
    )

    logger.info(
        "econ_release_signal",
        indicator=matched_release,
        latest=latest_val,
        previous=prev_val,
        change_pct=round(change_pct, 3),
        z_score=round(z_score, 2),
        threshold=threshold,
        p_yes=round(p_yes, 4),
        confidence=round(confidence, 3),
    )

    return (p_yes, confidence, reasoning)


def _parse_econ_threshold(question: str, indicator: str) -> Optional[float]:
    """Parse the threshold value from an economic market question.

    Examples:
      "Will CPI increase by more than 0.3% MoM?" → 0.3
      "Will unemployment rate be above 4.0%?" → 4.0
      "Will nonfarm payrolls exceed 200,000?" → 200000
    """
    q = question.lower()

    # Try common patterns
    patterns = [
        r"(?:above|over|exceed|more than|at least|higher than)\s+([\d,.]+)",
        r"([\d,.]+)\s*%",
        r"(?:below|under|less than|lower than)\s+([\d,.]+)",
        r"([\d,]+)\s+(?:or more|or higher|or greater)",
    ]

    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue

    return None
