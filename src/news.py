"""News fetching via web search for market context.

Priority: Tavily (best for AI agents) > Brave Search > DuckDuckGo fallback.
"""

import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urlencode

import httpx
import structlog

from .markets import Market
from .utils import BotConfig, RateLimiter


# ---------------------------------------------------------------------------
# Source credibility — weight authoritative sources higher
# ---------------------------------------------------------------------------

_HIGH_CREDIBILITY_DOMAINS = {
    "reuters.com", "apnews.com", "bloomberg.com", "wsj.com",
    "nytimes.com", "washingtonpost.com", "bbc.com", "bbc.co.uk",
    "ft.com", "economist.com", "cnbc.com", "politico.com",
    "thehill.com", "axios.com", "bls.gov", "federalreserve.gov",
    "treasury.gov", "congress.gov", "whitehouse.gov",
}

_MEDIUM_CREDIBILITY_DOMAINS = {
    "cnn.com", "foxnews.com", "msnbc.com", "npr.org",
    "theguardian.com", "usatoday.com", "abcnews.go.com",
    "cbsnews.com", "nbcnews.com", "pbs.org",
}


def _source_credibility_score(url: str) -> float:
    """Return a credibility score 0.0-1.0 for a news source URL."""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.replace("www.", "")
        parts = domain.split(".")
        if len(parts) >= 2:
            domain = ".".join(parts[-2:])
        if domain in _HIGH_CREDIBILITY_DOMAINS:
            return 1.0
        if domain in _MEDIUM_CREDIBILITY_DOMAINS:
            return 0.7
    except Exception:
        pass
    return 0.4


# ---------------------------------------------------------------------------
# Domain-specific query expansion
# ---------------------------------------------------------------------------

_DOMAIN_EXPANSIONS = {
    "fed": ["FOMC", "interest rate decision", "monetary policy"],
    "rate cut": ["FOMC meeting", "federal funds rate", "CME FedWatch"],
    "rate hike": ["FOMC meeting", "federal funds rate", "tightening"],
    "tariff": ["trade policy", "import duties", "trade war"],
    "shutdown": ["government shutdown", "continuing resolution", "funding"],
    "cpi": ["inflation data", "consumer price index", "BLS report"],
    "gdp": ["economic growth", "GDP report", "BEA data"],
    "jobs report": ["nonfarm payrolls", "unemployment rate", "BLS"],
    "election": ["polling", "election forecast", "voter turnout"],
    "impeach": ["impeachment", "congressional vote", "articles of impeachment"],
    "supreme court": ["SCOTUS", "court ruling", "oral arguments"],
    "executive order": ["presidential action", "White House directive"],
}


def _html_to_text(html: str, max_words: int = 500) -> str:
    """Strip HTML tags and return first max_words words of text."""
    # Remove script/style blocks
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Remove tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    return ' '.join(words[:max_words])


async def _fetch_article_text(url: str, timeout: float = 5.0, max_chars: int = 500) -> str:
    """Fetch full article text from a URL. Returns empty string on failure."""
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            text = _html_to_text(resp.text)
            return text[:max_chars]
    except Exception:
        return ""


@dataclass
class NewsArticle:
    """Represents a news article."""

    title: str
    summary: str
    url: str
    published: datetime
    source: str
    full_text: str = ""  # Full article content (fetched separately)

    def is_recent(self, hours: int = 72) -> bool:
        """Check if article is recent."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return self.published > cutoff

    def get_snippet(self, max_chars: int = 300) -> str:
        """Get a shortened snippet of the article."""
        text = f"{self.title}. {self.summary}"
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > max_chars * 0.8:
            truncated = truncated[:last_space]
        return truncated + "..."


class TavilySource:
    """Tavily Search API — built for AI agents. Free tier: 1,000 queries/month.
    
    Returns pre-cleaned summaries, ideal for LLM context.
    Sign up: https://tavily.com
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"

    async def search(self, query: str, max_articles: int = 8) -> List[NewsArticle]:
        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
                "max_results": min(max_articles, 10),
                "topic": "news",
                "days": 7,
            }
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(self.base_url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            articles = []
            for result in data.get("results", [])[:max_articles]:
                published = datetime.now(timezone.utc)
                pub_date = result.get("published_date")
                if pub_date:
                    try:
                        published = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                    except Exception:
                        pass

                articles.append(NewsArticle(
                    title=result.get("title", ""),
                    summary=result.get("content", ""),
                    url=result.get("url", ""),
                    published=published,
                    source="Tavily",
                ))
            return articles
        except Exception as e:
            structlog.get_logger().warning("tavily_search_error", error=str(e))
            return []


class BraveSearchSource:
    """Brave Search API — free tier gives 2,000 queries/month."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self._last_call: float = 0.0  # monotonic timestamp of last request
        self._cooldown_until: float = 0.0  # skip Brave until this time after 429

    async def search(self, query: str, max_articles: int = 8) -> List[NewsArticle]:
        # If in cooldown after a 429, skip immediately so Google RSS fallback takes over
        now = asyncio.get_event_loop().time()
        if now < self._cooldown_until:
            return []

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
            "count": min(max_articles, 20),
            "freshness": "pw",  # past week
            "text_decorations": "false",
        }

        try:
            # Enforce 2s minimum between Brave calls to avoid 429s
            elapsed = now - self._last_call
            if elapsed < 2.0:
                await asyncio.sleep(2.0 - elapsed)
            self._last_call = asyncio.get_event_loop().time()

            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(self.base_url, headers=headers, params=params)
                if resp.status_code == 429:
                    # Set 120s cooldown — all subsequent calls fall through to Google RSS
                    self._cooldown_until = asyncio.get_event_loop().time() + 120.0
                    structlog.get_logger().warning(
                        "brave_search_429_cooldown",
                        cooldown_seconds=120,
                    )
                    return []
                resp.raise_for_status()
                data = resp.json()

            articles = []
            for result in data.get("web", {}).get("results", [])[:max_articles]:
                published = datetime.now(timezone.utc)
                age = result.get("age")
                if age:
                    published = self._parse_age(age)

                articles.append(NewsArticle(
                    title=result.get("title", ""),
                    summary=result.get("description", ""),
                    url=result.get("url", ""),
                    published=published,
                    source="Brave Search",
                ))
            return articles
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                self._cooldown_until = asyncio.get_event_loop().time() + 120.0
            structlog.get_logger().warning("brave_search_error", error=str(e))
            return []
        except Exception as e:
            structlog.get_logger().warning("brave_search_error", error=str(e))
            return []

    @staticmethod
    def _parse_age(age_str: str) -> datetime:
        """Parse Brave's age strings like '2 hours ago', '3 days ago'."""
        now = datetime.now(timezone.utc)
        try:
            parts = age_str.lower().split()
            if len(parts) >= 2:
                num = int(parts[0])
                unit = parts[1]
                if "hour" in unit:
                    return now - timedelta(hours=num)
                elif "day" in unit:
                    return now - timedelta(days=num)
                elif "minute" in unit:
                    return now - timedelta(minutes=num)
                elif "week" in unit:
                    return now - timedelta(weeks=num)
                elif "month" in unit:
                    return now - timedelta(days=num * 30)
        except Exception:
            pass
        return now


class GoogleNewsRSSSource:
    """Google News RSS — free, no API key, reliable. Returns recent headlines."""

    async def search(self, query: str, max_articles: int = 8) -> List[NewsArticle]:
        try:
            encoded = quote_plus(query)
            url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
            async with httpx.AsyncClient(
                timeout=10.0,
                headers={"User-Agent": "Mozilla/5.0"},
                follow_redirects=True,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                xml = resp.text

            articles = []
            # Parse RSS items with regex (avoid xml library dependency)
            items = re.findall(r"<item>(.*?)</item>", xml, re.DOTALL)
            for item_xml in items[:max_articles]:
                title_m = re.search(r"<title>(.*?)</title>", item_xml, re.DOTALL)
                link_m = re.search(r"<link>(.*?)</link>", item_xml, re.DOTALL)
                pub_m = re.search(r"<pubDate>(.*?)</pubDate>", item_xml, re.DOTALL)
                source_m = re.search(r"<source[^>]*>(.*?)</source>", item_xml, re.DOTALL)

                title = title_m.group(1).strip() if title_m else ""
                link = link_m.group(1).strip() if link_m else ""
                source_name = source_m.group(1).strip() if source_m else "Google News"

                published = datetime.now(timezone.utc)
                if pub_m:
                    try:
                        from email.utils import parsedate_to_datetime
                        published = parsedate_to_datetime(pub_m.group(1).strip()).replace(
                            tzinfo=timezone.utc
                        ) if parsedate_to_datetime(pub_m.group(1).strip()).tzinfo is None else parsedate_to_datetime(pub_m.group(1).strip())
                    except Exception:
                        pass

                if title:
                    articles.append(NewsArticle(
                        title=title,
                        summary=title,  # RSS titles are often descriptive
                        url=link,
                        published=published,
                        source=f"Google News ({source_name})",
                    ))

            return articles
        except Exception as e:
            structlog.get_logger().warning("google_news_rss_error", error=str(e))
            return []


class DuckDuckGoSource:
    """DuckDuckGo HTML search — no API key needed, free fallback."""

    def __init__(self):
        self.base_url = "https://html.duckduckgo.com/html/"

    async def search(self, query: str, max_articles: int = 8) -> List[NewsArticle]:
        try:
            async with httpx.AsyncClient(
                timeout=15.0,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
                follow_redirects=True,
            ) as client:
                resp = await client.post(
                    self.base_url,
                    data={"q": f"{query} news", "t": "h_", "ia": "news"},
                )
                resp.raise_for_status()
                html = resp.text

            articles = []
            # Extract results from DDG HTML
            results = re.findall(
                r'class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>.*?'
                r'class="result__snippet"[^>]*>([^<]*)</span>',
                html,
                re.DOTALL,
            )

            for url, title, snippet in results[:max_articles]:
                # DDG wraps URLs in a redirect
                actual_url = self._extract_url(url)
                articles.append(NewsArticle(
                    title=title.strip(),
                    summary=snippet.strip(),
                    url=actual_url,
                    published=datetime.now(timezone.utc),  # DDG doesn't give dates
                    source="DuckDuckGo",
                ))

            return articles
        except Exception as e:
            structlog.get_logger().warning("ddg_search_error", error=str(e))
            return []

    @staticmethod
    def _extract_url(ddg_url: str) -> str:
        """Extract actual URL from DDG redirect."""
        match = re.search(r"uddg=([^&]+)", ddg_url)
        if match:
            from urllib.parse import unquote
            return unquote(match.group(1))
        return ddg_url


class NewsAggregator:
    """Aggregates news from web search sources."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = structlog.get_logger()

        news_config = config.news
        self.max_articles_per_query = news_config.get("max_articles_per_query", 8)
        self.article_age_hours = news_config.get("article_age_hours", 72)

        rate_limit_seconds = news_config.get("rate_limit_seconds", 1)
        self.rate_limiter = RateLimiter(60 // max(rate_limit_seconds, 1), 60.0)

        # Initialize sources — Brave Search > Google News RSS > DuckDuckGo
        # Tavily disabled (HTTP 432 errors on every call since late 2025)
        self.sources = []
        brave_key = os.getenv("BRAVE_SEARCH_API_KEY", "")

        if brave_key:
            self.sources.append(BraveSearchSource(brave_key))
            self.logger.info("news_source_init", source="brave_search")
        # Google News RSS as fallback (free, no API key)
        self.sources.append(GoogleNewsRSSSource())
        if not brave_key:
            self.sources.append(DuckDuckGoSource())
            self.logger.info("news_source_init", source="google_news_rss+ddg_fallback")
        else:
            self.logger.info("news_source_init", fallback="google_news_rss")

    async def get_market_news(self, market: Market) -> List[NewsArticle]:
        """Get news articles relevant to a market, including counterarguments."""
        if not self.sources:
            return []

        try:
            query = self._build_search_query(market)
            if not query:
                return []

            self.logger.debug("news_search", market_id=market.id, query=query)

            all_articles = []
            for source in self.sources:
                try:
                    await self.rate_limiter.acquire()
                    articles = await source.search(query, self.max_articles_per_query)
                    all_articles.extend(articles)
                    if articles:
                        break  # Got results from first source, stop
                except Exception as e:
                    self.logger.warning("news_source_error", source=type(source).__name__, error=str(e))
                    continue

            # Counterargument search — get opposing views for balanced analysis
            counter_query = self._build_counterargument_query(market)
            for source in self.sources:
                try:
                    await self.rate_limiter.acquire()
                    counter_articles = await source.search(counter_query, 3)
                    for a in counter_articles:
                        a.source = f"{a.source} [counter]"
                    all_articles.extend(counter_articles)
                    if counter_articles:
                        break
                except Exception:
                    continue

            # Filter recent + deduplicate
            recent = [a for a in all_articles if a.is_recent(self.article_age_hours)]
            unique = self._deduplicate(recent if recent else all_articles)

            # Sort by credibility first, then recency
            unique.sort(
                key=lambda a: (-_source_credibility_score(a.url), -a.published.timestamp()),
            )

            self.logger.info(
                "news_found",
                market_id=market.id,
                query=query,
                total=len(all_articles),
                recent=len(recent),
                unique=len(unique),
            )

            result = unique[:self.max_articles_per_query]

            # Fetch full article content for top 3 articles (reduced from 5 — saves 10s worst-case)
            fetch_full = self.config.news.get("fetch_full_articles", True)
            if fetch_full and result:
                max_article_chars = int(self.config.news.get("max_article_chars", 1500))
                top_articles = result[:3]
                tasks = [
                    _fetch_article_text(a.url, timeout=5.0, max_chars=max_article_chars)
                    for a in top_articles
                ]
                texts = await asyncio.gather(*tasks, return_exceptions=True)
                for i, text in enumerate(texts):
                    if isinstance(text, str) and text:
                        top_articles[i].full_text = text

            return result

        except Exception as e:
            self.logger.error("news_error", market_id=market.id, error=str(e))
            return []

    def _build_search_query(self, market: Market) -> str:
        """Build a search query from market question with domain-specific expansion.

        Strategy:
        1. Strip prediction-market framing
        2. Add domain-specific expansion terms for better results
        3. Cap at search-engine-friendly length
        """
        question = market.question or ""

        # Remove prediction-market framing
        question = re.sub(r"^Will\s+", "", question, flags=re.IGNORECASE)
        question = re.sub(r"\?$", "", question)

        # Remove date-range suffixes like "by June 30" or "in January"
        # but keep names/entities
        question = re.sub(
            r"\b(by|before|after|in)\s+(January|February|March|April|May|June|"
            r"July|August|September|October|November|December)\s*\d*\b",
            "",
            question,
            flags=re.IGNORECASE,
        )

        # Clean up
        question = re.sub(r"\s+", " ", question).strip()

        # Domain-specific query expansion
        q_lower = question.lower()
        expansion_terms = []
        for trigger, expansions in _DOMAIN_EXPANSIONS.items():
            if trigger in q_lower:
                # Add first expansion term that isn't already in the question
                for term in expansions:
                    if term.lower() not in q_lower:
                        expansion_terms.append(term)
                        break
                break  # Only add one expansion to keep query focused

        if expansion_terms:
            question = f"{question} {expansion_terms[0]}"

        # Cap length for search query
        if len(question) > 140:
            question = " ".join(question.split()[:14])

        return question

    def _build_counterargument_query(self, market: Market) -> str:
        """Build a search query for counterarguments / opposing views."""
        question = market.question or ""
        question = re.sub(r"^Will\s+", "", question, flags=re.IGNORECASE)
        question = re.sub(r"\?$", "", question)
        question = re.sub(r"\s+", " ", question).strip()

        # Add negation / skepticism framing
        q_lower = question.lower()
        if any(w in q_lower for w in ["likely", "expect", "will"]):
            question = f"{question} unlikely obstacles challenges"
        else:
            question = f"{question} skepticism criticism opposition"

        if len(question) > 140:
            question = " ".join(question.split()[:14])

        return question

    def _deduplicate(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles by title similarity."""
        if not articles:
            return []

        unique = []
        seen_titles = set()

        for article in articles:
            normalized = re.sub(r"[^\w\s]", "", article.title.lower())
            normalized = " ".join(normalized.split())

            is_dup = False
            for seen in seen_titles:
                words1 = set(normalized.split())
                words2 = set(seen.split())
                if words1 and words2:
                    intersection = len(words1 & words2)
                    union = len(words1 | words2)
                    if intersection / union > 0.7:
                        is_dup = True
                        break

            if not is_dup:
                unique.append(article)
                seen_titles.add(normalized)

        return unique

    def format_news_context(self, articles: List[NewsArticle], max_total_chars: int = 4000) -> str:
        """Format news articles for LLM context, including full text when available.

        Increased context window (was 2000, now 4000) and article content
        (was 400 chars, now 800) for better LLM analysis.
        """
        if not articles:
            return "No recent news articles found for this market."

        # Tag credibility
        parts = [f"Recent news ({len(articles)} articles, sorted by source credibility):"]
        total_chars = 0
        for i, article in enumerate(articles[:8], 1):
            snippet = article.get_snippet(350)
            source = article.source
            cred = _source_credibility_score(article.url)
            cred_tag = "HIGH" if cred >= 0.9 else "MED" if cred >= 0.6 else "LOW"
            entry = f"{i}. [{source}] [{cred_tag} credibility] {snippet}"
            if article.full_text:
                entry += f"\n   Content: {article.full_text[:800]}"
            if total_chars + len(entry) > max_total_chars:
                break
            parts.append(entry)
            total_chars += len(entry)

        return "\n".join(parts)
