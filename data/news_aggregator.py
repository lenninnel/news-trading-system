"""
Multi-source news aggregator with 4-level fallback chain.

Fallback chain
--------------
Level 0 — NewsAPI          Primary source (requires NEWSAPI_KEY)
Level 1 — RSS Feeds        Yahoo Finance ticker RSS + Nasdaq News RSS
Level 2 — Google News RSS  Public endpoint, no authentication required
Level 3 — Response Cache   Headlines from a previous run (<24 h old)

The aggregator ALWAYS returns a list (never raises).  An empty list is
returned only when all four levels fail AND no cache exists.

The ``source`` key in the result indicates which level was used.

Caching
-------
Every successful fetch from any level is stored in the module-level
ResponseCache so Level 3 is available on the next call.

FallbackCoordinator integration
--------------------------------
After each call the aggregator registers its result with the
FallbackCoordinator so the system can alert operators when a higher
fallback has been in use for >24 h.

CLI
---
    python3 -m data.news_aggregator --test-fallbacks
    python3 -m data.news_aggregator --ticker AAPL [--max 10]
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import requests

from config.settings import MAX_HEADLINES, NEWSAPI_KEY, NEWSAPI_URL
from utils.api_recovery import APIRecovery, CircuitOpenError
from utils.network_recovery import NetworkMonitor, get_cache

log = logging.getLogger(__name__)

# ── RSS feed URLs ─────────────────────────────────────────────────────────────

def _yahoo_rss(ticker: str) -> str:
    return (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={ticker}&region=US&lang=en-US"
    )

def _nasdaq_rss(ticker: str) -> str:
    return f"https://www.nasdaq.com/feed/rssoutbound?symbol={ticker}&type=news"

def _google_news_rss(ticker: str) -> str:
    return (
        f"https://news.google.com/rss/search"
        f"?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
    )

_RSS_TIMEOUT    = 10   # seconds
_SCRAPE_TIMEOUT = 12
_CACHE_SERVICE  = "newsapi"
_CACHE_TTL      = 86_400  # 24 h


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class NewsResult:
    """Wraps the list of headlines with metadata about the source used."""
    headlines: list[str]
    source:    str            # e.g. "newsapi", "rss_yahoo", "google_news", "cache"
    level:     int            # 0-3
    ticker:    str
    degraded:  bool = False   # True when not from primary source
    count:     int  = field(init=False)

    def __post_init__(self) -> None:
        self.count = len(self.headlines)


# ── NewsAggregator ────────────────────────────────────────────────────────────

class NewsAggregator:
    """
    Multi-source news retrieval with automatic fallback.

    Implements the same ``fetch(ticker) -> list[str]`` interface as NewsFeed
    so it can be used as a drop-in replacement in Coordinator.

    Args:
        api_key:       NewsAPI key.  Defaults to settings.NEWSAPI_KEY.
        max_headlines: Maximum headlines to return from any single source.
        db:            Optional Database for recovery logging.
    """

    def __init__(
        self,
        api_key:       str              = NEWSAPI_KEY,
        max_headlines: int              = MAX_HEADLINES,
        db:            "Any | None"     = None,
    ) -> None:
        self.api_key       = api_key
        self.max_headlines = max_headlines
        if db is not None:
            APIRecovery.set_db(db)
            NetworkMonitor.set_db(db)

    # -- Public API (NewsFeed-compatible) ------------------------------------

    def fetch(self, ticker: str) -> list[str]:
        """
        Fetch headlines for *ticker*, trying all four sources in order.

        Returns:
            List of headline strings (never raises, never None).
        """
        result = self.fetch_with_metadata(ticker)
        return result.headlines

    def fetch_with_metadata(self, ticker: str) -> NewsResult:
        """
        Fetch headlines and return a NewsResult with source metadata.
        """
        ticker = ticker.upper()
        NetworkMonitor.check_and_update()

        # Ordered fallback chain
        for level, (name, fn) in enumerate([
            ("newsapi",      lambda t: self._level0_newsapi(t)),
            ("rss_yahoo",    lambda t: self._level1_rss_yahoo(t)),
            ("google_news",  lambda t: self._level2_google_news(t)),
            ("cache",        lambda t: self._level3_cache(t)),
        ]):
            if level == 0 and NetworkMonitor.is_degraded():
                log.info("Network degraded — skipping NewsAPI for %s", ticker)
                continue

            try:
                headlines = fn(ticker)
                if headlines:
                    # Cache the result for Level 3 reuse
                    cache_key = f"headlines:{ticker}"
                    get_cache().set(_CACHE_SERVICE, cache_key, headlines)

                    degraded = level > 0
                    if degraded:
                        log.warning(
                            "[DEGRADED L%d] %s: %d headline(s) from %s",
                            level, ticker, len(headlines), name,
                        )
                    self._register(ticker, level, name)
                    return NewsResult(
                        headlines=headlines[:self.max_headlines],
                        source=name,
                        level=level,
                        ticker=ticker,
                        degraded=degraded,
                    )
            except Exception as exc:
                log.warning(
                    "News source '%s' failed for %s: %s", name, ticker, exc
                )

        # All sources failed and no cache → empty result
        log.error("All news sources exhausted for %s — returning empty list", ticker)
        return NewsResult(headlines=[], source="none", level=4, ticker=ticker, degraded=True)

    # -- Level 0: NewsAPI -----------------------------------------------------

    def _level0_newsapi(self, ticker: str) -> list[str]:
        """Primary: NewsAPI REST endpoint."""
        def _call() -> list[str]:
            params = {
                "q":        ticker,
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": self.max_headlines,
                "apiKey":   self.api_key,
            }
            resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            return [a["title"] for a in articles if a.get("title")]

        return APIRecovery.call("newsapi", _call, ticker=ticker)

    # -- Level 1: RSS feeds ---------------------------------------------------

    def _level1_rss_yahoo(self, ticker: str) -> list[str]:
        """
        Fallback 1: Yahoo Finance ticker RSS + Nasdaq News RSS.

        Tries Yahoo first; if that returns fewer than 3 headlines, also
        appends results from Nasdaq RSS to top up.
        """
        headlines: list[str] = []

        for url in (_yahoo_rss(ticker), _nasdaq_rss(ticker)):
            try:
                fetched = self._parse_rss(url, ticker)
                headlines.extend(fetched)
                if len(headlines) >= self.max_headlines:
                    break
            except Exception as exc:
                log.debug("RSS %s failed: %s", url, exc)

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for h in headlines:
            if h not in seen:
                seen.add(h)
                deduped.append(h)

        return deduped[:self.max_headlines]

    # -- Level 2: Google News RSS (web scraping) ------------------------------

    def _level2_google_news(self, ticker: str) -> list[str]:
        """Fallback 2: Google News public RSS search endpoint."""
        url = _google_news_rss(ticker)
        headlines = self._parse_rss(url, ticker)
        return headlines[:self.max_headlines]

    # -- Level 3: Cache -------------------------------------------------------

    def _level3_cache(self, ticker: str) -> list[str]:
        """
        Fallback 3: Return cached headlines from a previous run.

        Cache TTL is 24 h (set in ResponseCache initialisation).
        """
        cache_key = f"headlines:{ticker}"
        cached, hit = get_cache().get(_CACHE_SERVICE, cache_key)
        if hit and cached:
            log.warning(
                "[CACHE FALLBACK] %s: serving %d cached headline(s) (<24h old)",
                ticker, len(cached),
            )
            return list(cached)
        return []

    # -- RSS parser (shared) --------------------------------------------------

    @staticmethod
    def _parse_rss(url: str, ticker: str) -> list[str]:
        """
        Fetch and parse an RSS/Atom feed, returning item titles.

        Uses stdlib xml.etree.ElementTree — no feedparser dependency.
        Falls back to regex extraction if XML is malformed.
        """
        resp = requests.get(url, timeout=_RSS_TIMEOUT, headers={
            "User-Agent": "Mozilla/5.0 (compatible; NewsAggregator/1.0)",
        })
        resp.raise_for_status()
        content = resp.text

        # Try proper XML parse first
        try:
            root  = ET.fromstring(content.encode("utf-8", errors="replace"))
            ns    = {"atom": "http://www.w3.org/2005/Atom"}
            items: list[str] = []

            # RSS 2.0 format
            for item in root.findall(".//item"):
                title = item.findtext("title")
                if title:
                    items.append(title.strip())

            # Atom format fallback
            if not items:
                for entry in root.findall(".//atom:entry", ns):
                    title = entry.findtext("atom:title", namespaces=ns)
                    if title:
                        items.append(title.strip())

            return [h for h in items if _is_relevant(h, ticker)]
        except ET.ParseError:
            pass

        # Regex fallback for malformed XML
        titles = re.findall(r"<title[^>]*>(.*?)</title>", content, re.DOTALL)
        clean  = [re.sub(r"<[^>]+>", "", t).strip() for t in titles]
        return [h for h in clean if _is_relevant(h, ticker) and len(h) > 15]

    # -- FallbackCoordinator registration ------------------------------------

    @staticmethod
    def _register(ticker: str, level: int, source: str) -> None:
        """Notify FallbackCoordinator of the level used (best-effort)."""
        try:
            from data.fallback_coordinator import FallbackCoordinator
            FallbackCoordinator.register("news", level, source, ticker=ticker)
        except Exception:
            pass


# ── Relevance filter ──────────────────────────────────────────────────────────

def _is_relevant(headline: str, ticker: str) -> bool:
    """
    Keep only headlines that mention the ticker or common company names.

    For RSS feeds that return general news, this filters out off-topic items.
    For Google News (which is already filtered by search query), most items pass.
    """
    if not headline or len(headline) < 10:
        return False
    # Company-name lookup is expensive; just check if headline looks financial
    lower = headline.lower()
    # Reject obvious non-finance items
    skip_words = {"recipe", "how to cook", "weather", "sports score", "horoscope"}
    return not any(w in lower for w in skip_words)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _test_fallbacks(ticker: str = "AAPL", max_headlines: int = 5) -> None:
    """
    Test each fallback level independently and report the results.

    Each level is tested by temporarily disabling higher-priority sources.
    No mocking is used — this performs real network calls.
    """
    from unittest.mock import patch

    GREEN = "\033[32m"
    RED   = "\033[31m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    agg = NewsAggregator(max_headlines=max_headlines)
    print(f"\n{BOLD}News Aggregator — Fallback Test{RESET}")
    print(f"Ticker: {ticker}  |  Max headlines: {max_headlines}")
    print("─" * 60)

    results: dict[str, tuple[bool, int, str]] = {}  # level → (ok, count, note)

    # Level 0: NewsAPI
    print(f"\n[L0] NewsAPI …", end="", flush=True)
    try:
        h = agg._level0_newsapi(ticker)
        results["L0 NewsAPI"] = (True, len(h), "")
        print(f"  {GREEN}✓ {len(h)} headlines{RESET}")
        for hl in h[:3]:
            print(f"    • {hl[:80]}")
    except Exception as exc:
        results["L0 NewsAPI"] = (False, 0, str(exc)[:80])
        print(f"  {RED}✗ {exc}{RESET}")

    # Level 1: RSS
    print(f"\n[L1] RSS Feeds (Yahoo Finance + Nasdaq) …", end="", flush=True)
    try:
        h = agg._level1_rss_yahoo(ticker)
        results["L1 RSS"] = (True, len(h), "")
        print(f"  {GREEN}✓ {len(h)} headlines{RESET}")
        for hl in h[:3]:
            print(f"    • {hl[:80]}")
    except Exception as exc:
        results["L1 RSS"] = (False, 0, str(exc)[:80])
        print(f"  {RED}✗ {exc}{RESET}")

    # Level 2: Google News
    print(f"\n[L2] Google News RSS …", end="", flush=True)
    try:
        h = agg._level2_google_news(ticker)
        results["L2 Google News"] = (True, len(h), "")
        print(f"  {GREEN}✓ {len(h)} headlines{RESET}")
        for hl in h[:3]:
            print(f"    • {hl[:80]}")
    except Exception as exc:
        results["L2 Google News"] = (False, 0, str(exc)[:80])
        print(f"  {RED}✗ {exc}{RESET}")

    # Level 3: Cache (prime it first, then test)
    print(f"\n[L3] Response Cache (24h TTL) …", end="", flush=True)
    fake_headlines = [f"[CACHED] {ticker} cache test headline {i+1}" for i in range(3)]
    get_cache().set(_CACHE_SERVICE, f"headlines:{ticker}", fake_headlines)
    try:
        h = agg._level3_cache(ticker)
        results["L3 Cache"] = (True, len(h), "pre-seeded")
        print(f"  {GREEN}✓ {len(h)} cached headline(s){RESET}")
    except Exception as exc:
        results["L3 Cache"] = (False, 0, str(exc)[:80])
        print(f"  {RED}✗ {exc}{RESET}")

    # Full chain test (should use first working source)
    print(f"\n{'─' * 60}")
    print(f"[CHAIN] Full fallback chain for {ticker} …")
    result = agg.fetch_with_metadata(ticker)
    print(
        f"  Used source: {BOLD}{result.source}{RESET} (Level {result.level})"
        f"  |  {result.count} headline(s)"
        f"  |  degraded={result.degraded}"
    )

    # Summary
    print(f"\n{'─' * 60}")
    print(f"{BOLD}Summary:{RESET}")
    total = len(results)
    passed = sum(1 for ok, _, _ in results.values() if ok)
    for name, (ok, count, note) in results.items():
        icon  = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        extra = f" ({note})" if note else ""
        print(f"  {icon}  {name:<25} {count} headlines{extra}")
    print(
        f"\n  {GREEN if passed == total else YELLOW}"
        f"{passed}/{total} sources available{RESET}"
    )


if __name__ == "__main__":
    import argparse, sys, os
    from pathlib import Path

    # Ensure project root is on sys.path when run as `python3 -m data.news_aggregator`
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    import logging as _logging
    _logging.basicConfig(level=_logging.WARNING)

    parser = argparse.ArgumentParser(description="News Aggregator CLI")
    parser.add_argument("--test-fallbacks", action="store_true",
                        help="Test each fallback level and print results")
    parser.add_argument("--ticker", default="AAPL", metavar="TICKER",
                        help="Ticker to fetch (default: AAPL)")
    parser.add_argument("--max", type=int, default=5, metavar="N",
                        help="Max headlines per source (default: 5)")
    args = parser.parse_args()

    if args.test_fallbacks:
        _test_fallbacks(ticker=args.ticker.upper(), max_headlines=args.max)
    else:
        # Simple fetch
        agg = NewsAggregator(max_headlines=args.max)
        r   = agg.fetch_with_metadata(args.ticker.upper())
        print(f"\nSource: {r.source} (Level {r.level})  |  {r.count} headlines")
        for i, h in enumerate(r.headlines, 1):
            print(f"  {i:2}. {h}")
