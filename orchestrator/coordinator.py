"""
Coordinator — the central orchestrator for the News Trading System.

The Coordinator wires together all data sources, agents, and storage into a
single, reproducible analysis pipeline:

    1. NewsFeed     → fetch headlines for a ticker
    2. MarketData   → fetch current price and fundamentals (optional context)
    3. SentimentAgent → score each headline via Claude
    4. Aggregate scores → compute average sentiment score
    5. Signal       → map average score to BUY / SELL / HOLD
    6. Database     → persist the run and every individual score

The coordinator owns no business logic itself; it delegates to its
collaborators and assembles their outputs into a structured report dict.
"""

from __future__ import annotations

import json

from agents.sentiment_agent import SentimentAgent
from config.settings import BUY_THRESHOLD, SELL_THRESHOLD
from data.market_data import MarketData
from data.news_feed import NewsFeed
from storage.database import Database


class Coordinator:
    """
    Orchestrates agents and data sources to produce a trading signal.

    Args:
        news_feed:       NewsFeed instance. Created automatically if omitted.
        market_data:     MarketData instance. Created automatically if omitted.
        sentiment_agent: SentimentAgent instance. Created automatically if omitted.
        db:              Database instance. Created automatically if omitted.

    All collaborators accept constructor injection so they can be replaced
    with test doubles in unit tests without touching application code.

    Example::

        report = Coordinator().run("AAPL")
        print(report["signal"])  # "BUY" | "SELL" | "HOLD"
    """

    def __init__(
        self,
        news_feed: NewsFeed | None = None,
        market_data: MarketData | None = None,
        sentiment_agent: SentimentAgent | None = None,
        db: Database | None = None,
    ) -> None:
        self.news_feed = news_feed or NewsFeed()
        self.market_data = market_data or MarketData()
        self.sentiment_agent = sentiment_agent or SentimentAgent()
        self.db = db or Database()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _aggregate(self, scored: list[dict]) -> float:
        """Return the mean numeric score across all scored headlines."""
        if not scored:
            return 0.0
        return sum(s["score"] for s in scored) / len(scored)

    def _signal(self, avg_score: float) -> str:
        """Map an average score to a trading signal string."""
        if avg_score >= BUY_THRESHOLD:
            return "BUY"
        if avg_score <= SELL_THRESHOLD:
            return "SELL"
        return "HOLD"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, ticker: str, verbose: bool = True) -> dict:
        """
        Execute the full analysis pipeline for *ticker*.

        Args:
            ticker:  Stock ticker symbol (e.g. "AAPL").
            verbose: When True, print step-by-step progress to stdout.

        Returns:
            dict with keys:
                ticker             (str):   The analysed ticker.
                market             (dict):  Output of MarketData.fetch().
                headlines_fetched  (int):   Raw count from NewsFeed.
                scored             (list):  List of per-headline score dicts.
                avg_score          (float): Aggregated sentiment score.
                signal             (str):   BUY / SELL / HOLD.
                run_id             (int):   Database primary key for this run.
        """
        ticker = ticker.upper()

        # Step 1 — market context
        market = {}
        try:
            market = self.market_data.fetch(ticker)
            if verbose and market.get("price"):
                currency = market.get("currency", "")
                print(f"  {ticker}  |  {market['name']}  |  {market['price']} {currency}\n")
        except Exception as exc:
            if verbose:
                print(f"  [market data unavailable: {exc}]\n")

        # Step 2 — headlines
        if verbose:
            print(f"Fetching headlines for {ticker}...")
        headlines = self.news_feed.fetch(ticker)
        if verbose:
            print(f"Found {len(headlines)} headline(s).\n")

        # Step 3 — sentiment scoring
        scored: list[dict] = []
        for i, headline in enumerate(headlines, 1):
            if verbose:
                print(f"[{i}/{len(headlines)}] Analysing: {headline}")
            try:
                result = self.sentiment_agent.run(headline, ticker)
                scored.append(result)
                if verbose:
                    icon = {"bullish": "+", "bearish": "-", "neutral": "~"}.get(
                        result["sentiment"], "?"
                    )
                    print(f"         [{icon}] {result['sentiment'].upper()} — {result['reason']}")
            except (json.JSONDecodeError, KeyError, Exception) as exc:
                if verbose:
                    print(f"         [!] Skipped ({exc})")

        # Step 4 — aggregate
        avg_score = self._aggregate(scored)

        # Step 5 — signal
        signal = self._signal(avg_score)

        # Step 6 — persist
        run_id = self.db.log_run(
            ticker=ticker,
            headlines_fetched=len(headlines),
            headlines_analysed=len(scored),
            avg_score=avg_score,
            signal=signal,
        )
        for s in scored:
            self.db.log_headline_score(
                run_id=run_id,
                headline=s["headline"],
                sentiment=s["sentiment"],
                score=s["score"],
                reason=s.get("reason", ""),
            )

        return {
            "ticker": ticker,
            "market": market,
            "headlines_fetched": len(headlines),
            "scored": scored,
            "avg_score": avg_score,
            "signal": signal,
            "run_id": run_id,
        }
