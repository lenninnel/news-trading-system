"""
Coordinator — the central orchestrator for the News Trading System.

Single-agent pipeline (unchanged, backward-compatible):

    1. NewsFeed       → fetch headlines for a ticker
    2. MarketData     → fetch current price and fundamentals
    3. SentimentAgent → score each headline via Claude
    4. Aggregate      → compute average sentiment score → BUY / SELL / HOLD
    5. Database       → persist run + headline scores

Combined pipeline (run_combined):

    1–5 above (sentiment)
    6. TechnicalAgent → RSI, MACD, Bollinger, SMA → BUY / SELL / HOLD
    7. Fusion         → combine both signals into STRONG BUY / WEAK SELL / etc.
    8. RiskAgent      → position size, stop-loss, take-profit
    9. Database       → persist to combined_signals + risk_calculations tables

Signal fusion matrix
--------------------
Sentiment  Technical  Combined
─────────  ─────────  ─────────────
BUY        BUY        STRONG BUY
SELL       SELL       STRONG SELL
BUY        HOLD       WEAK BUY
SELL       HOLD       WEAK SELL
BUY        SELL       CONFLICTING
SELL       BUY        CONFLICTING
HOLD       *          HOLD
"""

from __future__ import annotations

import asyncio
import json
import logging
import time as _time


log = logging.getLogger(__name__)

from agents.bull_bear_debate import BullBearDebate, DebateResult
from agents.regime_agent import RegimeAgent
from agents.regime_detector import RegimeDetector
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from analytics.signal_logger import SignalLogger
from config.settings import (
    BUY_THRESHOLD,
    ENABLE_REGIME_FILTER,
    SELL_THRESHOLD,
    SOURCE_WEIGHTS,
    SOURCE_WEIGHTS_NO_NEWSAPI,
    SOURCE_WEIGHTS_NO_REDDIT,
    SOURCE_WEIGHTS_NO_REDDIT_NO_NEWSAPI,
)
from data.events_feed import get_days_to_earnings
from data.market_data import MarketData
from data.news_feed import NewsFeed
from data.marketaux_feed import MarketauxFeed
from data.social_feed import AdanosFeed, ApeWisdomFeed, RedditFeed, StockTwitsFeed, is_reddit_configured
from execution.broker_factory import create_trader
from orchestrator.cluster_detector import ClusterDetector
from storage.database import Database
from strategies.base import StrategyResult
from strategies.momentum import MomentumStrategy
from strategies.news_catalyst import NewsCatalystStrategy
from strategies.pead_strategy import PEADStrategy
from strategies.pullback import PullbackStrategy
from strategies.router import get_strategy, strategy_label

# Maps (sentiment_signal, technical_signal) → combined label
_FUSION_TABLE: dict[tuple[str, str], str] = {
    ("BUY",  "BUY"):  "STRONG BUY",
    ("SELL", "SELL"): "STRONG SELL",
    ("BUY",  "HOLD"): "WEAK BUY",
    ("SELL", "HOLD"): "WEAK SELL",
    ("BUY",  "SELL"): "CONFLICTING",
    ("SELL", "BUY"):  "CONFLICTING",
    ("HOLD", "BUY"):  "HOLD",
    ("HOLD", "SELL"): "HOLD",
    ("HOLD", "HOLD"): "HOLD",
}


_news_catalyst = NewsCatalystStrategy()


class Coordinator:
    """
    Orchestrates agents and data sources to produce a trading signal.

    Args:
        news_feed:        NewsFeed instance.  Created automatically if omitted.
        market_data:      MarketData instance. Created automatically if omitted.
        sentiment_agent:  SentimentAgent.      Created automatically if omitted.
        technical_agent:  TechnicalAgent.      Created automatically if omitted.
        db:               Database.            Created automatically if omitted.

    All collaborators accept constructor injection for testing.

    Examples::

        # Combined (default) mode
        report = Coordinator().run_combined("AAPL")
        print(report["combined_signal"])  # "STRONG BUY", "WEAK SELL", ...

        # Sentiment only (backward-compatible)
        report = Coordinator().run("AAPL")
        print(report["signal"])           # "BUY" | "SELL" | "HOLD"
    """

    def __init__(
        self,
        news_feed: NewsFeed | None = None,
        market_data: MarketData | None = None,
        sentiment_agent: SentimentAgent | None = None,
        technical_agent: TechnicalAgent | None = None,
        risk_agent: RiskAgent | None = None,
        regime_agent: RegimeAgent | None = None,
        db: Database | None = None,
        paper_trader=None,
        reddit_feed: RedditFeed | None = None,
        stocktwits_feed: StockTwitsFeed | None = None,
        marketaux_feed: MarketauxFeed | None = None,
        apewisdom_feed: ApeWisdomFeed | None = None,
        adanos_feed: AdanosFeed | None = None,
        debate_agent: BullBearDebate | None = None,
        macro_context: str = "",
    ) -> None:
        self.db = db or Database()
        self.news_feed = news_feed or NewsFeed()
        self.market_data = market_data or MarketData()
        self.sentiment_agent = sentiment_agent or SentimentAgent()
        # All agents share the same DB instance
        self.technical_agent = technical_agent or TechnicalAgent(db=self.db)
        self.risk_agent = risk_agent or RiskAgent(db=self.db)
        self.regime_agent = regime_agent or RegimeAgent()
        self.regime_detector = RegimeDetector()
        self.debate_agent = debate_agent or BullBearDebate()
        self.paper_trader = paper_trader or create_trader(db=self.db)
        self.reddit_feed = reddit_feed or RedditFeed()
        self.stocktwits_feed = stocktwits_feed or StockTwitsFeed()
        self.marketaux_feed = marketaux_feed or MarketauxFeed()
        self.apewisdom_feed = apewisdom_feed or ApeWisdomFeed()
        self.adanos_feed = adanos_feed or AdanosFeed()
        self.signal_logger = SignalLogger(db=self.db)
        # Once-per-session macro context injected by the scheduler. Empty
        # string when ENABLE_MACRO_CONTEXT is off, the session is non-US,
        # or the Haiku call failed — debate prompts stay legacy.
        self.macro_context = macro_context or ""

        # PEAD strategy (no Claude API needed — pure data)
        from config.settings import PEAD_ENABLED, PEAD_TICKERS, PEAD_EARNINGS_CACHE_PATH
        self._pead_enabled = PEAD_ENABLED
        self._pead_tickers = set(t.upper() for t in PEAD_TICKERS)
        self._pead_strategy = PEADStrategy(cache_path=PEAD_EARNINGS_CACHE_PATH)

        # Phase 2b — multi-strategy cluster detector as the combination layer.
        # Momentum/Pullback strategies are stateless, so one instance each is
        # enough.  NewsCatalyst already lives at module scope as `_news_catalyst`.
        self._cluster_detector = ClusterDetector()
        self._momentum_strategy = MomentumStrategy()
        self._pullback_strategy = PullbackStrategy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _has_alpaca_position(self, ticker: str) -> bool:
        """Check Alpaca directly for an open position in *ticker*.

        Alpaca is the source of truth — the local DB can be out of sync
        after bracket orders fill or expire.  Returns False on any error
        so the caller falls through to the local-DB guard.
        """
        try:
            api = self.paper_trader._api
            pos = api.get_position(ticker.upper())
            qty = int(pos.qty)
            if qty > 0:
                log.info(
                    "[%s] Alpaca position exists (%d shares) — blocking duplicate BUY",
                    ticker, qty,
                )
                return True
        except Exception:
            # get_position raises 404 if no position — that's the happy path
            pass
        return False

    def _last_combined_signal(self, ticker: str) -> str:
        """Return the last combined_signal for *ticker* from DB, or ''."""
        try:
            with self.db._connect() as conn:
                row = conn.execute(
                    "SELECT combined_signal FROM combined_signals "
                    "WHERE ticker = ? ORDER BY id DESC LIMIT 1",
                    (ticker.upper(),),
                ).fetchone()
            return row[0] if row else ""
        except Exception:
            return ""

    @staticmethod
    def _aggregate(scored: list[dict]) -> float:
        """Return the mean numeric score across all scored headlines."""
        if not scored:
            return 0.0
        return sum(s["score"] for s in scored) / len(scored)

    @staticmethod
    def _active_weights() -> dict[str, float]:
        """Return the source-weight map appropriate for the current config.

        Considers both Reddit availability and NewsAPI rate-limit status.
        When Reddit credentials are present and NewsAPI is within budget,
        the standard ``SOURCE_WEIGHTS`` are used.  Otherwise an appropriate
        fallback weight map compensates for the missing source(s).
        """
        from data.news_feed import is_newsapi_limit_reached

        reddit_ok = is_reddit_configured()
        newsapi_ok = not is_newsapi_limit_reached()

        if reddit_ok and newsapi_ok:
            return SOURCE_WEIGHTS
        elif reddit_ok and not newsapi_ok:
            return SOURCE_WEIGHTS_NO_NEWSAPI
        elif not reddit_ok and newsapi_ok:
            return SOURCE_WEIGHTS_NO_REDDIT
        else:
            return SOURCE_WEIGHTS_NO_REDDIT_NO_NEWSAPI

    @staticmethod
    def _weighted_aggregate(scored: list[dict], weights: dict[str, float] | None = None) -> float:
        """Return a weighted average score using *weights* (default: SOURCE_WEIGHTS)."""
        if not scored:
            return 0.0
        if weights is None:
            weights = SOURCE_WEIGHTS
        total_weight = 0.0
        weighted_sum = 0.0
        for s in scored:
            w = weights.get(s.get("source", "newsapi"), 1.0)
            weighted_sum += s["score"] * w
            total_weight += w
        return weighted_sum / total_weight if total_weight else 0.0

    @staticmethod
    def _source_breakdown(scored: list[dict]) -> dict:
        """Compute per-source count and average from scored items."""
        buckets: dict[str, list[int]] = {}
        for s in scored:
            src = s.get("source", "newsapi")
            buckets.setdefault(src, []).append(s["score"])
        return {
            src: {
                "count": len(scores),
                "avg": round(sum(scores) / len(scores), 4) if scores else 0.0,
            }
            for src, scores in buckets.items()
        }

    def _signal(self, avg_score: float) -> str:
        """Map an average score to a trading signal string."""
        if avg_score >= BUY_THRESHOLD:
            return "BUY"
        if avg_score <= SELL_THRESHOLD:
            return "SELL"
        return "HOLD"

    def _log_signal_event(self, result: dict, *, session: str | None = None) -> None:
        """Extract signal context and log it. Never raises."""
        try:
            technical = result.get("technical") or {}
            indicators = technical.get("indicators") or {}
            sentiment = result.get("sentiment") or {}
            debate = result.get("debate")

            price = indicators.get("price")
            sma_50 = indicators.get("sma_50")
            sma_ratio = (price / sma_50) if (price and sma_50 and sma_50 > 0) else None
            rvol = indicators.get("rvol")

            # Compute news vs social scores from source breakdown
            breakdown = sentiment.get("source_breakdown") or {}
            news_sources = ("newsapi", "marketaux")
            social_sources = ("reddit", "stocktwits", "apewisdom", "adanos")
            news_scores = [v["avg"] for k, v in breakdown.items() if k in news_sources and v.get("count")]
            social_scores = [v["avg"] for k, v in breakdown.items() if k in social_sources and v.get("count")]
            news_score = (sum(news_scores) / len(news_scores) + 1) / 2 if news_scores else None  # map -1..1 to 0..1
            social_score = (sum(social_scores) / len(social_scores) + 1) / 2 if social_scores else None

            avg = sentiment.get("avg_score", 0)
            # PEAD signals have no sentiment — show null instead of 0.5 default
            is_pead = result.get("strategy_name") == "PEAD"
            sentiment_score = None if is_pead else (avg + 1) / 2  # map -1..1 to 0..1

            execution = result.get("execution") or {}

            # Determine debate outcome
            if result.get("is_scanner_candidate"):
                debate_outcome = "scanner_no_debate"
            elif debate:
                if debate.degraded:
                    debate_outcome = "skipped"
                elif debate.final_signal == debate.original_signal:
                    debate_outcome = "agree"
                elif debate.adjusted_confidence < debate.original_confidence:
                    debate_outcome = "cautious"
                else:
                    debate_outcome = "disagree"
            else:
                debate_outcome = "skipped"

            self.signal_logger.log({
                "ticker": result.get("ticker"),
                "session": session,
                "strategy": "Combined",
                "signal": result.get("combined_signal"),
                "confidence": result.get("confidence"),
                "rsi": indicators.get("rsi"),
                "sma_ratio": sma_ratio,
                "volume_ratio": rvol,
                "sentiment_score": sentiment_score,
                "news_score": news_score,
                "social_score": social_score,
                "bull_case": debate.bull_case if debate else None,
                "bear_case": debate.bear_case if debate else None,
                "debate_outcome": debate_outcome,
                "price_at_signal": price,
                "trade_executed": 1 if execution.get("trade_id") else 0,
                "trade_id": execution.get("trade_id"),
                # getattr fallback: some tests use Coordinator.__new__() to
                # build a stub that skips __init__, so macro_context may not
                # exist as an attribute. Default to empty/False.
                "macro_context_used": bool(getattr(self, "macro_context", "")),
                "signal_path": result.get("signal_path"),
            })
        except Exception as exc:
            log.warning("Signal event logging failed (non-fatal): %s", exc)

    def _log_strategy_result(self, ticker: str, strategy_result, *, session: str | None = None, regime: str | None = None) -> None:
        """Log an individual strategy result to signal_events. Never raises.

        Every strategy result is logged regardless of signal type or
        confidence so the dashboard can show sub-strategy signals.
        """
        try:
            indicators = strategy_result.indicators or {}
            price = indicators.get("price")
            sma_50 = indicators.get("sma50") or indicators.get("sma_50")
            sma_ratio = (price / sma_50) if (price and sma_50 and sma_50 > 0) else None
            vol_ratio = indicators.get("vol_ratio") or indicators.get("rvol")

            self.signal_logger.log({
                "ticker": ticker,
                "session": session,
                "strategy": strategy_result.strategy_name,
                "signal": strategy_result.signal,
                "confidence": strategy_result.confidence / 100.0,
                "rsi": indicators.get("rsi"),
                "sma_ratio": sma_ratio,
                "volume_ratio": vol_ratio,
                "sentiment_score": None,
                "news_score": indicators.get("news_score"),
                "social_score": None,
                "bull_case": None,
                "bear_case": None,
                "regime": regime,
                "debate_outcome": None,
                "price_at_signal": price,
                "trade_executed": 0,
                "trade_id": None,
            })
        except Exception as exc:
            log.warning("Strategy result logging failed (non-fatal): %s", exc)

    def _run_pead(self, ticker: str, *, session: str | None = None) -> dict | None:
        """Run PEAD strategy for a ticker if eligible. Returns result dict or None."""
        if not self._pead_enabled:
            return None
        if ticker.upper() not in self._pead_tickers:
            return None
        from datetime import date as _date

        result = self._pead_strategy.generate_signal(ticker, _date.today())
        if result is None:
            return None
        self._log_strategy_result(ticker, result, session=session)
        log.info("[%s] PEAD signal: %s (%.0f%%)", ticker, result.signal, result.confidence)
        return {
            "signal": result.signal,
            "confidence": result.confidence / 100.0,
            "strategy_name": "PEAD",
            "reasoning": result.reasoning,
            "indicators": result.indicators,
        }

    async def _run_pead_only_async(
        self,
        ticker: str,
        *,
        account_balance: float = 10_000.0,
        execute: bool = False,
        data_semaphore: asyncio.Semaphore,
        db_lock: asyncio.Lock,
        session: str | None = None,
    ) -> dict:
        """PEAD-only fast path — skips news/sentiment/technical pipeline."""
        t0 = _time.monotonic()
        ticker = ticker.upper()

        pead_result = self._run_pead(ticker, session=session)

        if pead_result is None or pead_result["signal"] != "BUY":
            elapsed = _time.monotonic() - t0
            return {
                "ticker": ticker,
                "combined_signal": "HOLD",
                "confidence": 0.0,
                "strategy_name": "PEAD",
                "sentiment": {"signal": "N/A", "avg_score": 0.0},
                "technical": {"signal": "N/A", "indicators": {}},
                "risk": {"skipped": True},
                "execution": None,
                "debate": None,
                "elapsed_s": round(elapsed, 2),
            }

        combined_signal = "BUY"
        conf = pead_result["confidence"]

        # Fetch price for risk sizing
        market: dict = {}
        async with data_semaphore:
            try:
                market = await asyncio.to_thread(self.market_data.fetch, ticker)
            except Exception as exc:
                log.warning("[%s] PEAD market data fetch failed: %s", ticker, exc)

        price = (market or {}).get("price")
        price_is_live = bool(price) and not (market or {}).get("degraded")

        # Risk sizing
        async with db_lock:
            risk = self.risk_agent.run(
                ticker=ticker,
                signal=combined_signal,
                confidence=conf * 100,
                current_price=price,
                account_balance=account_balance,
            )

        # Execution
        execution = None
        if execute and not risk["skipped"] and price and price > 0 and price_is_live:
            if risk.get("stop_loss") and risk["stop_loss"] > 0 \
                    and risk.get("take_profit") and risk["take_profit"] > 0:
                async with db_lock:
                    if risk["direction"] == "BUY" and (
                        self._has_alpaca_position(ticker)
                        or self.db.get_portfolio_position(ticker)
                    ):
                        log.info("[%s] PEAD duplicate BUY blocked", ticker)
                    else:
                        execution = self.paper_trader.track_trade(
                            ticker=ticker,
                            action=risk["direction"],
                            shares=risk["shares"],
                            price=price,
                            stop_loss=risk["stop_loss"],
                            take_profit=risk["take_profit"],
                        )

        # Store forward signal for execution session
        if combined_signal not in ("HOLD", "CONFLICTING"):
            try:
                self.signal_logger.store_forward_signal({
                    "source_session": session or "PEAD_OPEN",
                    "target_session": "US_OPEN",
                    "ticker": ticker,
                    "signal": combined_signal,
                    "confidence": conf,
                    "price_at_signal": price,
                    "strategy_name": "PEAD",
                    "stop_loss": risk.get("stop_loss") if not risk.get("skipped") else None,
                    "take_profit": risk.get("take_profit") if not risk.get("skipped") else None,
                })
            except Exception as exc:
                log.warning("[%s] PEAD forward signal storage failed: %s", ticker, exc)

        elapsed = _time.monotonic() - t0

        final_result = {
            "ticker": ticker,
            "combined_signal": combined_signal,
            "confidence": conf,
            "strategy_name": "PEAD",
            "sentiment": {"signal": "PEAD", "avg_score": 0.0},
            "technical": {"signal": "N/A", "indicators": {}},
            "risk": risk,
            "execution": execution,
            "debate": None,
            "elapsed_s": round(elapsed, 2),
        }
        self._log_signal_event(final_result, session=session)
        return final_result

    @staticmethod
    def _build_news_data(sentiment: dict) -> dict | None:
        """Build *news_data* dict for NewsCatalystStrategy from a sentiment result.

        Returns None when the sentiment result has no scored headlines.
        """
        avg = sentiment.get("avg_score", 0.0)
        scored = sentiment.get("scored")
        if not scored:
            return None
        return {
            "news_score": (avg + 1) / 2,  # map -1..+1 to 0..1
            "headline_count": len(scored),
            "sentiment_direction": sentiment.get("signal", "HOLD"),
        }

    def _run_news_catalyst(
        self,
        ticker: str,
        bars,
        sentiment: dict,
        *,
        session: str | None = None,
    ) -> StrategyResult | None:
        """Run NewsCatalystStrategy and log the result. Never raises."""
        try:
            if bars is None or (hasattr(bars, "empty") and bars.empty):
                return None
            news_data = self._build_news_data(sentiment)
            result = _news_catalyst.analyze(
                ticker, bars, sentiment.get("signal", "HOLD"),
                news_data=news_data,
            )
            log.info(
                "[%s] NewsCatalyst → %s (%.0f%%)",
                ticker, result.signal, result.confidence,
            )
            self._log_strategy_result(ticker, result, session=session)
            return result
        except Exception as exc:
            log.warning("[%s] NewsCatalyst failed (non-fatal): %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Multi-strategy cluster fusion (Phase 2b)
    # ------------------------------------------------------------------

    def _gather_strategy_votes(
        self,
        ticker: str,
        bars,
        sentiment: dict,
        *,
        session: str | None = None,
        regime: str | None = None,
    ) -> list[StrategyResult]:
        """Run Momentum, Pullback, and NewsCatalyst and return the successful
        results.  Each vote is logged to signal_events individually.  Never
        raises — a failing strategy is skipped, the others still contribute.
        """
        votes: list[StrategyResult] = []

        if bars is None or (hasattr(bars, "empty") and bars.empty):
            # No price history → strategies can't vote.  NewsCatalyst also
            # short-circuits on empty bars, so we return empty.
            return votes

        sentiment_label = sentiment.get("signal", "HOLD")

        for strategy, name in (
            (self._momentum_strategy, "Momentum"),
            (self._pullback_strategy, "Pullback"),
        ):
            try:
                result = strategy.analyze(ticker, bars, sentiment_label)
                votes.append(result)
                log.info(
                    "[%s] Strategy %s → %s (%.0f%%)",
                    ticker, name, result.signal, result.confidence,
                )
                self._log_strategy_result(
                    ticker, result, session=session, regime=regime,
                )
            except Exception as exc:
                log.warning("[%s] Strategy %s failed (non-fatal): %s",
                            ticker, name, exc)

        # _run_news_catalyst handles its own logging and returns None on
        # failure or when bars are empty.
        nc_result = self._run_news_catalyst(
            ticker, bars, sentiment, session=session,
        )
        if nc_result is not None:
            votes.append(nc_result)

        return votes

    # The combined pipeline always asks three strategies to vote
    # (Momentum, Pullback, NewsCatalyst).  Anything less means one or more
    # strategies failed/short-circuited and the cluster is partial.
    _EXPECTED_VOTE_COUNT = 3

    def _fuse_signals(
        self,
        ticker: str,
        strategy_votes: list[StrategyResult],
        sentiment_signal: str,
        sentiment_confidence: float,
        fallback_technical_signal: str,
        fallback_technical_confidence: float | None,
    ) -> tuple[str, float, str]:
        """Combine strategy votes via ClusterDetector, falling back to
        combine_signals() on exception.

        Returns (combined_label, combined_confidence, signal_path) where
        signal_path is one of:
          * ``CLUSTER``          — all three strategies voted and
                                   ClusterDetector produced the verdict.
          * ``CLUSTER_PARTIAL``  — fewer than three strategies voted
                                   (one or more failed/short-circuited)
                                   and ClusterDetector still produced a
                                   verdict on the surviving votes.
          * ``FUSION_FALLBACK``  — no strategies voted or ClusterDetector
                                   raised; ``combine_signals()`` produced
                                   the verdict.
        """
        if strategy_votes:
            try:
                cluster = self._cluster_detector.detect(strategy_votes)
                path = (
                    "CLUSTER"
                    if len(strategy_votes) >= self._EXPECTED_VOTE_COUNT
                    else "CLUSTER_PARTIAL"
                )
                if path == "CLUSTER_PARTIAL":
                    voters = [r.strategy_name for r in strategy_votes]
                    log.info(
                        "[%s] Cluster partial (%d/%d votes: %s)",
                        ticker, len(strategy_votes),
                        self._EXPECTED_VOTE_COUNT, voters,
                    )
                return (
                    cluster.cluster_signal,
                    round(max(0.0, min(1.0, cluster.confidence)), 2),
                    path,
                )
            except Exception as exc:
                log.warning(
                    "[%s] ClusterDetector failed, falling back to combine_signals: %s",
                    ticker, exc,
                )

        label, conf = self.combine_signals(
            sentiment_signal,
            fallback_technical_signal,
            sentiment_confidence=sentiment_confidence,
            technical_confidence=fallback_technical_confidence,
        )
        return label, conf, "FUSION_FALLBACK"

    # ------------------------------------------------------------------
    # Signal fusion (static — easily unit-testable)
    # ------------------------------------------------------------------

    @staticmethod
    def combine_signals(
        sentiment_signal: str,
        technical_signal: str,
        sentiment_confidence: float = 0.5,
        technical_confidence: float = 0.5,
    ) -> tuple[str, float]:
        """
        Fuse sentiment and technical signals into a combined label and
        a combined confidence score.

        The label comes from _FUSION_TABLE (unchanged 3x3 matrix).
        The confidence is computed from the two input confidences:

          * Both BUY or both SELL  → max(s, t) * 1.1  (agreement bonus,
                                                       capped at 1.0)
          * BUY vs SELL            → abs(s - t) * 0.7 (disagreement penalty)
          * One HOLD, one non-HOLD → non-HOLD confidence * 0.8
                                     (both HOLD: max(s, t) * 0.8)

        Args:
            sentiment_signal:     "BUY" | "SELL" | "HOLD"
            technical_signal:     "BUY" | "SELL" | "HOLD"
            sentiment_confidence: 0.0 - 1.0 (default 0.5)
            technical_confidence: 0.0 - 1.0 (default 0.5)

        Returns:
            (combined_label, combined_confidence) — label is one of
            STRONG BUY, STRONG SELL, WEAK BUY, WEAK SELL, CONFLICTING,
            HOLD; confidence is clamped to [0.0, 1.0] and rounded to
            two decimal places.
        """
        label = _FUSION_TABLE.get((sentiment_signal, technical_signal), "HOLD")

        s = 0.5 if sentiment_confidence is None else sentiment_confidence
        t = 0.5 if technical_confidence is None else technical_confidence
        s = max(0.0, min(1.0, s))
        t = max(0.0, min(1.0, t))

        sent_dir = sentiment_signal in ("BUY", "SELL")
        tech_dir = technical_signal in ("BUY", "SELL")

        if sent_dir and tech_dir and sentiment_signal == technical_signal:
            combined_conf = min(1.0, max(s, t) * 1.1)
        elif sent_dir and tech_dir:
            combined_conf = abs(s - t) * 0.7
        else:
            if sentiment_signal == "HOLD" and technical_signal == "HOLD":
                non_hold = max(s, t)
            elif sentiment_signal == "HOLD":
                non_hold = t
            else:
                non_hold = s
            combined_conf = non_hold * 0.8

        return label, round(max(0.0, min(1.0, combined_conf)), 2)

    @staticmethod
    def confidence(
        combined_signal: str,
        avg_score: float,
        volume_confirmed: bool = False,
        rvol: "float | None" = None,
        technical_confidence: "float | None" = None,
    ) -> float:
        """
        Compute a confidence score (0.0-1.0) for the combined signal.

        The score blends sentiment strength with the technical agent's own
        confidence so that both agents contribute to the final number.

        STRONG signals start at 0.60 and scale up with a blend of sentiment
        strength (60%) and technical confidence (40%).  Range [0.60, 1.00].

        WEAK signals start at 0.35 and scale with the same blend.  Range
        [0.35, 0.60].  A normal trending market with moderate sentiment and
        a directional technical signal now lands in the 40-60% band.

        CONFLICTING is fixed at 0.10 (agents actively disagree).
        HOLD is fixed at 0.25 (no directional conviction).

        Volume adjustment (applied after base calculation):
            +0.10 when volume_confirmed is True (RVOL > 1.5 + OBV aligns).

        Args:
            combined_signal:      Output of combine_signals().
            avg_score:            Raw sentiment average (-1.0 to +1.0).
            volume_confirmed:     True when volume supports the signal direction.
            rvol:                 Relative volume (current / 20-day avg).
            technical_confidence: Technical agent's own confidence (0.0 - 1.0).
                                  When None the calculation falls back to
                                  sentiment-only scaling.

        Returns:
            Confidence float rounded to two decimal places, clamped to [0, 1].
        """
        sent_strength = abs(avg_score)  # 0.0 - 1.0
        tech_strength = technical_confidence if technical_confidence is not None else sent_strength

        # Blend: 60% sentiment, 40% technical
        blended = sent_strength * 0.6 + tech_strength * 0.4

        if combined_signal in ("STRONG BUY", "STRONG SELL"):
            base = min(1.0, 0.6 + blended * 0.4)
        elif combined_signal in ("WEAK BUY", "WEAK SELL"):
            base = min(0.6, 0.35 + blended * 0.25)
        elif combined_signal == "CONFLICTING":
            base = 0.10
        else:
            base = 0.25  # HOLD

        # Volume adjustment (only for directional signals)
        # Note: low-rvol penalty removed -- rvol compares partial intraday
        # volume to full-day average, making it structurally < 1.0 during
        # market hours and penalising every signal incorrectly.
        if combined_signal not in ("HOLD", "CONFLICTING"):
            if volume_confirmed:
                base += 0.10

        return round(max(0.0, min(1.0, base)), 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, ticker: str, verbose: bool = True) -> dict:
        """
        Execute the multi-source sentiment pipeline for *ticker*.

        Collects headlines from NewsAPI, StockTwits, and Reddit, scores each
        via Claude, and computes a weighted average using SOURCE_WEIGHTS.

        Logs current NewsAPI usage at each invocation to track rate limits.

        Args:
            ticker:  Stock ticker symbol (e.g. "AAPL").
            verbose: When True, print step-by-step progress to stdout.

        Returns:
            dict with keys: ticker, market, headlines_fetched, scored,
                            avg_score, signal, run_id, source_breakdown
        """
        ticker = ticker.upper()

        # Log NewsAPI usage at session start
        self.news_feed.log_usage()

        # Step 1 — market context
        market: dict = {}
        try:
            market = self.market_data.fetch(ticker)
            if verbose and market.get("price"):
                currency = market.get("currency", "")
                print(f"  {ticker}  |  {market['name']}  |  {market['price']} {currency}\n")
        except Exception as exc:
            if verbose:
                print(f"  [market data unavailable: {exc}]\n")

        # Step 2 — collect from all sources
        items: list[dict] = []  # each: {"text": ..., "source": ...}

        # NewsAPI
        if verbose:
            print(f"Fetching headlines for {ticker}...")
        newsapi_headlines = self.news_feed.fetch(ticker)
        for h in newsapi_headlines:
            items.append({"text": h, "source": "newsapi"})
        if verbose:
            print(f"  NewsAPI: {len(newsapi_headlines)} headline(s)")

        # StockTwits
        stocktwits_items = self.stocktwits_feed.fetch(ticker)
        for st in stocktwits_items:
            items.append({"text": st["text"], "source": "stocktwits"})
        if verbose:
            print(f"  StockTwits: {len(stocktwits_items)} message(s)")

        # Reddit
        reddit_items = self.reddit_feed.fetch(ticker)
        for rd in reddit_items:
            items.append({"text": rd["text"], "source": "reddit"})
        if verbose:
            print(f"  Reddit: {len(reddit_items)} post(s)")

        # Marketaux (pass last signal hint to skip HOLD tickers and save quota)
        _last_sig = self._last_combined_signal(ticker)
        marketaux_items = self.marketaux_feed.fetch(ticker, signal_hint=_last_sig)
        for mx in marketaux_items:
            items.append({"text": mx["text"], "source": "marketaux"})
        if verbose:
            print(f"  Marketaux: {len(marketaux_items)} article(s)")

        # ApeWisdom
        apewisdom_items = self.apewisdom_feed.fetch(ticker)
        for aw in apewisdom_items:
            items.append({"text": aw["text"], "source": "apewisdom"})
        if verbose:
            print(f"  ApeWisdom: {len(apewisdom_items)} mention(s)")

        # Adanos
        adanos_items = self.adanos_feed.fetch(ticker)
        for ad in adanos_items:
            items.append({"text": ad["text"], "source": "adanos"})
        if verbose:
            print(f"  Adanos: {len(adanos_items)} signal(s)")
            print(f"  Total: {len(items)} item(s)\n")

        # Step 3 — sentiment scoring
        scored: list[dict] = []
        for i, item in enumerate(items, 1):
            text = item["text"]
            source = item["source"]
            label = f"[{source}]"
            if verbose:
                print(f"[{i}/{len(items)}] {label} {text[:80]}")
            try:
                result = self.sentiment_agent.run(text, ticker)
                result["source"] = source
                scored.append(result)
                if verbose:
                    icon = {"bullish": "+", "bearish": "-", "neutral": "~"}.get(
                        result["sentiment"], "?"
                    )
                    print(f"         [{icon}] {result['sentiment'].upper()} — {result['reason']}")
            except (json.JSONDecodeError, KeyError, Exception) as exc:
                if verbose:
                    print(f"         [!] Skipped ({exc})")

        # Step 4 — weighted aggregate + signal
        avg_score = self._weighted_aggregate(scored, self._active_weights())
        signal = self._signal(avg_score)
        breakdown = self._source_breakdown(scored)

        # Step 5 — persist
        run_id = self.db.log_run(
            ticker=ticker,
            headlines_fetched=len(items),
            headlines_analysed=len(scored),
            avg_score=avg_score,
            signal=signal,
            source_breakdown=breakdown,
        )
        for s in scored:
            self.db.log_headline_score(
                run_id=run_id,
                headline=s["headline"],
                sentiment=s["sentiment"],
                score=s["score"],
                reason=s.get("reason", ""),
                source=s.get("source", "newsapi"),
            )

        return {
            "ticker": ticker,
            "market": market,
            "headlines_fetched": len(items),
            "scored": scored,
            "avg_score": avg_score,
            "signal": signal,
            "run_id": run_id,
            "source_breakdown": breakdown,
        }

    def run_combined(
        self,
        ticker: str,
        verbose: bool = True,
        account_balance: float = 10_000.0,
        execute: bool = False,
        session: str | None = None,
    ) -> dict:
        """
        Run sentiment, technical, and risk agents, then fuse their signals.

        Args:
            ticker:          Stock ticker symbol (e.g. "AAPL").
            verbose:         When True, print step-by-step progress to stdout.
            account_balance: Account size in USD used for position sizing.
            execute:         When True, execute the trade via PaperTrader.

        Returns:
            dict with keys:
                ticker           (str):   The analysed ticker.
                sentiment        (dict):  Full output of run() — sentiment pipeline.
                technical        (dict):  Full output of TechnicalAgent.run().
                combined_signal  (str):   Fused signal label.
                confidence       (float): Confidence score 0.0–1.0.
                combined_id      (int):   DB primary key for combined_signals row.
                risk             (dict):  Full output of RiskAgent.run().
                execution        (dict|None): PaperTrader result if execute=True.
        """
        ticker = ticker.upper()

        # --- Market regime (once per session, cached) ---
        regime_info: dict = {}
        try:
            regime_info = self.regime_agent.run()
            if verbose:
                regime = regime_info.get("regime", "UNKNOWN")
                cached = " (cached)" if regime_info.get("cached") else ""
                print(f"\n  Market regime: {regime}{cached}")
        except Exception as exc:
            if verbose:
                print(f"\n  [regime detection unavailable: {exc}]")

        # --- Sentiment pipeline ---
        if verbose:
            print(f"\n[1/3] Sentiment analysis for {ticker}...")
        sentiment = self.run(ticker, verbose=verbose)

        # --- Technical pipeline (always run for DB record) ---
        if verbose:
            print(f"\n[2/3] Technical analysis for {ticker}...")
        technical = self.technical_agent.run(ticker)
        if verbose:
            ind = technical["indicators"]
            p = ind.get("price")
            r = ind.get("rsi")
            fmt_p = f"{p:.2f}" if p is not None else "N/A"
            fmt_r = f"{r:.1f}" if r is not None else "N/A"
            print(f"  Price: {fmt_p}  RSI: {fmt_r}  →  {technical['signal']}")

        # --- Per-ticker regime detection ---
        ticker_regime = None
        if ENABLE_REGIME_FILTER:
            try:
                bars_for_regime = technical.get("bars")
                vix_val = regime_info.get("vix")
                if bars_for_regime is not None and not bars_for_regime.empty:
                    ticker_regime = self.regime_detector.detect(
                        ticker, bars_for_regime, vix=vix_val,
                    )
                    if verbose:
                        print(f"  Regime: {ticker_regime.regime} "
                              f"(ADX={ticker_regime.adx}, VIX={ticker_regime.vix}, "
                              f"ATR_pct={ticker_regime.atr_percentile:.0f}, "
                              f"size={ticker_regime.size_multiplier:.0%})")
            except Exception as exc:
                log.warning("[%s] Regime detection failed: %s", ticker, exc)

        # --- Strategy votes (Momentum + Pullback + NewsCatalyst) ---
        # strat_name is kept for downstream labels (debate skip logic, result
        # dict).  The router-selected name names the "primary" strategy for
        # this ticker even though all votes feed into the cluster detector.
        strat_name = strategy_label(ticker)
        _regime_name = ticker_regime.regime if ticker_regime else None

        bars = technical.get("bars")
        strategy_votes = self._gather_strategy_votes(
            ticker, bars, sentiment,
            session=session, regime=_regime_name,
        )
        if verbose and strategy_votes:
            summary = ", ".join(
                f"{r.strategy_name}={r.signal}({r.confidence:.0f}%)"
                for r in strategy_votes
            )
            print(f"  Strategy votes: {summary}")

        # Router-selected vote — still drives the downstream stop-loss /
        # take-profit override so the risk layer keeps its pre-cluster
        # behaviour (the cluster decides direction/confidence, the primary
        # strategy provides the SL/TP levels).
        strategy_result = next(
            (r for r in strategy_votes if r.strategy_name == strat_name),
            None,
        )

        # --- Fuse via cluster detector (falls back to combine_signals) ---
        combined_signal, conf, signal_path = self._fuse_signals(
            ticker,
            strategy_votes,
            sentiment_signal=sentiment["signal"],
            sentiment_confidence=abs(sentiment["avg_score"]),
            fallback_technical_signal=technical["signal"],
            fallback_technical_confidence=technical.get("adjusted_confidence"),
        )

        # --- Bull/Bear Debate (optional — skip for PEAD, pure data-driven) ---
        debate_result = None
        is_pead = strat_name == "PEAD"
        if self.debate_agent.is_enabled() and not is_pead:
            if combined_signal == "HOLD" and conf < 0.35:
                log.info("[%s] Debate skipped — HOLD signal", ticker)
                debate_result = DebateResult(
                    ticker=ticker,
                    original_signal=combined_signal,
                    original_confidence=conf,
                    final_signal=combined_signal,
                    adjusted_confidence=conf,
                    debate_summary="Debate skipped — HOLD signal.",
                )
            else:
                if verbose:
                    print(f"\n  [DEBATE] Running bull/bear debate for {ticker}...")
                debate_result = self.debate_agent.run(
                    ticker=ticker,
                    signal=combined_signal,
                    confidence=conf,
                    technical_data=technical.get("indicators", {}),
                    sentiment_data={"signal": sentiment["signal"], "avg_score": sentiment["avg_score"]},
                    macro_context=self.macro_context,
                )
            combined_signal = debate_result.final_signal
            conf = debate_result.adjusted_confidence

            # Enforce signal-type minimum floors after debate penalty.
            # The debate can reduce confidence but must not push it below
            # the floor for the resulting signal type.
            _SIGNAL_FLOORS = {
                "STRONG BUY": 0.60, "STRONG SELL": 0.60,
                "WEAK BUY": 0.35, "WEAK SELL": 0.35,
                "HOLD": 0.25,
                "CONFLICTING": 0.10,
            }
            floor = _SIGNAL_FLOORS.get(combined_signal, 0.0)
            if conf < floor:
                log.debug(
                    "[%s] Debate pushed confidence %.2f below %s floor %.2f — clamping",
                    ticker, conf, combined_signal, floor,
                )
                conf = floor

            if verbose:
                print(f"  [DEBATE] {debate_result.debate_summary}")
                print(f"  [DEBATE] Signal: {debate_result.original_signal} → {debate_result.final_signal}"
                      f"  Confidence: {debate_result.original_confidence:.0%} → {debate_result.adjusted_confidence:.0%}")

        # --- Persist combined ---
        combined_id = self.db.log_combined_signal(
            ticker=ticker,
            combined_signal=combined_signal,
            sentiment_signal=sentiment["signal"],
            technical_signal=technical["signal"],
            sentiment_score=sentiment["avg_score"],
            confidence=conf,
            run_id=sentiment["run_id"],
            technical_id=technical["signal_id"],
        )

        # --- Risk sizing ---
        if verbose:
            print(f"\n[3/3] Risk sizing for {ticker}...")
        # Prefer live market price; fall back to technical close price for analysis only
        market_info = sentiment.get("market") or {}
        market_price = market_info.get("price")
        tech_price = technical["indicators"].get("price")
        price = market_price or tech_price
        price_is_live = bool(market_price) and not market_info.get("degraded")
        if not price_is_live and verbose:
            print(f"  [WARNING] No live market price for {ticker} — using technical close")
        # Cross-validate: if both prices exist but diverge >20%, abort loudly
        if market_price and tech_price and tech_price > 0:
            divergence = abs(market_price - tech_price) / tech_price
            if divergence > 0.20:
                msg = (
                    f"[{ticker}] PRICE MISMATCH ABORT: "
                    f"market=${market_price:.2f} vs technical=${tech_price:.2f} "
                    f"({divergence * 100:.0f}% divergence) — possible ghost/stale price"
                )
                log.critical(msg)
                raise RuntimeError(msg)

        # Pre-compute event risk so we can downgrade signals before sizing
        days_to_earn = get_days_to_earnings(ticker)
        if days_to_earn is not None and days_to_earn <= 2:
            event_risk_flag = "earnings_imminent"
        elif days_to_earn is not None and days_to_earn <= 5:
            event_risk_flag = "earnings_week"
        else:
            event_risk_flag = "none"

        # Use per-ticker regime when available, fall back to broad market
        _effective_regime = (ticker_regime.regime if ticker_regime
                             else regime_info.get("regime"))
        risk = self.risk_agent.run(
            ticker=ticker,
            signal=combined_signal,
            confidence=conf * 100,          # convert 0–1 → 0–100
            current_price=price,
            account_balance=account_balance,
            regime=_effective_regime,
        )

        # Override SL/TP with strategy-specific values when available
        if strategy_result is not None and not risk["skipped"]:
            if strategy_result.stop_loss:
                risk["stop_loss"] = strategy_result.stop_loss
            if strategy_result.take_profit:
                risk["take_profit"] = strategy_result.take_profit

        if verbose:
            if risk["skipped"]:
                print(f"  No position — {risk['skip_reason']}")
            else:
                print(
                    f"  ${risk['position_size_usd']:,.2f}  "
                    f"({risk['shares']} shares)  "
                    f"SL: ${risk['stop_loss']:.2f}  "
                    f"TP: ${risk['take_profit']:.2f}"
                    f"  [{strat_name}]"
                )

        # --- Paper trade execution ---
        execution = None
        if execute and not risk["skipped"]:
            # Guard: abort if price is invalid or not live
            if not price or price <= 0:
                if verbose:
                    print(f"\n  [ABORTED] No valid price for {ticker} — skipping trade")
            elif not price_is_live:
                log.warning(
                    "[%s] Trade blocked — no live market price (price=%s, source=%s, degraded=%s)",
                    ticker, price, market_info.get("source", "none"),
                    market_info.get("degraded", "key_missing"),
                )
                if verbose:
                    print(f"\n  [ABORTED] No live market price for {ticker} — refusing to trade on stale data")
            # Guard: abort if stop_loss or take_profit missing/zero
            elif not risk.get("stop_loss") or risk["stop_loss"] <= 0 \
                    or not risk.get("take_profit") or risk["take_profit"] <= 0:
                if verbose:
                    print(f"\n  [ABORTED] Missing/invalid stop_loss/take_profit for {ticker}"
                          f" (SL={risk.get('stop_loss')}, TP={risk.get('take_profit')})")
            # Guard: skip if position already open (duplicate trade prevention)
            # Check Alpaca first (source of truth), fall back to local DB
            elif risk["direction"] == "BUY" and (
                self._has_alpaca_position(ticker)
                or self.db.get_portfolio_position(ticker)
            ):
                log.info("[%s] Duplicate BUY blocked — position already open", ticker)
                if verbose:
                    print(f"\n  [SKIPPED] Position already open for {ticker}")
            else:
                execution = self.paper_trader.track_trade(
                    ticker=ticker,
                    action=risk["direction"],
                    shares=risk["shares"],
                    price=price,
                    stop_loss=risk["stop_loss"],
                    take_profit=risk["take_profit"],
                )
                if verbose:
                    print(f"\n  [EXECUTED] Paper trade #{execution['trade_id']}: "
                          f"{risk['direction']} {risk['shares']} {ticker} "
                          f"@ ${price:.2f}")

        final_result = {
            "ticker": ticker,
            "sentiment": sentiment,
            "technical": technical,
            "combined_signal": combined_signal,
            "confidence": conf,
            "combined_id": combined_id,
            "risk": risk,
            "account_balance": account_balance,
            "execution": execution,
            "event_risk_flag": event_risk_flag,
            "regime": regime_info,
            "strategy_name": strat_name,
            "signal_path": signal_path,
            "debate": debate_result,
        }

        # Signal analytics logging (fire-and-forget)
        self._log_signal_event(final_result, session=session)

        return final_result

    # ------------------------------------------------------------------
    # Async pipeline (for batch / scheduler usage)
    # ------------------------------------------------------------------

    async def analyse_ticker_async(
        self,
        ticker: str,
        *,
        account_balance: float = 10_000.0,
        execute: bool = False,
        api_semaphore: asyncio.Semaphore,
        data_semaphore: asyncio.Semaphore,
        db_lock: asyncio.Lock,
        debate_semaphore: asyncio.Semaphore | None = None,
        session: str | None = None,
        session_type: str = "signal",
        scanner_candidates: set[str] | None = None,
    ) -> dict:
        """
        Async version of ``run_combined()`` with semaphore-controlled phases.

        Breaks the pipeline into four phases, each guarded by the
        appropriate concurrency primitive:

        1. **Data fetch** (market data, headlines, social) — *data_semaphore*
        2. **Sentiment scoring** (Claude API per headline) — *api_semaphore*
        3. **Technical analysis** (yfinance indicators) — *data_semaphore*
        4. **DB writes, risk sizing, execution** — *db_lock*

        The existing synchronous ``run_combined()`` is unchanged.

        Args:
            ticker:          Stock ticker symbol.
            account_balance: Account size in USD.
            execute:         When True, execute via broker.
            api_semaphore:   Limits concurrent Claude API calls.
            data_semaphore:  Limits concurrent yfinance / HTTP fetches.
            db_lock:         Serialises all database writes.
            session_type:    "signal" | "execution" | "monitor".

        Returns:
            Same dict shape as ``run_combined()``, plus ``elapsed_s``.
        """
        # ── Dispatch by session type ──────────────────────────────────
        if session_type == "monitor":
            return await self._monitor_position_async(
                ticker,
                data_semaphore=data_semaphore,
                db_lock=db_lock,
                session=session,
            )
        if session_type == "execution":
            return await self._execute_forward_signals_async(
                ticker,
                account_balance=account_balance,
                execute=execute,
                api_semaphore=api_semaphore,
                data_semaphore=data_semaphore,
                db_lock=db_lock,
                debate_semaphore=debate_semaphore,
                session=session,
                scanner_candidates=scanner_candidates,
            )
        if session_type == "pre_signal":
            return await self._pre_signal_refresh_async(
                ticker,
                api_semaphore=api_semaphore,
                data_semaphore=data_semaphore,
                db_lock=db_lock,
                debate_semaphore=debate_semaphore,
                session=session,
            )
        if session_type == "pead":
            return await self._run_pead_only_async(
                ticker,
                account_balance=account_balance,
                execute=execute,
                data_semaphore=data_semaphore,
                db_lock=db_lock,
                session=session,
            )

        t0 = _time.monotonic()
        ticker = ticker.upper()

        # --- Market regime (cached, fast) ---
        regime_info: dict = {}
        try:
            regime_info = self.regime_agent.run()
        except Exception:
            pass

        # ── Phase 1: Data fetches ──────────────────────────────────────
        async with data_semaphore:
            market: dict = {}
            try:
                market = await asyncio.to_thread(self.market_data.fetch, ticker)
            except Exception as exc:
                log.warning("[%s] Market data fetch failed: %s", ticker, exc)

            newsapi_headlines = await asyncio.to_thread(
                self.news_feed.fetch, ticker,
            )
            stocktwits_items = await asyncio.to_thread(
                self.stocktwits_feed.fetch, ticker,
            )
            reddit_items = await asyncio.to_thread(
                self.reddit_feed.fetch, ticker,
            )
            _last_sig = self._last_combined_signal(ticker)
            marketaux_items = await asyncio.to_thread(
                self.marketaux_feed.fetch, ticker, _last_sig,
            )
            apewisdom_items = await asyncio.to_thread(
                self.apewisdom_feed.fetch, ticker,
            )
            adanos_items = await asyncio.to_thread(
                self.adanos_feed.fetch, ticker,
            )

        # Build items list
        items: list[dict] = []
        for h in newsapi_headlines:
            items.append({"text": h, "source": "newsapi"})
        for st_item in stocktwits_items:
            items.append({"text": st_item["text"], "source": "stocktwits"})
        for rd in reddit_items:
            items.append({"text": rd["text"], "source": "reddit"})
        for mx in marketaux_items:
            items.append({"text": mx["text"], "source": "marketaux"})
        for aw in apewisdom_items:
            items.append({"text": aw["text"], "source": "apewisdom"})
        for ad in adanos_items:
            items.append({"text": ad["text"], "source": "adanos"})

        # ── Phase 2: Sentiment scoring (Claude API) ────────────────────
        scored: list[dict] = []
        for item in items:
            async with api_semaphore:
                try:
                    result = await asyncio.to_thread(
                        self.sentiment_agent.run, item["text"], ticker,
                    )
                    result["source"] = item["source"]
                    scored.append(result)
                except Exception:
                    pass

        avg_score = self._weighted_aggregate(scored, self._active_weights())
        sentiment_signal = self._signal(avg_score)
        breakdown = self._source_breakdown(scored)

        # ── Phase 3: Technical analysis (yfinance) ─────────────────────
        async with data_semaphore:
            technical = await asyncio.to_thread(
                self.technical_agent.run, ticker,
            )

        # ── Per-ticker regime detection ────────────────────────────────
        ticker_regime = None
        if ENABLE_REGIME_FILTER:
            try:
                bars_for_regime = technical.get("bars")
                vix_val = regime_info.get("vix")
                if bars_for_regime is not None and not bars_for_regime.empty:
                    ticker_regime = self.regime_detector.detect(
                        ticker, bars_for_regime, vix=vix_val,
                    )
            except Exception as exc:
                log.warning("[%s] Regime detection failed: %s", ticker, exc)

        # ── Strategy votes (Momentum + Pullback + NewsCatalyst) ─────────
        # strat_name is kept for downstream labels (debate skip logic, result
        # dict).  The router-selected name names the "primary" strategy for
        # this ticker even though all votes feed into the cluster detector.
        strat_name = strategy_label(ticker)
        _regime_name = ticker_regime.regime if ticker_regime else None

        bars = technical.get("bars")
        sentiment_for_cluster = {
            "avg_score": avg_score,
            "scored": scored,
            "signal": sentiment_signal,
        }
        strategy_votes = await asyncio.to_thread(
            self._gather_strategy_votes,
            ticker, bars, sentiment_for_cluster,
            session=session, regime=_regime_name,
        )

        # Router-selected vote — drives the downstream stop-loss /
        # take-profit override (the cluster decides direction/confidence,
        # the primary strategy provides the SL/TP levels).
        strategy_result = next(
            (r for r in strategy_votes if r.strategy_name == strat_name),
            None,
        )

        # ── PEAD check (no API calls — pure data) ─────────────────────
        pead_result = self._run_pead(ticker, session=session)

        # ── Fuse signals ───────────────────────────────────────────────
        if pead_result is not None and pead_result["signal"] == "BUY":
            # PEAD BUY acts as a strong additional vote — bypass the cluster
            # and use the legacy sentiment-vs-PEAD fusion so PEAD's confidence
            # dominates (matches pre-Phase-2b PEAD override semantics).
            combined_signal, conf = self.combine_signals(
                sentiment_signal, "BUY",
                sentiment_confidence=abs(avg_score),
                technical_confidence=pead_result["confidence"],
            )
            signal_path = "FUSION_FALLBACK"
            strat_name = "PEAD"
            log.info("[%s] PEAD overriding strategy: %s (%.0f%%)",
                     ticker, combined_signal, conf * 100)
        else:
            combined_signal, conf, signal_path = self._fuse_signals(
                ticker,
                strategy_votes,
                sentiment_signal=sentiment_signal,
                sentiment_confidence=abs(avg_score),
                fallback_technical_signal=technical["signal"],
                fallback_technical_confidence=technical.get("adjusted_confidence"),
            )

        # ── Bull/Bear Debate (optional — skip for PEAD, pure data-driven) ──
        debate_result = None
        is_pead = strat_name == "PEAD"
        is_scanner_candidate = ticker.upper() in (scanner_candidates or set())
        if is_scanner_candidate:
            log.info("[%s] Scanner candidate (Tier 2) — skipping bull/bear debate", ticker)
        if self.debate_agent.is_enabled() and not is_pead and not is_scanner_candidate:
            if combined_signal == "HOLD" and conf < 0.35:
                log.info("[%s] Debate skipped — HOLD signal", ticker)
                debate_result = DebateResult(
                    ticker=ticker,
                    original_signal=combined_signal,
                    original_confidence=conf,
                    final_signal=combined_signal,
                    adjusted_confidence=conf,
                    debate_summary="Debate skipped — HOLD signal.",
                )
            else:
                _dsem = debate_semaphore or api_semaphore
                async with _dsem:
                    debate_result = await self.debate_agent.run_async(
                        ticker=ticker,
                        signal=combined_signal,
                        confidence=conf,
                        technical_data=technical.get("indicators", {}),
                        sentiment_data={"signal": sentiment_signal, "avg_score": avg_score},
                        macro_context=self.macro_context,
                    )
            combined_signal = debate_result.final_signal
            conf = debate_result.adjusted_confidence

        # ── Phase 4: DB writes + risk sizing ───────────────────────────
        async with db_lock:
            run_id = self.db.log_run(
                ticker=ticker,
                headlines_fetched=len(items),
                headlines_analysed=len(scored),
                avg_score=avg_score,
                signal=sentiment_signal,
                source_breakdown=breakdown,
            )
            for s in scored:
                self.db.log_headline_score(
                    run_id=run_id,
                    headline=s["headline"],
                    sentiment=s["sentiment"],
                    score=s["score"],
                    reason=s.get("reason", ""),
                    source=s.get("source", "newsapi"),
                )

            combined_id = self.db.log_combined_signal(
                ticker=ticker,
                combined_signal=combined_signal,
                sentiment_signal=sentiment_signal,
                technical_signal=technical["signal"],
                sentiment_score=avg_score,
                confidence=conf,
                run_id=run_id,
                technical_id=technical["signal_id"],
            )

        # --- Risk sizing ---
        market_price = (market or {}).get("price")
        tech_price = technical["indicators"].get("price")
        price = market_price or tech_price
        price_is_live = bool(market_price) and not (market or {}).get("degraded")
        # Cross-validate: if both prices exist but diverge >20%, abort loudly
        if market_price and tech_price and tech_price > 0:
            divergence = abs(market_price - tech_price) / tech_price
            if divergence > 0.20:
                msg = (
                    f"[{ticker}] PRICE MISMATCH ABORT: "
                    f"market=${market_price:.2f} vs technical=${tech_price:.2f} "
                    f"({divergence * 100:.0f}% divergence) — possible ghost/stale price"
                )
                log.critical(msg)
                raise RuntimeError(msg)

        days_to_earn = get_days_to_earnings(ticker)
        if days_to_earn is not None and days_to_earn <= 2:
            event_risk_flag = "earnings_imminent"
        elif days_to_earn is not None and days_to_earn <= 5:
            event_risk_flag = "earnings_week"
        else:
            event_risk_flag = "none"

        async with db_lock:
            risk = self.risk_agent.run(
                ticker=ticker,
                signal=combined_signal,
                confidence=conf * 100,
                current_price=price,
                account_balance=account_balance,
                regime=regime_info.get("regime"),
            )

        # Override SL/TP with strategy-specific values when available
        if strategy_result is not None and not risk["skipped"]:
            if strategy_result.stop_loss:
                risk["stop_loss"] = strategy_result.stop_loss
            if strategy_result.take_profit:
                risk["take_profit"] = strategy_result.take_profit

        # --- Execution ---
        execution = None
        if execute and not risk["skipped"]:
            # Guard: abort if price is invalid
            if not price or price <= 0:
                pass  # skip — no valid price
            # Guard: abort if price is not live (stale/fallback data)
            elif not price_is_live:
                log.warning(
                    "[%s] Trade blocked — no live market price (price=%s, source=%s, degraded=%s)",
                    ticker, price, (market or {}).get("source", "none"),
                    (market or {}).get("degraded", "key_missing"),
                )
            # Guard: abort if stop_loss or take_profit missing/zero
            elif not risk.get("stop_loss") or risk["stop_loss"] <= 0 \
                    or not risk.get("take_profit") or risk["take_profit"] <= 0:
                pass  # skip — missing or invalid protective levels
            else:
                # Atomic check-then-execute under a single lock acquisition
                async with db_lock:
                    # Check Alpaca first (source of truth), fall back to local DB
                    if risk["direction"] == "BUY" and (
                        self._has_alpaca_position(ticker)
                        or self.db.get_portfolio_position(ticker)
                    ):
                        log.info("[%s] Duplicate BUY blocked — position already open", ticker)
                    else:
                        execution = self.paper_trader.track_trade(
                            ticker=ticker,
                            action=risk["direction"],
                            shares=risk["shares"],
                            price=price,
                            stop_loss=risk["stop_loss"],
                            take_profit=risk["take_profit"],
                        )

        elapsed = _time.monotonic() - t0

        sentiment_result = {
            "ticker": ticker,
            "market": market,
            "headlines_fetched": len(items),
            "scored": scored,
            "avg_score": avg_score,
            "signal": sentiment_signal,
            "run_id": run_id,
            "source_breakdown": breakdown,
        }

        final_result = {
            "ticker": ticker,
            "sentiment": sentiment_result,
            "technical": technical,
            "combined_signal": combined_signal,
            "confidence": conf,
            "combined_id": combined_id,
            "risk": risk,
            "account_balance": account_balance,
            "execution": execution,
            "event_risk_flag": event_risk_flag,
            "regime": regime_info,
            "strategy_name": strat_name,
            "signal_path": signal_path,
            "debate": debate_result,
            "is_scanner_candidate": is_scanner_candidate,
            "elapsed_s": round(elapsed, 2),
        }

        # Signal analytics logging (fire-and-forget)
        self._log_signal_event(final_result, session=session)

        # ── Forward signal storage (EOD → US_OPEN pipeline) ──────────
        # In signal mode, store actionable signals as forward signals
        # so the next execution session can validate and trade them.
        if session_type == "signal" and combined_signal not in ("HOLD", "CONFLICTING"):
            try:
                self.signal_logger.store_forward_signal({
                    "source_session": session or "EOD",
                    "target_session": "US_OPEN",
                    "ticker": ticker,
                    "signal": combined_signal,
                    "confidence": conf,
                    "price_at_signal": price,
                    "strategy_name": strat_name,
                    "stop_loss": risk.get("stop_loss") if not risk.get("skipped") else None,
                    "take_profit": risk.get("take_profit") if not risk.get("skipped") else None,
                })
            except Exception as exc:
                log.warning("[%s] Forward signal storage failed (non-fatal): %s", ticker, exc)

        return final_result

    # ------------------------------------------------------------------
    # Monitor mode (MIDDAY — position check only)
    # ------------------------------------------------------------------

    async def _monitor_position_async(
        self,
        ticker: str,
        *,
        data_semaphore: asyncio.Semaphore,
        db_lock: asyncio.Lock,
        session: str | None = None,
    ) -> dict:
        """Lightweight position check — no signal generation."""
        t0 = _time.monotonic()
        ticker = ticker.upper()

        position = None
        async with db_lock:
            position = self.db.get_portfolio_position(ticker)

        if not position:
            return {
                "ticker": ticker,
                "session_type": "monitor",
                "combined_signal": "MONITOR",
                "confidence": 0,
                "has_position": False,
                "elapsed_s": round(_time.monotonic() - t0, 2),
            }

        # Fetch current price
        current_price = None
        async with data_semaphore:
            try:
                market = await asyncio.to_thread(self.market_data.fetch, ticker)
                current_price = market.get("price")
            except Exception as exc:
                log.warning("[%s] Monitor price fetch failed: %s", ticker, exc)

        # Load stop-loss / take-profit from most recent trade
        trade_info: dict = {}
        async with db_lock:
            try:
                with self.db._connect() as conn:
                    row = conn.execute(
                        "SELECT stop_loss, take_profit FROM trade_history "
                        "WHERE ticker = ? ORDER BY id DESC LIMIT 1",
                        (ticker,),
                    ).fetchone()
                    if row:
                        trade_info = dict(row)
            except Exception:
                pass

        stop_loss = trade_info.get("stop_loss")
        take_profit = trade_info.get("take_profit")

        # Compute distances
        dist_to_stop = None
        dist_to_tp = None
        if current_price and stop_loss:
            dist_to_stop = round((current_price - stop_loss) / current_price * 100, 2)
        if current_price and take_profit:
            dist_to_tp = round((take_profit - current_price) / current_price * 100, 2)

        elapsed = _time.monotonic() - t0

        return {
            "ticker": ticker,
            "session_type": "monitor",
            "combined_signal": "MONITOR",
            "confidence": 0,
            "has_position": True,
            "current_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "distance_to_stop_pct": dist_to_stop,
            "distance_to_tp_pct": dist_to_tp,
            "elapsed_s": round(elapsed, 2),
        }

    # ------------------------------------------------------------------
    # Pre-signal mode (XETRA_PRE, US_PRE — news + sentiment refresh)
    # ------------------------------------------------------------------

    _PRE_CONVICTION_DELTA = 0.10  # only update forward signal if >10% change

    async def _pre_signal_refresh_async(
        self,
        ticker: str,
        *,
        api_semaphore: asyncio.Semaphore,
        data_semaphore: asyncio.Semaphore,
        db_lock: asyncio.Lock,
        debate_semaphore: asyncio.Semaphore | None = None,
        session: str | None = None,
    ) -> dict:
        """
        Lightweight signal refresh — news + sentiment + debate only.

        Skips: technical analysis, position sizing, execution.
        Only updates the forward signal if conviction changed by >10%.
        """
        t0 = _time.monotonic()
        ticker = ticker.upper()
        log.info("[%s] Refreshing signals (pre-session)", ticker)

        # ── Phase 1: Data fetches ─────────────────────────────────────
        async with data_semaphore:
            newsapi_headlines = await asyncio.to_thread(
                self.news_feed.fetch, ticker,
            )
            stocktwits_items = await asyncio.to_thread(
                self.stocktwits_feed.fetch, ticker,
            )
            marketaux_items = await asyncio.to_thread(
                self.marketaux_feed.fetch, ticker,
                self._last_combined_signal(ticker),
            )

        items: list[dict] = []
        for h in newsapi_headlines:
            items.append({"text": h, "source": "newsapi"})
        for st_item in stocktwits_items:
            items.append({"text": st_item["text"], "source": "stocktwits"})
        for mx in marketaux_items:
            items.append({"text": mx["text"], "source": "marketaux"})

        # ── Phase 2: Sentiment scoring (Claude API) ───────────────────
        scored: list[dict] = []
        for item in items:
            async with api_semaphore:
                try:
                    result = await asyncio.to_thread(
                        self.sentiment_agent.run, item["text"], ticker,
                    )
                    result["source"] = item["source"]
                    scored.append(result)
                except Exception:
                    pass

        avg_score = self._weighted_aggregate(scored, self._active_weights())
        sentiment_signal = self._signal(avg_score)

        # ── Fuse with sentiment-only (no TA, no strategies) ───────────
        combined_signal, conf = self.combine_signals(
            sentiment_signal, "HOLD",
            sentiment_confidence=abs(avg_score),
            technical_confidence=0.5,
        )
        signal_path = "FUSION_FALLBACK"

        # ── Bull/Bear Debate (optional) ───────────────────────────────
        debate_result = None
        if self.debate_agent.is_enabled():
            if combined_signal == "HOLD" and conf < 0.35:
                log.info("[%s] Debate skipped — HOLD signal", ticker)
                debate_result = DebateResult(
                    ticker=ticker,
                    original_signal=combined_signal,
                    original_confidence=conf,
                    final_signal=combined_signal,
                    adjusted_confidence=conf,
                    debate_summary="Debate skipped — HOLD signal.",
                )
            else:
                _dsem = debate_semaphore or api_semaphore
                async with _dsem:
                    debate_result = await self.debate_agent.run_async(
                        ticker=ticker,
                        signal=combined_signal,
                        confidence=conf,
                        technical_data={},
                        sentiment_data={"signal": sentiment_signal, "avg_score": avg_score},
                        macro_context=self.macro_context,
                    )
            combined_signal = debate_result.final_signal
            conf = debate_result.adjusted_confidence

        # ── Check existing forward signal for conviction delta ────────
        updated_forward = False
        if combined_signal not in ("HOLD", "CONFLICTING"):
            try:
                # Determine target execution session
                target_session = "XETRA_OPEN" if session and "XETRA" in session else "US_OPEN"
                pending = self.signal_logger.get_pending_forward_signals(
                    target_session, ticker,
                )
                if pending:
                    old_conf = pending[0].get("confidence", 0)
                    delta = abs(conf - old_conf)
                    if delta > self._PRE_CONVICTION_DELTA:
                        log.info(
                            "[%s] PRE conviction changed %.0f%% → %.0f%% (delta %.0f%%) — updating forward signal",
                            ticker, old_conf * 100, conf * 100, delta * 100,
                        )
                        self.signal_logger.store_forward_signal({
                            "source_session": session or "PRE",
                            "target_session": target_session,
                            "ticker": ticker,
                            "signal": combined_signal,
                            "confidence": conf,
                            "price_at_signal": None,
                        })
                        updated_forward = True
                    else:
                        log.info(
                            "[%s] PRE conviction stable (%.0f%% → %.0f%%, delta %.0f%%) — no update",
                            ticker, old_conf * 100, conf * 100, delta * 100,
                        )
                else:
                    # No existing forward signal — store new one
                    log.info(
                        "[%s] PRE new forward signal: %s (%.0f%%)",
                        ticker, combined_signal, conf * 100,
                    )
                    self.signal_logger.store_forward_signal({
                        "source_session": session or "PRE",
                        "target_session": "XETRA_OPEN" if session and "XETRA" in session else "US_OPEN",
                        "ticker": ticker,
                        "signal": combined_signal,
                        "confidence": conf,
                        "price_at_signal": None,
                    })
                    updated_forward = True
            except Exception as exc:
                log.warning("[%s] PRE forward signal update failed: %s", ticker, exc)

        elapsed = _time.monotonic() - t0

        return {
            "ticker": ticker,
            "session_type": "pre_signal",
            "combined_signal": combined_signal,
            "confidence": conf,
            "headlines_fetched": len(items),
            "headlines_scored": len(scored),
            "avg_score": avg_score,
            "sentiment_signal": sentiment_signal,
            "debate": debate_result,
            "updated_forward": updated_forward,
            "elapsed_s": round(elapsed, 2),
        }

    # ------------------------------------------------------------------
    # Execution mode (US_OPEN — validate and execute forward signals)
    # ------------------------------------------------------------------

    _FORWARD_PRICE_DRIFT_PCT = 2.0  # max acceptable price drift

    async def _execute_forward_signals_async(
        self,
        ticker: str,
        *,
        account_balance: float = 10_000.0,
        execute: bool = False,
        api_semaphore: asyncio.Semaphore,
        data_semaphore: asyncio.Semaphore,
        db_lock: asyncio.Lock,
        debate_semaphore: asyncio.Semaphore | None = None,
        session: str | None = None,
        scanner_candidates: set[str] | None = None,
    ) -> dict:
        """
        Execute trades using cached US_PRE signals (fast path).

        1. Read cached signal from signal_events (max 90 min old).
        2. Fetch current price.
        3. If price moved >2% since cache, re-run debate only.
        4. Execute trade based on cached/refreshed signal.

        Falls back to the full pipeline only when no cached signal exists.
        """
        from storage.database import Database

        t0 = _time.monotonic()
        ticker = ticker.upper()

        # ── Step 1: Try cached US_PRE signal ─────────────────────────────
        cached = self.db.get_cached_signal(ticker, max_age_minutes=90)

        if not cached or not cached.get("signal"):
            log.info(
                "[%s] No valid US_PRE cache — running full pipeline as fallback",
                ticker,
            )
            fallback = await self.analyse_ticker_async(
                ticker,
                account_balance=account_balance,
                execute=execute,
                api_semaphore=api_semaphore,
                data_semaphore=data_semaphore,
                db_lock=db_lock,
                debate_semaphore=debate_semaphore,
                session=session,
                session_type="signal",
                scanner_candidates=scanner_candidates,
            )
            fallback["session_type"] = "execution"
            fallback["cache_status"] = "miss"
            fallback["elapsed_s"] = round(_time.monotonic() - t0, 2)
            return fallback

        cached_signal = cached["signal"]
        cached_conf = cached.get("confidence") or 0.5
        cached_price = cached.get("price_at_signal")

        log.info(
            "[%s] Cached signal: %s (%.0f%%) @ $%.2f",
            ticker, cached_signal, cached_conf * 100,
            cached_price or 0,
        )

        # ── Step 2: Fetch current price ──────────────────────────────────
        current_price = None
        async with data_semaphore:
            try:
                market = await asyncio.to_thread(self.market_data.fetch, ticker)
                current_price = market.get("price")
            except Exception as exc:
                log.warning("[%s] Execution mode price fetch failed: %s", ticker, exc)

        if not current_price or current_price <= 0:
            return {
                "ticker": ticker,
                "session_type": "execution",
                "combined_signal": cached_signal,
                "confidence": cached_conf,
                "cache_status": "used_no_price",
                "execution": None,
                "elapsed_s": round(_time.monotonic() - t0, 2),
            }

        # ── Step 3: Staleness check — re-run debate if >2% drift ────────
        combined_signal = cached_signal
        conf = cached_conf
        debate_refreshed = False
        is_scanner_candidate = ticker.upper() in (scanner_candidates or set())

        if Database.is_price_stale(current_price, cached_price or 0, threshold=0.02):
            log.info(
                "[%s] Price stale: cached $%.2f → current $%.2f (>2%%) — refreshing debate",
                ticker, cached_price or 0, current_price,
            )
            if is_scanner_candidate:
                log.info(
                    "[%s] Scanner candidate (Tier 2) — skipping debate refresh on drift",
                    ticker,
                )
            elif self.debate_agent.is_enabled() and cached_signal not in ("HOLD",):
                _dsem = debate_semaphore or api_semaphore
                async with _dsem:
                    debate_result = await self.debate_agent.run_async(
                        ticker=ticker,
                        signal=cached_signal,
                        confidence=cached_conf,
                        technical_data={},
                        sentiment_data={
                            "signal": cached_signal,
                            "avg_score": cached.get("sentiment_score", 0),
                        },
                        macro_context=self.macro_context,
                    )
                combined_signal = debate_result.final_signal
                conf = debate_result.adjusted_confidence
                debate_refreshed = True
                log.info(
                    "[%s] Debate refresh: %s %.0f%% → %s %.0f%%",
                    ticker, cached_signal, cached_conf * 100,
                    combined_signal, conf * 100,
                )
            else:
                log.info("[%s] Debate disabled or HOLD — using cached signal as-is", ticker)

        # ── Step 4: Process pending forward signals ──────────────────────
        pending = self.signal_logger.get_pending_forward_signals(
            target_session="US_OPEN", ticker=ticker,
        )
        forward_results: list[dict] = []
        for fwd in pending:
            fwd_id = fwd["id"]
            signal_price = fwd.get("price_at_signal")
            fwd_signal = fwd.get("signal", "")

            if not current_price or not signal_price:
                self.signal_logger.update_forward_signal(
                    fwd_id, "invalidated", "no price data available",
                )
                forward_results.append({"id": fwd_id, "status": "invalidated", "reason": "no_price"})
                continue

            is_buy = "BUY" in fwd_signal.upper()
            if is_buy:
                adverse_drift = (signal_price - current_price) / signal_price * 100
            else:
                adverse_drift = (current_price - signal_price) / signal_price * 100

            if adverse_drift > self._FORWARD_PRICE_DRIFT_PCT:
                reason = (
                    f"price moved {adverse_drift:.1f}% against signal "
                    f"(was {signal_price:.2f}, now {current_price:.2f})"
                )
                self.signal_logger.update_forward_signal(fwd_id, "invalidated", reason)
                log.info("[%s] Forward signal invalidated: %s", ticker, reason)
                forward_results.append({"id": fwd_id, "status": "invalidated", "reason": reason})
                continue

            self.signal_logger.update_forward_signal(fwd_id, "confirmed")
            forward_results.append({"id": fwd_id, "status": "confirmed"})

        # ── Step 5: Execute trade based on cached/refreshed signal ───────
        execution = None
        risk = {"skipped": True, "skip_reason": "no actionable signal"}

        if combined_signal not in ("HOLD", "CONFLICTING") and execute:
            async with db_lock:
                risk = self.risk_agent.run(
                    ticker=ticker,
                    signal=combined_signal,
                    confidence=conf * 100,
                    current_price=current_price,
                    account_balance=account_balance,
                )

            if not risk["skipped"]:
                # Use forward signal SL/TP if available
                for fwd in pending:
                    if fwd.get("stop_loss"):
                        risk["stop_loss"] = fwd["stop_loss"]
                    if fwd.get("take_profit"):
                        risk["take_profit"] = fwd["take_profit"]
                    break  # use first matching forward signal

                if (risk.get("stop_loss") and risk["stop_loss"] > 0
                        and risk.get("take_profit") and risk["take_profit"] > 0):
                    async with db_lock:
                        if risk["direction"] == "BUY" and (
                            self._has_alpaca_position(ticker)
                            or self.db.get_portfolio_position(ticker)
                        ):
                            log.info("[%s] BUY blocked — position already open", ticker)
                        else:
                            execution = self.paper_trader.track_trade(
                                ticker=ticker,
                                action=risk["direction"],
                                shares=risk["shares"],
                                price=current_price,
                                stop_loss=risk["stop_loss"],
                                take_profit=risk["take_profit"],
                            )
                            log.info(
                                "[%s] Cached signal executed: %s %d shares @ %.2f",
                                ticker, risk["direction"], risk["shares"], current_price,
                            )
                            for fr in forward_results:
                                if fr["status"] == "confirmed":
                                    self.signal_logger.update_forward_signal(
                                        fr["id"], "executed",
                                    )

        elapsed = _time.monotonic() - t0

        return {
            "ticker": ticker,
            "session_type": "execution",
            "combined_signal": combined_signal,
            "confidence": conf,
            "risk": risk,
            "execution": execution,
            "forward_signals": forward_results,
            "cache_status": "refreshed" if debate_refreshed else "hit",
            "cached_price": cached_price,
            "current_price": current_price,
            "elapsed_s": round(elapsed, 2),
        }
