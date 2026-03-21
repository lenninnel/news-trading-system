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

from data.alpaca_data import AlpacaDataClient

log = logging.getLogger(__name__)

from agents.bull_bear_debate import BullBearDebate
from agents.regime_agent import RegimeAgent
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from config.settings import (
    BUY_THRESHOLD,
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
from storage.database import Database
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
    ) -> None:
        self.db = db or Database()
        self.news_feed = news_feed or NewsFeed()
        self.market_data = market_data or MarketData()
        self.sentiment_agent = sentiment_agent or SentimentAgent()
        # All agents share the same DB instance
        self.technical_agent = technical_agent or TechnicalAgent(db=self.db)
        self.risk_agent = risk_agent or RiskAgent(db=self.db)
        self.regime_agent = regime_agent or RegimeAgent()
        self.debate_agent = debate_agent or BullBearDebate()
        self.paper_trader = paper_trader or create_trader(db=self.db)
        self.reddit_feed = reddit_feed or RedditFeed()
        self.stocktwits_feed = stocktwits_feed or StockTwitsFeed()
        self.marketaux_feed = marketaux_feed or MarketauxFeed()
        self.apewisdom_feed = apewisdom_feed or ApeWisdomFeed()
        self.adanos_feed = adanos_feed or AdanosFeed()

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

    # ------------------------------------------------------------------
    # Signal fusion (static — easily unit-testable)
    # ------------------------------------------------------------------

    @staticmethod
    def combine_signals(sentiment_signal: str, technical_signal: str) -> str:
        """
        Fuse sentiment and technical signals into a combined label.

        Args:
            sentiment_signal: "BUY" | "SELL" | "HOLD"
            technical_signal: "BUY" | "SELL" | "HOLD"

        Returns:
            One of: STRONG BUY, STRONG SELL, WEAK BUY, WEAK SELL,
                    CONFLICTING, HOLD
        """
        return _FUSION_TABLE.get((sentiment_signal, technical_signal), "HOLD")

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

        # --- Strategy override ---
        strategy = get_strategy(ticker)
        strat_name = strategy_label(ticker)
        strategy_result = None

        if strategy is not None:
            try:
                bars = AlpacaDataClient().get_bars(ticker, timeframe="1Day", limit=252)
                strategy_result = strategy.analyze(
                    ticker, bars, sentiment["signal"],
                )
                log.info(
                    "[%s] Strategy %s → %s (%.0f%%)",
                    ticker, strat_name,
                    strategy_result.signal, strategy_result.confidence,
                )
                if verbose:
                    print(f"  Strategy [{strat_name}]: "
                          f"{strategy_result.signal} ({strategy_result.confidence:.0f}%)")
            except Exception as exc:
                log.warning(
                    "[%s] Strategy %s failed, using generic TA: %s",
                    ticker, strat_name, exc,
                )

        # --- Fuse ---
        if strategy_result is not None:
            # Map strategy signal → simple BUY/SELL/HOLD for fusion table
            _ss = strategy_result.signal.upper()
            if _ss in ("STRONG BUY", "BUY", "WEAK BUY"):
                tech_for_fusion = "BUY"
            elif _ss in ("SELL", "STRONG SELL", "WEAK SELL"):
                tech_for_fusion = "SELL"
            else:
                tech_for_fusion = "HOLD"
            combined_signal = self.combine_signals(sentiment["signal"], tech_for_fusion)
            conf = strategy_result.confidence / 100.0  # 0-100 → 0.0-1.0
        else:
            combined_signal = self.combine_signals(sentiment["signal"], technical["signal"])
            conf = self.confidence(
                combined_signal,
                sentiment["avg_score"],
                volume_confirmed=technical.get("volume_confirmed", False),
                rvol=technical.get("indicators", {}).get("rvol"),
                technical_confidence=technical.get("adjusted_confidence"),
            )

        # --- Bull/Bear Debate (optional) ---
        debate_result = None
        if self.debate_agent.is_enabled():
            if verbose:
                print(f"\n  [DEBATE] Running bull/bear debate for {ticker}...")
            debate_result = self.debate_agent.run(
                ticker=ticker,
                signal=combined_signal,
                confidence=conf,
                technical_data=technical.get("indicators", {}),
                sentiment_data={"signal": sentiment["signal"], "avg_score": sentiment["avg_score"]},
            )
            combined_signal = debate_result.final_signal
            conf = debate_result.adjusted_confidence
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

        # Downgrade STRONG signals when earnings are imminent
        sizing_signal = combined_signal
        if event_risk_flag == "earnings_imminent":
            if combined_signal == "STRONG BUY":
                sizing_signal = "WEAK BUY"
            elif combined_signal == "STRONG SELL":
                sizing_signal = "WEAK SELL"

        risk = self.risk_agent.run(
            ticker=ticker,
            signal=sizing_signal,
            confidence=conf * 100,          # convert 0–1 → 0–100
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

        return {
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
            "debate": debate_result,
        }

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

        Returns:
            Same dict shape as ``run_combined()``, plus ``elapsed_s``.
        """
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

        # ── Strategy override ────────────────────────────────────────
        strategy = get_strategy(ticker)
        strat_name = strategy_label(ticker)
        strategy_result = None

        if strategy is not None:
            try:
                _alpaca = AlpacaDataClient()
                async with data_semaphore:
                    bars = await asyncio.to_thread(
                        _alpaca.get_bars, ticker, "1Day", 252,
                    )
                strategy_result = await asyncio.to_thread(
                    strategy.analyze, ticker, bars, sentiment_signal,
                )
                log.info(
                    "[%s] Strategy %s → %s (%.0f%%)",
                    ticker, strat_name,
                    strategy_result.signal, strategy_result.confidence,
                )
            except Exception as exc:
                log.warning(
                    "[%s] Strategy %s failed, using generic TA: %s",
                    ticker, strat_name, exc,
                )

        # ── Fuse signals ───────────────────────────────────────────────
        if strategy_result is not None:
            _ss = strategy_result.signal.upper()
            if _ss in ("STRONG BUY", "BUY", "WEAK BUY"):
                tech_for_fusion = "BUY"
            elif _ss in ("SELL", "STRONG SELL", "WEAK SELL"):
                tech_for_fusion = "SELL"
            else:
                tech_for_fusion = "HOLD"
            combined_signal = self.combine_signals(sentiment_signal, tech_for_fusion)
            conf = strategy_result.confidence / 100.0
        else:
            combined_signal = self.combine_signals(
                sentiment_signal, technical["signal"],
            )
            conf = self.confidence(
                combined_signal,
                avg_score,
                volume_confirmed=technical.get("volume_confirmed", False),
                rvol=technical.get("indicators", {}).get("rvol"),
                technical_confidence=technical.get("adjusted_confidence"),
            )

        # ── Bull/Bear Debate (optional) ──────────────────────────────
        debate_result = None
        if self.debate_agent.is_enabled():
            async with api_semaphore:
                debate_result = await self.debate_agent.run_async(
                    ticker=ticker,
                    signal=combined_signal,
                    confidence=conf,
                    technical_data=technical.get("indicators", {}),
                    sentiment_data={"signal": sentiment_signal, "avg_score": avg_score},
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

        sizing_signal = combined_signal
        if event_risk_flag == "earnings_imminent":
            if combined_signal == "STRONG BUY":
                sizing_signal = "WEAK BUY"
            elif combined_signal == "STRONG SELL":
                sizing_signal = "WEAK SELL"

        async with db_lock:
            risk = self.risk_agent.run(
                ticker=ticker,
                signal=sizing_signal,
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

        return {
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
            "debate": debate_result,
            "elapsed_s": round(elapsed, 2),
        }
