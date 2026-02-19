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

Strategy pipeline (run_strategy):

    Delegates to StrategyCoordinator which runs three agents in parallel:
    MomentumAgent, MeanReversionAgent, SwingAgent → ensemble signal → RiskAgent
    DB tables: strategy_signals + strategy_performance

Signal fusion matrix (combined pipeline)
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

import json

from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from config.settings import BUY_THRESHOLD, SELL_THRESHOLD
from data.market_data import MarketData
from data.news_aggregator import NewsAggregator
from data.news_feed import NewsFeed
from storage.database import Database

# Optional — only imported when execute/notify mode is active to avoid circular deps
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from execution.paper_trader import PaperTrader
    from notifications.telegram_bot import TelegramNotifier
    from orchestrator.strategy_coordinator import StrategyCoordinator

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
        news_feed: "NewsFeed | NewsAggregator | None" = None,
        market_data: MarketData | None = None,
        sentiment_agent: SentimentAgent | None = None,
        technical_agent: TechnicalAgent | None = None,
        risk_agent: RiskAgent | None = None,
        db: Database | None = None,
        paper_trader: "PaperTrader | None" = None,
        notifier: "TelegramNotifier | None" = None,
        strategy_coordinator: "StrategyCoordinator | None" = None,
    ) -> None:
        self.db = db or Database()
        self.news_feed = news_feed or NewsAggregator(db=self.db)
        self.market_data = market_data or MarketData()
        self.sentiment_agent = sentiment_agent or SentimentAgent()
        # All agents share the same DB instance
        self.technical_agent = technical_agent or TechnicalAgent(db=self.db)
        self.risk_agent = risk_agent or RiskAgent(db=self.db)
        self.paper_trader = paper_trader
        self.notifier = notifier
        self._strategy_coordinator = strategy_coordinator

    # ------------------------------------------------------------------
    # Strategy pipeline
    # ------------------------------------------------------------------

    def run_strategy(
        self,
        ticker: str,
        account_balance: float = 10_000.0,
        verbose: bool = True,
        strategy: str = "all",
    ) -> dict:
        """
        Run the multi-strategy pipeline (momentum + mean-reversion + swing).

        Delegates entirely to StrategyCoordinator; creates one on demand when
        no coordinator was injected at construction time.

        Args:
            ticker:          Stock ticker symbol (e.g. "AAPL").
            account_balance: Account size in USD for position sizing.
            verbose:         Print progress to stdout.
            strategy:        Which agents to run: "momentum" | "mean-reversion" |
                             "swing" | "all" (default).

        Returns:
            dict — same structure as StrategyCoordinator.run():
                ticker, strategy, strategy_signals, ranked_signals,
                combined_strategy_signal, ensemble_confidence,
                consensus, risk, account_balance, strategy_run_id, errors
        """
        if self._strategy_coordinator is None:
            from orchestrator.strategy_coordinator import StrategyCoordinator
            self._strategy_coordinator = StrategyCoordinator(db=self.db)

        return self._strategy_coordinator.run(
            ticker=ticker,
            account_balance=account_balance,
            verbose=verbose,
            strategy=strategy,
        )

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
    def confidence(combined_signal: str, avg_score: float) -> float:
        """
        Compute a confidence score (0.0–1.0) for the combined signal.

        STRONG signals scale with |avg_score| in the upper half [0.6, 1.0].
        WEAK signals scale with |avg_score| in the lower half [0.0, 0.6].
        CONFLICTING is fixed at 0.10 (agents actively disagree).
        HOLD is fixed at 0.25 (no directional conviction).

        Args:
            combined_signal: Output of combine_signals().
            avg_score:       Raw sentiment average (−1.0 to +1.0).

        Returns:
            Confidence float rounded to two decimal places.
        """
        strength = abs(avg_score)  # 0.0 – 1.0
        if combined_signal in ("STRONG BUY", "STRONG SELL"):
            return round(min(1.0, 0.6 + strength * 0.4), 2)
        if combined_signal in ("WEAK BUY", "WEAK SELL"):
            return round(min(0.6, 0.2 + strength * 0.4), 2)
        if combined_signal == "CONFLICTING":
            return 0.10
        return 0.25  # HOLD

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, ticker: str, verbose: bool = True) -> dict:
        """
        Execute the sentiment-only pipeline for *ticker*.

        Unchanged from the original implementation — fully backward-compatible.

        Args:
            ticker:  Stock ticker symbol (e.g. "AAPL").
            verbose: When True, print step-by-step progress to stdout.

        Returns:
            dict with keys: ticker, market, headlines_fetched, scored,
                            avg_score, signal, run_id
        """
        ticker = ticker.upper()

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

        # Step 4 — aggregate + signal
        avg_score = self._aggregate(scored)
        signal = self._signal(avg_score)

        # Step 5 — persist
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

    def run_combined(
        self,
        ticker: str,
        verbose: bool = True,
        account_balance: float = 10_000.0,
    ) -> dict:
        """
        Run sentiment, technical, and risk agents, then fuse their signals.

        Args:
            ticker:          Stock ticker symbol (e.g. "AAPL").
            verbose:         When True, print step-by-step progress to stdout.
            account_balance: Account size in USD used for position sizing.

        Returns:
            dict with keys:
                ticker           (str):   The analysed ticker.
                sentiment        (dict):  Full output of run() — sentiment pipeline.
                technical        (dict):  Full output of TechnicalAgent.run().
                combined_signal  (str):   Fused signal label.
                confidence       (float): Confidence score 0.0–1.0.
                combined_id      (int):   DB primary key for combined_signals row.
                risk             (dict):  Full output of RiskAgent.run().
        """
        ticker = ticker.upper()

        # --- Sentiment pipeline ---
        if verbose:
            print(f"\n[1/3] Sentiment analysis for {ticker}...")
        sentiment = self.run(ticker, verbose=verbose)

        # --- Technical pipeline ---
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

        # --- Fuse ---
        combined_signal = self.combine_signals(sentiment["signal"], technical["signal"])
        conf = self.confidence(combined_signal, sentiment["avg_score"])

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

        # --- Telegram: signal alert ---
        if self.notifier is not None:
            # Build a short reasoning string from the top headline scores
            top = sorted(
                sentiment.get("scored", []),
                key=lambda s: abs(s.get("score", 0)),
                reverse=True,
            )[:2]
            reasoning = "  |  ".join(
                h.get("headline", "")[:80] for h in top if h.get("headline")
            )
            self.notifier.send_signal(
                ticker=ticker,
                signal=combined_signal,
                confidence=conf * 100,
                reasoning=reasoning,
            )

        # --- Risk sizing ---
        if verbose:
            print(f"\n[3/3] Risk sizing for {ticker}...")
        # Prefer live market price; fall back to technical close price
        price = (sentiment.get("market") or {}).get("price") \
            or technical["indicators"].get("price")
        risk = self.risk_agent.run(
            ticker=ticker,
            signal=combined_signal,
            confidence=conf * 100,          # convert 0–1 → 0–100
            current_price=price,
            account_balance=account_balance,
        )
        if verbose:
            if risk["skipped"]:
                print(f"  No position — {risk['skip_reason']}")
            else:
                print(
                    f"  ${risk['position_size_usd']:,.2f}  "
                    f"({risk['shares']} shares)  "
                    f"SL: ${risk['stop_loss']:.2f}  "
                    f"TP: ${risk['take_profit']:.2f}"
                )

        # --- Paper execution ---
        trade_id = None
        if self.paper_trader is not None and not risk["skipped"]:
            trade_id = self.paper_trader.track_trade(
                ticker=ticker,
                action=risk["direction"],
                shares=risk["shares"],
                price=price,
                stop_loss=risk["stop_loss"],
                take_profit=risk["take_profit"],
            )
            if verbose:
                print(f"\n  [PAPER TRADE LOGGED]  trade_history id=#{trade_id}")

            # --- Telegram: trade executed alert ---
            if self.notifier is not None:
                self.notifier.send_trade_executed(
                    ticker=ticker,
                    action=risk["direction"],
                    shares=risk["shares"],
                    price=price,
                    stop_loss=risk["stop_loss"],
                    take_profit=risk["take_profit"],
                )

        return {
            "ticker": ticker,
            "sentiment": sentiment,
            "technical": technical,
            "combined_signal": combined_signal,
            "confidence": conf,
            "combined_id": combined_id,
            "risk": risk,
            "account_balance": account_balance,
            "trade_id": trade_id,
        }
