"""
News Trading System — command-line entry point.

Usage
-----
    python main.py <TICKER> [--balance N] [--agent sentiment|technical]
                            [--strategy momentum|mean-reversion|swing|all]
                            [--execute]

Examples
--------
    python main.py AAPL                                    # combined mode, $10k account
    python main.py NVDA --balance 25000                    # combined mode, $25k account
    python main.py TSLA --agent sentiment                  # sentiment only
    python main.py AAPL --agent technical                  # technical only
    python main.py AAPL --balance 10000 --execute          # combined + log to DB
    python main.py AAPL --strategy all --balance 10000     # all 3 strategy agents
    python main.py AAPL --strategy momentum                # momentum agent only
    python main.py AAPL --strategy all --execute           # strategy + paper trade
"""

import argparse

from orchestrator.coordinator import Coordinator


def _fmt(val: "float | None", decimals: int = 2) -> str:
    return f"{val:.{decimals}f}" if val is not None else "N/A"


# ---------------------------------------------------------------------------
# Report printers
# ---------------------------------------------------------------------------

def print_combined_report(report: dict) -> None:
    """Print the full combined (sentiment + technical + risk) report."""
    sent = report["sentiment"]
    tech = report["technical"]
    ind  = tech["indicators"]
    risk = report["risk"]
    sig  = report["combined_signal"]
    conf = report["confidence"]

    scored = sent["scored"]
    counts: dict[str, int] = {"bullish": 0, "bearish": 0, "neutral": 0}
    for s in scored:
        counts[s["sentiment"]] = counts.get(s["sentiment"], 0) + 1

    balance = report.get("account_balance") or 0.0
    bar = "=" * 62

    print(f"\n{bar}")
    print(f"  {report['ticker']}  —  Combined Analysis  (ID #{report['combined_id']})")
    print(bar)

    # -- Sentiment section --
    print("\n  [SENTIMENT]")
    print(f"  Headlines : {len(scored)} analysed  "
          f"(+{counts['bullish']} bullish  "
          f"-{counts['bearish']} bearish  "
          f"~{counts['neutral']} neutral)")
    print(f"  Avg Score : {sent['avg_score']:+.2f}   →   {sent['signal']}")

    # -- Technical section --
    print("\n  [TECHNICAL]")
    print(f"  Price     : {_fmt(ind.get('price'))}  |  RSI (14): {_fmt(ind.get('rsi'))}")
    print(f"  MACD      : {_fmt(ind.get('macd'))}  |  Signal: {_fmt(ind.get('macd_signal'))}")
    print(f"  SMA 20/50 : {_fmt(ind.get('sma_20'))} / {_fmt(ind.get('sma_50'))}")
    print(f"  Bollinger : {_fmt(ind.get('bb_lower'))} — {_fmt(ind.get('bb_upper'))}")
    print(f"  Notes     : {tech['reasoning'][0]}")
    print(f"              →   {tech['signal']}")

    # -- Combined signal --
    print(f"\n  [COMBINED SIGNAL]")
    print(f"  {sig}   (confidence: {conf:.0%})")

    # -- Risk section --
    print(f"\n  [RISK MANAGEMENT]  (account: ${balance:,.2f})")
    if risk["skipped"]:
        print(f"  No position — {risk['skip_reason']}")
    else:
        pct_of_acct = (risk["risk_amount"] / balance * 100) if balance else 0
        sl_pct  = risk["stop_pct"] * 100 if risk["stop_pct"] else 0
        tp_pct  = sl_pct * 2
        direction_arrow = "▲" if risk["direction"] == "BUY" else "▼"
        print(f"  Direction     : {direction_arrow} {risk['direction']}")
        print(f"  Position Size : ${risk['position_size_usd']:,.2f}  "
              f"({risk['shares']} shares @ ${_fmt(ind.get('price'))})")
        print(f"  Stop Loss     : ${risk['stop_loss']:.2f}  (-{sl_pct:.2f}%)")
        print(f"  Take Profit   : ${risk['take_profit']:.2f}  (+{tp_pct:.2f}%)")
        print(f"  Risk Amount   : ${risk['risk_amount']:.2f}  "
              f"({pct_of_acct:.2f}% of portfolio)")
        print(f"  Kelly Frac.   : {risk['kelly_fraction']:.2%}  "
              f"(capped at {min(risk['kelly_fraction'], 0.10):.2%})")

    print(f"\n{bar}\n")


def print_sentiment_report(report: dict) -> None:
    """Print a formatted summary of a sentiment-only run."""
    scored = report["scored"]
    counts: dict[str, int] = {"bullish": 0, "bearish": 0, "neutral": 0}
    for s in scored:
        counts[s["sentiment"]] = counts.get(s["sentiment"], 0) + 1

    print("\n" + "=" * 60)
    print(f"  Ticker:       {report['ticker']}")
    print(f"  Run ID:       {report['run_id']}")
    print(f"  Headlines:    {len(scored)} analysed")
    print(f"  Bullish:      {counts['bullish']}")
    print(f"  Bearish:      {counts['bearish']}")
    print(f"  Neutral:      {counts['neutral']}")
    print(f"  Avg Score:    {report['avg_score']:+.2f}  (range -1.00 to +1.00)")
    print(f"  Signal:       {report['signal']}")
    print("=" * 60)
    print()


def print_technical_report(result: dict) -> None:
    """Print a formatted summary of a technical-only run."""
    ind = result["indicators"]

    print("\n" + "=" * 60)
    print(f"  Ticker:       {result['ticker']}")
    print(f"  Signal ID:    {result['signal_id']}")
    print(f"  Signal:       {result['signal']}")
    print()
    print(f"  Price:        {_fmt(ind.get('price'))}")
    print(f"  RSI (14):     {_fmt(ind.get('rsi'))}")
    print(f"  MACD:         {_fmt(ind.get('macd'))}  |  Signal: {_fmt(ind.get('macd_signal'))}")
    print(f"  SMA 20:       {_fmt(ind.get('sma_20'))}")
    print(f"  SMA 50:       {_fmt(ind.get('sma_50'))}")
    print(f"  BB Upper:     {_fmt(ind.get('bb_upper'))}")
    print(f"  BB Lower:     {_fmt(ind.get('bb_lower'))}")
    print()
    print("  Reasoning:")
    for reason in result["reasoning"]:
        print(f"    * {reason}")
    print("=" * 60)
    print()


def print_strategy_report(report: dict) -> None:
    """Print the full multi-strategy coordinator report."""
    ticker   = report["ticker"]
    signals  = report["strategy_signals"]
    ranked   = report["ranked_signals"]
    combined = report["combined_strategy_signal"]
    ensemble = report["ensemble_confidence"]
    consensus = report["consensus"]
    risk     = report["risk"]
    errors   = report.get("errors", [])
    run_id   = report.get("strategy_run_id")

    bar = "=" * 66
    print(f"\n{bar}")
    print(f"  {ticker}  —  Multi-Strategy Analysis  (run #{run_id})")
    print(bar)

    # -- Individual agent signals --
    print("\n  [STRATEGY AGENTS]")
    icon_map = {"BUY": "+", "SELL": "-", "HOLD": "~"}
    for sig in signals:
        icon = icon_map.get(sig["signal"], "?")
        print(
            f"  [{sig['strategy'].upper():17s}] [{icon}] {sig['signal']:<4}  "
            f"conf {sig['confidence']:.0f}%  ({sig['timeframe']})"
        )
        for reason in sig.get("reasoning", []):
            print(f"    → {reason}")
    if errors:
        for err in errors:
            print(f"  [!] {err}")

    # -- Ensemble result --
    print(f"\n  [ENSEMBLE SIGNAL]")
    print(f"  Combined : {combined}  (confidence: {ensemble:.1f}%,  consensus: {consensus})")
    if ranked:
        print(f"  Ranked   : {', '.join(s['strategy'] for s in ranked)}")

    # -- Risk --
    balance = report.get("account_balance") or risk.get("account_balance") or 0.0
    print(f"\n  [RISK MANAGEMENT]  (account: ${balance:,.2f})")
    if risk["skipped"]:
        print(f"  No position — {risk['skip_reason']}")
    else:
        sl_pct = (risk["stop_pct"] or 0.0) * 100
        tp_pct = sl_pct * 2
        direction_arrow = "▲" if risk["direction"] == "BUY" else "▼"
        print(f"  Direction     : {direction_arrow} {risk['direction']}")
        price = risk.get("current_price") or 0.0
        print(
            f"  Position Size : ${risk['position_size_usd']:,.2f}  "
            f"({risk['shares']} shares)"
        )
        if price:
            print(f"  Entry Price   : ${price:.2f}")
        print(f"  Stop Loss     : ${risk['stop_loss']:.2f}  (-{sl_pct:.2f}%)")
        print(f"  Take Profit   : ${risk['take_profit']:.2f}  (+{tp_pct:.2f}%)")
        print(f"  Risk Amount   : ${risk['risk_amount']:.2f}")
        print(f"  Kelly Frac.   : {risk['kelly_fraction']:.2%}")

    print(f"\n{bar}\n")


# ---------------------------------------------------------------------------
# Strategy-mode dispatcher
# ---------------------------------------------------------------------------

def _run_strategy_mode(ticker: str, args) -> None:
    """
    Execute the multi-strategy pipeline for a single ticker.

    When --strategy is a specific agent name only that agent runs standalone.
    When --strategy all (or omitted default) all three agents run via
    StrategyCoordinator and the ensemble result is displayed.
    """
    strategy = args.strategy  # "momentum" | "mean-reversion" | "swing" | "all"
    balance  = args.balance

    if strategy == "all":
        from orchestrator.strategy_coordinator import StrategyCoordinator
        from storage.database import Database
        db = Database()

        paper_trader = None
        if args.execute:
            from execution.paper_trader import PaperTrader
            paper_trader = PaperTrader()

        print(f"\nRunning all strategy agents for {ticker}  (balance: ${balance:,.2f})...")
        coordinator = StrategyCoordinator(db=db)
        report = coordinator.run(ticker=ticker, account_balance=balance, verbose=True)
        print_strategy_report(report)

        if args.execute and paper_trader is not None:
            risk = report["risk"]
            if not risk["skipped"]:
                price = risk.get("current_price") or 0.0
                trade_id = paper_trader.track_trade(
                    ticker=ticker,
                    action=risk["direction"],
                    shares=risk["shares"],
                    price=price,
                    stop_loss=risk["stop_loss"],
                    take_profit=risk["take_profit"],
                )
                print(f"  Paper trade logged  (trade_history id=#{trade_id})")
                positions = paper_trader.get_portfolio()
                if positions:
                    print("\n  Current portfolio:")
                    for pos in positions:
                        print(
                            f"    {pos['ticker']:6s}  "
                            f"{pos['shares']} shares @ "
                            f"${pos['avg_price']:.2f}  "
                            f"(value: ${pos['current_value']:,.2f})"
                        )
            else:
                print("  No trade logged — risk agent skipped position.")

    else:
        # Single-agent mode
        from storage.database import Database
        db = Database()
        agent_map = {
            "momentum":      ("agents.momentum_agent",       "MomentumAgent"),
            "mean-reversion": ("agents.mean_reversion_agent", "MeanReversionAgent"),
            "swing":         ("agents.swing_agent",          "SwingAgent"),
        }
        module_name, class_name = agent_map[strategy]
        import importlib
        mod   = importlib.import_module(module_name)
        agent = getattr(mod, class_name)(db=db)

        print(f"\nRunning {class_name} for {ticker}...")
        sig = agent.run(ticker)

        bar = "=" * 60
        icon_map = {"BUY": "+", "SELL": "-", "HOLD": "~"}
        icon = icon_map.get(sig.signal, "?")
        print(f"\n{bar}")
        print(f"  {ticker}  —  {class_name}")
        print(bar)
        print(f"  Signal     : [{icon}] {sig.signal}  (confidence: {sig.confidence:.0f}%)")
        print(f"  Timeframe  : {sig.timeframe}")
        print(f"  Indicators :")
        for k, v in sig.indicators.items():
            if v is not None:
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"    {k:<22s}: {val}")
        print(f"  Reasoning  :")
        for reason in sig.reasoning:
            print(f"    → {reason}")
        print(f"{bar}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="News Trading System — analyse a stock ticker"
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument(
        "--agent",
        choices=["sentiment", "technical"],
        default=None,
        help="Run a single agent only. Omit to run both agents (combined mode).",
    )
    parser.add_argument(
        "--strategy",
        choices=["momentum", "mean-reversion", "swing", "all"],
        default=None,
        metavar="STRATEGY",
        help=(
            "Run the multi-strategy technical pipeline instead of the sentiment pipeline. "
            "Choices: momentum | mean-reversion | swing | all (default: all three)."
        ),
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10_000.0,
        metavar="USD",
        help="Account balance in USD for position sizing (default: 10000).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Log the trade to the database (paper execution). "
             "Only applies to combined mode. Without this flag the "
             "recommendation is shown but nothing is recorded.",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send Telegram notifications for signals and trades. "
             "Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars "
             "and telegram.enabled=true in config/watchlist.yaml.",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()

    if args.strategy is not None:
        _run_strategy_mode(ticker, args)
        return

    if args.agent == "sentiment":
        report = Coordinator().run(ticker)
        print_sentiment_report(report)

    elif args.agent == "technical":
        from agents.technical_agent import TechnicalAgent
        print(f"\nRunning technical analysis for {ticker}...")
        result = TechnicalAgent().run(ticker)
        print_technical_report(result)

    else:
        paper_trader = None
        if args.execute:
            from execution.paper_trader import PaperTrader
            paper_trader = PaperTrader()

        notifier = None
        if args.notify:
            import yaml
            from pathlib import Path
            from notifications.telegram_bot import TelegramNotifier
            _cfg_path = Path(__file__).parent / "config" / "watchlist.yaml"
            with open(_cfg_path, encoding="utf-8") as _fh:
                _cfg = yaml.safe_load(_fh)
            _cfg.setdefault("telegram", {})["enabled"] = True
            notifier = TelegramNotifier.from_config(_cfg)
            if notifier is None:
                print("  [warn] --notify: Telegram credentials missing — notifications disabled.")

        coordinator = Coordinator(paper_trader=paper_trader, notifier=notifier)
        report = coordinator.run_combined(ticker, account_balance=args.balance)
        print_combined_report(report)

        if args.execute:
            trade_id = report.get("trade_id")
            if trade_id is not None:
                print(f"  Paper trade logged  (trade_history id=#{trade_id})")
                positions = paper_trader.get_portfolio()
                if positions:
                    print("\n  Current portfolio:")
                    for pos in positions:
                        print(
                            f"    {pos['ticker']:6s}  "
                            f"{pos['shares']} shares @ "
                            f"${pos['avg_price']:.2f}  "
                            f"(value: ${pos['current_value']:,.2f})"
                        )
            else:
                print("  No trade logged (signal skipped or no position).")


if __name__ == "__main__":
    main()
