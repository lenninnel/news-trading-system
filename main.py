"""
News Trading System — command-line entry point.

Usage
-----
    python main.py <TICKER> [--balance N] [--agent sentiment|technical]

Examples
--------
    python main.py AAPL                          # combined mode, $10k account
    python main.py NVDA --balance 25000          # combined mode, $25k account
    python main.py TSLA --agent sentiment        # sentiment only
    python main.py AAPL --agent technical        # technical only
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
        "--balance",
        type=float,
        default=10_000.0,
        metavar="USD",
        help="Account balance in USD for position sizing (default: 10000).",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    coordinator = Coordinator()

    if args.agent == "sentiment":
        report = coordinator.run(ticker)
        print_sentiment_report(report)

    elif args.agent == "technical":
        from agents.technical_agent import TechnicalAgent
        print(f"\nRunning technical analysis for {ticker}...")
        result = TechnicalAgent().run(ticker)
        print_technical_report(result)

    else:
        report = coordinator.run_combined(ticker, account_balance=args.balance)
        print_combined_report(report)


if __name__ == "__main__":
    main()
