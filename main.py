"""
News Trading System — command-line entry point.

Usage
-----
    python main.py <TICKER>                   # combined mode (default)
    python main.py <TICKER> --agent sentiment # sentiment only
    python main.py <TICKER> --agent technical # technical only

Examples
--------
    python main.py AAPL
    python main.py NVDA --agent technical
    python main.py TSLA --agent sentiment
"""

import argparse

from orchestrator.coordinator import Coordinator

# Signal labels that carry a directional view
_DIRECTIONAL = {"STRONG BUY", "STRONG SELL", "WEAK BUY", "WEAK SELL", "CONFLICTING"}


def _fmt(val: "float | None", decimals: int = 2) -> str:
    return f"{val:.{decimals}f}" if val is not None else "N/A"


# ---------------------------------------------------------------------------
# Report printers
# ---------------------------------------------------------------------------

def print_combined_report(report: dict) -> None:
    """Print the full combined (sentiment + technical + fusion) report."""
    sent  = report["sentiment"]
    tech  = report["technical"]
    ind   = tech["indicators"]
    sig   = report["combined_signal"]
    conf  = report["confidence"]

    scored  = sent["scored"]
    counts: dict[str, int] = {"bullish": 0, "bearish": 0, "neutral": 0}
    for s in scored:
        counts[s["sentiment"]] = counts.get(s["sentiment"], 0) + 1

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
    print(f"  Price     : {_fmt(ind.get('price'))}  |  "
          f"RSI (14): {_fmt(ind.get('rsi'))}")
    print(f"  MACD      : {_fmt(ind.get('macd'))}  |  "
          f"Signal: {_fmt(ind.get('macd_signal'))}")
    print(f"  SMA 20/50 : {_fmt(ind.get('sma_20'))} / {_fmt(ind.get('sma_50'))}")
    print(f"  Bollinger : {_fmt(ind.get('bb_lower'))} — {_fmt(ind.get('bb_upper'))}")
    print(f"  Notes     : {tech['reasoning'][0]}")
    print(f"              →   {tech['signal']}")

    # -- Combined section --
    print(f"\n  [COMBINED SIGNAL]")
    print(f"  {sig}   (confidence: {conf:.0%})")
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
        help="Run a single agent only. Omit to run both (combined mode).",
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
        # Default: combined mode
        report = coordinator.run_combined(ticker)
        print_combined_report(report)


if __name__ == "__main__":
    main()
