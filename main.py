"""
News Trading System — command-line entry point.

Usage
-----
    python main.py <TICKER>

Example
-------
    python main.py AAPL

The script delegates all logic to the Coordinator, which handles headline
fetching, sentiment scoring, database persistence, and signal generation.
"""

import sys

from orchestrator.coordinator import Coordinator


def print_report(report: dict) -> None:
    """Print a formatted summary of a completed analysis run."""
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


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <TICKER>")
        print("Example: python main.py AAPL")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    coordinator = Coordinator()
    report = coordinator.run(ticker)
    print_report(report)


if __name__ == "__main__":
    main()
