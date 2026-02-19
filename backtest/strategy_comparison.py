"""
Strategy Comparison CLI — compare Momentum, Mean Reversion, and Swing strategies
on the same ticker and date range.

Usage
-----
  # Compare all 3 strategies on AAPL for 2024:
  python3 backtest/strategy_comparison.py --ticker AAPL --start 2024-01-01 --end 2025-01-01

  # Auto-test on AAPL, TSLA, NVDA and print winner summary:
  python3 backtest/strategy_comparison.py

  # Save chart as HTML:
  python3 backtest/strategy_comparison.py --ticker NVDA --start 2024-01-01 --end 2025-01-01 \\
      --save-html /tmp/nvda_comparison.html

Output format
-------------
  ┌──────────────┬──────────┬────────┬─────────┬──────────┬────────┬──────────┐
  │ Strategy     │ Return   │ Sharpe │ Max DD  │ Win Rate │ Trades │ Avg Hold │
  ├──────────────┼──────────┼────────┼─────────┼──────────┼────────┼──────────┤
  │ Momentum     │  +12.3%  │   2.10 │  -4.2%  │    58%   │    42  │   8.3d   │
  │ Mean Rev.    │   +8.7%  │   1.80 │  -3.1%  │    62%   │    67  │   5.1d   │
  │ Swing        │  +15.2%  │   2.40 │  -5.8%  │    54%   │    28  │  12.7d   │
  │ Buy & Hold   │  +35.6%  │   1.60 │ -12.3%  │   100%   │     1  │ 253.0d   │
  └──────────────┴──────────┴────────┴─────────┴──────────┴────────┴──────────┘
  Winner: Swing (best risk-adjusted return, Sharpe 2.40)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import BacktestEngine  # noqa: E402


# ── Formatting helpers ────────────────────────────────────────────────────────

_STRATEGY_LABELS = {
    "momentum":       "Momentum",
    "mean_reversion": "Mean Rev.",
    "swing":          "Swing",
    "buy_and_hold":   "Buy & Hold",
}

_COL_W = {
    "Strategy":     14,
    "Return (%)":   10,
    "Sharpe":        8,
    "Max DD (%)":    9,
    "Win Rate (%)":  10,
    "Trades":        8,
    "Avg Hold (d)": 10,
}


def _hline(char: str = "─") -> str:
    return (
        "├"
        + "┼".join(char * (w + 2) for w in _COL_W.values())
        + "┤"
    )


def _top() -> str:
    return (
        "┌"
        + "┬".join("─" * (w + 2) for w in _COL_W.values())
        + "┐"
    )


def _bottom() -> str:
    return (
        "└"
        + "┴".join("─" * (w + 2) for w in _COL_W.values())
        + "┘"
    )


def _row(cells: list, is_header: bool = False) -> str:
    parts = []
    for cell, width in zip(cells, _COL_W.values()):
        parts.append(f" {str(cell):{'^' if is_header else '>'}{ width}} ")
    return "│" + "│".join(parts) + "│"


def _print_table(ticker: str, comparison: dict) -> None:
    df      = comparison["comparison_df"]
    winner  = comparison["winner"]
    results = comparison["strategies"]
    bh      = comparison["buy_and_hold"]

    winner_label  = _STRATEGY_LABELS.get(winner, winner)
    winner_sharpe = results[winner].sharpe_ratio if winner in results else bh["sharpe"]

    print(f"\n{'─' * 66}")
    print(f"  {ticker}  Strategy Comparison  "
          f"{comparison['start_date']} → {comparison['end_date']}")
    print(f"{'─' * 66}")

    headers = list(_COL_W.keys())
    print(_top())
    print(_row(headers, is_header=True))
    print(_hline("─"))

    for _, row in df.iterrows():
        strat = row["Strategy"]
        cells = [
            strat,
            f"{row['Return (%)']:+.1f}%",
            f"{row['Sharpe']:.2f}",
            f"{row['Max DD (%)']:.1f}%",
            f"{row['Win Rate (%)']:.0f}%",
            str(int(row["Trades"])),
            f"{row['Avg Hold (d)']:.1f}d",
        ]
        print(_row(cells))

    print(_bottom())
    print(f"\n  Winner: {winner_label} (best risk-adjusted return, Sharpe {winner_sharpe:.2f})\n")


def _plot_comparison(
    engine:    BacktestEngine,
    comparison: dict,
    save_path:  str | None = None,
    show:       bool       = True,
) -> None:
    """Render and optionally save the multi-line equity chart."""
    fig = engine.plot_comparison(comparison, show=show, save_path=save_path)
    return fig


def _auto_test(
    tickers:   list,
    start:     str,
    end:       str,
    balance:   float,
    no_plot:   bool       = False,
    save_html: str | None = None,
) -> None:
    """
    Run strategy comparison on each ticker and print a winner summary.

    Prints e.g. "Momentum wins on 2/3 tickers".
    """
    win_counts: dict = {}
    total = len(tickers)

    for ticker in tickers:
        print(f"\n{'═' * 66}")
        print(f"  Auto-test: {ticker}  ({start} → {end})")
        print(f"{'═' * 66}")
        try:
            engine = BacktestEngine(
                ticker          = ticker,
                start_date      = start,
                end_date        = end,
                initial_balance = balance,
                verbose         = True,
            )
            comparison = engine.compare_strategies()
            _print_table(ticker, comparison)

            winner = comparison["winner"]
            win_counts[winner] = win_counts.get(winner, 0) + 1

            if not no_plot:
                path = save_html or str(
                    PROJECT_ROOT / "backtest"
                    / f"{ticker}_strategy_comparison_{start}_{end}.html"
                )
                _plot_comparison(engine, comparison, save_path=path, show=False)
                print(f"  Chart saved → {path}")

        except Exception as exc:
            print(f"  [!] {ticker} failed: {exc}")

    # ── Winner summary ────────────────────────────────────────────────────────
    print(f"\n{'═' * 66}")
    print(f"  AUTO-TEST SUMMARY  ({start} → {end},  {total} tickers)")
    print(f"{'═' * 66}")
    for strat, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        label = _STRATEGY_LABELS.get(strat, strat)
        print(f"  {label:16s}  wins on  {count}/{total} tickers")
    if win_counts:
        top = max(win_counts, key=win_counts.get)
        print(
            f"\n  Overall winner: {_STRATEGY_LABELS.get(top, top)} "
            f"(best Sharpe on {win_counts[top]}/{total} tickers)"
        )
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Momentum, Mean Reversion, and Swing strategies "
            "on the same ticker and date range."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Single ticker to compare (e.g. AAPL). "
             "Omit for auto-test on AAPL, TSLA, NVDA.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Multiple tickers (overrides --ticker and auto-test list).",
    )
    parser.add_argument(
        "--start",
        default="2024-01-01",
        help="Start date YYYY-MM-DD (default: 2024-01-01).",
    )
    parser.add_argument(
        "--end",
        default="2025-01-01",
        help="End date YYYY-MM-DD (default: 2025-01-01).",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10_000.0,
        metavar="USD",
        help="Initial balance in USD (default: 10000).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip chart display.",
    )
    parser.add_argument(
        "--save-html",
        metavar="PATH",
        help="Save comparison chart as HTML. "
             "Auto-named per ticker when omitted.",
    )
    args = parser.parse_args()

    # ── Resolve ticker list ───────────────────────────────────────────────────
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = []  # auto-test mode

    # ── Auto-test mode (no ticker specified) ──────────────────────────────────
    if not tickers:
        _auto_test(
            tickers   = ["AAPL", "TSLA", "NVDA"],
            start     = args.start,
            end       = args.end,
            balance   = args.balance,
            no_plot   = args.no_plot,
            save_html = args.save_html,
        )
        return

    # ── Single / multi ticker mode ────────────────────────────────────────────
    for ticker in tickers:
        engine = BacktestEngine(
            ticker          = ticker,
            start_date      = args.start,
            end_date        = args.end,
            initial_balance = args.balance,
            verbose         = True,
        )
        comparison = engine.compare_strategies()
        _print_table(ticker, comparison)

        if not args.no_plot:
            save_path = args.save_html or str(
                PROJECT_ROOT / "backtest"
                / f"{ticker}_strategy_comparison_{args.start}_{args.end}.html"
            )
            _plot_comparison(engine, comparison, save_path=save_path, show=True)


if __name__ == "__main__":
    main()
