"""
Strategy comparison report generator.

Produces a ranked comparison report printed to terminal AND saved as
a Markdown file in backtest/results/.

Report sections:
  1. Overall Winner
  2. Strategy Rankings
  3. Ticker Rankings
  4. Sector Analysis
  5. Crypto vs Stocks
  6. Top 10 Combinations
  7. Bottom 10 Combinations
  8. Recommendation
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "backtest" / "results"


def generate_report(data: dict[str, Any]) -> Path:
    """Generate the full comparison report.  Prints to stdout and saves .md."""
    results = [r for r in data["results"] if r["error"] is None]
    meta = data["meta"]

    if not results:
        print("\nNo successful backtests — cannot generate report.\n")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"strategy_comparison_{date.today().isoformat()}.md"
        path.write_text("# Strategy Comparison\n\nNo successful backtests.\n")
        return path

    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    out(f"# Strategy Comparison Report")
    out(f"**Period:** {meta['start_date']} to {meta['end_date']}  ")
    out(f"**Tickers:** {meta.get('succeeded', len(results))} succeeded, "
        f"{meta.get('failed', 0)} failed  ")
    out(f"**Balance:** ${meta['balance']:,.0f}  ")
    out()

    # ── 1. Overall Winner ────────────────────────────────────────────
    strat_avg = _avg_by_field(results, "strategy", "sharpe")
    best_strat = max(strat_avg, key=strat_avg.get)
    out(f"## 1. Overall Winner")
    out(f"**{best_strat}** — avg Sharpe {strat_avg[best_strat]:+.2f} across all tickers")
    out()

    # ── 2. Strategy Rankings ─────────────────────────────────────────
    out(f"## 2. Strategy Rankings")
    out()
    out(f"| Strategy | Avg Sharpe | Avg Return | Avg Drawdown | Win Rate | Best Ticker | Worst Ticker |")
    out(f"|----------|-----------|------------|--------------|----------|-------------|--------------|")

    for sname in sorted(strat_avg, key=strat_avg.get, reverse=True):
        s_results = [r for r in results if r["strategy"] == sname]
        avg_ret = _mean([r["total_return_pct"] for r in s_results])
        avg_dd = _mean([r["max_drawdown_pct"] for r in s_results])
        avg_wr = _mean([r["win_rate"] for r in s_results])
        best_t = max(s_results, key=lambda r: r["sharpe"])["ticker"]
        worst_t = min(s_results, key=lambda r: r["sharpe"])["ticker"]
        out(f"| {sname:20s} | {strat_avg[sname]:+9.2f} | {avg_ret:+9.1f}% | "
            f"{avg_dd:11.1f}% | {avg_wr:6.0f}% | {best_t:11s} | {worst_t:12s} |")
    out()

    # ── 3. Ticker Rankings ───────────────────────────────────────────
    out(f"## 3. Ticker Rankings")
    out()
    out(f"| Ticker | Sector | Best Strategy | Best Sharpe | Best Return |")
    out(f"|--------|--------|---------------|-------------|-------------|")

    ticker_best: dict[str, dict] = {}
    for r in results:
        t = r["ticker"]
        if t not in ticker_best or r["sharpe"] > ticker_best[t]["sharpe"]:
            ticker_best[t] = r
    for t in sorted(ticker_best, key=lambda x: ticker_best[x]["sharpe"], reverse=True):
        b = ticker_best[t]
        out(f"| {t:12s} | {b['sector']:10s} | {b['strategy']:18s} | "
            f"{b['sharpe']:+10.2f} | {b['total_return_pct']:+10.1f}% |")
    out()

    # ── 4. Sector Analysis ───────────────────────────────────────────
    out(f"## 4. Sector Analysis")
    out()
    sectors = defaultdict(list)
    for r in results:
        sectors[r["sector"]].append(r)

    for sector in sorted(sectors):
        sector_results = sectors[sector]
        sector_strat_avg = _avg_by_field(sector_results, "strategy", "sharpe")
        if not sector_strat_avg:
            continue
        best_s = max(sector_strat_avg, key=sector_strat_avg.get)
        out(f"- **{sector}**: Best strategy = **{best_s}** "
            f"(avg Sharpe {sector_strat_avg[best_s]:+.2f})")
    out()

    # ── 5. Crypto vs Stocks ──────────────────────────────────────────
    out(f"## 5. Crypto vs Stocks")
    out()
    crypto = [r for r in results if r["sector"] == "CRYPTO"]
    stocks = [r for r in results if r["sector"] != "CRYPTO"]

    if crypto:
        out(f"- **Crypto** ({len(crypto)} backtests): "
            f"avg Sharpe {_mean([r['sharpe'] for r in crypto]):+.2f}, "
            f"avg return {_mean([r['total_return_pct'] for r in crypto]):+.1f}%")
    if stocks:
        out(f"- **Stocks** ({len(stocks)} backtests): "
            f"avg Sharpe {_mean([r['sharpe'] for r in stocks]):+.2f}, "
            f"avg return {_mean([r['total_return_pct'] for r in stocks]):+.1f}%")
    out()

    # ── 6. Top 10 Combinations ───────────────────────────────────────
    out(f"## 6. Top 10 Combinations (by Sharpe)")
    out()
    out(f"| # | Ticker | Strategy | Sharpe | Return | Drawdown | Trades |")
    out(f"|---|--------|----------|--------|--------|----------|--------|")
    top10 = sorted(results, key=lambda r: r["sharpe"], reverse=True)[:10]
    for i, r in enumerate(top10, 1):
        out(f"| {i:2d} | {r['ticker']:12s} | {r['strategy']:18s} | "
            f"{r['sharpe']:+.2f} | {r['total_return_pct']:+.1f}% | "
            f"{r['max_drawdown_pct']:.1f}% | {r['total_trades']:5d} |")
    out()

    # ── 7. Bottom 10 Combinations ────────────────────────────────────
    out(f"## 7. Bottom 10 Combinations (what to avoid)")
    out()
    out(f"| # | Ticker | Strategy | Sharpe | Return | Drawdown | Trades |")
    out(f"|---|--------|----------|--------|--------|----------|--------|")
    bottom10 = sorted(results, key=lambda r: r["sharpe"])[:10]
    for i, r in enumerate(bottom10, 1):
        out(f"| {i:2d} | {r['ticker']:12s} | {r['strategy']:18s} | "
            f"{r['sharpe']:+.2f} | {r['total_return_pct']:+.1f}% | "
            f"{r['max_drawdown_pct']:.1f}% | {r['total_trades']:5d} |")
    out()

    # ── 8. Recommendation ────────────────────────────────────────────
    out(f"## 8. Recommendation")
    out()
    for sector in sorted(sectors):
        sector_results = sectors[sector]
        sector_strat_avg = _avg_by_field(sector_results, "strategy", "sharpe")
        if not sector_strat_avg:
            continue
        best_s = max(sector_strat_avg, key=sector_strat_avg.get)
        tickers_str = ", ".join(sorted({r["ticker"] for r in sector_results}))
        out(f"- Use **{best_s}** for {sector} ({tickers_str})")
    out()

    # ── Save ─────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"strategy_comparison_{date.today().isoformat()}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ── Helpers ───────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _avg_by_field(
    results: list[dict], group_key: str, value_key: str
) -> dict[str, float]:
    """Compute mean of *value_key* grouped by *group_key*."""
    groups: dict[str, list[float]] = defaultdict(list)
    for r in results:
        groups[r[group_key]].append(r[value_key])
    return {k: _mean(v) for k, v in groups.items()}
