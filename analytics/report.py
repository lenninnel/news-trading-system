"""
Signal analytics report — CLI summary of signal performance.

Usage::

    python3 -m analytics.report              # last 30 days
    python3 -m analytics.report --days 7     # last 7 days
    python3 -m analytics.report --ticker AAPL # single ticker
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict

from analytics.signal_logger import SignalLogger

log = logging.getLogger(__name__)


def _pct(n: float) -> str:
    return f"{n:.0f}%"


def _fmt(val, width: int = 8) -> str:
    if val is None:
        return "n/a".center(width)
    if isinstance(val, float):
        return f"{val:.0f}%".rjust(width)
    return str(val).rjust(width)


def generate_report(
    days: int = 30,
    ticker: str | None = None,
) -> str:
    """Generate a text report and return it as a string."""
    logger = SignalLogger()
    signals = logger.get_signals(ticker=ticker, days=days)

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  Signal Analytics Report")
    lines.append("=" * 60)
    lines.append(f"  Period: last {days} days | Total signals: {len(signals)}")
    if ticker:
        lines.append(f"  Ticker filter: {ticker}")
    lines.append("")

    if not signals:
        lines.append("  No signals found for this period.")
        lines.append("")
        return "\n".join(lines)

    # ── Per-ticker table ─────────────────────────────────────────────
    ticker_stats: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "conf_sum": 0.0, "buys": 0, "correct_3d": 0, "has_outcome": 0}
    )
    for s in signals:
        t = s["ticker"]
        ticker_stats[t]["count"] += 1
        ticker_stats[t]["conf_sum"] += (s.get("confidence") or 0)
        sig = (s.get("signal") or "").upper()
        if "BUY" in sig:
            ticker_stats[t]["buys"] += 1
        if s.get("outcome_3d_pct") is not None:
            ticker_stats[t]["has_outcome"] += 1
            if s.get("outcome_correct") == 1:
                ticker_stats[t]["correct_3d"] += 1

    lines.append("  TICKER   | SIGNALS | AVG CONF | BUY RATE | 3D HIT RATE")
    lines.append("  " + "-" * 55)
    for t in sorted(ticker_stats, key=lambda x: ticker_stats[x]["count"], reverse=True):
        st = ticker_stats[t]
        avg_conf = (st["conf_sum"] / st["count"] * 100) if st["count"] else 0
        buy_rate = (st["buys"] / st["count"] * 100) if st["count"] else 0
        if st["has_outcome"] >= 3:
            hit = f"{st['correct_3d'] / st['has_outcome'] * 100:.0f}%"
        else:
            hit = "n/a (need more data)"
        lines.append(
            f"  {t:<9}|{st['count']:>8} |{avg_conf:>7.0f}%  |{buy_rate:>7.0f}%  | {hit}"
        )

    lines.append("")

    # ── Per-strategy table ───────────────────────────────────────────
    strat_stats: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "conf_sum": 0.0, "correct_hi": 0, "has_hi": 0}
    )
    for s in signals:
        strat = s.get("strategy") or "Combined"
        strat_stats[strat]["count"] += 1
        strat_stats[strat]["conf_sum"] += (s.get("confidence") or 0)
        conf = s.get("confidence") or 0
        if conf > 0.35 and s.get("outcome_3d_pct") is not None:
            strat_stats[strat]["has_hi"] += 1
            if s.get("outcome_correct") == 1:
                strat_stats[strat]["correct_hi"] += 1

    lines.append("  STRATEGY      | SIGNALS | AVG CONF | HIT RATE (when >35%)")
    lines.append("  " + "-" * 55)
    for st_name in sorted(strat_stats, key=lambda x: strat_stats[x]["count"], reverse=True):
        st = strat_stats[st_name]
        avg_conf = (st["conf_sum"] / st["count"] * 100) if st["count"] else 0
        if st["has_hi"] >= 3:
            hit = f"{st['correct_hi'] / st['has_hi'] * 100:.0f}%"
        else:
            hit = "n/a"
        lines.append(
            f"  {st_name:<14} |{st['count']:>8} |{avg_conf:>7.0f}%  | {hit}"
        )

    lines.append("")

    # ── Confidence vs outcome correlation ────────────────────────────
    conf_buckets: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "correct": 0, "has_outcome": 0}
    )
    for s in signals:
        conf = s.get("confidence") or 0
        if conf < 0.20:
            bucket = "<20%"
        elif conf < 0.35:
            bucket = "20-35%"
        elif conf < 0.50:
            bucket = "35-50%"
        elif conf < 0.65:
            bucket = "50-65%"
        else:
            bucket = "65%+"
        conf_buckets[bucket]["count"] += 1
        if s.get("outcome_3d_pct") is not None:
            conf_buckets[bucket]["has_outcome"] += 1
            if s.get("outcome_correct") == 1:
                conf_buckets[bucket]["correct"] += 1

    lines.append("  CONFIDENCE BAND | SIGNALS | 3D HIT RATE")
    lines.append("  " + "-" * 40)
    for bucket in ["<20%", "20-35%", "35-50%", "50-65%", "65%+"]:
        if bucket not in conf_buckets:
            continue
        b = conf_buckets[bucket]
        if b["has_outcome"] >= 3:
            hit = f"{b['correct'] / b['has_outcome'] * 100:.0f}%"
        else:
            hit = "n/a"
        lines.append(f"  {bucket:<17} |{b['count']:>8} | {hit}")

    lines.append("")

    # ── Actionable signals (>35% confidence) ─────────────────────────
    actionable = [
        s for s in signals
        if (s.get("confidence") or 0) > 0.35
        and "BUY" in (s.get("signal") or "").upper()
    ]
    actionable_tickers: dict[str, int] = defaultdict(int)
    for s in actionable:
        actionable_tickers[s["ticker"]] += 1

    if actionable_tickers:
        lines.append("  MOST ACTIONABLE TICKERS (>35% conf, BUY signals):")
        for t, cnt in sorted(actionable_tickers.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {t}: {cnt} signal(s)")
    else:
        lines.append("  No actionable signals (>35% confidence BUY) in this period.")

    lines.append("")
    lines.append("=" * 60)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Signal Analytics Report",
    )
    parser.add_argument("--days", type=int, default=30, help="Lookback period in days.")
    parser.add_argument("--ticker", type=str, default=None, help="Filter by ticker.")
    args = parser.parse_args()

    print(generate_report(days=args.days, ticker=args.ticker))


if __name__ == "__main__":
    main()
