"""
Signal analytics report — comprehensive CLI summary.

Usage::

    python3 -m analytics.report              # last 30 days
    python3 -m analytics.report --days 7     # last 7 days
    python3 -m analytics.report --ticker AAPL # single ticker
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict

from analytics.signal_logger import SignalLogger

log = logging.getLogger(__name__)

W = 62  # report width


def _header(title: str) -> str:
    return f"\n  {'=' * (W - 4)}\n  {title}\n  {'=' * (W - 4)}"


def _subheader(title: str) -> str:
    return f"\n  --- {title} ---"


def _bar(n: int, total: int, width: int = 20) -> str:
    if total == 0:
        return ""
    filled = round(n / total * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def generate_report(
    days: int = 30,
    ticker: str | None = None,
) -> str:
    """Generate the full analytics report and return it as a string."""
    logger = SignalLogger()
    signals = logger.get_signals(ticker=ticker, days=days)

    lines: list[str] = []

    # ==================================================================
    # 1. OVERVIEW
    # ==================================================================
    lines.append(_header("SIGNAL QUALITY REPORT"))

    if not signals:
        lines.append(f"\n  Period: last {days} days | No signals found.")
        if ticker:
            lines.append(f"  Ticker filter: {ticker}")
        return "\n".join(lines)

    timestamps = [s["timestamp"] for s in signals if s.get("timestamp")]
    date_min = min(timestamps)[:10] if timestamps else "?"
    date_max = max(timestamps)[:10] if timestamps else "?"
    tickers = sorted({s["ticker"] for s in signals})
    sessions = sorted({s.get("session") or "N/A" for s in signals})

    lines.append(f"\n  Total signals:  {len(signals)}")
    lines.append(f"  Date range:     {date_min} to {date_max}")
    lines.append(f"  Sessions:       {', '.join(sessions)}")
    lines.append(f"  Tickers ({len(tickers)}):   {', '.join(tickers)}")
    if ticker:
        lines.append(f"  Filter:         {ticker}")

    # ==================================================================
    # 2. SIGNAL DISTRIBUTION
    # ==================================================================
    lines.append(_header("SIGNAL DISTRIBUTION"))

    # By signal type
    lines.append(_subheader("By Signal Type"))
    sig_counts = Counter(s.get("signal", "?") for s in signals)
    for sig, cnt in sig_counts.most_common():
        pct = cnt / len(signals) * 100
        lines.append(f"  {sig:<14} {cnt:>4}  ({pct:4.1f}%)  {_bar(cnt, len(signals))}")

    # By strategy
    lines.append(_subheader("By Strategy"))
    strat_counts = Counter(s.get("strategy") or "Combined" for s in signals)
    for strat, cnt in strat_counts.most_common():
        pct = cnt / len(signals) * 100
        lines.append(f"  {strat:<14} {cnt:>4}  ({pct:4.1f}%)  {_bar(cnt, len(signals))}")

    # By session
    lines.append(_subheader("By Session"))
    sess_counts = Counter(s.get("session") or "N/A" for s in signals)
    for sess, cnt in sess_counts.most_common():
        pct = cnt / len(signals) * 100
        lines.append(f"  {sess:<14} {cnt:>4}  ({pct:4.1f}%)  {_bar(cnt, len(signals))}")

    # ==================================================================
    # 3. CONFIDENCE ANALYSIS
    # ==================================================================
    lines.append(_header("CONFIDENCE ANALYSIS"))

    # Avg confidence per signal type
    lines.append(_subheader("Avg Confidence by Signal Type"))
    sig_conf: dict[str, list[float]] = defaultdict(list)
    for s in signals:
        c = s.get("confidence")
        if c is not None:
            sig_conf[s.get("signal", "?")].append(c)
    lines.append(f"  {'SIGNAL':<14} {'AVG':>6}  {'MIN':>6}  {'MAX':>6}  {'COUNT':>5}")
    lines.append("  " + "-" * 46)
    for sig in sorted(sig_conf, key=lambda k: -(sum(sig_conf[k]) / len(sig_conf[k]))):
        vals = sig_conf[sig]
        avg = sum(vals) / len(vals) * 100
        mn = min(vals) * 100
        mx = max(vals) * 100
        lines.append(f"  {sig:<14} {avg:5.1f}%  {mn:5.1f}%  {mx:5.1f}%  {len(vals):>5}")

    # Avg confidence per strategy
    lines.append(_subheader("Avg Confidence by Strategy"))
    strat_conf: dict[str, list[float]] = defaultdict(list)
    for s in signals:
        c = s.get("confidence")
        if c is not None:
            strat_conf[s.get("strategy") or "Combined"].append(c)
    lines.append(f"  {'STRATEGY':<14} {'AVG':>6}  {'MIN':>6}  {'MAX':>6}  {'COUNT':>5}")
    lines.append("  " + "-" * 46)
    for strat in sorted(strat_conf, key=lambda k: -(sum(strat_conf[k]) / len(strat_conf[k]))):
        vals = strat_conf[strat]
        avg = sum(vals) / len(vals) * 100
        mn = min(vals) * 100
        mx = max(vals) * 100
        lines.append(f"  {strat:<14} {avg:5.1f}%  {mn:5.1f}%  {mx:5.1f}%  {len(vals):>5}")

    # Confidence bucket distribution
    lines.append(_subheader("Confidence Distribution"))
    buckets = [
        ("0-30%", 0.0, 0.30),
        ("30-50%", 0.30, 0.50),
        ("50-70%", 0.50, 0.70),
        ("70%+", 0.70, 1.01),
    ]
    bucket_counts: dict[str, int] = {}
    for label, lo, hi in buckets:
        cnt = sum(1 for s in signals if s.get("confidence") is not None and lo <= s["confidence"] < hi)
        bucket_counts[label] = cnt
    for label, _, _ in buckets:
        cnt = bucket_counts[label]
        pct = cnt / len(signals) * 100 if signals else 0
        lines.append(f"  {label:<10} {cnt:>4}  ({pct:4.1f}%)  {_bar(cnt, len(signals))}")

    # ==================================================================
    # 4. CLUSTER ANALYSIS
    # ==================================================================
    lines.append(_header("CLUSTER ANALYSIS"))

    # Group sub-strategy signals by ticker + date to find agreement
    combined = [s for s in signals if (s.get("strategy") or "Combined") == "Combined"]
    subs = [s for s in signals if (s.get("strategy") or "Combined") != "Combined"]

    # Build ticker+date → list of sub-strategy signals
    sub_by_td: dict[str, list[dict]] = defaultdict(list)
    for s in subs:
        ts = s.get("timestamp", "")[:10]
        key = f"{s['ticker']}__{ts}"
        sub_by_td[key].append(s)

    agreement_counts = Counter()  # "2/2 agree", "1/2 agree", etc.
    combo_counts = Counter()  # ("Momentum:BUY", "Pullback:BUY"), etc.

    for key, group in sub_by_td.items():
        strats = {s.get("strategy"): s.get("signal") for s in group}
        directions = set()
        combo_parts = []
        for strat_name, sig in sorted(strats.items()):
            combo_parts.append(f"{strat_name}:{sig}")
            sig_upper = (sig or "").upper()
            if "BUY" in sig_upper:
                directions.add("bullish")
            elif sig_upper == "SELL":
                directions.add("bearish")
            else:
                directions.add("neutral")

        n_strats = len(strats)
        if len(directions) == 1 and "neutral" not in directions:
            agreement_counts[f"{n_strats}/{n_strats} agree"] += 1
        elif len(directions) <= 2:
            agreement_counts[f"partial ({len(strats)} strats)"] += 1
        else:
            agreement_counts["disagree"] += 1

        combo_counts[" + ".join(combo_parts)] += 1

    if agreement_counts:
        lines.append(_subheader("Strategy Agreement"))
        for label, cnt in agreement_counts.most_common():
            lines.append(f"  {label:<28} {cnt:>4}")
    else:
        lines.append("\n  No sub-strategy data available for cluster analysis.")

    if combo_counts:
        lines.append(_subheader("Most Common Combinations (top 10)"))
        for combo, cnt in combo_counts.most_common(10):
            lines.append(f"  {combo:<45} {cnt:>3}x")

    # ==================================================================
    # 5. TOP SIGNALS
    # ==================================================================
    lines.append(_header("TOP SIGNALS"))

    # Top 10 highest confidence Combined signals
    lines.append(_subheader("Top 10 Highest Confidence (Combined)"))
    combined_sorted = sorted(
        [s for s in combined if s.get("confidence") is not None],
        key=lambda s: s["confidence"],
        reverse=True,
    )[:10]
    if combined_sorted:
        lines.append(f"  {'TICKER':<10} {'SIGNAL':<14} {'CONF':>5}  {'DATE':<12} {'SESSION'}")
        lines.append("  " + "-" * 56)
        for s in combined_sorted:
            conf_pct = s["confidence"] * 100
            date = s.get("timestamp", "")[:10]
            sess = s.get("session") or "N/A"
            lines.append(f"  {s['ticker']:<10} {s.get('signal','?'):<14} {conf_pct:4.0f}%  {date:<12} {sess}")
    else:
        lines.append("  No Combined signals found.")

    # Most active tickers
    lines.append(_subheader("Most Active Tickers"))
    ticker_counts = Counter(s["ticker"] for s in signals)
    for t, cnt in ticker_counts.most_common(10):
        lines.append(f"  {t:<10} {cnt:>4} signals")

    # ==================================================================
    # 6. SESSION HEATMAP
    # ==================================================================
    lines.append(_header("SESSION HEATMAP"))
    lines.append("  Most frequent signal per ticker x session:\n")

    # Build ticker × session → most common signal
    heatmap: dict[str, dict[str, str]] = {}
    for s in signals:
        t = s["ticker"]
        sess = s.get("session") or "N/A"
        if t not in heatmap:
            heatmap[t] = {}
        key = f"{t}__{sess}"
        if key not in heatmap[t]:
            heatmap[t][sess] = Counter()
        heatmap[t][sess][s.get("signal", "?")] += 1

    all_sessions = sorted({s.get("session") or "N/A" for s in signals})
    # Header
    hdr = f"  {'TICKER':<10}"
    for sess in all_sessions:
        hdr += f" {sess:>12}"
    lines.append(hdr)
    lines.append("  " + "-" * (10 + 13 * len(all_sessions)))

    for t in sorted(heatmap):
        row = f"  {t:<10}"
        for sess in all_sessions:
            counter = heatmap[t].get(sess)
            if counter:
                most_common = counter.most_common(1)[0][0]
                row += f" {most_common:>12}"
            else:
                row += f" {'---':>12}"
        lines.append(row)

    lines.append("")
    lines.append("  " + "=" * (W - 4))
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
