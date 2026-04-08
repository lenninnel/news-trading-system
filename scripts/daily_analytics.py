"""
Daily analytics — print today's system stats to the terminal.

Reads from the production Postgres database (via DATABASE_URL + psycopg2)
when available, otherwise falls back to the local SQLite file used by
storage.database.Database.

Usage::

    python3 scripts/daily_analytics.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()


# ── Cost model ────────────────────────────────────────────────────────────
# Sonnet 4.6 pricing rough estimate per debate call (bull or bear):
#   ~5K input tokens (mostly cached) + ~300 output tokens
#   ≈ $0.0075 / call → $0.015 per debate (2 calls)
COST_PER_DEBATE_CALL_USD = 0.0075


# ── DB connection layer ───────────────────────────────────────────────────


class _DBClient:
    """Thin adapter that runs ?-style SQL against either Postgres or SQLite."""

    def __init__(self) -> None:
        self.kind: str
        self._conn: object
        url = os.environ.get("DATABASE_URL", "").strip()
        if url:
            try:
                import psycopg2
                import psycopg2.extras

                self._conn = psycopg2.connect(url)
                self._dict_cursor = psycopg2.extras.RealDictCursor
                self.kind = "postgres"
                return
            except Exception as exc:
                print(
                    f"[warn] DATABASE_URL set but psycopg2 connect failed ({exc}); "
                    "falling back to local SQLite",
                    file=sys.stderr,
                )

        # SQLite fallback — make sure SignalLogger has run its migrations so
        # the regime column exists on signal_events.
        from analytics.signal_logger import SignalLogger  # noqa: F401
        SignalLogger()

        from storage.database import Database
        import sqlite3

        db = Database()
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        self._conn = conn
        self.kind = "sqlite"

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        if self.kind == "postgres":
            sql_pg = sql.replace("?", "%s")
            with self._conn.cursor(cursor_factory=self._dict_cursor) as cur:  # type: ignore[attr-defined]
                cur.execute(sql_pg, params)
                if cur.description is None:
                    return []
                return [dict(row) for row in cur.fetchall()]
        # sqlite
        cur = self._conn.execute(sql, params)  # type: ignore[union-attr]
        rows = cur.fetchall()
        return [dict(r) for r in rows]


# ── Helpers ───────────────────────────────────────────────────────────────


def _parse_ts(ts: object) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def _fmt_pct(num: float | None, denom: float | None) -> str:
    if not denom:
        return "n/a"
    return f"{(num or 0) / denom * 100:.1f}%"


def _fmt_secs(s: float | None) -> str:
    if s is None:
        return "n/a"
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}m{int(s % 60):02d}s"


# ── Data fetch ────────────────────────────────────────────────────────────


def fetch_signal_events(db: _DBClient, since: datetime) -> list[dict]:
    rows = db.query(
        "SELECT * FROM signal_events WHERE timestamp >= ? ORDER BY id ASC",
        (since.isoformat(),),
    )
    for r in rows:
        r["_ts"] = _parse_ts(r.get("timestamp"))
    return rows


def fetch_portfolio_positions(db: _DBClient) -> list[dict]:
    return db.query("SELECT * FROM portfolio_positions", ())


def fetch_trade_history(db: _DBClient, since: datetime) -> list[dict]:
    rows = db.query(
        "SELECT * FROM trade_history WHERE created_at >= ? ORDER BY id ASC",
        (since.isoformat(),),
    )
    for r in rows:
        r["_ts"] = _parse_ts(r.get("created_at"))
    return rows


# ── Sections ──────────────────────────────────────────────────────────────


def print_session_performance(events_today: list[dict], events_7d: list[dict]) -> None:
    _section("SESSION PERFORMANCE")

    # Group today's events by session
    by_session_today: dict[str, list[dict]] = defaultdict(list)
    for e in events_today:
        by_session_today[e.get("session") or "?"].append(e)

    if not by_session_today:
        print("No sessions ran today.")
        return

    # 7-day session elapsed (per day, per session)
    session_day_elapsed: dict[str, list[float]] = defaultdict(list)
    bucket: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    for e in events_7d:
        ts = e.get("_ts")
        if not ts:
            continue
        day = ts.date().isoformat()
        sess = e.get("session") or "?"
        bucket[(sess, day)].append(ts)
    today_iso = datetime.now(timezone.utc).date().isoformat()
    for (sess, day), times in bucket.items():
        if day == today_iso:
            continue  # exclude today from baseline
        if len(times) < 2:
            continue
        elapsed = (max(times) - min(times)).total_seconds()
        session_day_elapsed[sess].append(elapsed)

    header = f"{'Session':<12} {'Tickers':>8} {'Signals':>8} {'Time':>10} {'7d Avg':>10} {'Δ':>10}"
    print(header)
    print("-" * len(header))

    # Order sessions by their first timestamp today
    ordered = sorted(
        by_session_today.items(),
        key=lambda kv: min(
            (e["_ts"] for e in kv[1] if e.get("_ts")),
            default=datetime.now(timezone.utc),
        ),
    )

    for sess, evs in ordered:
        tickers = {e["ticker"] for e in evs if e.get("ticker")}
        # "signals generated" = final Combined-strategy rows
        signals = [e for e in evs if (e.get("strategy") or "") == "Combined"]
        times = [e["_ts"] for e in evs if e.get("_ts")]
        elapsed = (max(times) - min(times)).total_seconds() if len(times) >= 2 else None
        baseline_runs = session_day_elapsed.get(sess, [])
        avg_7d = sum(baseline_runs) / len(baseline_runs) if baseline_runs else None
        if elapsed is not None and avg_7d is not None:
            delta = elapsed - avg_7d
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{_fmt_secs(delta)}"
        else:
            delta_str = "n/a"
        print(
            f"{sess:<12} {len(tickers):>8} {len(signals):>8} "
            f"{_fmt_secs(elapsed):>10} {_fmt_secs(avg_7d):>10} {delta_str:>10}"
        )


def print_signal_distribution(events_today: list[dict]) -> None:
    _section("SIGNAL DISTRIBUTION")

    final = [e for e in events_today if (e.get("strategy") or "") == "Combined"]
    type_counts = Counter((e.get("signal") or "?").upper() for e in final)
    type_order = ["STRONG BUY", "BUY", "WEAK BUY", "HOLD", "WEAK SELL", "SELL", "STRONG SELL"]
    print(f"By signal type (Combined, total={len(final)}):")
    if not final:
        print("  (no signals today)")
    else:
        for t in type_order:
            if type_counts.get(t):
                print(f"  {t:<12} {type_counts[t]:>4}")
        # any unexpected types
        for t, c in type_counts.items():
            if t not in type_order:
                print(f"  {t:<12} {c:>4}")

    print()
    print("By strategy:")
    strat_counts = Counter((e.get("strategy") or "?") for e in events_today)
    for strat in ("Momentum", "Pullback", "NewsCatalyst", "Combined", "PEAD"):
        print(f"  {strat:<13} {strat_counts.get(strat, 0):>4}")
    other = {k: v for k, v in strat_counts.items()
             if k not in {"Momentum", "Pullback", "NewsCatalyst", "Combined", "PEAD"}}
    for k, v in other.items():
        print(f"  {k:<13} {v:>4}")

    print()
    print("Debate outcomes (Combined signals):")
    outcomes = Counter((e.get("debate_outcome") or "skipped") for e in final)
    total = len(final) or 1
    for k in ("agree", "cautious", "disagree", "skipped"):
        c = outcomes.get(k, 0)
        print(f"  {k:<10} {c:>4}  ({c / total * 100:>5.1f}%)")
    skipped = outcomes.get("skipped", 0)
    print(f"  HOLD skip rate: {skipped / total * 100:.1f}%  ({skipped}/{len(final)})")


def print_market_conditions(events_today: list[dict]) -> None:
    _section("MARKET CONDITIONS")

    final = [e for e in events_today if (e.get("strategy") or "") == "Combined"]
    regimes = Counter((e.get("regime") or "unknown") for e in final)
    print("Regime distribution (Combined signals):")
    if not regimes:
        print("  (no signals today)")
    else:
        for regime, count in sorted(regimes.items(), key=lambda kv: -kv[1]):
            print(f"  {regime:<14} {count:>4}")

    print()
    print("Average confidence by session:")
    by_session: dict[str, list[float]] = defaultdict(list)
    for e in final:
        c = e.get("confidence")
        if c is not None:
            by_session[e.get("session") or "?"].append(float(c))
    if not by_session:
        print("  (no confidence data)")
    else:
        for sess, vals in sorted(by_session.items()):
            avg = sum(vals) / len(vals)
            print(f"  {sess:<12} {avg * 100:>5.1f}%  (n={len(vals)})")

    print()
    vols = [float(e["volume_ratio"]) for e in events_today
            if e.get("volume_ratio") is not None]
    if vols:
        print(
            f"Volume ratio: avg={sum(vols) / len(vols):.2f}  "
            f"min={min(vols):.2f}  max={max(vols):.2f}  (n={len(vols)})"
        )
    else:
        print("Volume ratio: no data")


def print_pead_status(
    db: _DBClient,
    events_today: list[dict],
    portfolio: list[dict],
    trades_recent: list[dict],
) -> None:
    _section("PEAD STATUS")

    # Identify which open positions came from PEAD by joining with the most
    # recent executed PEAD signal for that ticker.
    pead_signals_recent = db.query(
        "SELECT ticker, MAX(timestamp) AS last_ts FROM signal_events "
        "WHERE strategy = ? AND trade_executed = 1 GROUP BY ticker",
        ("PEAD",),
    )
    pead_tickers = {row["ticker"] for row in pead_signals_recent}

    open_pead = [p for p in portfolio if p.get("ticker") in pead_tickers and (p.get("shares") or 0) > 0]

    # Build entry-date lookup from trade_history (most recent BUY per ticker)
    last_buy: dict[str, dict] = {}
    for t in trades_recent:
        if (t.get("action") or "").upper() != "BUY":
            continue
        last_buy[t["ticker"]] = t  # last one wins (rows ordered ASC)

    print(f"Open PEAD positions: {len(open_pead)}")
    if open_pead:
        print(
            f"  {'Ticker':<10} {'Entry':>9} {'Current':>9} "
            f"{'Days':>5} {'Stop':>9}"
        )
        now = datetime.now(timezone.utc)
        for pos in open_pead:
            ticker = pos["ticker"]
            entry = pos.get("avg_price") or 0.0
            shares = pos.get("shares") or 0
            current_value = pos.get("current_value") or 0.0
            current = (current_value / shares) if shares else 0.0
            buy = last_buy.get(ticker, {})
            buy_ts = buy.get("_ts")
            days = (now - buy_ts).days if buy_ts else "?"
            stop = buy.get("stop_loss")
            stop_str = f"${stop:.2f}" if stop else "n/a"
            print(
                f"  {ticker:<10} ${entry:>8.2f} ${current:>8.2f} "
                f"{str(days):>5} {stop_str:>9}"
            )
        print("  (trailing-stop state lives in PositionManager memory; "
              "shown 'Stop' is the original stop_loss)")

    pead_today = [e for e in events_today if (e.get("strategy") or "") == "PEAD"]
    checked = {e["ticker"] for e in pead_today if e.get("ticker")}
    print()
    print(f"PEAD signals today: {len(pead_today)} rows  ({len(checked)} tickers checked)")
    if pead_today:
        result_counts = Counter((e.get("signal") or "?") for e in pead_today)
        for sig, c in result_counts.most_common():
            print(f"  {sig:<12} {c}")


def print_outcome_tracker(db: _DBClient) -> None:
    _section("OUTCOME TRACKER")

    now = datetime.now(timezone.utc)

    def _window(days_ago: int) -> tuple[str, str]:
        start = (now - timedelta(days=days_ago)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = start + timedelta(days=1)
        return start.isoformat(), end.isoformat()

    def _bucket(days_ago: int, col: str) -> None:
        start, end = _window(days_ago)
        rows = db.query(
            f"SELECT {col} AS pct, outcome_correct FROM signal_events "
            f"WHERE strategy = 'Combined' AND signal != 'HOLD' "
            f"AND timestamp >= ? AND timestamp < ?",
            (start, end),
        )
        total = len(rows)
        resolved = [r for r in rows if r["pct"] is not None]
        wins = sum(1 for r in resolved if (r.get("outcome_correct") or 0) == 1)
        avg_ret = (sum(r["pct"] for r in resolved) / len(resolved)) if resolved else None
        print(
            f"Signals from {days_ago} days ago: {len(resolved)}/{total} resolved"
        )
        if resolved:
            print(
                f"  win rate: {wins / len(resolved) * 100:.1f}%   "
                f"avg return: {avg_ret:+.2f}%"
            )

    _bucket(3, "outcome_3d_pct")
    _bucket(5, "outcome_5d_pct")

    overall = db.query(
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN outcome_correct = 1 THEN 1 ELSE 0 END) AS wins "
        "FROM signal_events "
        "WHERE strategy = 'Combined' AND outcome_correct IS NOT NULL",
        (),
    )
    if overall and overall[0]["total"]:
        total = overall[0]["total"]
        wins = overall[0]["wins"] or 0
        print(
            f"Overall outcome_correct rate: {wins / total * 100:.1f}%  "
            f"({wins}/{total} resolved signals)"
        )
    else:
        print("Overall outcome_correct rate: no resolved signals yet")


def print_cost_estimate(events_today: list[dict], events_7d: list[dict]) -> None:
    _section("COST ESTIMATE")

    final_today = [e for e in events_today if (e.get("strategy") or "") == "Combined"]
    non_hold_today = [e for e in final_today if (e.get("signal") or "").upper() != "HOLD"]
    debate_calls_today = len(non_hold_today) * 2
    cost_today = debate_calls_today * COST_PER_DEBATE_CALL_USD

    # 7-day baseline (excluding today)
    today_iso = datetime.now(timezone.utc).date().isoformat()
    by_day_calls: dict[str, int] = defaultdict(int)
    for e in events_7d:
        if (e.get("strategy") or "") != "Combined":
            continue
        ts = e.get("_ts")
        if not ts:
            continue
        day = ts.date().isoformat()
        if day == today_iso:
            continue
        if (e.get("signal") or "").upper() != "HOLD":
            by_day_calls[day] += 2
    if by_day_calls:
        avg_calls_7d = sum(by_day_calls.values()) / len(by_day_calls)
        avg_cost_7d = avg_calls_7d * COST_PER_DEBATE_CALL_USD
    else:
        avg_calls_7d = 0.0
        avg_cost_7d = 0.0

    total_today = len(final_today)
    hold_today = sum(1 for e in final_today if (e.get("signal") or "").upper() == "HOLD")
    skip_pct = (hold_today / total_today * 100) if total_today else 0.0

    print(f"Combined signals today:    {total_today}")
    print(f"Non-HOLD signals today:    {len(non_hold_today)}")
    print(f"Debate calls (×2):         {debate_calls_today}")
    print(f"Estimated cost today:      ${cost_today:.4f}")
    print(
        f"7-day avg debate calls:    {avg_calls_7d:.1f}  "
        f"(~${avg_cost_7d:.4f}/day)"
    )
    print(f"HOLD skip rate:            {skip_pct:.1f}%  ({hold_today}/{total_today})")


# ── Entry point ───────────────────────────────────────────────────────────


def main() -> None:
    db = _DBClient()
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    seven_days_ago = today_start - timedelta(days=7)

    print(f"Daily Analytics — {now.date().isoformat()} (UTC)   [{db.kind}]")

    events_7d = fetch_signal_events(db, seven_days_ago)
    events_today = [e for e in events_7d if e.get("_ts") and e["_ts"] >= today_start]

    portfolio = fetch_portfolio_positions(db)
    trades_recent = fetch_trade_history(db, today_start - timedelta(days=30))

    print_session_performance(events_today, events_7d)
    print_signal_distribution(events_today)
    print_market_conditions(events_today)
    print_pead_status(db, events_today, portfolio, trades_recent)
    print_outcome_tracker(db)
    print_cost_estimate(events_today, events_7d)
    print()


if __name__ == "__main__":
    main()
