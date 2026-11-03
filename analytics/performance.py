"""Track-record performance for the dashboard and the MCP (2026-11-03).

Builds a daily NAV series for the IBKR paper account since inception and
derives portfolio-overview metrics in the style of TrendSpider / broker
statements: period returns (1W, 1M, 3M, 6M, YTD, since inception) next to
benchmarks (SPY, QQQ), drawdown, volatility, Sharpe/Sortino, beta, monthly
returns and closed-trade statistics.

NAV sources, in order of precedence per day
-------------------------------------------
1. ``account_snapshots`` (since 2026-09-23): the broker's NetLiquidation.
   The ``kind='eod'`` row of the day wins, otherwise the last row.
2. ``risk_calculations.account_balance`` (since inception): NetLiquidation
   at the time of each risk calculation, last row of the day. Intraday,
   not end of day: good enough for a daily track record, labelled
   ``source='risk_calc'`` so a reader can tell.
3. Today only: the live portfolio view (cash + marked positions + accruals).

Days without any row are simply absent (no interpolation). Returns between
two points therefore span the gap; volatility treats each step as one
observation and says how many points it used.

Stdlib + sqlite3 only, like analytics/portfolio_view.py: api/main.py and
mcp_server/nts_mcp.py must not import the trading stack.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any

TRADING_DAYS = 252
DEFAULT_INCEPTION = "2026-05-04"
DEFAULT_BENCHMARKS = ("SPY", "QQQ")

# Max days the base point may lie before the period start (series gaps).
_BASE_TOLERANCE_DAYS = {"1w": 4, "1m": 5, "3m": 7, "6m": 10}

PERIODS: tuple[tuple[str, int | None], ...] = (
    ("1w", 7),
    ("1m", 30),
    ("3m", 91),
    ("6m", 182),
    ("ytd", None),
    ("inception", None),
)


# ── small helpers ───────────────────────────────────────────────────────


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _r(v: float | None, nd: int = 4) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(v, nd)


def _parse_day(s: str) -> date:
    return date.fromisoformat(s[:10])


# ── NAV series ──────────────────────────────────────────────────────────


def equity_series(
    conn: sqlite3.Connection,
    inception: str = DEFAULT_INCEPTION,
    live_nav: float | None = None,
    today: date | None = None,
) -> list[dict]:
    """Daily NAV points ``[{date, nav, source}]`` ascending, since inception."""
    tables = _tables(conn)
    by_day: dict[str, tuple[float, str]] = {}

    if "risk_calculations" in tables and {"account_balance", "created_at"} <= _columns(conn, "risk_calculations"):
        rows = conn.execute(
            "SELECT substr(created_at, 1, 10) AS d, account_balance FROM risk_calculations "
            "WHERE created_at >= ? AND account_balance > 0 ORDER BY created_at, rowid",
            (inception,),
        ).fetchall()
        for d, bal in rows:  # last row of the day wins
            by_day[d] = (float(bal), "risk_calc")

    if "account_snapshots" in tables:
        rows = conn.execute(
            "SELECT substr(ts, 1, 10) AS d, net_liquidation, COALESCE(kind, '') FROM account_snapshots "
            "WHERE ts >= ? AND net_liquidation > 0 ORDER BY ts, id",
            (inception,),
        ).fetchall()
        snap_day: dict[str, tuple[float, str]] = {}
        for d, nl, kind in rows:
            prev = snap_day.get(d)
            if prev is not None and prev[1] == "eod" and kind != "eod":
                continue  # keep the day's EOD row over later intraday rows
            snap_day[d] = (float(nl), "eod" if kind == "eod" else "snapshot")
        for d, (nl, kind) in snap_day.items():
            by_day[d] = (nl, "broker_eod" if kind == "eod" else "broker")

    if live_nav and live_nav > 0:
        t = (today or datetime.now(timezone.utc).date()).isoformat()
        if t >= inception:
            by_day[t] = (float(live_nav), "live")

    return [
        {"date": d, "nav": round(v, 2), "source": src}
        for d, (v, src) in sorted(by_day.items())
        if d >= inception
    ]


def benchmark_closes(conn: sqlite3.Connection, tickers, start: str) -> dict[str, dict[str, float]]:
    """``{ticker: {date: close}}`` from daily_ohlc (adj_close when present)."""
    out: dict[str, dict[str, float]] = {t: {} for t in tickers}
    if "daily_ohlc" not in _tables(conn):
        return out
    for t in tickers:
        rows = conn.execute(
            "SELECT date, COALESCE(adj_close, close) FROM daily_ohlc "
            "WHERE ticker = ? AND date >= ? ORDER BY date",
            (t, start),
        ).fetchall()
        out[t] = {d: float(c) for d, c in rows if c}
    return out


def _value_on_or_before(series: list[tuple[str, float]], day: str) -> tuple[str, float] | None:
    best = None
    for d, v in series:
        if d <= day:
            best = (d, v)
        else:
            break
    return best


# ── metrics ─────────────────────────────────────────────────────────────


def _period_start(key: str, days: int | None, last_day: date, inception: str) -> str | None:
    if key == "inception":
        return inception
    if key == "ytd":
        return date(last_day.year - 1, 12, 31).isoformat()
    return (last_day - timedelta(days=days or 0)).isoformat()


def period_returns(points: list[tuple[str, float]], inception: str) -> dict[str, dict | None]:
    """Return per period: ``{start_date, start, end, return}`` or None.

    The base is the last value on or before the period start; when the
    series begins after the start (period longer than the history), the
    period is None.  YTD with inception in the same year starts at
    inception and says so (``from_inception``).
    """
    out: dict[str, dict | None] = {}
    if len(points) < 2:
        return {k: None for k, _ in PERIODS}
    last_d, last_v = points[-1]
    last_day = _parse_day(last_d)
    first_d = points[0][0]
    for key, days in PERIODS:
        start = _period_start(key, days, last_day, inception)
        from_inception = False
        if key == "ytd" and start < first_d:
            start, from_inception = first_d, True
        if key == "inception":
            start = first_d
        base = _value_on_or_before(points, start)
        if base is None or base[0] == last_d:
            out[key] = None
            continue
        # A base far older than the period start (a gap in the series)
        # would pass a longer window off as this period: no value then.
        tol = _BASE_TOLERANCE_DAYS.get(key)
        if tol is not None and (_parse_day(start) - _parse_day(base[0])).days > tol:
            out[key] = None
            continue
        out[key] = {
            "start_date": base[0],
            "start": round(base[1], 2),
            "end": round(last_v, 2),
            "return": _r(last_v / base[1] - 1),
            **({"from_inception": True} if from_inception else {}),
        }
    return out


def _step_returns(points: list[tuple[str, float]]) -> list[tuple[str, float]]:
    return [
        (points[i][0], points[i][1] / points[i - 1][1] - 1)
        for i in range(1, len(points))
        if points[i - 1][1] > 0
    ]


def drawdowns(points: list[tuple[str, float]]) -> dict[str, Any]:
    peak_v, peak_d = -math.inf, None
    max_dd, max_peak_d, max_trough_d = 0.0, None, None
    series = []
    for d, v in points:
        if v > peak_v:
            peak_v, peak_d = v, d
        dd = v / peak_v - 1 if peak_v > 0 else 0.0
        series.append({"date": d, "drawdown": _r(dd)})
        if dd < max_dd:
            max_dd, max_peak_d, max_trough_d = dd, peak_d, d
    current = series[-1]["drawdown"] if series else None
    return {
        "max_drawdown": _r(max_dd),
        "max_drawdown_peak_date": max_peak_d,
        "max_drawdown_trough_date": max_trough_d,
        "current_drawdown": current,
        "peak_nav": _r(peak_v if peak_v > -math.inf else None, 2),
        "peak_date": peak_d,
        "series": series,
    }


def risk_stats(points: list[tuple[str, float]], bench: list[tuple[str, float]] | None = None) -> dict[str, Any]:
    rets = [r for _, r in _step_returns(points)]
    n = len(rets)
    out: dict[str, Any] = {"observations": n}
    if n < 2:
        out.update({"volatility": None, "sharpe": None, "sortino": None,
                    "best_day": None, "worst_day": None, "beta": None, "correlation": None})
        return out
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    vol = math.sqrt(var) * math.sqrt(TRADING_DAYS)
    downside = [min(0.0, r) for r in rets]
    dvar = sum(r * r for r in downside) / (n - 1)
    dvol = math.sqrt(dvar) * math.sqrt(TRADING_DAYS)
    out.update({
        "volatility": _r(vol),
        "sharpe": _r(mean * TRADING_DAYS / vol, 2) if vol > 0 else None,
        "sortino": _r(mean * TRADING_DAYS / dvol, 2) if dvol > 0 else None,
        "best_day": _r(max(rets)),
        "worst_day": _r(min(rets)),
        "risk_free_rate": 0.0,
    })
    out["beta"], out["correlation"] = None, None
    if bench:
        pairs = []
        for i in range(1, len(points)):
            d0, d1 = points[i - 1][0], points[i][0]
            b0, b1 = _value_on_or_before(bench, d0), _value_on_or_before(bench, d1)
            if b0 and b1 and b0[0] != b1[0] and points[i - 1][1] > 0:
                pairs.append((points[i][1] / points[i - 1][1] - 1, b1[1] / b0[1] - 1))
        if len(pairs) >= 10:
            px = [p for p, _ in pairs]
            bx = [b for _, b in pairs]
            mp, mb = sum(px) / len(px), sum(bx) / len(bx)
            cov = sum((p - mp) * (b - mb) for p, b in pairs) / (len(pairs) - 1)
            vb = sum((b - mb) ** 2 for b in bx) / (len(bx) - 1)
            vp = sum((p - mp) ** 2 for p in px) / (len(px) - 1)
            out["beta"] = _r(cov / vb, 2) if vb > 0 else None
            out["correlation"] = _r(cov / math.sqrt(vb * vp), 2) if vb > 0 and vp > 0 else None
    return out


def monthly_returns(points: list[tuple[str, float]]) -> list[dict]:
    """Month-over-month change of the last NAV in each month (first month vs inception)."""
    by_month: dict[str, float] = {}
    for d, v in points:
        by_month[d[:7]] = v
    months = sorted(by_month)
    out = []
    prev = points[0][1] if points else None
    for m in months:
        v = by_month[m]
        out.append({"month": m, "return": _r(v / prev - 1) if prev else None, "end_nav": round(v, 2)})
        prev = v
    return out


def trade_stats(conn: sqlite3.Connection, inception: str) -> dict[str, Any]:
    if "trade_history" not in _tables(conn):
        return {"closed_trades": 0}
    rows = conn.execute(
        "SELECT ticker, action, shares, price, pnl, created_at FROM trade_history "
        "WHERE created_at >= ? ORDER BY created_at, id",
        (inception,),
    ).fetchall()
    sells = [float(r[4] or 0) for r in rows if (r[1] or "").upper() == "SELL"]
    wins = [p for p in sells if p > 0]
    losses = [p for p in sells if p < 0]
    holds: list[float] = []
    last_buy: dict[str, datetime] = {}
    for t, action, _sh, _px, _pnl, ts in rows:
        try:
            when = datetime.fromisoformat(ts)
        except (TypeError, ValueError):
            continue
        if (action or "").upper() == "BUY":
            last_buy[t] = when
        elif (action or "").upper() == "SELL" and t in last_buy:
            holds.append((when - last_buy.pop(t)).total_seconds() / 86400)
    gross_win, gross_loss = sum(wins), -sum(losses)
    n = len(sells)
    return {
        "closed_trades": n,
        "win_rate": _r(len(wins) / n) if n else None,
        "avg_win": _r(gross_win / len(wins), 2) if wins else None,
        "avg_loss": _r(-gross_loss / len(losses), 2) if losses else None,
        "largest_win": _r(max(wins), 2) if wins else None,
        "largest_loss": _r(min(losses), 2) if losses else None,
        "profit_factor": _r(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "expectancy": _r(sum(sells) / n, 2) if n else None,
        "realized_pnl": _r(sum(sells), 2),
        "avg_holding_days": _r(sum(holds) / len(holds), 1) if holds else None,
        "holding_pairs": len(holds),
    }


def exposure_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    if "account_snapshots" not in _tables(conn) or "gross_position_value" not in _columns(conn, "account_snapshots"):
        return {"avg_exposure": None, "since": None}
    rows = conn.execute(
        "SELECT substr(ts,1,10), gross_position_value, net_liquidation FROM account_snapshots "
        "WHERE net_liquidation > 0 AND gross_position_value IS NOT NULL ORDER BY ts, id"
    ).fetchall()
    per_day: dict[str, float] = {}
    for d, gpv, nl in rows:
        per_day[d] = float(gpv) / float(nl)
    if not per_day:
        return {"avg_exposure": None, "since": None}
    vals = list(per_day.values())
    return {"avg_exposure": _r(sum(vals) / len(vals)), "since": min(per_day), "days": len(vals)}


# ── public entry points ─────────────────────────────────────────────────


def build_equity_history(
    conn: sqlite3.Connection,
    inception: str = DEFAULT_INCEPTION,
    benchmarks=DEFAULT_BENCHMARKS,
    live_nav: float | None = None,
) -> dict[str, Any]:
    """NAV series plus benchmarks rebased to the first NAV date (=100)."""
    pts = equity_series(conn, inception, live_nav=live_nav)
    out: dict[str, Any] = {"inception": inception, "points": pts, "benchmarks": {}}
    if not pts:
        return out
    first_d, first_v = pts[0]["date"], pts[0]["nav"]
    for p in pts:
        p["index"] = round(p["nav"] / first_v * 100, 3)
    closes = benchmark_closes(conn, benchmarks, (_parse_day(first_d) - timedelta(days=7)).isoformat())
    for t, cmap in closes.items():
        ser = sorted(cmap.items())
        base = _value_on_or_before(ser, first_d)
        if not base:
            out["benchmarks"][t] = []
            continue
        out["benchmarks"][t] = [
            {"date": d, "close": round(c, 4), "index": round(c / base[1] * 100, 3)}
            for d, c in ser if d >= base[0]
        ]
    return out


def build_performance_summary(
    conn: sqlite3.Connection,
    inception: str = DEFAULT_INCEPTION,
    benchmarks=DEFAULT_BENCHMARKS,
    live_nav: float | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    pts_d = equity_series(conn, inception, live_nav=live_nav, today=now.date())
    pts = [(p["date"], p["nav"]) for p in pts_d]
    closes = benchmark_closes(conn, benchmarks, (_parse_day(inception) - timedelta(days=10)).isoformat())
    bench_series = {t: sorted(c.items()) for t, c in closes.items()}

    # 2026-11-03: compare over identical end dates.  Today's NAV is live,
    # today's benchmark bar only arrives with the 22:30 UTC ingest; until
    # then the returns table ends at the last day all series share.  The
    # header numbers (current_nav) stay live.
    last_bench = [s[-1][0] for s in bench_series.values() if s]
    returns_as_of = pts[-1][0] if pts else None
    pts_cmp = pts
    if pts and last_bench:
        cut = min(min(last_bench), pts[-1][0])
        trimmed = [p for p in pts if p[0] <= cut]
        if len(trimmed) >= 2:
            pts_cmp, returns_as_of = trimmed, trimmed[-1][0]

    returns = {"portfolio": period_returns(pts_cmp, inception)}
    for t, ser in bench_series.items():
        # Benchmark periods use the portfolio's own base and end dates so the
        # comparison covers the identical window.
        br: dict[str, dict | None] = {}
        for key, pr in returns["portfolio"].items():
            if not pr or not ser:
                br[key] = None
                continue
            b0 = _value_on_or_before(ser, pr["start_date"])
            b1 = _value_on_or_before(ser, pts_cmp[-1][0])
            br[key] = (
                {"start_date": b0[0], "end_date": b1[0], "return": _r(b1[1] / b0[1] - 1)}
                if b0 and b1 and b0[0] != b1[0] else None
            )
        returns[t] = br

    dd = drawdowns(pts)
    dd_series = dd.pop("series")
    return {
        "as_of": now.isoformat(timespec="seconds"),
        "inception": inception,
        "inception_nav": pts[0][1] if pts else None,
        "current_nav": pts[-1][1] if pts else None,
        "current_date": pts[-1][0] if pts else None,
        "nav_points": len(pts),
        "returns": returns,
        "returns_as_of": returns_as_of,
        "drawdown": dd,
        "drawdown_series": dd_series,
        "risk": risk_stats(pts, bench_series.get("SPY")),
        "monthly": monthly_returns(pts),
        "monthly_benchmarks": {
            t: monthly_returns([x for x in s if pts and x[0] >= (_value_on_or_before(s, pts[0][0]) or (pts[0][0],))[0]])
            for t, s in bench_series.items() if s
        },
        "trades": trade_stats(conn, inception),
        "exposure": exposure_stats(conn),
        "notes": [
            "NAV before 2026-09-23 is NetLiquidation at the time of each risk calculation (intraday); from 2026-09-23 the broker's daily account snapshot.",
            "Sharpe and Sortino assume a risk-free rate of 0 and annualise with 252 trading days.",
        ],
    }
