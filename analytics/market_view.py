"""Market overview and per-ticker chart data for the dashboard (2026-11-03).

Two read-only views over the NTS database, both stdlib + sqlite3 only so
api/main.py and the MCP server can use them without the trading stack:

``build_market_view``
    Benchmarks (SPY, QQQ) with last close, 1-day change, distance to the
    50/200-day averages, 20-day realised volatility and a short sparkline;
    the market regime the RegimeAgent last recorded (risk_calculations);
    and one row per universe ticker with price changes, the latest
    Combined signal, the per-ticker regime and whether NTS holds it.

``build_ticker_chart``
    Daily OHLCV bars from ``daily_ohlc`` plus the trades (trade_history)
    and Combined BUY-type signals (signal_events) for one ticker, and the
    open position's entry / stop / target when there is one. The
    dashboard draws these as candles with markers and price lines.

Nothing here is computed from signals to *look* like market data: prices
come from daily_ohlc, regimes from what the agents recorded.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

BUY_SIGNALS = ("STRONG BUY", "BUY", "WEAK BUY")


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _r(v: float | None, nd: int = 4) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(v, nd)


def _closes(conn: sqlite3.Connection, ticker: str, limit: int) -> list[tuple[str, float]]:
    rows = conn.execute(
        "SELECT date, COALESCE(adj_close, close) FROM daily_ohlc WHERE ticker = ? "
        "ORDER BY date DESC LIMIT ?",
        (ticker, limit),
    ).fetchall()
    return [(d, float(c)) for d, c in reversed(rows) if c]


def _change(closes: list[tuple[str, float]], back: int) -> float | None:
    if len(closes) <= back:
        return None
    return _r(closes[-1][1] / closes[-1 - back][1] - 1)


def _vs_sma(closes: list[tuple[str, float]], n: int) -> float | None:
    if len(closes) < n:
        return None
    sma = sum(c for _, c in closes[-n:]) / n
    return _r(closes[-1][1] / sma - 1)


def _realised_vol(closes: list[tuple[str, float]], n: int = 20) -> float | None:
    if len(closes) < n + 1:
        return None
    rets = [closes[i][1] / closes[i - 1][1] - 1 for i in range(len(closes) - n, len(closes))]
    mu = sum(rets) / n
    var = sum((x - mu) ** 2 for x in rets) / (n - 1)
    return _r(math.sqrt(var) * math.sqrt(252))


def _price_block(closes: list[tuple[str, float]], spark: int) -> dict[str, Any]:
    if not closes:
        return {"last": None, "date": None}
    hi = max(c for _, c in closes[-252:])
    return {
        "last": round(closes[-1][1], 2),
        "date": closes[-1][0],
        "change_1d": _change(closes, 1),
        "change_5d": _change(closes, 5),
        "change_1m": _change(closes, 21),
        "vs_sma50": _vs_sma(closes, 50),
        "vs_sma200": _vs_sma(closes, 200),
        "from_52w_high": _r(closes[-1][1] / hi - 1),
        "vol_20d": _realised_vol(closes),
        "spark": [round(c, 2) for _, c in closes[-spark:]],
    }


def _market_regime(conn: sqlite3.Connection) -> dict[str, Any] | None:
    if "risk_calculations" not in _tables(conn) or "regime" not in _columns(conn, "risk_calculations"):
        return None
    row = conn.execute(
        "SELECT regime, created_at FROM risk_calculations "
        "WHERE regime IS NOT NULL AND regime != '' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return {"regime": row[0], "as_of": row[1]} if row else None


def _latest_signals(conn: sqlite3.Connection, since: str) -> dict[str, dict[str, Any]]:
    """Latest Combined signal_events row per ticker since ``since``."""
    if "signal_events" not in _tables(conn):
        return {}
    has_regime = "regime" in _columns(conn, "signal_events")
    rows = conn.execute(
        "SELECT ticker, signal, confidence, timestamp, session, trade_executed"
        + (", regime" if has_regime else ", NULL")
        + " FROM signal_events WHERE strategy = 'Combined' AND timestamp >= ? ORDER BY id",
        (since,),
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for t, sig, conf, ts, sess, ex, reg in rows:
        out[t] = {
            "signal": sig,
            "confidence": _r(conf, 3),
            "timestamp": ts,
            "session": sess,
            "trade_executed": bool(ex),
            "regime": reg,
        }
    return out


def _ticker_regimes(conn: sqlite3.Connection, since: str) -> dict[str, str]:
    if "signal_events" not in _tables(conn) or "regime" not in _columns(conn, "signal_events"):
        return {}
    rows = conn.execute(
        "SELECT ticker, regime FROM signal_events WHERE regime IS NOT NULL AND regime != '' "
        "AND timestamp >= ? ORDER BY id",
        (since,),
    ).fetchall()
    return {t: r for t, r in rows}


def _held(conn: sqlite3.Connection) -> set[str]:
    if "portfolio_positions" not in _tables(conn):
        return set()
    cols = _columns(conn, "portfolio_positions")
    qty = "quantity" if "quantity" in cols else ("shares" if "shares" in cols else None)
    if qty is None:
        return {r[0] for r in conn.execute("SELECT ticker FROM portfolio_positions")}
    return {r[0] for r in conn.execute(f"SELECT ticker FROM portfolio_positions WHERE {qty} > 0")}


def _position_levels(open_lots: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Entry (share-weighted), first entry date and the latest stop/target
    of the BUY lots since the last SELL."""
    if not open_lots:
        return None
    sh = sum(l["shares"] or 0 for l in open_lots)
    cost = sum((l["shares"] or 0) * (l["price"] or 0) for l in open_lots)
    last = open_lots[-1]
    return {
        "shares": sh,
        "entry": _r(cost / sh, 2) if sh else None,
        "entry_date": open_lots[0]["date"],
        "stop_loss": last["stop_loss"],
        "take_profit": last["take_profit"],
    }


def _open_lots_all(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    if "trade_history" not in _tables(conn):
        return {}
    lots: dict[str, list[dict[str, Any]]] = {}
    for t, action, shares, price, stop, tp, ts in conn.execute(
        "SELECT ticker, action, shares, price, stop_loss, take_profit, created_at "
        "FROM trade_history ORDER BY created_at, id"
    ):
        a = (action or "").upper()
        if a == "BUY":
            lots.setdefault(t, []).append({
                "shares": shares, "price": price, "date": (ts or "")[:10],
                "stop_loss": _r(stop, 2), "take_profit": _r(tp, 2),
            })
        elif a == "SELL":
            lots.pop(t, None)
    return lots


def build_market_view(
    conn: sqlite3.Connection,
    benchmarks=("SPY", "QQQ"),
    universe: list[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    tables = _tables(conn)
    bench = {}
    if "daily_ohlc" in tables:
        for t in benchmarks:
            bench[t] = _price_block(_closes(conn, t, 260), spark=60)

    held = _held(conn)
    if universe is None:
        universe = (
            [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM daily_ohlc ORDER BY ticker")]
            if "daily_ohlc" in tables else []
        )
    names = sorted({t for t in universe if t not in benchmarks} | held)
    since = (now - timedelta(days=5)).isoformat()
    sigs = _latest_signals(conn, since)
    regimes = _ticker_regimes(conn, since)
    lots = _open_lots_all(conn)

    rows = []
    for t in names:
        closes = _closes(conn, t, 260) if "daily_ohlc" in tables else []
        block = _price_block(closes, spark=30)
        rows.append({
            "ticker": t,
            **block,
            "held": t in held,
            "position": _position_levels(lots.get(t, [])) if t in held else None,
            "regime": regimes.get(t),
            "latest_signal": sigs.get(t),
        })

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "benchmarks": bench,
        "market_regime": _market_regime(conn),
        "tickers": rows,
        "notes": [
            "Prices are daily closes from daily_ohlc (adjusted where available); today's bar arrives with the 22:30 UTC ingest.",
            "Market regime is the RegimeAgent's last classification (SPY trend, VIX, realised vol); ticker regime is the per-ticker detector on the latest signal.",
        ],
    }


def build_ticker_chart(
    conn: sqlite3.Connection,
    ticker: str,
    days: int = 365,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    ticker = ticker.upper()
    tables = _tables(conn)
    start = (now - timedelta(days=days)).date().isoformat()

    bars = []
    if "daily_ohlc" in tables:
        for d, o, h, l, c, v in conn.execute(
            "SELECT date, open, high, low, close, volume FROM daily_ohlc "
            "WHERE ticker = ? AND date >= ? ORDER BY date",
            (ticker, start),
        ):
            bars.append({"time": d, "open": o, "high": h, "low": l, "close": c, "volume": v or 0})

    trades = []
    open_lots: list[dict[str, Any]] = []
    if "trade_history" in tables:
        cols = _columns(conn, "trade_history")
        strat = "strategy" if "strategy" in cols else "NULL"
        for action, shares, price, stop, tp, pnl, ts, st in conn.execute(
            f"SELECT action, shares, price, stop_loss, take_profit, pnl, created_at, {strat} "
            "FROM trade_history WHERE ticker = ? ORDER BY created_at, id",
            (ticker,),
        ):
            a = (action or "").upper()
            row = {
                "action": a,
                "date": (ts or "")[:10],
                "timestamp": ts,
                "shares": shares,
                "price": _r(price, 2),
                "stop_loss": _r(stop, 2),
                "take_profit": _r(tp, 2),
                "pnl": _r(pnl, 2) if a == "SELL" else None,
                "strategy": st,
            }
            if a == "BUY":
                open_lots.append(row)
            elif a == "SELL":
                open_lots = []
            if (ts or "")[:10] >= start:
                trades.append(row)

    signals = []
    if "signal_events" in tables:
        for sig, conf, ts, sess, ex, px in conn.execute(
            "SELECT signal, confidence, timestamp, session, trade_executed, price_at_signal "
            "FROM signal_events WHERE ticker = ? AND strategy = 'Combined' AND timestamp >= ? "
            f"AND signal IN ({','.join('?' * len(BUY_SIGNALS))}) ORDER BY id",
            (ticker, start, *BUY_SIGNALS),
        ):
            signals.append({
                "date": (ts or "")[:10],
                "timestamp": ts,
                "signal": sig,
                "confidence": _r(conf, 3),
                "session": sess,
                "trade_executed": bool(ex),
                "price": _r(px, 2),
            })
        # one marker per day: the strongest signal of the day
        rank = {s: i for i, s in enumerate(BUY_SIGNALS)}
        best: dict[str, dict[str, Any]] = {}
        for s in signals:
            cur = best.get(s["date"])
            if cur is None or (rank[s["signal"]], -(s["confidence"] or 0)) < (rank[cur["signal"]], -(cur["confidence"] or 0)) or s["trade_executed"]:
                best[s["date"]] = s
        signals = [best[d] for d in sorted(best)]

    position = _position_levels(open_lots) if ticker in _held(conn) else None

    return {
        "ticker": ticker,
        "as_of": now.isoformat(timespec="seconds"),
        "bars": bars,
        "trades": trades,
        "signals": signals,
        "position": position,
    }
