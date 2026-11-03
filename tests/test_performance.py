"""analytics/performance.py — track record, period returns, risk, trades (2026-11-03)."""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from analytics.performance import (
    build_equity_history,
    build_performance_summary,
    drawdowns,
    equity_series,
    period_returns,
    trade_stats,
)
from storage.database import Database


@pytest.fixture
def conn(tmp_path):
    db = Database(str(tmp_path / "p.db"))
    c = sqlite3.connect(db.db_path)
    # Alpaca-era test balance before inception must be ignored.
    c.execute("INSERT INTO risk_calculations (ticker, signal, confidence, current_price, account_balance, created_at) "
              "VALUES ('X','BUY',0.5,1,98412,'2026-05-01T13:00:00+00:00')")
    start = date(2026, 5, 4)
    nav = 100_000.0
    d = start
    i = 0
    while d <= date(2026, 11, 3):
        if d.weekday() < 5:
            nav *= 1.001 if i % 3 else 0.998
            ts = f"{d.isoformat()}T13:15:00+00:00"
            if d < date(2026, 9, 23):
                c.execute("INSERT INTO risk_calculations (ticker, signal, confidence, current_price, account_balance, created_at) "
                          "VALUES ('X','BUY',0.5,1,?,?)", (nav, ts))
            else:
                c.execute("INSERT INTO account_snapshots (ts, net_liquidation, total_cash, gross_position_value, kind, source) "
                          "VALUES (?,?,?,?,?,?)", (f"{d.isoformat()}T22:45:00+00:00", nav, nav * 0.9, nav * 0.1, "eod", "ibkr"))
            for t, px in (("SPY", 500 * (1 + i * 0.0005)), ("QQQ", 400 * (1 + i * 0.0008))):
                c.execute("INSERT INTO daily_ohlc (ticker, date, open, high, low, close, adj_close, volume, source, ingested_at) "
                          "VALUES (?,?,?,?,?,?,?,?,?,?)", (t, d.isoformat(), px, px, px, px, px, 1, "alpaca", ts))
            i += 1
        d += timedelta(days=1)
    for row in (("AAA", "BUY", 10, 100, 0, "2026-06-01T14:30:00+00:00"),
                ("AAA", "SELL", 10, 110, 100, "2026-06-04T14:30:00+00:00"),
                ("BBB", "BUY", 10, 100, 0, "2026-07-01T14:30:00+00:00"),
                ("BBB", "SELL", 10, 95, -50, "2026-07-03T14:30:00+00:00")):
        c.execute("INSERT INTO trade_history (ticker, action, shares, price, pnl, created_at) VALUES (?,?,?,?,?,?)", row)
    c.commit()
    yield c
    c.close()


def test_series_starts_at_inception_and_prefers_broker_rows(conn):
    pts = equity_series(conn, "2026-05-04")
    assert pts[0]["date"] == "2026-05-04"
    assert all(p["nav"] > 50_000 for p in pts)          # no $98k Alpaca row
    assert {p["source"] for p in pts if p["date"] >= "2026-09-23"} == {"broker_eod"}
    assert {p["source"] for p in pts if p["date"] < "2026-09-23"} == {"risk_calc"}


def test_live_nav_is_todays_point(conn):
    pts = equity_series(conn, "2026-05-04", live_nav=123_456.0, today=date(2026, 11, 4))
    assert pts[-1] == {"date": "2026-11-04", "nav": 123456.0, "source": "live"}


def test_period_returns_and_ytd_from_inception():
    pts = [("2026-05-04", 100.0), ("2026-08-01", 110.0), ("2026-10-01", 99.0), ("2026-11-03", 104.5)]
    r = period_returns(pts, "2026-05-04")
    assert r["inception"]["return"] == pytest.approx(0.045)
    assert r["ytd"]["from_inception"] is True
    assert r["1m"]["start_date"] == "2026-10-01"
    assert r["1m"]["return"] == pytest.approx(104.5 / 99 - 1, abs=1e-4)
    assert r["6m"]["start_date"] == "2026-05-04"          # 182 days back = 05-05, base 05-04
    assert r["1w"] is None                                # base 10-01 is 26 days before 10-27: a gap, not a week


def test_drawdown():
    dd = drawdowns([("a", 100.0), ("b", 120.0), ("c", 90.0), ("d", 100.0)])
    assert dd["max_drawdown"] == pytest.approx(-0.25)
    assert dd["max_drawdown_peak_date"] == "b" and dd["max_drawdown_trough_date"] == "c"
    assert dd["current_drawdown"] == pytest.approx(100 / 120 - 1, abs=1e-4)


def test_trade_stats(conn):
    t = trade_stats(conn, "2026-05-04")
    assert t["closed_trades"] == 2 and t["win_rate"] == 0.5
    assert t["profit_factor"] == 2.0 and t["expectancy"] == 25.0
    assert t["avg_holding_days"] == pytest.approx(2.5, abs=0.01)


def test_summary_has_benchmarks_on_same_window(conn):
    s = build_performance_summary(conn, now=datetime(2026, 11, 3, 23, tzinfo=timezone.utc))
    assert s["inception"] == "2026-05-04" and s["nav_points"] > 100
    for t in ("SPY", "QQQ"):
        r = s["returns"][t]["inception"]
        assert r["start_date"] == s["returns"]["portfolio"]["inception"]["start_date"]
        assert r["return"] > 0
    assert s["risk"]["beta"] is not None
    assert s["exposure"]["avg_exposure"] == pytest.approx(0.1)
    assert s["monthly"][0]["month"] == "2026-05"


def test_equity_history_rebased(conn):
    h = build_equity_history(conn)
    assert h["points"][0]["index"] == 100.0
    assert h["benchmarks"]["SPY"][0]["index"] == 100.0
    assert h["benchmarks"]["SPY"][0]["date"] <= h["points"][0]["date"]


def test_empty_db_is_safe(tmp_path):
    db = Database(str(tmp_path / "e.db"))
    c = sqlite3.connect(db.db_path)
    s = build_performance_summary(c)
    assert s["nav_points"] == 0 and s["returns"]["portfolio"]["inception"] is None


def test_returns_end_on_last_common_day_when_benchmark_lags(conn):
    # Today's benchmark bar is not ingested yet; the live NAV is.
    conn.execute("DELETE FROM daily_ohlc WHERE date = '2026-11-03'")
    conn.commit()
    s = build_performance_summary(conn, live_nav=123_456.0, now=datetime(2026, 11, 3, 18, tzinfo=timezone.utc))
    assert s["current_nav"] == 123_456.0 and s["current_date"] == "2026-11-03"
    assert s["returns_as_of"] == "2026-11-02"
    assert s["returns"]["portfolio"]["1w"]["end"] != 123_456.0
    assert s["returns"]["SPY"]["1w"]["end_date"] == "2026-11-02"
