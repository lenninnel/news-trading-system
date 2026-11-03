"""analytics/market_view.py — market overview and ticker chart data (2026-11-03)."""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from analytics.signal_logger import SignalLogger
from analytics.market_view import build_market_view, build_ticker_chart
from storage.database import Database

NOW = datetime(2026, 11, 3, 21, tzinfo=timezone.utc)


@pytest.fixture
def conn(tmp_path):
    db = Database(str(tmp_path / "m.db"))
    SignalLogger(db)  # creates signal_events
    c = sqlite3.connect(db.db_path)
    d, i = date(2025, 11, 3), 0
    while d <= date(2026, 11, 2):
        if d.weekday() < 5:
            for t, px in (("SPY", 500 + i * 0.5), ("QQQ", 400 + i * 0.2), ("AAA", 100 + i * 0.1)):
                c.execute("INSERT INTO daily_ohlc (ticker, date, open, high, low, close, adj_close, volume, source, ingested_at) "
                          "VALUES (?,?,?,?,?,?,?,?,?,?)", (t, d.isoformat(), px, px + 1, px - 1, px, px, 1000, "alpaca", d.isoformat()))
            i += 1
        d += timedelta(days=1)
    c.execute("INSERT INTO risk_calculations (ticker, signal, confidence, current_price, account_balance, regime, created_at) "
              "VALUES ('AAA','BUY',0.5,1,1,'TRENDING_BULL','2026-11-03T13:15:00+00:00')")
    for ts, sig, conf, ex in (("2026-10-20T13:15:00+00:00", "WEAK BUY", 0.4, 0),
                              ("2026-10-20T18:00:00+00:00", "BUY", 0.6, 1),
                              ("2026-11-03T13:15:00+00:00", "HOLD", 0.3, 0)):
        c.execute("INSERT INTO signal_events (timestamp, session, ticker, strategy, signal, confidence, trade_executed, regime) "
                  "VALUES (?,?,?,?,?,?,?,?)", (ts, "US_PRE", "AAA", "Combined", sig, conf, ex, "TRENDING_UP"))
    c.execute("INSERT INTO trade_history (ticker, action, shares, price, stop_loss, take_profit, pnl, created_at) "
              "VALUES ('AAA','BUY',10,120,110,140,0,'2026-10-20T14:30:00+00:00')")
    c.execute("INSERT INTO portfolio_positions (ticker, shares, avg_price, current_value, updated_at) "
              "VALUES ('AAA',10,120,1250,'2026-11-03T21:00:00+00:00')")
    c.commit()
    yield c
    c.close()


def test_market_view(conn):
    m = build_market_view(conn, now=NOW)
    spy = m["benchmarks"]["SPY"]
    assert spy["date"] == "2026-11-02" and spy["change_1d"] > 0 and spy["vs_sma200"] > 0
    assert len(spy["spark"]) == 60
    assert m["market_regime"]["regime"] == "TRENDING_BULL"
    (row,) = m["tickers"]
    assert row["ticker"] == "AAA" and row["held"] and row["regime"] == "TRENDING_UP"
    assert row["latest_signal"]["signal"] == "HOLD"


def test_ticker_chart(conn):
    ch = build_ticker_chart(conn, "aaa", days=90, now=NOW)
    assert ch["ticker"] == "AAA" and ch["bars"][-1]["time"] == "2026-11-02"
    assert all(b["time"] >= "2026-08-05" for b in ch["bars"])
    assert [t["action"] for t in ch["trades"]] == ["BUY"]
    # one marker per day: the executed BUY wins over the earlier WEAK BUY
    assert len(ch["signals"]) == 1 and ch["signals"][0]["signal"] == "BUY"
    assert ch["position"] == {"shares": 10, "entry": 120.0, "entry_date": "2026-10-20",
                              "stop_loss": 110.0, "take_profit": 140.0}


def test_empty_db_is_safe(tmp_path):
    c = sqlite3.connect(Database(str(tmp_path / "e.db")).db_path)
    assert build_market_view(c, now=NOW)["tickers"] == []
    assert build_ticker_chart(c, "X", now=NOW)["bars"] == []


def test_api_endpoints(conn, tmp_path, monkeypatch):
    import api.main as api_main
    from fastapi import HTTPException

    path = conn.execute("PRAGMA database_list").fetchone()[2]
    monkeypatch.setattr(api_main, "_DB_PATH", path)
    assert api_main.market()["market_regime"]["regime"] == "TRENDING_BULL"
    assert api_main.chart("AAA", days=90)["position"]["entry"] == 120.0
    with pytest.raises(HTTPException):
        api_main.chart("AAA;DROP", days=90)
