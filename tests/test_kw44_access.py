"""NTS access additions (2026-10-28): server-side strategy filter on
/api/signals, ``days`` on /api/trades, MCP ``get_trades``.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import api.main as api_main  # noqa: E402
from analytics.signal_logger import SignalLogger  # noqa: E402
from mcp_server import nts_mcp  # noqa: E402
from storage.database import Database  # noqa: E402


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "kw44.db"
    db = Database(str(path))
    lg = SignalLogger(db)
    now = datetime.now(timezone.utc)
    # 30 Momentum rows newer than 1 NewsCatalyst row: a client-side filter
    # after LIMIT 10 would miss the NewsCatalyst row entirely.
    lg.log({"timestamp": (now - timedelta(days=2)).isoformat(), "session": "US_PRE",
            "ticker": "CVX", "strategy": "NewsCatalyst", "signal": "WEAK BUY",
            "confidence": 0.42, "price_at_signal": 230.0})
    for i in range(30):
        lg.log({"timestamp": (now - timedelta(hours=i)).isoformat(), "session": "US_PRE",
                "ticker": "AAPL", "strategy": "Momentum", "signal": "HOLD",
                "confidence": 0.25, "price_at_signal": 340.0})
    with db._connect() as conn:
        conn.execute("INSERT INTO trade_history (ticker, action, shares, price, stop_loss, "
                     "take_profit, pnl, created_at) VALUES (?,?,?,?,?,?,?,?)",
                     ("CVX", "BUY", 130, 207.33, 200.74, 220.33, 0.0,
                      (now - timedelta(days=30)).isoformat()))
        conn.execute("INSERT INTO trade_history (ticker, action, shares, price, stop_loss, "
                     "take_profit, pnl, created_at) VALUES (?,?,?,?,?,?,?,?)",
                     ("CVX", "SELL", 130, 220.34, None, None, 1690.30,
                      (now - timedelta(days=14)).isoformat()))
        conn.execute("INSERT INTO trade_history (ticker, action, shares, price, stop_loss, "
                     "take_profit, pnl, created_at) VALUES (?,?,?,?,?,?,?,?)",
                     ("OLD", "SELL", 1, 1.0, None, None, -5.0,
                      (now - timedelta(days=200)).isoformat()))
    monkeypatch.setattr(api_main, "_DB_PATH", str(path))
    monkeypatch.setenv("DB_PATH", str(path))
    monkeypatch.delenv("NTS_API_URL", raising=False)
    return path


def test_api_signals_strategy_filter_applies_before_limit(db_path):
    rows = api_main.signals(limit=10, ticker=None, days=7, strategy="newscatalyst")
    assert len(rows) == 1 and rows[0]["strategy"] == "NewsCatalyst"


def test_api_signals_without_strategy_unchanged(db_path):
    rows = api_main.signals(limit=10, ticker=None, days=7, strategy=None)
    assert len(rows) == 10


def test_api_trades_days_window(db_path):
    assert len(api_main.trades(limit=50, days=None)["trades"]) == 3
    recent = api_main.trades(limit=50, days=60)["trades"]
    assert {t["ticker"] for t in recent} == {"CVX"}


def test_mcp_get_trades_sqlite(db_path):
    out = asyncio.run(nts_mcp.get_trades(days=60))
    assert "2 row(s), 1 closed, realized $+1,690.30, 1/1 closed in profit" in out
    assert "OLD" not in out
    assert "SL=200.74 TP=220.33" in out


def test_mcp_get_trades_http_passes_days(monkeypatch):
    monkeypatch.setenv("NTS_API_URL", "http://stub.test")
    now = datetime.now(timezone.utc)
    seen = {}

    class _Resp:
        status_code = 200
        def raise_for_status(self): return None
        def json(self):
            return {"trades": [
                {"ticker": "MSFT", "action": "SELL", "shares": 51, "price": 539.53,
                 "pnl": 396.29, "timestamp": (now - timedelta(days=3)).isoformat()},
                {"ticker": "OLD", "action": "SELL", "shares": 1, "price": 1.0,
                 "pnl": -5.0, "timestamp": (now - timedelta(days=300)).isoformat()},
            ]}

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def get(self, url, params=None):
            seen["url"], seen["params"] = url, params
            return _Resp()

    monkeypatch.setattr(nts_mcp.httpx, "AsyncClient", _Client)
    out = asyncio.run(nts_mcp.get_trades(days=30))
    assert seen["url"].endswith("/api/trades") and seen["params"]["days"] == 30
    assert "MSFT" in out and "OLD" not in out


def test_mcp_signals_http_sends_strategy(monkeypatch):
    monkeypatch.setenv("NTS_API_URL", "http://stub.test")
    seen = {}

    class _Resp:
        status_code = 200
        def raise_for_status(self): return None
        def json(self): return []

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def get(self, url, params=None):
            seen["params"] = params
            return _Resp()

    monkeypatch.setattr(nts_mcp.httpx, "AsyncClient", _Client)
    asyncio.run(nts_mcp.get_signals(days=60, strategy="NewsCatalyst"))
    assert seen["params"]["strategy"] == "NewsCatalyst"
