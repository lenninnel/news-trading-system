"""GET /api/portfolio goes through analytics.portfolio_view (2026-09-23):
NAV = cash + positions at live marks; the dashboard's legacy keys stay.

The endpoint functions are called directly (they are plain sync
functions): fastapi's TestClient trips over the anyio portal on Python
3.14 once other suites have touched the asyncio policy ('NoneType' has
no attribute 'set_name'), which is a test-harness artefact, not the API.

Run: python3 -m pytest tests/test_api_portfolio.py -v
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import api.main as api_main  # noqa: E402
from storage.database import Database  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = tmp_path / "api.db"
    db = Database(str(path))
    db.set_portfolio_position(ticker="AAPL", shares=81, avg_price=333.45,
                              current_value=27009.45, mark_price=333.44, mark_source="fill")
    db.mark_portfolio_position("AAPL", 81, 333.45, 342.98, "yfinance_1m")
    db.sync_portfolio_position("AAPL", 81, 333.45)      # session-start sync keeps the mark
    db.record_account_snapshot(net_liquidation=271095.71, total_cash=243314.33,
                               prev_day_equity=270000.0, kind="session",
                               ts="2026-09-23T13:15:00+00:00")
    monkeypatch.setattr(api_main, "_DB_PATH", str(path))

    class _Client:
        def get(self, route):
            fn = {"/api/portfolio": api_main.portfolio, "/api/status": api_main.status}[route]
            data = fn()

            class _Resp:
                def json(self):
                    return data
            return _Resp()
    return _Client()


def test_portfolio_endpoint_reports_nav_cash_and_daily_pnl(client):
    data = client.get("/api/portfolio").json()
    # legacy keys the dashboard SummaryBar reads
    assert data["value"] == 271095.71          # NAV, not positions-only
    assert data["cash"] == 243314.33
    assert data["daily_pnl"] == 1095.71 and data["daily_pnl_pct"] == 0.41
    assert data["positions"][0]["ticker"] == "AAPL"
    assert data["positions"][0]["current"] == 342.98
    # explicit parts + reconciliation
    assert data["positions_value"] == 27781.38
    assert data["prev_close_nav"] == 270000.0
    assert data["broker_nav"] == 271095.71 and data["nav_minus_broker"] == 0.0
    assert data["cash_source"] == "ibkr:TotalCashValue"
    assert data["notes"] == []


def test_portfolio_endpoint_and_mcp_sqlite_mode_agree(client, monkeypatch):
    """Same DB, same numbers — the MCP tool in SQLite mode reads the same view."""
    import asyncio
    from mcp_server import nts_mcp
    monkeypatch.delenv("NTS_API_URL", raising=False)
    monkeypatch.setenv("DB_PATH", api_main._DB_PATH)
    via_api = client.get("/api/portfolio").json()
    via_mcp = asyncio.run(nts_mcp._fetch_portfolio())
    for key in ("value", "cash", "daily_pnl", "positions_value", "prev_close_nav", "broker_nav"):
        assert via_api[key] == via_mcp[key]


def test_status_endpoint_next_session_has_a_date(client):
    data = client.get("/api/status").json()
    assert data["next_session"] in {e["name"] for e in api_main._SCHEDULE}
    assert len(data["next_run_at"]) >= 16 and "T" in data["next_run_at"]
    assert "calendar_note" in data
