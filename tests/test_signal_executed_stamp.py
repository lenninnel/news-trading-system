"""US_OPEN fast path stamps the cached US_PRE row as executed (2026-10-28).

KW44 review: 35 gate-passed Combined signals, 7 real buys, 0 rows with
trade_executed=1 — the cached-signal executor never touched signal_events.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

from analytics.signal_logger import SignalLogger
from orchestrator.coordinator import Coordinator


def _insert_us_pre_row(db, ticker="NVDA") -> int:
    SignalLogger(db).log({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session": "US_PRE",
        "ticker": ticker,
        "strategy": "Combined",
        "signal": "BUY",
        "confidence": 0.75,
        "price_at_signal": 100.0,
    })
    with db._connect() as conn:
        return conn.execute(
            "SELECT id FROM signal_events WHERE ticker=? ORDER BY id DESC LIMIT 1",
            (ticker,),
        ).fetchone()[0]


def _coord(db, cached_row_id, trade_id=99):
    coord = Coordinator.__new__(Coordinator)
    coord.db = db
    coord.signal_logger = SignalLogger(db)
    real_get = db.get_cached_signal
    coord.db.get_cached_signal = lambda t, max_age_minutes=90: real_get(t, max_age_minutes=max_age_minutes)
    coord.market_data = MagicMock()
    coord.market_data.fetch.return_value = {"price": 100.0}
    coord.risk_agent = MagicMock()
    coord.risk_agent.run.return_value = {
        "skipped": False, "direction": "BUY", "shares": 10,
        "position_size_usd": 1_000.0, "stop_loss": 95.0, "take_profit": 108.0,
    }
    coord.paper_trader = MagicMock()
    coord.paper_trader.track_trade.return_value = (
        {"trade_id": trade_id, "price": 100.0} if trade_id else None
    )
    coord._portfolio_manager = MagicMock()
    coord._portfolio_manager.can_add_position.return_value = (True, "")
    coord._has_alpaca_position = MagicMock(return_value=False)
    coord.db.get_portfolio_position = lambda t: None
    coord._record_forward_attribution = MagicMock()
    return coord


def _run(coord):
    async def go():
        return await coord._execute_forward_signals_async(
            "NVDA", account_balance=10_000.0, execute=True,
            api_semaphore=asyncio.Semaphore(4), data_semaphore=asyncio.Semaphore(4),
            db_lock=asyncio.Lock(), session="US_OPEN",
        )
    return asyncio.run(go())


def _row(db, rid):
    with db._connect() as conn:
        conn.row_factory = None
        return conn.execute(
            "SELECT trade_executed, trade_id FROM signal_events WHERE id=?", (rid,)
        ).fetchone()


def test_executed_cached_signal_is_stamped(tmp_db):
    rid = _insert_us_pre_row(tmp_db)
    result = _run(_coord(tmp_db, rid))
    assert result["execution"]["trade_id"] == 99
    assert tuple(_row(tmp_db, rid)) == (1, "99")


def test_no_trade_leaves_row_unstamped(tmp_db):
    rid = _insert_us_pre_row(tmp_db)
    _run(_coord(tmp_db, rid, trade_id=None))
    assert _row(tmp_db, rid)[0] == 0


def test_mark_signal_executed_never_raises(tmp_db):
    assert tmp_db.mark_signal_executed(0, "1") is False
    assert tmp_db.mark_signal_executed(123456, "1") is False
