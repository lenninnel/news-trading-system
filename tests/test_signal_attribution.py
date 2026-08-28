"""
Tests for B3 signal-attribution wiring (R spec 2026-08-27, A1-A6).

Covers the spec's test list:
    (a) 2-vote cluster → both suppliers persisted, boost 0.10,
        directional_count 2, strongest_supplier = max()-ranked voter.
    (b) solo vote → one directional supplier, boost 0.00, the two HOLD
        votes recorded, directional_count 1.
    (c) integration: an executed BUY produces exactly one
        signal_attribution row with a valid trade_id.
    (d) REGRESSION: signal decisions byte-identical — _fuse_signals'
        (signal, conf, path) triple pinned for every fusion exit.
    (e) forward READ-BACK round trip: vector written at signal time is
        carried into the attribution row with staleness_minutes > 0.
    (f) A4: forward row WITHOUT a vector → attribution row still written,
        status='no_vote_vector', reason populated.
    (g) CONFLICTING and FUSION_FALLBACK → NULL vote fields, never
        fabricated values.
    (h) attribution write failure → ERROR logged, trade completes and
        the track_trade result is returned intact.

Mocking conventions follow tests/test_level_gate.py: bare Coordinator
(no __init__), every external collaborator a MagicMock, real
ClusterDetector and (where the test needs persistence) a real tmp-file
Database / SignalLogger.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from analytics.signal_logger import SignalLogger
from orchestrator.cluster_detector import ClusterDetector
from strategies.base import StrategyResult


# ── Vote fixtures ────────────────────────────────────────────────────────


def _vote(name: str, signal: str, confidence: float) -> StrategyResult:
    return StrategyResult(
        signal=signal,
        confidence=confidence,
        strategy_name=name,
    )


TWO_VOTE = [
    _vote("Momentum", "BUY", 65.0),
    _vote("Pullback", "HOLD", 40.0),
    _vote("NewsCatalyst", "WEAK BUY", 55.0),
]

SOLO_VOTE = [
    _vote("Momentum", "BUY", 65.0),
    _vote("Pullback", "HOLD", 40.0),
    _vote("NewsCatalyst", "HOLD", 45.0),
]

CONFLICTING_VOTE = [
    _vote("Momentum", "BUY", 65.0),
    _vote("Pullback", "HOLD", 40.0),
    _vote("NewsCatalyst", "SELL", 55.0),
]


def _bare_coord(db=None):
    """Coordinator with real cluster fusion and a pluggable db."""
    from orchestrator.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    coord._cluster_detector = ClusterDetector()
    coord.db = db if db is not None else MagicMock()
    return coord


def _fuse(coord, votes):
    return coord._fuse_signals(
        "NVDA",
        votes,
        sentiment_signal="BUY",
        sentiment_confidence=0.5,
        fallback_technical_signal="BUY",
        fallback_technical_confidence=0.6,
    )


def _attribution_rows(db):
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM signal_attribution ORDER BY id",
        ).fetchall()
        return [dict(r) for r in rows]


# ── (a) 2-vote cluster ───────────────────────────────────────────────────


def test_two_vote_cluster_attribution(tmp_db):
    coord = _bare_coord(tmp_db)
    signal, conf, path, vote_ctx = _fuse(coord, TWO_VOTE)

    assert path == "CLUSTER"
    assert vote_ctx is not None
    assert vote_ctx["directional_count"] == 2
    assert vote_ctx["strongest_supplier"] == "Momentum"
    assert vote_ctx["boost_applied"] == pytest.approx(0.10)

    coord._record_signal_attribution(
        execution={"trade_id": 7, "price": 100.0},
        direction="BUY",
        ticker="NVDA",
        session="US_OPEN",
        vote_ctx=vote_ctx,
        combined_conf=conf,
        conf_pre_floor=conf,
        conf_post_floor=conf,
    )

    rows = _attribution_rows(tmp_db)
    assert len(rows) == 1
    row = rows[0]
    assert row["trade_id"] == 7
    assert row["session"] == "US_OPEN"
    # Both directional suppliers persisted
    assert row["momentum_signal"] == "BUY"
    assert row["momentum_conf"] == pytest.approx(0.65)
    assert row["newscatalyst_signal"] == "WEAK BUY"
    assert row["newscatalyst_conf"] == pytest.approx(0.55)
    # The HOLD vote is recorded too
    assert row["pullback_signal"] == "HOLD"
    assert row["pullback_conf"] == pytest.approx(0.40)
    assert row["pead_signal"] is None
    assert row["directional_count"] == 2
    assert row["strongest_supplier"] == "Momentum"
    assert row["boost_applied"] == pytest.approx(0.10)
    assert row["attribution_status"] == "complete"
    # Same-run stamps (A3)
    assert row["evaluated_at"] == row["executed_at"]
    assert row["staleness_minutes"] == 0.0


# ── (b) solo vote ────────────────────────────────────────────────────────


def test_solo_vote_attribution(tmp_db):
    coord = _bare_coord(tmp_db)
    signal, conf, path, vote_ctx = _fuse(coord, SOLO_VOTE)

    assert vote_ctx["directional_count"] == 1
    assert vote_ctx["boost_applied"] == pytest.approx(0.0)
    assert vote_ctx["strongest_supplier"] == "Momentum"

    coord._record_signal_attribution(
        execution={"trade_id": 8},
        direction="BUY",
        ticker="NVDA",
        session="EOD",
        vote_ctx=vote_ctx,
        combined_conf=conf,
        conf_pre_floor=conf,
        conf_post_floor=conf,
    )

    row = _attribution_rows(tmp_db)[0]
    assert row["momentum_signal"] == "BUY"
    assert row["directional_count"] == 1
    assert row["boost_applied"] == 0.0
    assert row["strongest_supplier"] == "Momentum"
    # The two HOLD votes are recorded
    assert row["pullback_signal"] == "HOLD"
    assert row["pullback_conf"] == pytest.approx(0.40)
    assert row["newscatalyst_signal"] == "HOLD"
    assert row["newscatalyst_conf"] == pytest.approx(0.45)
    assert row["attribution_status"] == "complete"


# ── (c) integration: executed BUY → exactly one attribution row ──────────


def _integration_coord(monkeypatch, tmp_db):
    """run_combined harness: real db + real fusion, everything else mocked."""
    coord = _bare_coord(tmp_db)

    monkeypatch.setattr(
        "orchestrator.coordinator.strategy_label", lambda t: "Momentum",
    )
    monkeypatch.setattr(
        "orchestrator.coordinator.get_days_to_earnings", lambda t: None,
    )

    coord.regime_agent = MagicMock()
    coord.regime_agent.run.return_value = {}
    coord.regime_detector = MagicMock()

    coord.run = MagicMock(return_value={
        "ticker": "NVDA",
        "signal": "BUY",
        "avg_score": 0.5,
        "run_id": 1,
        "market": {"price": 100.0},
    })

    coord.technical_agent = MagicMock()
    coord.technical_agent.run.return_value = {
        "signal": "BUY",
        "signal_id": 1,
        "indicators": {"price": 100.0},
        "bars": None,
        "adjusted_confidence": 0.6,
    }

    coord.risk_agent = MagicMock()
    coord.risk_agent.run.return_value = {
        "skipped": False,
        "direction": "BUY",
        "shares": 10,
        "position_size_usd": 1_000.0,
        "stop_loss": 95.0,
        "take_profit": 108.0,
    }

    coord.paper_trader = MagicMock()
    coord.paper_trader.track_trade.return_value = {
        "trade_id": 42, "price": 100.0,
    }

    coord.signal_logger = MagicMock()
    coord._portfolio_manager = MagicMock()
    coord._portfolio_manager.can_add_position.return_value = (True, "")
    coord._has_alpaca_position = MagicMock(return_value=False)
    coord._log_signal_event = MagicMock()
    coord._gather_strategy_votes = MagicMock(return_value=TWO_VOTE)
    return coord


def test_executed_buy_writes_exactly_one_attribution_row(
    monkeypatch, tmp_db,
):
    coord = _integration_coord(monkeypatch, tmp_db)

    result = coord.run_combined(
        "NVDA", verbose=False, account_balance=10_000.0,
        execute=True, session="US_OPEN",
    )

    assert result["execution"] == {"trade_id": 42, "price": 100.0}
    rows = _attribution_rows(tmp_db)
    assert len(rows) == 1
    row = rows[0]
    assert row["trade_id"] == 42
    assert row["ticker"] == "NVDA"
    assert row["attribution_status"] == "complete"
    assert row["directional_count"] == 2
    assert row["strongest_supplier"] == "Momentum"


# ── (d) REGRESSION: fusion decisions byte-identical ──────────────────────


def test_fusion_decisions_pinned():
    """Pin (signal, conf, path) for every fusion exit — the attribution
    carry-along must not move a single decision."""
    coord = _bare_coord()

    # 2-vote agreement: base 0.65 + 0.10 boost
    assert _fuse(coord, TWO_VOTE)[:3] == ("BUY", 0.75, "CLUSTER")
    # solo: base 0.65, no boost
    assert _fuse(coord, SOLO_VOTE)[:3] == ("BUY", 0.65, "CLUSTER")
    # conflicting: fixed 0.10
    assert _fuse(coord, CONFLICTING_VOTE)[:3] == (
        "CONFLICTING", 0.10, "CLUSTER",
    )
    # partial cluster (2 of 3 votes)
    assert _fuse(coord, TWO_VOTE[:2])[:3] == ("BUY", 0.65, "CLUSTER_PARTIAL")
    # no votes → combine_signals fallback (BUY/BUY agreement bonus:
    # max(0.5, 0.6) * 1.1 = 0.66)
    label, conf, path, ctx = _fuse(coord, [])
    assert path == "FUSION_FALLBACK"
    assert ctx is None
    assert label == "STRONG BUY"
    assert conf == pytest.approx(0.66)

    # ClusterDetector outputs themselves unchanged
    det = ClusterDetector()
    res = det.detect(TWO_VOTE)
    assert (res.cluster_signal, res.confidence, res.cluster_strength) == (
        "BUY", 0.75, 2,
    )
    hold = det.detect([_vote("Momentum", "HOLD", 40.0)])
    assert hold.cluster_signal == "HOLD"
    assert hold.confidence == pytest.approx(0.32)  # max(0.40) * 0.8


# ── (e) forward READ-BACK round trip ─────────────────────────────────────


def _forward_exec_coord(monkeypatch, tmp_db):
    """_execute_forward_signals_async harness with real db + logger."""
    coord = _bare_coord(tmp_db)
    coord.signal_logger = SignalLogger(tmp_db)

    coord.db.get_cached_signal = lambda ticker, max_age_minutes=90: {
        "signal": "BUY",
        "confidence": 0.75,
        "price_at_signal": 100.0,
        "strategy": "Combined",
    }

    coord.market_data = MagicMock()
    coord.market_data.fetch.return_value = {"price": 100.0}

    coord.risk_agent = MagicMock()
    coord.risk_agent.run.return_value = {
        "skipped": False,
        "direction": "BUY",
        "shares": 10,
        "position_size_usd": 1_000.0,
        "stop_loss": 95.0,
        "take_profit": 108.0,
    }

    coord.paper_trader = MagicMock()
    coord.paper_trader.track_trade.return_value = {
        "trade_id": 99, "price": 100.0,
    }

    coord._portfolio_manager = MagicMock()
    coord._portfolio_manager.can_add_position.return_value = (True, "")
    coord._has_alpaca_position = MagicMock(return_value=False)
    coord.db.get_portfolio_position = lambda t: None
    return coord


def _drive_forward_exec(coord):
    async def _run():
        return await coord._execute_forward_signals_async(
            "NVDA",
            account_balance=10_000.0,
            execute=True,
            api_semaphore=asyncio.Semaphore(4),
            data_semaphore=asyncio.Semaphore(4),
            db_lock=asyncio.Lock(),
            session="US_OPEN",
        )
    return asyncio.run(_run())


def test_forward_readback_round_trip(monkeypatch, tmp_db):
    coord = _forward_exec_coord(monkeypatch, tmp_db)

    # Signal session an hour ago: store the forward row WITH the vector.
    evaluated_at = (
        datetime.now(timezone.utc) - timedelta(minutes=60)
    ).isoformat()
    vector = [
        {"strategy_name": "Momentum", "signal": "BUY", "confidence": 0.65},
        {"strategy_name": "Pullback", "signal": "HOLD", "confidence": 0.40},
        {"strategy_name": "NewsCatalyst", "signal": "WEAK BUY",
         "confidence": 0.55},
    ]
    fwd_id = coord.signal_logger.store_forward_signal({
        "source_session": "EOD",
        "target_session": "US_OPEN",
        "ticker": "NVDA",
        "signal": "BUY",
        "confidence": 0.75,
        "price_at_signal": 100.0,
        "strategy_name": "Momentum",
        "stop_loss": 95.0,
        "take_profit": 108.0,
        "evaluated_at": evaluated_at,
        "vote_vector_json": json.dumps(vector),
        "directional_count": 2,
        "strongest_supplier": "Momentum",
        "boost_applied": 0.10,
        "conf_pre_floor": 0.75,
        "conf_post_floor": 0.75,
    })
    assert fwd_id is not None

    # Later simulated run executes it.
    result = _drive_forward_exec(coord)
    assert result["execution"] == {"trade_id": 99, "price": 100.0}

    rows = _attribution_rows(tmp_db)
    assert len(rows) == 1
    row = rows[0]
    assert row["trade_id"] == 99
    assert row["session"] == "US_OPEN"
    # Original vector carried through, not recomputed
    assert row["momentum_signal"] == "BUY"
    assert row["momentum_conf"] == pytest.approx(0.65)
    assert row["pullback_signal"] == "HOLD"
    assert row["newscatalyst_signal"] == "WEAK BUY"
    assert row["directional_count"] == 2
    assert row["strongest_supplier"] == "Momentum"
    assert row["boost_applied"] == pytest.approx(0.10)
    assert row["attribution_status"] == "complete"
    # Staleness computed from the original evaluated_at (A2)
    assert row["evaluated_at"] == evaluated_at
    assert row["staleness_minutes"] == pytest.approx(60.0, abs=1.0)
    assert row["staleness_minutes"] > 0


# ── (f) A4: forward row WITHOUT a vector ─────────────────────────────────


def test_forward_row_without_vector_still_writes_attribution(
    monkeypatch, tmp_db,
):
    coord = _forward_exec_coord(monkeypatch, tmp_db)

    # Pre-deploy-shaped forward row: no vector, no evaluated_at.
    fwd_id = coord.signal_logger.store_forward_signal({
        "source_session": "EOD",
        "target_session": "US_OPEN",
        "ticker": "NVDA",
        "signal": "BUY",
        "confidence": 0.75,
        "price_at_signal": 100.0,
        "stop_loss": 95.0,
        "take_profit": 108.0,
    })
    assert fwd_id is not None

    result = _drive_forward_exec(coord)
    assert result["execution"]["trade_id"] == 99

    rows = _attribution_rows(tmp_db)
    assert len(rows) == 1
    row = rows[0]
    assert row["attribution_status"] == "no_vote_vector"
    assert row["attribution_reason"]  # reason populated
    assert "no vote vector" in row["attribution_reason"]
    for col in (
        "momentum_signal", "momentum_conf", "pullback_signal",
        "pullback_conf", "newscatalyst_signal", "newscatalyst_conf",
        "pead_signal", "pead_conf", "directional_count",
        "strongest_supplier", "boost_applied", "staleness_minutes",
    ):
        assert row[col] is None, f"{col} must be NULL, not fabricated"


def test_execution_without_any_forward_row_records_missing(
    monkeypatch, tmp_db,
):
    """Cached-signal execution with zero pending forward rows must still
    write a visible no_vote_vector row (A4 — no silent skip)."""
    coord = _forward_exec_coord(monkeypatch, tmp_db)

    result = _drive_forward_exec(coord)
    assert result["execution"]["trade_id"] == 99

    rows = _attribution_rows(tmp_db)
    assert len(rows) == 1
    assert rows[0]["attribution_status"] == "no_vote_vector"
    assert "no pending forward row" in rows[0]["attribution_reason"]


# ── (g) CONFLICTING / FUSION_FALLBACK → NULL vote fields ─────────────────


def test_conflicting_and_fallback_produce_null_vote_fields(tmp_db):
    coord = _bare_coord(tmp_db)

    # CONFLICTING — vote_ctx must be None (a strongest supplier would be
    # fabricated) …
    signal, conf, path, vote_ctx = _fuse(coord, CONFLICTING_VOTE)
    assert signal == "CONFLICTING"
    assert vote_ctx is None

    # FUSION_FALLBACK — no vector exists at all.
    signal2, conf2, path2, vote_ctx2 = _fuse(coord, [])
    assert path2 == "FUSION_FALLBACK"
    assert vote_ctx2 is None

    # … and both persist NULL vote fields, never fabricated values.
    for i, (ctx, reason) in enumerate(
        [(vote_ctx, "CONFLICTING"), (vote_ctx2, "FUSION_FALLBACK")],
    ):
        coord._record_signal_attribution(
            execution={"trade_id": 100 + i},
            direction="BUY",
            ticker="NVDA",
            session="US_OPEN",
            vote_ctx=ctx,
            combined_conf=0.5,
            no_vector_reason=f"signal_path={reason}",
        )

    rows = _attribution_rows(tmp_db)
    assert len(rows) == 2
    for row in rows:
        assert row["attribution_status"] == "no_vote_vector"
        assert row["attribution_reason"]
        for col in (
            "momentum_signal", "momentum_conf", "pullback_signal",
            "pullback_conf", "newscatalyst_signal", "newscatalyst_conf",
            "pead_signal", "pead_conf", "directional_count",
            "strongest_supplier", "boost_applied",
        ):
            assert row[col] is None, f"{col} must be NULL, not fabricated"


# ── (h) attribution write failure → trade unaffected ─────────────────────


def test_attribution_failure_logs_error_and_trade_completes(
    monkeypatch, tmp_db, caplog,
):
    coord = _integration_coord(monkeypatch, tmp_db)
    coord.db.insert_signal_attribution = MagicMock(
        side_effect=RuntimeError("disk full"),
    )

    with caplog.at_level(logging.ERROR, logger="orchestrator.coordinator"):
        result = coord.run_combined(
            "NVDA", verbose=False, account_balance=10_000.0,
            execute=True, session="US_OPEN",
        )

    # Trade completed and the track_trade result is returned intact.
    assert result["execution"] == {"trade_id": 42, "price": 100.0}
    coord.paper_trader.track_trade.assert_called_once()

    errors = [
        r for r in caplog.records
        if r.levelno == logging.ERROR and "signal_attribution" in r.getMessage()
    ]
    assert len(errors) == 1
    msg = errors[0].getMessage()
    assert "NVDA" in msg and "42" in msg


# ── Sells and skipped trades never write rows ────────────────────────────


def test_sell_and_missing_trade_id_write_nothing(tmp_db):
    coord = _bare_coord(tmp_db)
    _, conf, _, vote_ctx = _fuse(coord, TWO_VOTE)

    coord._record_signal_attribution(
        execution={"trade_id": 1},
        direction="SELL",
        ticker="NVDA",
        session="US_OPEN",
        vote_ctx=vote_ctx,
        combined_conf=conf,
    )
    coord._record_signal_attribution(
        execution=None,
        direction="BUY",
        ticker="NVDA",
        session="US_OPEN",
        vote_ctx=vote_ctx,
        combined_conf=conf,
    )
    coord._record_signal_attribution(
        execution={"trade_id": None},
        direction="BUY",
        ticker="NVDA",
        session="US_OPEN",
        vote_ctx=vote_ctx,
        combined_conf=conf,
    )

    assert _attribution_rows(tmp_db) == []


# ── store_forward_signal stays non-raising, forward row survives ─────────


def test_vote_vector_write_failure_keeps_forward_row(tmp_db, monkeypatch):
    """A vote-vector write failure must not lose the forward row (A1a)."""
    logger = SignalLogger(tmp_db)

    fwd_id = logger.store_forward_signal({
        "source_session": "EOD",
        "target_session": "US_OPEN",
        "ticker": "NVDA",
        "signal": "BUY",
        "confidence": 0.75,
        "evaluated_at": "2026-08-28T12:00:00+00:00",
        # Un-serialisable vector object forces the secondary write to
        # fail while the primary INSERT has already succeeded.
        "vote_vector_json": {"bad": object()},
    })

    assert fwd_id is not None
    pending = logger.get_pending_forward_signals("US_OPEN", "NVDA")
    assert len(pending) == 1
    assert pending[0]["id"] == fwd_id
    assert pending[0]["vote_vector_json"] is None
