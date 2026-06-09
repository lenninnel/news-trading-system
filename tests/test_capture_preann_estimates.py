"""
Tests for the Benzinga pre-announcement estimate snapshot capture (Q-013).

RECORDED-ONLY, fully isolated daily job.  These tests pin the contract that
keeps it freeze-safe:

* Migration idempotency — ``_init_schema`` may run any number of times and
  ``schema_migrations`` carries exactly one
  ``20260609_benzinga_estimate_preann_snapshot`` row; the table exists.
* Fail-safe — the body (``run``) is free to raise, but ``main`` swallows ALL
  exceptions and exits 0; an exception part-way through the insert loop rolls
  back with NO partial row, and touches NO other table.
* Unit boundary — ``estimated_eps`` is stored VERBATIM in Benzinga's native
  fraction units (NO ×100), unlike the Q-005 live sidecar's mapping boundary.
* Isolation — a run writes ONLY ``benzinga_estimate_preann_snapshot``; no
  other table's row count changes.

No network: the calendar is injected via ``fetch_fn`` (or ``requests.get`` is
monkeypatched) so these are offline unit tests.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from scripts import capture_preann_estimates as cap
from storage.database import Database

_MIGRATION = "20260609_benzinga_estimate_preann_snapshot"
_TABLE = "benzinga_estimate_preann_snapshot"

# Fixed "now" → tomorrow = 2026-06-10 (deterministic upcoming dates).
_NOW = datetime(2026, 6, 9, 11, 0, 0, tzinfo=timezone.utc)


# ── Helpers ──────────────────────────────────────────────────────────


def _event(**overrides) -> dict:
    """A live-shaped UPCOMING calendar row (date > today, actual_eps absent).

    Values mirror the Q-013 STEP 0 probe (e.g. ORCL est=1.89, confirmed).
    """
    rec = {
        "ticker": "ORCL",
        "date": "2026-06-10",
        "date_status": "confirmed",
        "estimated_eps": 1.89,
        "importance": 5,
        "fiscal_period": "Q4",
        "fiscal_year": 2026,
        "benzinga_id": "abc123",
        "company_name": "Oracle",
        "last_updated": "2026-06-01T23:01:51Z",
        "time": "16:15:00",
    }
    rec.update(overrides)
    return rec


def _table_counts(path: str) -> dict:
    """Row count of every user table in the DB."""
    con = sqlite3.connect(path)
    try:
        names = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        ]
        return {n: con.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0]
                for n in names}
    finally:
        con.close()


def _snapshot_rows(path: str) -> list:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(f"SELECT * FROM {_TABLE}")]
    finally:
        con.close()


# ── Migration idempotency ────────────────────────────────────────────


class TestMigration:
    def test_table_created_and_registered_once(self, tmp_path):
        path = str(tmp_path / "m.db")
        # Run _init_schema twice on the same file.
        Database(path)
        Database(path)

        con = sqlite3.connect(path)
        try:
            # Table exists.
            assert con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                f"AND name='{_TABLE}'"
            ).fetchone() is not None
            # Exactly one migration row, no duplicate from the second run.
            n = con.execute(
                "SELECT COUNT(*) FROM schema_migrations "
                "WHERE migration_name=?", (_MIGRATION,)
            ).fetchone()[0]
            assert n == 1
        finally:
            con.close()

    def test_unique_constraint_columns(self, tmp_db):
        # The UNIQUE key is (ticker, scheduled_report_date,
        # capture_timestamp_utc) — two captures of the same event at
        # DIFFERENT timestamps must both persist (each daily snapshot is
        # its own row), while a same-timestamp re-insert is ignored.
        path = tmp_db.db_path
        cap.run(db_path=path, api_key="k", fetch_fn=lambda: [_event()],
                now=_NOW)
        # Re-run at the SAME now → same capture_ts → UNIQUE-ignored.
        cap.run(db_path=path, api_key="k", fetch_fn=lambda: [_event()],
                now=_NOW)
        assert len(_snapshot_rows(path)) == 1
        # A later capture → new timestamp → second row.
        later = datetime(2026, 6, 9, 12, 0, 0, tzinfo=timezone.utc)
        cap.run(db_path=path, api_key="k", fetch_fn=lambda: [_event()],
                now=later)
        assert len(_snapshot_rows(path)) == 2


# ── Unit boundary (VERBATIM, no ×100) ────────────────────────────────


class TestUnitBoundary:
    def test_estimated_eps_stored_verbatim(self, tmp_db):
        path = tmp_db.db_path
        # A fraction-shaped value that would be unmistakably wrong if ×100'd.
        cap.run(db_path=path, api_key="k",
                fetch_fn=lambda: [_event(estimated_eps=0.0361)], now=_NOW)
        rows = _snapshot_rows(path)
        assert len(rows) == 1
        # Stored EXACTLY as received — Benzinga's native units, NOT 3.61.
        assert rows[0]["estimated_eps"] == 0.0361
        assert rows[0]["estimated_eps"] != pytest.approx(3.61)

    def test_columns_and_raw_json_roundtrip(self, tmp_db):
        path = tmp_db.db_path
        ev = _event(estimated_eps=1.89, importance=5, fiscal_year=2026,
                    fiscal_period="Q4", benzinga_id="zzz")
        cap.run(db_path=path, api_key="k", fetch_fn=lambda: [ev], now=_NOW)
        row = _snapshot_rows(path)[0]
        assert row["ticker"] == "ORCL"
        assert row["scheduled_report_date"] == "2026-06-10"
        assert row["date_status"] == "confirmed"
        assert row["estimated_eps"] == 1.89
        assert row["importance"] == 5
        assert row["fiscal_period"] == "Q4"
        assert row["fiscal_year"] == "2026"   # int → TEXT column
        assert row["benzinga_id"] == "zzz"
        assert row["capture_timestamp_utc"] == _NOW.isoformat()
        # raw_json is the full untouched record.
        assert json.loads(row["raw_json"]) == ev

    def test_absent_estimate_stored_as_null(self, tmp_db):
        # 'projected' rows often omit estimated_eps entirely → NULL, not error.
        path = tmp_db.db_path
        ev = _event(date_status="projected")
        ev.pop("estimated_eps")
        cap.run(db_path=path, api_key="k", fetch_fn=lambda: [ev], now=_NOW)
        row = _snapshot_rows(path)[0]
        assert row["estimated_eps"] is None
        assert row["eps_method"] is None   # absent on upcoming rows


# ── Universe cap ─────────────────────────────────────────────────────


class TestCap:
    def test_caps_by_importance_and_logs_drop(self, tmp_db, monkeypatch,
                                               caplog):
        path = tmp_db.db_path
        monkeypatch.setattr(cap, "PREANN_MAX_NAMES", 2)
        events = [
            _event(ticker="LOW", importance=1),
            _event(ticker="HIGH", importance=5),
            _event(ticker="MID", importance=3),
        ]
        with caplog.at_level("WARNING"):
            res = cap.run(db_path=path, api_key="k",
                          fetch_fn=lambda: events, now=_NOW)
        assert res["fetched"] == 3
        assert res["inserted"] == 2
        assert res["dropped"] == 1
        kept = {r["ticker"] for r in _snapshot_rows(path)}
        assert kept == {"HIGH", "MID"}   # top-2 by importance, LOW dropped
        assert any("DROPPED" in m for m in caplog.messages)


# ── Early-report flag ────────────────────────────────────────────────


class TestEarlyReport:
    def test_early_event_recorded_and_warned(self, tmp_db, caplog):
        path = tmp_db.db_path
        # Dated BEFORE tomorrow (2026-06-09 < 2026-06-10) → flagged + recorded.
        ev = _event(ticker="EARLY", date="2026-06-09")
        with caplog.at_level("WARNING"):
            res = cap.run(db_path=path, api_key="k",
                          fetch_fn=lambda: [ev], now=_NOW)
        assert res["early"] == 1
        assert len(_snapshot_rows(path)) == 1   # still recorded
        assert any("EARLY-REPORT" in m for m in caplog.messages)


# ── Isolation ────────────────────────────────────────────────────────


class TestIsolation:
    def test_writes_only_its_own_table(self, tmp_db):
        path = tmp_db.db_path
        # Seed pre-existing rows in OTHER tables.
        tmp_db.log_run("AAA", 1, 1, 0.5, "BUY")   # → runs
        con = sqlite3.connect(path)
        con.execute(
            "INSERT INTO pead_signal_log (timestamp, ticker, created_at) "
            "VALUES (?, ?, ?)",
            (_NOW.isoformat(), "ZZZ", _NOW.isoformat()),
        )
        con.commit()
        con.close()

        before = _table_counts(path)
        cap.run(db_path=path, api_key="k",
                fetch_fn=lambda: [_event(ticker="A"), _event(ticker="B")],
                now=_NOW)
        after = _table_counts(path)

        # Only the snapshot table grew; everything else byte-for-byte stable.
        assert after[_TABLE] - before.get(_TABLE, 0) == 2
        for tbl, cnt in after.items():
            if tbl == _TABLE:
                continue
            assert cnt == before[tbl], f"{tbl} row count changed"


# ── Fail-safe ────────────────────────────────────────────────────────


class TestFailSafe:
    def test_main_swallows_and_exits_zero(self, tmp_db, monkeypatch):
        # An API-layer failure (the #1 real risk) must be swallowed: main()
        # returns 0, nothing is written, no other table is touched.
        path = tmp_db.db_path
        tmp_db.log_run("AAA", 1, 1, 0.5, "BUY")
        before = _table_counts(path)

        monkeypatch.setenv("DB_PATH", path)
        monkeypatch.setenv("POLYGON_API_KEY", "k")

        def _boom(*a, **k):
            raise RuntimeError("benzinga down")

        monkeypatch.setattr(cap, "_fetch_calendar", _boom)

        rc = cap.main()
        assert rc == 0
        after = _table_counts(path)
        assert after.get(_TABLE, 0) == 0          # no rows written
        assert after == before                     # NOTHING changed anywhere

    def test_midloop_exception_rolls_back_no_partial_row(self, tmp_db):
        # The body is free to raise; it commits once at the end, so an
        # exception part-way through the loop leaves NO partial/garbage row
        # and touches no other table.
        path = tmp_db.db_path
        tmp_db.log_run("AAA", 1, 1, 0.5, "BUY")
        before = _table_counts(path)

        # Second element is a non-dict → rec.get raises mid-loop, AFTER the
        # first good row was inserted into the (uncommitted) transaction.
        poisoned = [_event(ticker="GOOD"), "POISON"]
        with pytest.raises(AttributeError):
            cap.run(db_path=path, api_key="k", fetch_fn=lambda: poisoned,
                    now=_NOW)

        after = _table_counts(path)
        assert after.get(_TABLE, 0) == 0          # rolled back, no partial row
        assert after == before                     # no other table touched

    def test_missing_api_key_is_swallowed_by_main(self, tmp_db, monkeypatch):
        path = tmp_db.db_path
        monkeypatch.setenv("DB_PATH", path)
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        # run() raises RuntimeError("POLYGON_API_KEY missing"); main swallows.
        assert cap.main() == 0
        assert _snapshot_rows(path) == []
