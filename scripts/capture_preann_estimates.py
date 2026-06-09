#!/usr/bin/env python3
"""Q-013 Step 4 — Benzinga PRE-ANNOUNCEMENT estimate snapshot capture.

RECORDED-ONLY, fully isolated observability job.  Once a day (a standalone
``nts-preann-estimates`` systemd --user timer, NOT a trading-session hook) it
snapshots Benzinga's CURRENT consensus estimate at T-1 for the next batch of
reporters into ``benzinga_estimate_preann_snapshot``.  A FUTURE analysis can
then compare each captured T-1 estimate against Benzinga's HISTORICAL record
for the same event and certify whether Benzinga preserves the
pre-announcement estimate (point-in-time) or silently restates it.

NOTHING in the live trading / signal / execution path reads or writes this
table.  This script imports NOTHING from the trading path: it talks to the
Benzinga earnings endpoint and writes ONLY its own table via a private
``sqlite3`` connection.

Source (Q-013 STEP 0, 2026-06-09 — gate PASSED)
------------------------------------------------
There is NO separate consensus/estimates endpoint (consensus_ratings /
analyst_insights / estimates all 404/403).  The ONLY source of
``estimated_eps`` is the earnings endpoint itself, and STEP 0 confirmed that
UPCOMING (date > today) calendar rows already carry a non-null
``estimated_eps`` while ``actual_eps`` is still null — i.e. the
pre-announcement estimate genuinely exists to capture at T-1.  STEP 0 also
confirmed the endpoint accepts a BARE forward date range with no ticker, so
the broad calendar is pulled in ONE call::

    GET https://api.polygon.io/benzinga/v1/earnings
        ?date.gte={tomorrow}&date.lte={tomorrow+1d}&sort=date.asc
        &limit=1000&apiKey={POLYGON_API_KEY}

Units (load-bearing)
--------------------
``estimated_eps`` and ``raw_json`` are stored VERBATIM in Benzinga's NATIVE
units.  This table does NO unit conversion and computes NO surprise — unlike
the Q-005 LIVE sidecar (``data/benzinga_feed.py``), which ×100s the surprise
FRACTION at its mapping boundary to stay unit-consistent with yfinance.  Here
there is no second source to reconcile against, so the raw Benzinga value is
the correct thing to preserve.

Fail-safe contract (non-negotiable — same class as the Q-005 sidecar)
---------------------------------------------------------------------
The ENTIRE job body is wrapped in try/except in ``main()``.  ANY failure
(API error, empty calendar, parse error, DB error) is logged and the process
exits 0.  This job can NEVER raise into anything else.  The capture body
(``run``) is free to raise — ``main`` swallows it — and it commits exactly
once at the end, so an exception part-way through the insert loop rolls back
with no partial/garbage rows persisted.

Exit code: always 0 (recorded-only, isolated).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("capture_preann_estimates")

_BASE_URL = "https://api.polygon.io/benzinga/v1/earnings"
_TABLE = "benzinga_estimate_preann_snapshot"

# Universe bound — NEVER pull unbounded.  The forward calendar is broad (any
# liquid US reporter, not just PEAD names).  If it returns more than this cap,
# keep the top-N by Benzinga 'importance' (desc) and LOG how many were
# dropped.  Env-overridable; default 200.
PREANN_MAX_NAMES = int(os.environ.get("PREANN_MAX_NAMES", "200"))

# Production DB path.  Mirrors scripts/backup_db.py's explicit VPS fallback so
# this standalone job writes to the SAME live DB the daemon uses.  Tests
# override via the DB_PATH env var (pointed at a temp file).
_DB_PATH_DEFAULT = "/home/trading/trading-data/news_trading.db"


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Create the snapshot table if absent.

    Self-sufficient: the job runs correctly even if invoked before the
    storage-layer schema migration (and is harmless after it).  Kept in exact
    sync with the canonical definition in ``storage/database.py``.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benzinga_estimate_preann_snapshot (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker                TEXT    NOT NULL,
            scheduled_report_date TEXT    NOT NULL,
            date_status           TEXT,
            capture_timestamp_utc TEXT    NOT NULL,
            estimated_eps         REAL,
            eps_method            TEXT,
            importance            INTEGER,
            fiscal_period         TEXT,
            fiscal_year           TEXT,
            benzinga_id           TEXT,
            raw_json              TEXT    NOT NULL,
            UNIQUE(ticker, scheduled_report_date, capture_timestamp_utc)
        )
        """
    )


def _fetch_calendar(
    api_key: str,
    gte: str,
    lte: str,
    *,
    timeout: float = 15.0,
) -> list:
    """Pull the broad forward earnings calendar in ONE call.

    STEP 0 confirmed a bare date range with no ticker is accepted.  May raise
    (network / HTTP / parse) — the caller's fail-safe wrapper handles it.
    """
    params = {
        "date.gte": gte,
        "date.lte": lte,
        "sort": "date.asc",
        "limit": 1000,
        "apiKey": api_key,
    }
    resp = requests.get(_BASE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("results") or []


def run(
    *,
    db_path: "str | None" = None,
    api_key: "str | None" = None,
    fetch_fn=None,
    now: "datetime | None" = None,
) -> dict:
    """Capture body — may raise freely (``main`` wraps it).

    Args:
        db_path:  Target SQLite path.  Defaults to DB_PATH env / VPS path.
        api_key:  POLYGON_API_KEY.  Defaults to the env var.
        fetch_fn: Zero-arg callable returning the calendar rows (for tests).
                  Defaults to a live ``_fetch_calendar`` call.
        now:      Injectable UTC ``datetime`` (for tests).  Defaults to now.

    Returns:
        dict {fetched, inserted, dropped, early, capture_ts}.
    """
    now = now or datetime.now(timezone.utc)
    today = now.date()
    tomorrow = today + timedelta(days=1)
    gte = tomorrow.isoformat()
    lte = (tomorrow + timedelta(days=1)).isoformat()  # tomorrow + 1 day
    capture_ts = now.isoformat()

    api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY missing")

    fetch = fetch_fn or (lambda: _fetch_calendar(api_key, gte, lte))
    events = fetch()
    fetched = len(events)
    logger.info(
        "fetched %d forward calendar rows (gte=%s lte=%s)", fetched, gte, lte
    )

    # Bound the universe: keep the top PREANN_MAX_NAMES by importance (desc).
    dropped = 0
    if fetched > PREANN_MAX_NAMES:
        events = sorted(
            events, key=lambda r: (r.get("importance") or 0), reverse=True
        )[:PREANN_MAX_NAMES]
        dropped = fetched - PREANN_MAX_NAMES
        logger.warning(
            "calendar returned %d rows > cap %d — kept top %d by importance, "
            "DROPPED %d", fetched, PREANN_MAX_NAMES, PREANN_MAX_NAMES, dropped
        )

    db_path = db_path or os.environ.get("DB_PATH", _DB_PATH_DEFAULT)
    inserted = 0
    early = 0

    # Private connection — writes ONLY this table.  One transaction, one
    # commit at the end: an exception mid-loop rolls back with no partial rows.
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        _ensure_table(conn)
        for rec in events:
            sched = rec.get("date")
            ticker = rec.get("ticker")
            if not ticker or not sched:
                logger.warning("skipping row missing ticker/date: %r", rec)
                continue

            # EARLY-REPORT flag.  gte=tomorrow should preclude this, but a
            # 'projected' date can be fuzzy.  WARN + still record; downstream
            # analysis derives reliability from scheduled_report_date vs
            # capture_timestamp_utc (no dedicated column — the relationship is
            # self-describing).
            try:
                if date.fromisoformat(str(sched)[:10]) < tomorrow:
                    early += 1
                    logger.warning(
                        "EARLY-REPORT %s sched=%s < tomorrow=%s — event may "
                        "report before T-1 capture; PIT snapshot may be "
                        "post-hoc for this row", ticker, sched, tomorrow
                    )
            except (ValueError, TypeError):
                pass

            fy = rec.get("fiscal_year")
            cur = conn.execute(
                f"""
                INSERT OR IGNORE INTO {_TABLE}
                    (ticker, scheduled_report_date, date_status,
                     capture_timestamp_utc, estimated_eps, eps_method,
                     importance, fiscal_period, fiscal_year, benzinga_id,
                     raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    sched,
                    rec.get("date_status"),
                    capture_ts,
                    # VERBATIM — Benzinga's native units, NO ×100, NO surprise
                    # computation.  raw_json below holds the full record.
                    rec.get("estimated_eps"),
                    rec.get("eps_method"),
                    rec.get("importance"),
                    rec.get("fiscal_period"),
                    str(fy) if fy is not None else None,
                    rec.get("benzinga_id"),
                    json.dumps(rec, sort_keys=True, default=str),
                ),
            )
            inserted += cur.rowcount  # 1 on insert, 0 if UNIQUE-ignored
        conn.commit()
    finally:
        conn.close()

    logger.info(
        "capture done: fetched=%d inserted=%d dropped=%d early_report=%d "
        "capture_ts=%s", fetched, inserted, dropped, early, capture_ts
    )
    return {
        "fetched": fetched,
        "inserted": inserted,
        "dropped": dropped,
        "early": early,
        "capture_ts": capture_ts,
    }


def main() -> int:
    """Fail-safe entry point.  ALWAYS returns 0 — recorded-only, isolated."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        run()
    except Exception:  # noqa: BLE001 — intentional total swallow
        # Recorded-only, fully isolated job: ANY failure is logged and
        # swallowed.  This job can NEVER raise into anything else.
        logger.exception(
            "capture_preann_estimates FAILED — swallowed (recorded-only, "
            "exit 0)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
