"""
Tests for the Q2 earnings source eval capture stack (R-spec 2026-08-24).

Standalone, capture-only, NON-LIVE job — these tests pin its contract:

* Schema — the three R-spec tables (cal_capture / eps_capture / run_log)
  are created with exactly the specified columns, on the script's OWN DB
  (never news_trading.db; path injected via a tmp sqlite file here).
* Field mapping — both providers' fixture payload shapes map to the
  normalized columns, incl. eps_method, date_status, provider_status_raw,
  time_of_day and days_ahead.
* first_seen_ts — set on the FIRST capture where actual_eps is non-null
  for (provider, ticker, report_date); carried forward afterwards.
* Fail-soft — one provider raising never stops the other; BOTH outcomes
  land in run_log.
* Retry cap — initial attempt + max 2 backoff retries (3 calls total);
  a further retry never fires.
* Universe — sourced at runtime from config/watchlist.yaml (>0 US names)
  and config.settings.PEAD_TICKERS (exactly 15) — nothing hard-coded.
* Secrets — error strings are sanitized (no query params / API keys) and
  run_log stores endpoint paths only.

No network: fetchers are injected, or ``requests.get`` is monkeypatched.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from scripts import earnings_source_eval as ev

# Fixed "now" — deterministic date math throughout.
_NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
_NOW_EPS = datetime(2026, 8, 24, 23, 0, 0, tzinfo=timezone.utc)

_UNIVERSE = {"AAPL", "CASY", "TSLA"}

_NO_SLEEP = lambda _s: None  # noqa: E731


# ── Fixture payloads (real provider response shapes, both providers) ─

# Finnhub GET /calendar/earnings → {"earningsCalendar": [...]}
FINNHUB_CAL_ROWS = [
    {"date": "2026-09-03", "epsActual": None, "epsEstimate": 2.10,
     "hour": "amc", "quarter": 3, "revenueActual": None,
     "revenueEstimate": 90000000000, "symbol": "AAPL", "year": 2026},
    {"date": "2026-09-08", "epsActual": None, "epsEstimate": 4.55,
     "hour": "", "quarter": 2, "revenueActual": None,
     "revenueEstimate": 4200000000, "symbol": "CASY", "year": 2026},
    # Out-of-universe row — must be filtered out.
    {"date": "2026-09-05", "epsActual": None, "epsEstimate": 1.00,
     "hour": "bmo", "quarter": 3, "revenueActual": None,
     "revenueEstimate": 1, "symbol": "ZZZQ", "year": 2026},
]

# FMP GET /api/v3/earning_calendar → bare list
FMP_CAL_ROWS = [
    {"date": "2026-09-03", "symbol": "AAPL", "eps": None,
     "epsEstimated": 2.08, "time": "amc", "revenue": None,
     "revenueEstimated": 89500000000, "fiscalDateEnding": "2026-08-31",
     "updatedFromDate": "2026-08-20"},
    {"date": "2026-09-08", "symbol": "CASY", "eps": None,
     "epsEstimated": 4.60, "time": "--", "revenue": None,
     "revenueEstimated": 4150000000, "fiscalDateEnding": "2026-07-31",
     "updatedFromDate": "2026-08-20"},
    {"date": "2026-09-05", "symbol": "ZZZQ", "eps": None,
     "epsEstimated": 1.0, "time": "bmo", "revenue": None,
     "revenueEstimated": 1, "fiscalDateEnding": "2026-08-31",
     "updatedFromDate": "2026-08-20"},
]

FINNHUB_EPS_ROWS = [
    # Reported today (capture day) — available_same_day = 1.
    {"date": "2026-08-24", "epsActual": 2.35, "epsEstimate": 2.10,
     "hour": "bmo", "quarter": 3, "symbol": "AAPL", "year": 2026},
    # Reported yesterday, actual present now — available_same_day = 0.
    {"date": "2026-08-23", "epsActual": 4.80, "epsEstimate": 4.55,
     "hour": "amc", "quarter": 2, "symbol": "CASY", "year": 2026},
    # Actual still missing — available_same_day / first_seen_ts NULL.
    {"date": "2026-08-24", "epsActual": None, "epsEstimate": 1.50,
     "hour": "amc", "quarter": 3, "symbol": "TSLA", "year": 2026},
]

FMP_EPS_ROWS = [
    {"date": "2026-08-24", "symbol": "AAPL", "eps": 2.34,
     "epsEstimated": 2.08, "time": "bmo", "revenue": 91000000000,
     "revenueEstimated": 89500000000, "fiscalDateEnding": "2026-08-31",
     "updatedFromDate": "2026-08-24"},
]


def _fetcher(rows, status=200, headers=None):
    return lambda frm, to: (rows, status, headers or {})


@pytest.fixture
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "earnings_source_eval.db")
    yield c
    c.close()


# ── Schema ───────────────────────────────────────────────────────────

def _columns(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def test_schema_creates_exact_rspec_tables(conn):
    ev.ensure_schema(conn)
    ev.ensure_schema(conn)  # idempotent
    assert _columns(conn, "cal_capture") == [
        "capture_ts", "provider", "ticker", "report_date", "time_of_day",
        "date_status", "days_ahead", "raw_payload_hash",
        "provider_status_raw"]
    assert _columns(conn, "eps_capture") == [
        "capture_ts", "provider", "ticker", "report_date", "estimate_eps",
        "actual_eps", "surprise_pct", "eps_method", "available_same_day",
        "first_seen_ts"]
    assert _columns(conn, "run_log") == [
        "run_ts", "provider", "endpoint", "http_status", "rows_returned",
        "rate_limit_headers", "error_text"]


# ── cal mode: field mapping, days_ahead, universe filter ─────────────

def test_cal_capture_field_mapping_both_providers(conn):
    fetchers = {"finnhub": _fetcher(FINNHUB_CAL_ROWS),
                "fmp": _fetcher(FMP_CAL_ROWS)}
    summary = ev.run_capture("cal", conn, now=_NOW, fetchers=fetchers,
                             universe=_UNIVERSE, sleep_fn=_NO_SLEEP)
    assert summary["providers"]["finnhub"]["ok"]
    assert summary["providers"]["fmp"]["ok"]

    rows = conn.execute(
        "SELECT provider, ticker, report_date, time_of_day, date_status,"
        " days_ahead, raw_payload_hash, provider_status_raw"
        " FROM cal_capture ORDER BY provider, ticker").fetchall()
    # ZZZQ filtered out for both providers → 2 rows each.
    assert len(rows) == 4
    by_key = {(r[0], r[1]): r for r in rows}

    fh_aapl = by_key[("finnhub", "AAPL")]
    assert fh_aapl[2] == "2026-09-03"
    assert fh_aapl[3] == "amc"
    assert fh_aapl[4] == "scheduled"          # session known → scheduled
    assert fh_aapl[5] == 10                   # days_ahead from 2026-08-24
    assert len(fh_aapl[6]) == 64              # sha256 hex of raw row
    assert fh_aapl[7] == "amc"                # provider_status_raw verbatim

    fh_casy = by_key[("finnhub", "CASY")]
    assert fh_casy[3] == "unknown"            # empty hour → unknown
    assert fh_casy[4] == "tentative"          # no session → tentative
    assert fh_casy[7] == ""                   # raw field preserved

    fmp_casy = by_key[("fmp", "CASY")]
    assert fmp_casy[3] == "unknown"           # "--" → unknown
    assert fmp_casy[4] == "tentative"
    assert fmp_casy[7] == "--"
    assert fmp_casy[5] == 15

    # Distinct raw payloads → distinct hashes.
    assert fh_aapl[6] != by_key[("fmp", "AAPL")][6]


def test_cal_run_log_rows_and_no_query_params(conn):
    fetchers = {"finnhub": _fetcher(FINNHUB_CAL_ROWS,
                                    headers={"X-RateLimit-Remaining": "29",
                                             "X-RateLimit-Limit": "30",
                                             "Content-Type": "application/json"}),
                "fmp": _fetcher(FMP_CAL_ROWS)}
    ev.run_capture("cal", conn, now=_NOW, fetchers=fetchers,
                   universe=_UNIVERSE, sleep_fn=_NO_SLEEP)
    logs = conn.execute(
        "SELECT provider, endpoint, http_status, rows_returned,"
        " rate_limit_headers, error_text FROM run_log ORDER BY provider").fetchall()
    assert len(logs) == 2
    fh = [l for l in logs if l[0] == "finnhub"][0]
    assert "?" not in fh[1] and "apikey" not in fh[1] and "token" not in fh[1]
    assert fh[2] == 200
    assert fh[3] == len(FINNHUB_CAL_ROWS)     # pre-filter API row count
    rl = json.loads(fh[4])
    assert rl == {"X-RateLimit-Remaining": "29", "X-RateLimit-Limit": "30"}
    assert fh[5] is None


# ── eps mode: field mapping incl. eps_method / available_same_day ────

def test_eps_capture_field_mapping_both_providers(conn):
    fetchers = {"finnhub": _fetcher(FINNHUB_EPS_ROWS),
                "fmp": _fetcher(FMP_EPS_ROWS)}
    ev.run_capture("eps", conn, now=_NOW_EPS, fetchers=fetchers,
                   universe=_UNIVERSE, sleep_fn=_NO_SLEEP)
    rows = conn.execute(
        "SELECT provider, ticker, report_date, estimate_eps, actual_eps,"
        " surprise_pct, eps_method, available_same_day, first_seen_ts"
        " FROM eps_capture ORDER BY provider, ticker").fetchall()
    assert len(rows) == 4
    by_key = {(r[0], r[1]): r for r in rows}

    fh_aapl = by_key[("finnhub", "AAPL")]
    assert fh_aapl[3] == 2.10 and fh_aapl[4] == 2.35
    assert fh_aapl[5] == pytest.approx((2.35 - 2.10) / 2.10 * 100)
    assert fh_aapl[6] == "finnhub_calendar:epsActual/epsEstimate"
    assert fh_aapl[7] == 1                    # reported on capture day
    assert fh_aapl[8] == _NOW_EPS.isoformat()

    fh_casy = by_key[("finnhub", "CASY")]
    assert fh_casy[7] == 0                    # reported yesterday, seen today

    fh_tsla = by_key[("finnhub", "TSLA")]
    assert fh_tsla[4] is None                 # no actual yet
    assert fh_tsla[5] is None and fh_tsla[7] is None and fh_tsla[8] is None

    fmp_aapl = by_key[("fmp", "AAPL")]
    assert fmp_aapl[3] == 2.08 and fmp_aapl[4] == 2.34
    assert fmp_aapl[6] == "fmp_calendar:eps/epsEstimated"


def test_first_seen_ts_set_once_and_carried_forward(conn):
    fetchers = {"finnhub": _fetcher(FINNHUB_EPS_ROWS), "fmp": _fetcher([])}
    ev.run_capture("eps", conn, now=_NOW_EPS, fetchers=fetchers,
                   universe=_UNIVERSE, sleep_fn=_NO_SLEEP)
    # Second capture of the SAME (provider, ticker, report_date), next day.
    later = datetime(2026, 8, 25, 23, 0, 0, tzinfo=timezone.utc)
    ev.run_capture("eps", conn, now=later, fetchers=fetchers,
                   universe=_UNIVERSE, sleep_fn=_NO_SLEEP)

    seen = conn.execute(
        "SELECT capture_ts, first_seen_ts FROM eps_capture"
        " WHERE provider='finnhub' AND ticker='AAPL'"
        " AND report_date='2026-08-24' ORDER BY rowid").fetchall()
    assert len(seen) == 2
    assert seen[0] == (_NOW_EPS.isoformat(), _NOW_EPS.isoformat())
    # Second capture keeps the FIRST first_seen_ts, not its own capture_ts.
    assert seen[1][0] == later.isoformat()
    assert seen[1][1] == _NOW_EPS.isoformat()


# ── Fail-soft + retries ──────────────────────────────────────────────

def test_fail_soft_one_provider_raising_never_stops_the_other(conn):
    def _boom(frm, to):
        raise RuntimeError("finnhub down")

    fetchers = {"finnhub": _boom, "fmp": _fetcher(FMP_CAL_ROWS)}
    summary = ev.run_capture("cal", conn, now=_NOW, fetchers=fetchers,
                             universe=_UNIVERSE, sleep_fn=_NO_SLEEP)
    assert summary["providers"]["finnhub"]["ok"] is False
    assert summary["providers"]["fmp"]["ok"] is True

    # FMP rows still written.
    assert conn.execute(
        "SELECT COUNT(*) FROM cal_capture WHERE provider='fmp'"
    ).fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM cal_capture WHERE provider='finnhub'"
    ).fetchone()[0] == 0
    # BOTH outcomes land in run_log.
    logs = {r[0]: r for r in conn.execute(
        "SELECT provider, error_text, rows_returned FROM run_log")}
    assert len(logs) == 2
    assert "finnhub down" in logs["finnhub"][1]
    assert logs["fmp"][1] is None and logs["fmp"][2] == 3


def test_retry_cap_initial_plus_two_retries_then_stop(conn):
    calls = []
    sleeps = []

    def _always_fails(frm, to):
        calls.append(1)
        raise RuntimeError("still down")

    fetchers = {"finnhub": _always_fails, "fmp": _fetcher([])}
    ev.run_capture("cal", conn, now=_NOW, fetchers=fetchers,
                   universe=_UNIVERSE, sleep_fn=sleeps.append)
    # Max 2 retries after the initial attempt → exactly 3 calls, 2 backoff
    # sleeps; a 3rd retry never fires.
    assert len(calls) == 3
    assert len(sleeps) == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM run_log WHERE provider='finnhub'"
        " AND error_text IS NOT NULL").fetchone()[0] == 1


def test_retry_recovers_without_run_log_error(conn):
    attempts = []

    def _flaky(frm, to):
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("blip")
        return FMP_CAL_ROWS, 200, {}

    fetchers = {"finnhub": _fetcher([]), "fmp": _flaky}
    summary = ev.run_capture("cal", conn, now=_NOW, fetchers=fetchers,
                             universe=_UNIVERSE, sleep_fn=_NO_SLEEP)
    assert len(attempts) == 2
    assert summary["providers"]["fmp"]["ok"] is True
    assert conn.execute(
        "SELECT error_text FROM run_log WHERE provider='fmp'"
    ).fetchone()[0] is None


# ── Live fetchers: payload shapes + key placement (requests mocked) ──

class _FakeResp:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.status_code = 200
        self.headers = headers or {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_finnhub_key_in_header_not_url(monkeypatch):
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured.update(url=url, params=params, headers=headers)
        return _FakeResp({"earningsCalendar": FINNHUB_CAL_ROWS},
                         {"X-RateLimit-Remaining": "29"})

    monkeypatch.setattr(ev.requests, "get", fake_get)
    rows, status, headers = ev.fetch_finnhub("SECRET", "2026-08-24", "2026-09-23")
    assert rows == FINNHUB_CAL_ROWS and status == 200
    assert captured["headers"] == {"X-Finnhub-Token": "SECRET"}
    assert "SECRET" not in captured["url"]
    assert "SECRET" not in json.dumps(captured["params"])


def test_fetch_fmp_parses_bare_list(monkeypatch):
    monkeypatch.setattr(ev.requests, "get",
                        lambda *a, **kw: _FakeResp(FMP_CAL_ROWS))
    rows, status, _ = ev.fetch_fmp("SECRET", "2026-08-24", "2026-09-23")
    assert rows == FMP_CAL_ROWS and status == 200


def test_sanitize_error_strips_keys_and_query_strings():
    msg = ("500 Server Error for url: https://financialmodelingprep.com/api/"
           "v3/earning_calendar?from=2026-08-24&to=2026-09-23&apikey=SECRET123")
    out = ev._sanitize_error(msg)
    assert "SECRET123" not in out
    assert "from=2026-08-24" not in out


# ── Universe loaders (real repo configs, offline) ────────────────────

def test_universe_loader_finds_us_and_exactly_15_pead_names():
    us = ev.load_us_watchlist()
    assert len(us) > 0
    assert all(isinstance(t, str) and t for t in us)

    pead = ev.load_pead_tickers()
    assert len(pead) == 15

    universe = ev.load_universe()
    assert set(us) <= universe and set(pead) <= universe
