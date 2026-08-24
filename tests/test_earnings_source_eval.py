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
* Alpha Vantage — CSV calendar normalization; per-ticker eps path with
  the free-tier daily budget (shortfall WARNING + run_log.error_text);
  HTTP-200 JSON-note rate limiting treated as retriable; eps_method
  'unknown'; reportTime mapping; fail-soft vs the other two providers.
* NEWS_SENTIMENT probe — stdout only, summary fields present, no DB
  write, exit 1 on failure.

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


def _av_eps_fetcher(payloads, status=200, headers=None, calls=None):
    """Per-ticker AV eps fetcher: payloads maps ticker → payload."""
    def _fetch(ticker):
        if calls is not None:
            calls.append(ticker)
        return payloads[ticker], status, headers or {}
    return _fetch


# Empty stand-ins so tests focused on one provider still satisfy the
# three-provider loop.
_AV_NONE_CAL = _fetcher([])
_AV_NONE_EPS = lambda ticker: ({"symbol": ticker, "quarterlyEarnings": []}, 200, {})  # noqa: E731


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
                "fmp": _fetcher(FMP_CAL_ROWS),
                "alphavantage": _AV_NONE_CAL}
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
                "fmp": _fetcher(FMP_CAL_ROWS),
                "alphavantage": _AV_NONE_CAL}
    ev.run_capture("cal", conn, now=_NOW, fetchers=fetchers,
                   universe=_UNIVERSE, sleep_fn=_NO_SLEEP)
    logs = conn.execute(
        "SELECT provider, endpoint, http_status, rows_returned,"
        " rate_limit_headers, error_text FROM run_log ORDER BY provider").fetchall()
    assert len(logs) == 3
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
                "fmp": _fetcher(FMP_EPS_ROWS),
                "alphavantage": _AV_NONE_EPS}
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
    fetchers = {"finnhub": _fetcher(FINNHUB_EPS_ROWS), "fmp": _fetcher([]),
                "alphavantage": _AV_NONE_EPS}
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

    fetchers = {"finnhub": _boom, "fmp": _fetcher(FMP_CAL_ROWS),
                "alphavantage": _AV_NONE_CAL}
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
    assert len(logs) == 3
    assert "finnhub down" in logs["finnhub"][1]
    assert logs["fmp"][1] is None and logs["fmp"][2] == 3


def test_retry_cap_initial_plus_two_retries_then_stop(conn):
    calls = []
    sleeps = []

    def _always_fails(frm, to):
        calls.append(1)
        raise RuntimeError("still down")

    fetchers = {"finnhub": _always_fails, "fmp": _fetcher([]),
                "alphavantage": _AV_NONE_CAL}
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

    fetchers = {"finnhub": _fetcher([]), "fmp": _flaky,
                "alphavantage": _AV_NONE_CAL}
    summary = ev.run_capture("cal", conn, now=_NOW, fetchers=fetchers,
                             universe=_UNIVERSE, sleep_fn=_NO_SLEEP)
    assert len(attempts) == 2
    assert summary["providers"]["fmp"]["ok"] is True
    assert conn.execute(
        "SELECT error_text FROM run_log WHERE provider='fmp'"
    ).fetchone()[0] is None


# ── Live fetchers: payload shapes + key placement (requests mocked) ──

class _FakeResp:
    def __init__(self, payload, headers=None, text=None):
        self._payload = payload
        self.status_code = 200
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(payload)

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


# ── Alpha Vantage fixtures ───────────────────────────────────────────

# EARNINGS_CALENDAR CSV rows as csv.DictReader yields them (all strings).
AV_CAL_ROWS = [
    {"symbol": "CASY", "name": "Caseys General Stores Inc",
     "reportDate": "2026-09-08", "fiscalDateEnding": "2026-07-31",
     "estimate": "4.50", "currency": "USD"},
    # No reportDate → date_status "absent".
    {"symbol": "AAPL", "name": "Apple Inc", "reportDate": "",
     "fiscalDateEnding": "2026-09-30", "estimate": "", "currency": "USD"},
    # Out-of-universe row — must be filtered out.
    {"symbol": "ZZZQ", "name": "Zzz Corp", "reportDate": "2026-09-05",
     "fiscalDateEnding": "2026-08-31", "estimate": "1.00", "currency": "USD"},
]

AV_CAL_CSV = (
    "symbol,name,reportDate,fiscalDateEnding,estimate,currency\r\n"
    "CASY,Caseys General Stores Inc,2026-09-08,2026-07-31,4.50,USD\r\n"
)

# EARNINGS payload (per-ticker; whole history — only in-window rows count).
AV_EPS_AAPL = {
    "symbol": "AAPL",
    "annualEarnings": [{"fiscalDateEnding": "2025-09-30",
                        "reportedEPS": "7.4"}],
    "quarterlyEarnings": [
        {"fiscalDateEnding": "2026-06-30", "reportedDate": "2026-08-24",
         "reportedEPS": "2.35", "estimatedEPS": "2.10", "surprise": "0.25",
         "surprisePercentage": "11.9", "reportTime": "post-market"},
        {"fiscalDateEnding": "2020-06-30", "reportedDate": "2020-07-30",
         "reportedEPS": "0.65", "estimatedEPS": "0.51", "surprise": "0.14",
         "surprisePercentage": "27.45", "reportTime": "pre-market"},
    ],
}

AV_RATE_LIMIT_NOTE = {
    "Information": "Thank you for using Alpha Vantage! Our standard API "
                   "rate limit is 25 requests per day."
}

AV_NEWS_FIXTURE = {
    "items": "2",
    "sentiment_score_definition": "x <= -0.35: Bearish; ...",
    "relevance_score_definition": "0 < x <= 1, higher means more relevant",
    "feed": [
        {"title": "Apple beats on Q3 earnings",
         "url": "https://example.com/a",
         "time_published": "20260824T101500", "authors": ["Reporter A"],
         "summary": "Apple reported...", "source": "Benzinga",
         "overall_sentiment_score": 0.25,
         "overall_sentiment_label": "Somewhat-Bullish",
         "ticker_sentiment": [
             {"ticker": "AAPL", "relevance_score": "0.85",
              "ticker_sentiment_score": "0.31",
              "ticker_sentiment_label": "Somewhat-Bullish"}]},
        {"title": "Tech roundup", "url": "https://example.com/b",
         "time_published": "20260823T220000",
         "overall_sentiment_label": "Neutral",
         "ticker_sentiment": [
             {"ticker": "AAPL", "relevance_score": "0.42",
              "ticker_sentiment_label": "Neutral"},
             {"ticker": "MSFT", "relevance_score": "0.10",
              "ticker_sentiment_label": "Neutral"}]},
    ],
}


# ── Alpha Vantage: cal normalization ─────────────────────────────────

def test_av_cal_normalization_universe_filter_and_date_status(conn):
    fetchers = {"finnhub": _fetcher([]), "fmp": _fetcher([]),
                "alphavantage": _fetcher(AV_CAL_ROWS)}
    summary = ev.run_capture("cal", conn, now=_NOW, fetchers=fetchers,
                             universe=_UNIVERSE, sleep_fn=_NO_SLEEP)
    assert summary["providers"]["alphavantage"]["ok"]
    assert summary["providers"]["alphavantage"]["rows_returned"] == 3

    rows = conn.execute(
        "SELECT ticker, report_date, time_of_day, date_status, days_ahead,"
        " provider_status_raw FROM cal_capture"
        " WHERE provider='alphavantage' ORDER BY ticker").fetchall()
    assert len(rows) == 2                     # ZZZQ filtered out
    aapl, casy = rows[0], rows[1]

    assert casy[0] == "CASY"
    assert casy[1] == "2026-09-08"
    assert casy[2] == "unknown"               # AV CSV has no session field
    assert casy[3] == "scheduled"             # reportDate present
    assert casy[4] == 15                      # days_ahead from 2026-08-24
    raw = json.loads(casy[5])                 # raw CSV row as-is
    assert raw["estimate"] == "4.50" and raw["fiscalDateEnding"] == "2026-07-31"

    assert aapl[1] == "" and aapl[3] == "absent" and aapl[4] is None


def test_fetch_alphavantage_calendar_parses_csv_and_detects_note(monkeypatch):
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured.update(url=url, params=params)
        return _FakeResp(None, text=AV_CAL_CSV)

    monkeypatch.setattr(ev.requests, "get", fake_get)
    rows, status, _ = ev.fetch_alphavantage_calendar("SECRET")
    assert status == 200
    assert rows == [{"symbol": "CASY", "name": "Caseys General Stores Inc",
                     "reportDate": "2026-09-08",
                     "fiscalDateEnding": "2026-07-31",
                     "estimate": "4.50", "currency": "USD"}]
    assert captured["params"]["function"] == "EARNINGS_CALENDAR"
    assert captured["params"]["horizon"] == "3month"

    # Rate limiting: HTTP 200 with a JSON note instead of CSV.
    monkeypatch.setattr(
        ev.requests, "get",
        lambda *a, **kw: _FakeResp(None, text=json.dumps(AV_RATE_LIMIT_NOTE)))
    with pytest.raises(ev.AlphaVantageRateLimitError):
        ev.fetch_alphavantage_calendar("SECRET")


# ── Alpha Vantage: eps normalization ─────────────────────────────────

def test_normalize_av_report_time():
    assert ev.normalize_av_report_time("pre-market") == "bmo"
    assert ev.normalize_av_report_time("post-market") == "amc"
    assert ev.normalize_av_report_time("") == "unknown"
    assert ev.normalize_av_report_time(None) == "unknown"
    assert ev.normalize_av_report_time("whatever") == "unknown"


def test_av_eps_normalization_window_method_first_seen(conn):
    calls = []
    fetchers = {"finnhub": _fetcher([]), "fmp": _fetcher([]),
                "alphavantage": _av_eps_fetcher({"AAPL": AV_EPS_AAPL},
                                                calls=calls)}
    ev.run_capture("eps", conn, now=_NOW_EPS, fetchers=fetchers,
                   universe={"AAPL"}, av_priority=["AAPL"],
                   sleep_fn=_NO_SLEEP)
    assert calls == ["AAPL"]

    rows = conn.execute(
        "SELECT report_date, estimate_eps, actual_eps, surprise_pct,"
        " eps_method, available_same_day, first_seen_ts FROM eps_capture"
        " WHERE provider='alphavantage'").fetchall()
    # Only the in-window (today-1..today) quarter — the 2020 row dropped.
    assert len(rows) == 1
    r = rows[0]
    assert r[0] == "2026-08-24"
    assert r[1] == 2.10 and r[2] == 2.35      # AV strings → floats
    assert r[3] == 11.9                       # AV's own surprisePercentage
    assert r[4] == "unknown"                  # no gaap/adj distinction
    assert r[5] == 1                          # reported on capture day
    assert r[6] == _NOW_EPS.isoformat()

    # reportTime mapping is computed (informational — eps_capture has no
    # time_of_day column in the R-spec schema).
    norms = ev.normalize_av_eps_rows(
        AV_EPS_AAPL, "2026-08-23", "2026-08-24", _NOW_EPS.date())
    assert norms[0]["time_of_day"] == "amc"   # post-market → amc

    # Second capture next day: first_seen_ts carried forward.
    later = datetime(2026, 8, 25, 23, 0, 0, tzinfo=timezone.utc)
    ev.run_capture("eps", conn, now=later, fetchers=fetchers,
                   universe={"AAPL"}, av_priority=["AAPL"],
                   sleep_fn=_NO_SLEEP)
    seen = conn.execute(
        "SELECT first_seen_ts FROM eps_capture WHERE provider='alphavantage'"
        " AND report_date='2026-08-24' ORDER BY rowid").fetchall()
    assert [s[0] for s in seen] == [_NOW_EPS.isoformat()] * 2


# ── Alpha Vantage: budget shortfall + note-style rate limiting ───────

def test_av_budget_shortfall_warning_and_error_text(conn, caplog):
    calls = []
    fetchers = {
        "finnhub": _fetcher([]), "fmp": _fetcher([]),
        "alphavantage": _av_eps_fetcher(
            {"CASY": {"symbol": "CASY", "quarterlyEarnings": []},
             "TSLA": {"symbol": "TSLA", "quarterlyEarnings": []}},
            calls=calls),
    }
    with caplog.at_level("WARNING", logger="earnings_source_eval"):
        summary = ev.run_capture(
            "eps", conn, now=_NOW_EPS, fetchers=fetchers,
            universe=_UNIVERSE, av_priority=["CASY", "TSLA", "AAPL"],
            av_daily_limit=3, sleep_fn=_NO_SLEEP)

    # Budget = 3 daily - 1 reserved for cal = 2 < 3 tickers → AAPL skipped.
    assert calls == ["CASY", "TSLA"]
    assert summary["providers"]["alphavantage"]["skipped"] == 1

    joined = " ".join(r.getMessage() for r in caplog.records
                      if r.levelname == "WARNING")
    assert "3 tickers > remaining daily budget 2" in joined
    assert "AAPL" in joined

    err = conn.execute(
        "SELECT error_text FROM run_log WHERE provider='alphavantage'"
    ).fetchone()[0]
    assert "budget shortfall" in err
    assert "universe 3 > remaining budget 2" in err
    assert "AAPL" in err


def test_av_note_rate_limit_is_retriable_and_recorded(conn):
    calls = []

    def _rate_limited(ticker):
        calls.append(ticker)
        return AV_RATE_LIMIT_NOTE, 200, {}

    fetchers = {"finnhub": _fetcher([]), "fmp": _fetcher([]),
                "alphavantage": _rate_limited}
    summary = ev.run_capture("eps", conn, now=_NOW_EPS, fetchers=fetchers,
                             universe={"AAPL"}, av_priority=["AAPL"],
                             sleep_fn=_NO_SLEEP)
    # HTTP 200 + note is a retriable failure: initial + 2 retries.
    assert calls == ["AAPL"] * 3
    assert summary["providers"]["alphavantage"]["ok"] is False

    log = conn.execute(
        "SELECT rows_returned, rate_limit_headers, error_text FROM run_log"
        " WHERE provider='alphavantage'").fetchone()
    assert log[0] == 0
    assert "25 requests per day" in json.loads(log[1])["av_note"]
    assert "rate-limit note" in log[2]


def test_av_fail_soft_other_providers_unaffected(conn):
    def _boom(frm, to):
        raise RuntimeError("av boom")

    fetchers = {"finnhub": _fetcher(FINNHUB_CAL_ROWS),
                "fmp": _fetcher(FMP_CAL_ROWS), "alphavantage": _boom}
    summary = ev.run_capture("cal", conn, now=_NOW, fetchers=fetchers,
                             universe=_UNIVERSE, sleep_fn=_NO_SLEEP)
    assert summary["providers"]["alphavantage"]["ok"] is False
    assert summary["providers"]["finnhub"]["ok"] is True
    assert summary["providers"]["fmp"]["ok"] is True
    assert conn.execute(
        "SELECT COUNT(*) FROM cal_capture WHERE provider IN ('finnhub','fmp')"
    ).fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM run_log").fetchone()[0] == 3
    assert "av boom" in conn.execute(
        "SELECT error_text FROM run_log WHERE provider='alphavantage'"
    ).fetchone()[0]


# ── NEWS_SENTIMENT probe (stdout only, no DB) ────────────────────────

def test_news_probe_summary_and_no_db_write(tmp_path, monkeypatch, capsys):
    db = tmp_path / "probe-should-not-exist.db"
    monkeypatch.setenv("EARNINGS_EVAL_DB", str(db))

    rc = ev.run_news_probe("AAPL",
                           fetch_fn=lambda: (AV_NEWS_FIXTURE, 200, {}))
    out = capsys.readouterr().out
    assert rc == 0
    assert '"feed"' in out                    # raw JSON pretty-printed
    assert "articles: 2" in out
    assert "date_range: 20260823T220000 .. 20260824T101500" in out
    assert "fields_present" in out
    assert "title: 2/2" in out
    assert "authors: 1/2" in out              # field presence per article
    assert "ticker_relevance (AAPL): n=2" in out
    assert "min=0.4200" in out and "max=0.8500" in out
    assert '"Somewhat-Bullish": 1' in out and '"Neutral": 1' in out
    assert not db.exists()                    # stdout ONLY — no DB write


def test_news_probe_failure_exits_1(capsys):
    def _auth_fail():
        raise RuntimeError("401 unauthorized for url: https://x?apikey=SECRET")

    assert ev.run_news_probe("AAPL", fetch_fn=_auth_fail) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "SECRET" not in out

    # Provider note (rate limit) also fails the probe.
    assert ev.run_news_probe(
        "AAPL", fetch_fn=lambda: (AV_RATE_LIMIT_NOTE, 200, {})) == 1
    assert "provider note" in capsys.readouterr().out
