#!/usr/bin/env python3
"""Q2 earnings source evaluation — capture-only (R-spec 2026-08-24).

Standalone, NON-LIVE evaluation stack that captures earnings-calendar and
EPS-actuals data from four free providers (Finnhub + FMP + Alpha Vantage
+ yfinance) into its OWN SQLite database, so the sources can later be
compared for calendar accuracy and actuals latency.

Isolation contract (non-negotiable)
-----------------------------------
* Imports NOTHING from the live trading path: no ``storage/database.py``,
  no ``orchestrator/``, no ``scheduler/daily_runner.py``.  The only repo
  imports are read-only universe configs (``config/watchlist.yaml`` parsed
  directly, ``config.settings.PEAD_TICKERS`` — a constants module).
* Writes ONLY its own DB (``/home/trading/trading-data/
  earnings_source_eval.db``; override via ``EARNINGS_EVAL_DB`` for tests /
  local runs).  Never touches ``news_trading.db``.
* Fail-soft per provider: one provider's exception never stops the other;
  BOTH outcomes land in ``run_log``.  The process always exits 0.

Modes
-----
``--mode cal``  (12:00 UTC timer) — calendar capture, one RANGE call per
    provider covering today-2 .. today+30d, filtered to the universe.
``--mode eps``  (23:00 UTC timer) — actuals capture, one RANGE call per
    provider covering today-2 .. today, filtered to the universe.
    Alpha Vantage has no range actuals endpoint — its EARNINGS call is
    per-ticker, so eps mode walks the universe in priority order (PEAD
    first, then US watchlist) under the free-tier daily call budget
    (default 25/day, 1 reserved for the cal call); any shortfall is
    WARNING-logged with exact numbers and recorded in run_log.error_text.
``--mode news-probe --ticker T`` — one-shot Alpha Vantage NEWS_SENTIMENT
    probe: pretty-prints the raw JSON + a field summary to stdout ONLY
    (no DB write, no run_log row, no timer). Exit 1 on any failure.

Universe (sourced at runtime, never hard-coded)
-----------------------------------------------
* US watchlist: ``config/watchlist.yaml`` key ``us_tickers``.
* PEAD names:   ``config.settings.PEAD_TICKERS`` (15 names).
Rows whose symbol is not an exact match against the union are dropped.

Secrets discipline
------------------
API keys are read from env (``FINNHUB_API_KEY`` / ``FMP_API_KEY`` /
``ALPHA_VANTAGE_API_KEY``).  The Finnhub key travels in the
``X-Finnhub-Token`` header (never in the URL).  FMP and Alpha Vantage
require the key as a query param, so NO full URL is ever logged and
every error string is passed through ``_sanitize_error`` (strips query
strings and masks ``apikey=``/``token=`` values) before logging or storage.

Alpha Vantage quirks (handled explicitly)
-----------------------------------------
* EARNINGS_CALENDAR returns CSV (not JSON), covers ALL tickers in one
  call (horizon=3month) — parsed with csv.DictReader, filtered to the
  universe.
* Rate limiting arrives as HTTP 200 with a JSON "Note"/"Information"
  message, NOT a 429 — detected and treated as a retriable failure; any
  such note is captured into run_log.rate_limit_headers.
* EARNINGS gives no GAAP/adjusted distinction → eps_method="unknown".

Window coverage per provider (R spec 2026-08-26, [T-2, T+30])
-------------------------------------------------------------
* finnhub: from/to range params → full [T-2, T+30] cal, [T-2, T] eps.
* alphavantage: EARNINGS_CALENDAR is horizon-based and FORWARD-ONLY —
  it cannot return past dates; the limitation is recorded as a note in
  run_log.error_text on every cal run (T-2 coverage comes from the
  per-ticker EARNINGS endpoint in eps mode, filtered to [T-2, T]).
* fmp: 403 on the free tier, unchanged.
* yfinance: ``Ticker.calendar`` is forward-only (next report only) —
  same run_log note as AV; eps history filtered to [T-2, T].
days_ahead is stored as computed — negative values (report in the past)
are legal and never clamped.

yfinance (fourth provider, R spec 2026-08-26)
---------------------------------------------
The prod venv carries yfinance 0.2.58 whose earnings_dates endpoint is
broken, so yfinance data comes from an ISOLATED SUBPROCESS: the main
script executes ``scripts/_yf_eval_fetch.py`` with the interpreter at
``$YF_EVAL_PYTHON`` (default /home/trading/yfeval-venv/bin/python3,
created out of band) and parses ONE json object from its stdout.
yfinance is never imported here.  A missing interpreter, timeout
(120s), non-zero exit or malformed stdout is that provider's failure —
run_log row, other providers unaffected.  Single attempt, no retries
(a 120s-timeout retry would triple the run; the daily cadence is the
retry).  eps_method is "unknown" (no GAAP/adjusted distinction, same
as AV); time_of_day derives from the row timestamp's clock time when
present, else NULL.

run_log.client_version (R spec 2026-08-26)
------------------------------------------
Which client produced each run_log row: the three HTTP providers store
"requests <version>" (they all go through the requests library);
yfinance stores "yfinance <version>" as reported by the isolated
helper (NULL when the helper never got far enough to report one).

Retries
-------
Each HTTP provider call gets max 2 retries with exponential backoff
after the initial attempt (3 HTTP calls total), then the failure is
logged to ``run_log`` and the run continues with the other providers.
The yfinance subprocess is single-attempt (see above).
"""
from __future__ import annotations

# Allow manual CLI invocation from anywhere without PYTHONPATH=. (the
# systemd units set WorkingDirectory, an interactive shell doesn't).
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("earnings_source_eval")

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Own DB — NOT news_trading.db.  Env override for tests / local runs.
_DB_PATH_DEFAULT = "/home/trading/trading-data/earnings_source_eval.db"

# Capture windows (R spec 2026-08-26: widened to [T-2, T+30]).
_CAL_LOOKBACK_DAYS = 2   # cal mode: today-2 .. today+30d
_CAL_FORWARD_DAYS = 30
_EPS_LOOKBACK_DAYS = 2   # eps mode: today-2 .. today

_MAX_RETRIES = 2         # retries after the initial attempt (3 calls total)
_BACKOFF_BASE_S = 2.0    # 2s, then 4s
_HTTP_TIMEOUT_S = 20.0

# Endpoint identifiers stored in run_log — path only, NEVER a full URL
# with query params (the FMP / Alpha Vantage keys ride in the query string).
_FINNHUB_ENDPOINT = "finnhub.io/api/v1/calendar/earnings"
_FMP_ENDPOINT = "financialmodelingprep.com/api/v3/earning_calendar"
_AV_CAL_ENDPOINT = "alphavantage.co/query#EARNINGS_CALENDAR"
_AV_EPS_ENDPOINT = "alphavantage.co/query#EARNINGS"
_AV_NEWS_ENDPOINT = "alphavantage.co/query#NEWS_SENTIMENT"
_YF_CAL_ENDPOINT = "yfinance#Ticker.calendar"
_YF_EPS_ENDPOINT = "yfinance#Ticker.get_earnings_dates"

_FINNHUB_URL = "https://finnhub.io/api/v1/calendar/earnings"
_FMP_URL = "https://financialmodelingprep.com/api/v3/earning_calendar"
_AV_URL = "https://www.alphavantage.co/query"

_PROVIDERS = ("finnhub", "fmp", "alphavantage", "yfinance")

# run_log.client_version for the HTTP providers — they all speak
# through the requests library.  yfinance rows carry the version the
# isolated helper reports instead.
_HTTP_CLIENT_VERSION = f"requests {requests.__version__}"

# yfinance runs in an ISOLATED venv (prod's 0.2.58 earnings_dates is
# broken) — the helper is executed by this interpreter, never imported.
_YF_EVAL_PYTHON_DEFAULT = "/home/trading/yfeval-venv/bin/python3"
_YF_HELPER = _REPO_ROOT / "scripts" / "_yf_eval_fetch.py"
_YF_SUBPROCESS_TIMEOUT_S = 120.0

# Endpoint limitations vs the [T-2, T+30] target window — recorded in
# run_log.error_text on every successful cal run rather than silently
# narrowing (R spec 2026-08-26 item 3).
_WINDOW_NOTES = {
    ("alphavantage", "cal"):
        "cal endpoint forward-only; T-2 coverage comes from the "
        "per-ticker EARNINGS endpoint",
    ("yfinance", "cal"):
        "calendar endpoint forward-only (next report + estimate only); "
        "T-2 coverage comes from get_earnings_dates in eps mode",
}

# Alpha Vantage free-tier daily call budget.  The EARNINGS actuals
# endpoint is per-ticker, so eps mode spends one call per universe name;
# one call/day is reserved for the (same-day) EARNINGS_CALENDAR run.
_AV_DAILY_LIMIT = int(os.environ.get("ALPHA_VANTAGE_DAILY_LIMIT", "25"))
_AV_CAL_RESERVED_CALLS = 1


class AlphaVantageRateLimitError(RuntimeError):
    """AV signals rate limiting as HTTP 200 + a JSON note — retriable."""

    def __init__(self, note: str):
        super().__init__(f"alphavantage rate-limit note: {note}")
        self.note = note


def _av_note(payload) -> "str | None":
    """Extract AV's rate-limit/notice message from a JSON payload, if any."""
    if isinstance(payload, dict):
        for key in ("Note", "Information", "Error Message"):
            if payload.get(key):
                return str(payload[key])
    return None


# ── Secrets hygiene ──────────────────────────────────────────────────

def _sanitize_error(text: str) -> str:
    """Strip anything that could leak an API key from an error string.

    requests exceptions embed the full request URL (query params included),
    so every provider error is passed through here before it reaches a log
    line or the run_log table.
    """
    text = re.sub(r"(apikey|apiKey|token)=[^&\s'\"]+", r"\1=***", text)
    text = re.sub(r"\?\S+", "?<redacted>", text)
    return text


def _rate_limit_headers_json(headers) -> str:
    """Serialize whatever rate-limit-ish headers exist to a JSON string."""
    keep = {}
    for k, v in dict(headers or {}).items():
        kl = k.lower()
        if "ratelimit" in kl or "rate-limit" in kl or kl == "retry-after":
            keep[k] = v
    return json.dumps(keep, sort_keys=True)


# ── Universe (sourced at runtime from canonical repo configs) ────────

def load_us_watchlist(repo_root: Path = _REPO_ROOT) -> list[str]:
    """US watchlist from config/watchlist.yaml (key ``us_tickers``)."""
    path = repo_root / "config" / "watchlist.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    us = cfg.get("us_tickers") or []
    if not us:
        raise RuntimeError(f"no us_tickers found in {path}")
    return list(us)


def load_pead_tickers() -> list[str]:
    """The 15 PEAD names from config.settings.PEAD_TICKERS.

    config.settings is a constants module (env + literals) — importing it
    executes no trading-path code.
    """
    from config.settings import PEAD_TICKERS
    if not PEAD_TICKERS:
        raise RuntimeError("config.settings.PEAD_TICKERS is empty")
    return list(PEAD_TICKERS)


def load_universe() -> set[str]:
    return set(load_us_watchlist()) | set(load_pead_tickers())


# ── Schema ───────────────────────────────────────────────────────────

def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the three eval tables if absent (exact R-spec columns),
    then apply additive migrations (R Amendment A4: eps_capture.time_of_day;
    R spec 2026-08-26: run_log.client_version)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cal_capture (
            capture_ts          TEXT    NOT NULL,
            provider            TEXT    NOT NULL,
            ticker              TEXT    NOT NULL,
            report_date         TEXT    NOT NULL,
            time_of_day         TEXT,
            date_status         TEXT,
            days_ahead          INTEGER,
            raw_payload_hash    TEXT    NOT NULL,
            provider_status_raw TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eps_capture (
            capture_ts         TEXT    NOT NULL,
            provider           TEXT    NOT NULL,
            ticker             TEXT    NOT NULL,
            report_date        TEXT    NOT NULL,
            estimate_eps       REAL,
            actual_eps         REAL,
            surprise_pct       REAL,
            eps_method         TEXT,
            available_same_day INTEGER,
            first_seen_ts      TEXT
        )
        """
    )
    # R Amendment A4 (2026-08-25): persist AV's reportTime session.  SQLite
    # has no ADD COLUMN IF NOT EXISTS, so the migration is try/except on
    # the "duplicate column" error for DBs that already carry it.
    try:
        conn.execute("ALTER TABLE eps_capture ADD COLUMN time_of_day TEXT")
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_log (
            run_ts             TEXT    NOT NULL,
            provider           TEXT    NOT NULL,
            endpoint           TEXT    NOT NULL,
            http_status        INTEGER,
            rows_returned      INTEGER,
            rate_limit_headers TEXT,
            error_text         TEXT
        )
        """
    )
    # R spec 2026-08-26: which client produced the row (same idempotent
    # try/except pattern as the A4 migration above).
    try:
        conn.execute("ALTER TABLE run_log ADD COLUMN client_version TEXT")
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise


def _insert_run_log(
    conn: sqlite3.Connection,
    run_ts: str,
    provider: str,
    endpoint: str,
    http_status: "int | None",
    rows_returned: "int | None",
    rate_limit_headers: "str | None",
    error_text: "str | None",
    client_version: "str | None",
) -> None:
    conn.execute(
        "INSERT INTO run_log (run_ts, provider, endpoint, http_status,"
        " rows_returned, rate_limit_headers, error_text, client_version)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_ts, provider, endpoint, http_status, rows_returned,
         rate_limit_headers, error_text, client_version),
    )


# ── Provider fetchers ────────────────────────────────────────────────
# Each returns (rows, http_status, headers).  May raise freely — the
# retry wrapper + per-provider fail-soft handles it.

def fetch_finnhub(api_key: str, frm: str, to: str) -> tuple[list, int, dict]:
    """One RANGE call to Finnhub's earnings calendar.

    The key travels in the X-Finnhub-Token header — never in the URL.
    """
    resp = requests.get(
        _FINNHUB_URL,
        params={"from": frm, "to": to},
        headers={"X-Finnhub-Token": api_key},
        timeout=_HTTP_TIMEOUT_S,
    )
    resp.raise_for_status()
    rows = resp.json().get("earningsCalendar") or []
    return rows, resp.status_code, dict(resp.headers)


def fetch_fmp(api_key: str, frm: str, to: str) -> tuple[list, int, dict]:
    """One RANGE call to FMP's earning_calendar (v3).

    FMP only accepts the key as a query param — callers must never log
    the URL (see _sanitize_error).
    """
    resp = requests.get(
        _FMP_URL,
        params={"from": frm, "to": to, "apikey": api_key},
        timeout=_HTTP_TIMEOUT_S,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not isinstance(rows, list):
        raise ValueError(f"unexpected FMP payload type: {type(rows).__name__}")
    return rows, resp.status_code, dict(resp.headers)


def fetch_alphavantage_calendar(api_key: str) -> tuple[list, int, dict]:
    """One call to AV's EARNINGS_CALENDAR (horizon=3month, ALL tickers).

    Returns CSV parsed into a list of dicts.  A rate-limit response is
    HTTP 200 with a JSON note instead of CSV — detected and raised as
    AlphaVantageRateLimitError (retriable).  The key rides in the query
    string — callers must never log the URL (see _sanitize_error).
    """
    resp = requests.get(
        _AV_URL,
        params={"function": "EARNINGS_CALENDAR", "horizon": "3month",
                "apikey": api_key},
        timeout=_HTTP_TIMEOUT_S,
    )
    resp.raise_for_status()
    text = resp.text.strip()
    if text.startswith("{"):
        note = _av_note(json.loads(text))
        raise AlphaVantageRateLimitError(note or "unexpected JSON payload")
    rows = list(csv.DictReader(io.StringIO(text)))
    return rows, resp.status_code, dict(resp.headers)


def fetch_alphavantage_earnings(api_key: str, ticker: str) -> tuple[dict, int, dict]:
    """Per-ticker call to AV's EARNINGS actuals endpoint."""
    resp = requests.get(
        _AV_URL,
        params={"function": "EARNINGS", "symbol": ticker, "apikey": api_key},
        timeout=_HTTP_TIMEOUT_S,
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload, resp.status_code, dict(resp.headers)


def fetch_alphavantage_news(api_key: str, ticker: str) -> tuple[dict, int, dict]:
    """One call to AV's NEWS_SENTIMENT endpoint (probe mode only)."""
    resp = requests.get(
        _AV_URL,
        params={"function": "NEWS_SENTIMENT", "tickers": ticker,
                "apikey": api_key},
        timeout=_HTTP_TIMEOUT_S,
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload, resp.status_code, dict(resp.headers)


def _av_api_key() -> str:
    key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not key:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY missing")
    return key


def fetch_yfinance(mode: str, tickers: "list[str]") -> dict:
    """Run scripts/_yf_eval_fetch.py in the ISOLATED yfinance venv.

    Executes the helper with the interpreter at $YF_EVAL_PYTHON (default
    /home/trading/yfeval-venv/bin/python3, created out of band) and
    parses one json object {version, rows, errors} from its stdout.
    Raises on a missing interpreter, timeout (120s), non-zero exit or
    malformed stdout — the caller's fail-soft boundary turns that into
    a run_log row without touching the other providers.
    """
    interpreter = os.environ.get("YF_EVAL_PYTHON", _YF_EVAL_PYTHON_DEFAULT)
    if not Path(interpreter).exists():
        raise RuntimeError(
            f"yfinance eval interpreter missing: {interpreter} "
            "(yfeval-venv not provisioned?)")
    proc = subprocess.run(
        [interpreter, str(_YF_HELPER),
         "--mode", mode, "--tickers", ",".join(tickers)],
        capture_output=True, text=True, timeout=_YF_SUBPROCESS_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"yfinance helper exit {proc.returncode}: "
            f"{(proc.stderr or '').strip()[:500]}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"yfinance helper emitted malformed JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"yfinance helper payload is {type(payload).__name__}, "
            "expected object")
    return payload


def _default_fetchers(mode: str) -> dict:
    """Build the live fetchers from env keys.

    A missing key raises inside the provider's callable so it lands in
    run_log as that provider's failure without touching the others.
    Signatures per provider: finnhub/fmp take (frm, to); alphavantage
    takes (frm, to) in cal mode (AV's horizon is fixed, the range is
    ignored) but (ticker) in eps mode — its actuals endpoint is
    per-ticker.  yfinance takes (tickers) — one subprocess covers the
    whole universe in both modes — and returns the helper's payload
    dict {version, rows, errors}.
    """
    def _finnhub(frm: str, to: str):
        key = os.environ.get("FINNHUB_API_KEY", "")
        if not key:
            raise RuntimeError("FINNHUB_API_KEY missing")
        return fetch_finnhub(key, frm, to)

    def _fmp(frm: str, to: str):
        key = os.environ.get("FMP_API_KEY", "")
        if not key:
            raise RuntimeError("FMP_API_KEY missing")
        return fetch_fmp(key, frm, to)

    if mode == "cal":
        def _av(frm: str, to: str):
            return fetch_alphavantage_calendar(_av_api_key())
    else:
        def _av(ticker: str):
            return fetch_alphavantage_earnings(_av_api_key(), ticker)

    def _yf(tickers: "list[str]"):
        return fetch_yfinance(mode, tickers)

    return {"finnhub": _finnhub, "fmp": _fmp, "alphavantage": _av,
            "yfinance": _yf}


# ── Retry wrapper ────────────────────────────────────────────────────

def call_with_retries(fn, *, retries: int = _MAX_RETRIES, sleep_fn=time.sleep):
    """Initial attempt + up to ``retries`` backoff retries (default 2 →
    3 calls total); a further retry never fires.  Re-raises the last
    exception after the cap."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — fail-soft boundary
            last_exc = exc
            if attempt < retries:
                delay = _BACKOFF_BASE_S * (2 ** attempt)
                logger.warning(
                    "attempt %d/%d failed (%s) — retrying in %.0fs",
                    attempt + 1, retries + 1,
                    _sanitize_error(str(exc)), delay,
                )
                sleep_fn(delay)
    raise last_exc


# ── Row normalization ────────────────────────────────────────────────

_SESSIONS = ("bmo", "amc", "dmh")


def _payload_hash(row: dict) -> str:
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, default=str).encode()
    ).hexdigest()


def _norm_time_of_day(raw) -> str:
    tod = (raw or "").strip().lower()
    return tod if tod in _SESSIONS else "unknown"


def normalize_cal_row(provider: str, row: dict, today: date) -> "dict | None":
    """Map one raw calendar row to cal_capture fields.

    date_status: 'scheduled' when the provider commits to a session
    (bmo/amc/dmh), 'tentative' otherwise — neither Finnhub's nor FMP's
    free tier exposes an explicit confirmed/estimated flag, so session
    presence is the best available proxy.  provider_status_raw preserves
    the provider's raw timing/status field verbatim for later audit.

    Alpha Vantage: no session field at all → time_of_day 'unknown';
    date_status 'scheduled' if a reportDate exists, 'absent' otherwise
    (report_date stored as '' then — the column is NOT NULL);
    provider_status_raw carries the whole raw CSV row as JSON (the
    estimate and fiscalDateEnding context also live in the hashed
    payload).
    """
    if provider == "alphavantage":
        ticker, report_date = row.get("symbol"), row.get("reportDate")
        if not ticker:
            return None
        try:
            days_ahead = (
                date.fromisoformat(report_date[:10]) - today
            ).days if report_date else None
        except ValueError:
            days_ahead = None
        return {
            "ticker": ticker,
            "report_date": report_date[:10] if report_date else "",
            "time_of_day": "unknown",
            "date_status": "scheduled" if report_date else "absent",
            "days_ahead": days_ahead,
            "raw_payload_hash": _payload_hash(row),
            "provider_status_raw": json.dumps(row, sort_keys=True),
        }
    if provider == "finnhub":
        ticker, report_date, status_raw = (
            row.get("symbol"), row.get("date"), row.get("hour"))
    else:  # fmp
        ticker, report_date, status_raw = (
            row.get("symbol"), row.get("date"), row.get("time"))
    if not ticker or not report_date:
        return None
    time_of_day = _norm_time_of_day(status_raw)
    try:
        days_ahead = (date.fromisoformat(report_date[:10]) - today).days
    except ValueError:
        days_ahead = None
    return {
        "ticker": ticker,
        "report_date": report_date[:10],
        "time_of_day": time_of_day,
        "date_status": "scheduled" if time_of_day != "unknown" else "tentative",
        "days_ahead": days_ahead,
        "raw_payload_hash": _payload_hash(row),
        "provider_status_raw": status_raw,
    }


def normalize_eps_row(provider: str, row: dict, capture_day: date) -> "dict | None":
    """Map one raw calendar row to eps_capture fields.

    eps_method records which raw keys estimate/actual came from, so a
    later analysis knows exactly how each number was derived.
    available_same_day: 1 if the actual was already non-null when captured
    ON the report date itself, 0 if it only showed up on a later capture
    day, NULL while the actual is still missing.
    time_of_day is always None here — neither Finnhub's nor FMP's eps
    payload provides a session; only Alpha Vantage does (reportTime).
    """
    if provider == "finnhub":
        ticker, report_date = row.get("symbol"), row.get("date")
        estimate, actual = row.get("epsEstimate"), row.get("epsActual")
        method = "finnhub_calendar:epsActual/epsEstimate"
    else:  # fmp
        ticker, report_date = row.get("symbol"), row.get("date")
        estimate, actual = row.get("epsEstimated"), row.get("eps")
        method = "fmp_calendar:eps/epsEstimated"
    if not ticker or not report_date:
        return None
    report_date = report_date[:10]

    surprise_pct = None
    if actual is not None and estimate not in (None, 0):
        surprise_pct = (actual - estimate) / abs(estimate) * 100.0

    available_same_day = None
    if actual is not None:
        try:
            available_same_day = int(
                date.fromisoformat(report_date) == capture_day)
        except ValueError:
            available_same_day = None

    return {
        "ticker": ticker,
        "report_date": report_date,
        "estimate_eps": estimate,
        "actual_eps": actual,
        "surprise_pct": surprise_pct,
        "eps_method": method,
        "available_same_day": available_same_day,
        "time_of_day": None,
    }


def _av_float(value) -> "float | None":
    """AV serializes numbers as strings and missing values as 'None'."""
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_av_report_time(raw) -> str:
    """AV reportTime → the stack's session vocabulary.

    Persisted to eps_capture.time_of_day since R Amendment A4
    (2026-08-25); "pre-market"→"bmo", "post-market"→"amc", else "unknown".
    """
    rt = (raw or "").strip().lower()
    return {"pre-market": "bmo", "post-market": "amc"}.get(rt, "unknown")


def normalize_av_eps_rows(
    payload: dict, frm: str, to: str, capture_day: date,
) -> list[dict]:
    """Map one ticker's EARNINGS payload to eps_capture rows.

    Only quarterlyEarnings entries whose reportedDate falls inside the
    run's [frm, to] window are captured (the endpoint returns the whole
    history).  eps_method is 'unknown' — AV documents no GAAP/adjusted
    distinction for reportedEPS.  surprise_pct prefers AV's own
    surprisePercentage; falls back to computing from actual/estimate.
    """
    ticker = payload.get("symbol")
    if not ticker:
        return []
    out = []
    for q in payload.get("quarterlyEarnings") or []:
        report_date = (q.get("reportedDate") or "")[:10]
        if not report_date or not (frm <= report_date <= to):
            continue
        estimate = _av_float(q.get("estimatedEPS"))
        actual = _av_float(q.get("reportedEPS"))
        surprise_pct = _av_float(q.get("surprisePercentage"))
        if surprise_pct is None and actual is not None \
                and estimate not in (None, 0):
            surprise_pct = (actual - estimate) / abs(estimate) * 100.0
        available_same_day = None
        if actual is not None:
            try:
                available_same_day = int(
                    date.fromisoformat(report_date) == capture_day)
            except ValueError:
                available_same_day = None
        out.append({
            "ticker": ticker,
            "report_date": report_date,
            "estimate_eps": estimate,
            "actual_eps": actual,
            "surprise_pct": surprise_pct,
            "eps_method": "unknown",
            "available_same_day": available_same_day,
            "time_of_day": normalize_av_report_time(q.get("reportTime")),
        })
    return out


def normalize_yf_cal_row(row: dict, today: date) -> "dict | None":
    """Map one helper cal row to cal_capture fields.

    yfinance's calendar exposes no session and no confirmed/estimated
    flag → time_of_day 'unknown'; date_status 'scheduled' if a report
    date exists, 'absent' otherwise (report_date stored as '' then —
    the column is NOT NULL, mirroring the AV handling).
    provider_status_raw carries the helper's whole raw calendar dict.
    """
    ticker = row.get("ticker")
    if not ticker:
        return None
    report_date = (row.get("report_date") or "")[:10]
    try:
        days_ahead = (
            date.fromisoformat(report_date) - today
        ).days if report_date else None
    except ValueError:
        days_ahead = None
    return {
        "ticker": ticker,
        "report_date": report_date,
        "time_of_day": "unknown",
        "date_status": "scheduled" if report_date else "absent",
        "days_ahead": days_ahead,
        "raw_payload_hash": _payload_hash(row),
        "provider_status_raw": json.dumps(
            row.get("raw") or {}, sort_keys=True, default=str),
    }


def normalize_yf_report_time(raw) -> "str | None":
    """Helper 'HH:MM' (exchange-local row timestamp) → session, else NULL.

    Yahoo's earnings_dates timestamps carry a real clock time when the
    session is known; the mapping is positional vs US cash hours:
    before 09:30 → bmo, 16:00 or later → amc, in between → dmh.
    Missing/midnight-only/unparseable → None (NULL — R spec item 1d,
    unlike AV's 'unknown' which asserts the field existed).
    """
    if not raw:
        return None
    try:
        hour, minute = str(raw).split(":", 1)
        minutes = int(hour) * 60 + int(minute)
    except ValueError:
        return None
    if minutes < 9 * 60 + 30:
        return "bmo"
    if minutes >= 16 * 60:
        return "amc"
    return "dmh"


def normalize_yf_eps_row(
    row: dict, frm: str, to: str, capture_day: date,
) -> "dict | None":
    """Map one helper eps row to eps_capture fields.

    get_earnings_dates returns history (and future placeholders) — only
    rows inside the run's [frm, to] window are captured.  eps_method is
    'unknown' (Yahoo documents no GAAP/adjusted distinction, same as
    AV).  surprise_pct prefers Yahoo's own Surprise(%); falls back to
    computing from actual/estimate.
    """
    ticker = row.get("ticker")
    report_date = (row.get("report_date") or "")[:10]
    if not ticker or not report_date:
        return None
    if not (frm <= report_date <= to):
        return None
    estimate = _av_float(row.get("estimate_eps"))
    actual = _av_float(row.get("actual_eps"))
    surprise_pct = _av_float(row.get("surprise_pct"))
    if surprise_pct is None and actual is not None \
            and estimate not in (None, 0):
        surprise_pct = (actual - estimate) / abs(estimate) * 100.0
    available_same_day = None
    if actual is not None:
        try:
            available_same_day = int(
                date.fromisoformat(report_date) == capture_day)
        except ValueError:
            available_same_day = None
    return {
        "ticker": ticker,
        "report_date": report_date,
        "estimate_eps": estimate,
        "actual_eps": actual,
        "surprise_pct": surprise_pct,
        "eps_method": "unknown",
        "available_same_day": available_same_day,
        "time_of_day": normalize_yf_report_time(row.get("report_time")),
    }


def _first_seen_ts(
    conn: sqlite3.Connection,
    provider: str,
    ticker: str,
    report_date: str,
    capture_ts: str,
) -> str:
    """First capture_ts at which actual_eps was non-null for this
    (provider, ticker, report_date) — set once, carried forward after."""
    row = conn.execute(
        """
        SELECT first_seen_ts FROM eps_capture
        WHERE provider = ? AND ticker = ? AND report_date = ?
          AND first_seen_ts IS NOT NULL
        ORDER BY rowid LIMIT 1
        """,
        (provider, ticker, report_date),
    ).fetchone()
    return row[0] if row else capture_ts


def _insert_eps_row(
    conn: sqlite3.Connection, capture_ts: str, provider: str, norm: dict,
) -> None:
    first_seen = None
    if norm["actual_eps"] is not None:
        first_seen = _first_seen_ts(
            conn, provider, norm["ticker"], norm["report_date"], capture_ts)
    conn.execute(
        "INSERT INTO eps_capture (capture_ts, provider, ticker,"
        " report_date, estimate_eps, actual_eps, surprise_pct,"
        " eps_method, available_same_day, first_seen_ts, time_of_day)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (capture_ts, provider, norm["ticker"], norm["report_date"],
         norm["estimate_eps"], norm["actual_eps"], norm["surprise_pct"],
         norm["eps_method"], norm["available_same_day"], first_seen,
         norm["time_of_day"]),
    )


# ── Alpha Vantage eps path (per-ticker, budget-capped) ───────────────

def _av_priority_order() -> list[str]:
    """Universe walk order under the daily call budget: PEAD names first
    (the strategy the eval serves), then the US watchlist."""
    pead = load_pead_tickers()
    seen = set(pead)
    return pead + [t for t in load_us_watchlist() if t not in seen]


def _capture_av_eps(
    conn: sqlite3.Connection,
    *,
    capture_ts: str,
    today: date,
    frm: str,
    to: str,
    universe: set,
    fetcher,
    sleep_fn,
    priority_order: "list[str] | None" = None,
    daily_limit: "int | None" = None,
) -> dict:
    """AV actuals: one EARNINGS call per ticker under the free-tier
    daily budget (default 25/day, 1 reserved for the same-day cal run).

    A shortfall (universe > remaining budget) is the operational-ceiling
    result the eval is after: WARNING with exact numbers, process as many
    tickers as fit (PEAD first, then US watchlist), record the shortfall
    in run_log.error_text.  Per-ticker failures never abort the loop.
    """
    daily_limit = _AV_DAILY_LIMIT if daily_limit is None else daily_limit
    budget = max(0, daily_limit - _AV_CAL_RESERVED_CALLS)
    ordered = [t for t in (priority_order or _av_priority_order())
               if t in universe]
    skipped: list[str] = []
    if len(ordered) > budget:
        skipped = ordered[budget:]
        logger.warning(
            "alphavantage eps: universe %d tickers > remaining daily "
            "budget %d (daily limit %d - %d reserved for cal) — "
            "processing first %d in priority order (PEAD first), "
            "SKIPPING %d: %s",
            len(ordered), budget, daily_limit, _AV_CAL_RESERVED_CALLS,
            budget, len(skipped), ",".join(skipped),
        )
    process = ordered[:budget]

    rows_total = 0
    inserted = 0
    errors: list[str] = []
    note_seen: "str | None" = None
    last_status: "int | None" = None
    for ticker in process:
        def _one(t=ticker):
            payload, status, headers = fetcher(t)
            note = _av_note(payload)
            if note:
                raise AlphaVantageRateLimitError(note)
            return payload, status, headers

        try:
            payload, status, _headers = call_with_retries(
                _one, sleep_fn=sleep_fn)
        except Exception as exc:  # noqa: BLE001 — per-ticker fail-soft
            if isinstance(exc, AlphaVantageRateLimitError):
                note_seen = exc.note
            err = _sanitize_error(f"{ticker}: {type(exc).__name__}: {exc}")
            logger.error("alphavantage eps %s", err)
            errors.append(err)
            continue
        last_status = status
        rows_total += len(payload.get("quarterlyEarnings") or [])
        for norm in normalize_av_eps_rows(payload, frm, to, today):
            _insert_eps_row(conn, capture_ts, "alphavantage", norm)
            inserted += 1

    parts = []
    if skipped:
        parts.append(
            f"budget shortfall: universe {len(ordered)} > remaining "
            f"budget {budget} (daily limit {daily_limit}, "
            f"{_AV_CAL_RESERVED_CALLS} reserved for cal) — skipped "
            f"{len(skipped)}: {','.join(skipped)}")
    parts.extend(errors)
    error_text = " | ".join(parts) or None
    rate_limit = json.dumps({"av_note": note_seen}) if note_seen else "{}"

    _insert_run_log(
        conn, capture_ts, "alphavantage", _AV_EPS_ENDPOINT, last_status,
        rows_total, rate_limit, error_text, _HTTP_CLIENT_VERSION)
    conn.commit()
    logger.info(
        "alphavantage eps capture: %d/%d tickers processed, %d rows "
        "returned, %d in-window inserted, %d skipped, %d errors",
        len(process) - len(errors), len(ordered), rows_total, inserted,
        len(skipped), len(errors),
    )
    return {
        "ok": last_status is not None,
        "rows_returned": rows_total,
        "inserted": inserted,
        "skipped": len(skipped),
        "errors": len(errors),
    }


# ── yfinance path (isolated subprocess, both modes) ──────────────────

def _capture_yfinance(
    conn: sqlite3.Connection,
    *,
    mode: str,
    capture_ts: str,
    today: date,
    frm: str,
    to: str,
    universe: set,
    fetcher,
) -> dict:
    """yfinance capture: ONE subprocess call covers the whole universe.

    The fetcher returns the helper's payload {version, rows, errors} —
    per-ticker failures arrive in errors[] (recorded in
    run_log.error_text, sanitized) without failing the provider.  In
    cal mode the forward-only endpoint limitation note is prepended to
    error_text on every run (R spec item 3).  http_status is NULL — no
    HTTP happens in this process.
    """
    payload = fetcher(sorted(universe))
    version = payload.get("version")
    rows = payload.get("rows") or []
    helper_errors = payload.get("errors") or []
    endpoint = _YF_CAL_ENDPOINT if mode == "cal" else _YF_EPS_ENDPOINT

    inserted = 0
    for raw in rows:
        if mode == "cal":
            norm = normalize_yf_cal_row(raw, today)
            if norm is None or norm["ticker"] not in universe:
                continue
            conn.execute(
                "INSERT INTO cal_capture (capture_ts, provider, ticker,"
                " report_date, time_of_day, date_status, days_ahead,"
                " raw_payload_hash, provider_status_raw)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (capture_ts, "yfinance", norm["ticker"],
                 norm["report_date"], norm["time_of_day"],
                 norm["date_status"], norm["days_ahead"],
                 norm["raw_payload_hash"], norm["provider_status_raw"]),
            )
        else:
            norm = normalize_yf_eps_row(raw, frm, to, today)
            if norm is None or norm["ticker"] not in universe:
                continue
            _insert_eps_row(conn, capture_ts, "yfinance", norm)
        inserted += 1

    parts = []
    note = _WINDOW_NOTES.get(("yfinance", mode))
    if note:
        parts.append(note)
    parts.extend(
        _sanitize_error(f"{e.get('ticker')}: {e.get('error')}")
        for e in helper_errors)
    error_text = " | ".join(parts) or None

    _insert_run_log(
        conn, capture_ts, "yfinance", endpoint, None, len(rows), None,
        error_text, f"yfinance {version}" if version else None)
    conn.commit()
    logger.info(
        "yfinance %s capture OK: %d rows returned, %d in-universe "
        "inserted, %d helper errors (yfinance %s)",
        mode, len(rows), inserted, len(helper_errors), version,
    )
    return {
        "ok": True,
        "rows_returned": len(rows),
        "inserted": inserted,
        "helper_errors": len(helper_errors),
    }


# ── Capture body ─────────────────────────────────────────────────────

def run_capture(
    mode: str,
    conn: sqlite3.Connection,
    *,
    now: "datetime | None" = None,
    fetchers: "dict | None" = None,
    universe: "set[str] | None" = None,
    sleep_fn=time.sleep,
    av_priority: "list[str] | None" = None,
    av_daily_limit: "int | None" = None,
) -> dict:
    """Run one capture pass over all providers.

    Fail-soft: each provider is isolated in its own try/except; each
    outcome (success or final failure) gets a run_log row, and each
    provider's rows + run_log entry are committed independently so one
    provider's crash never loses the others' data.  Alpha Vantage eps
    runs through the budget-capped per-ticker path (_capture_av_eps);
    yfinance runs through the isolated-subprocess path
    (_capture_yfinance) in both modes.
    """
    if mode not in ("cal", "eps"):
        raise ValueError(f"unknown mode: {mode!r}")
    now = now or datetime.now(timezone.utc)
    today = now.date()
    capture_ts = now.isoformat()

    # [T-2, T+30] target window (R spec 2026-08-26) — days_ahead may go
    # negative for the lookback days and is stored unclamped.
    if mode == "cal":
        frm, to = (
            today - timedelta(days=_CAL_LOOKBACK_DAYS)).isoformat(), (
            today + timedelta(days=_CAL_FORWARD_DAYS)).isoformat()
    else:
        frm, to = (
            today - timedelta(days=_EPS_LOOKBACK_DAYS)).isoformat(), today.isoformat()

    fetchers = fetchers or _default_fetchers(mode)
    universe = universe if universe is not None else load_universe()
    endpoints = {
        "finnhub": _FINNHUB_ENDPOINT,
        "fmp": _FMP_ENDPOINT,
        "alphavantage": _AV_CAL_ENDPOINT if mode == "cal" else _AV_EPS_ENDPOINT,
        "yfinance": _YF_CAL_ENDPOINT if mode == "cal" else _YF_EPS_ENDPOINT,
    }

    ensure_schema(conn)
    summary: dict = {"mode": mode, "capture_ts": capture_ts, "providers": {}}

    for provider in _PROVIDERS:
        endpoint = endpoints[provider]

        if provider == "yfinance":
            # Isolated-subprocess path, single attempt (no retries).
            try:
                summary["providers"][provider] = _capture_yfinance(
                    conn, mode=mode, capture_ts=capture_ts, today=today,
                    frm=frm, to=to, universe=universe,
                    fetcher=fetchers[provider])
            except Exception as exc:  # noqa: BLE001 — fail-soft boundary
                err = _sanitize_error(f"{type(exc).__name__}: {exc}")
                logger.error("yfinance %s capture FAILED: %s", mode, err)
                _insert_run_log(conn, capture_ts, provider, endpoint,
                                None, None, None, err, None)
                conn.commit()
                summary["providers"][provider] = {"ok": False, "error": err}
            continue

        if provider == "alphavantage" and mode == "eps":
            # Per-ticker path with its own budget/fail-soft/run_log.
            try:
                summary["providers"][provider] = _capture_av_eps(
                    conn, capture_ts=capture_ts, today=today, frm=frm,
                    to=to, universe=universe,
                    fetcher=fetchers[provider], sleep_fn=sleep_fn,
                    priority_order=av_priority, daily_limit=av_daily_limit)
            except Exception as exc:  # noqa: BLE001 — fail-soft boundary
                err = _sanitize_error(f"{type(exc).__name__}: {exc}")
                logger.error("alphavantage eps capture FAILED: %s", err)
                _insert_run_log(conn, capture_ts, provider, endpoint,
                                None, None, None, err,
                                _HTTP_CLIENT_VERSION)
                conn.commit()
                summary["providers"][provider] = {"ok": False, "error": err}
            continue

        def _fetch_once(provider=provider):
            result = fetchers[provider](frm, to)
            if provider == "alphavantage":
                # AV rate limiting is HTTP 200 + a JSON note, not 429 —
                # surface it as a retriable failure.
                note = _av_note(result[0])
                if note:
                    raise AlphaVantageRateLimitError(note)
            return result

        try:
            rows, status, headers = call_with_retries(
                _fetch_once, sleep_fn=sleep_fn)
        except Exception as exc:  # noqa: BLE001 — fail-soft boundary
            status = getattr(getattr(exc, "response", None), "status_code", None)
            rate_limit = None
            if isinstance(exc, AlphaVantageRateLimitError):
                status = status or 200
                rate_limit = json.dumps({"av_note": exc.note})
            err = _sanitize_error(f"{type(exc).__name__}: {exc}")
            logger.error("%s %s capture FAILED: %s", provider, mode, err)
            _insert_run_log(conn, capture_ts, provider, endpoint, status,
                            None, rate_limit, err, _HTTP_CLIENT_VERSION)
            conn.commit()
            summary["providers"][provider] = {"ok": False, "error": err}
            continue

        inserted = 0
        for raw in rows:
            if mode == "cal":
                norm = normalize_cal_row(provider, raw, today)
                if norm is None or norm["ticker"] not in universe:
                    continue
                conn.execute(
                    "INSERT INTO cal_capture (capture_ts, provider, ticker,"
                    " report_date, time_of_day, date_status, days_ahead,"
                    " raw_payload_hash, provider_status_raw)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (capture_ts, provider, norm["ticker"], norm["report_date"],
                     norm["time_of_day"], norm["date_status"],
                     norm["days_ahead"], norm["raw_payload_hash"],
                     norm["provider_status_raw"]),
                )
            else:
                norm = normalize_eps_row(provider, raw, today)
                if norm is None or norm["ticker"] not in universe:
                    continue
                _insert_eps_row(conn, capture_ts, provider, norm)
            inserted += 1

        # A window-coverage limitation (AV's forward-only calendar) is
        # recorded even on success — error_text doubles as a note field.
        _insert_run_log(conn, capture_ts, provider, endpoint, status,
                        len(rows), _rate_limit_headers_json(headers),
                        _WINDOW_NOTES.get((provider, mode)),
                        _HTTP_CLIENT_VERSION)
        conn.commit()
        logger.info(
            "%s %s capture OK: %d rows returned, %d in-universe inserted",
            provider, mode, len(rows), inserted,
        )
        summary["providers"][provider] = {
            "ok": True, "rows_returned": len(rows), "inserted": inserted}

    return summary


# ── NEWS_SENTIMENT probe (one-shot, stdout only) ─────────────────────

def run_news_probe(ticker: str, fetch_fn=None, out=print) -> int:
    """One-shot Alpha Vantage NEWS_SENTIMENT probe.

    Pretty-prints the raw JSON response, then a summary block: article
    count, date range, fields present per article, ticker relevance
    scores, sentiment labels.  stdout ONLY — no DB write, no run_log
    row, no retries.  Returns 1 on any failure (rate limit, auth,
    network), 0 otherwise.  Purpose: show what the endpoint gives, not
    store it.
    """
    fetch = fetch_fn or (lambda: fetch_alphavantage_news(_av_api_key(), ticker))
    try:
        payload, status, _headers = fetch()
    except Exception as exc:  # noqa: BLE001 — probe boundary
        out(f"NEWS_SENTIMENT probe FAILED: "
            f"{_sanitize_error(f'{type(exc).__name__}: {exc}')}")
        return 1
    note = _av_note(payload)
    if note:
        out(f"NEWS_SENTIMENT probe FAILED (provider note): {note}")
        return 1

    out("── raw response " + "─" * 40)
    out(json.dumps(payload, indent=2, sort_keys=True))

    feed = payload.get("feed") or []
    out("── summary " + "─" * 45)
    out(f"http_status: {status}")
    out(f"articles: {len(feed)}")

    times = sorted(a.get("time_published", "") for a in feed
                   if a.get("time_published"))
    out(f"date_range: {times[0]} .. {times[-1]}" if times
        else "date_range: n/a")

    field_counts: dict = {}
    for a in feed:
        for k in a:
            field_counts[k] = field_counts.get(k, 0) + 1
    out("fields_present (field: articles_with_field/articles):")
    for k in sorted(field_counts):
        out(f"  {k}: {field_counts[k]}/{len(feed)}")

    relevances = []
    overall_labels: dict = {}
    ticker_labels: dict = {}
    for a in feed:
        lbl = a.get("overall_sentiment_label")
        if lbl:
            overall_labels[lbl] = overall_labels.get(lbl, 0) + 1
        for ts in a.get("ticker_sentiment") or []:
            if ts.get("ticker") == ticker:
                try:
                    relevances.append(float(ts.get("relevance_score")))
                except (TypeError, ValueError):
                    pass
                tlbl = ts.get("ticker_sentiment_label")
                if tlbl:
                    ticker_labels[tlbl] = ticker_labels.get(tlbl, 0) + 1
    if relevances:
        out(f"ticker_relevance ({ticker}): n={len(relevances)} "
            f"min={min(relevances):.4f} "
            f"mean={sum(relevances) / len(relevances):.4f} "
            f"max={max(relevances):.4f}")
    else:
        out(f"ticker_relevance ({ticker}): none found")
    out(f"overall_sentiment_labels: {json.dumps(overall_labels, sort_keys=True)}")
    out(f"ticker_sentiment_labels ({ticker}): "
        f"{json.dumps(ticker_labels, sort_keys=True)}")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: "list[str] | None" = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True,
                        choices=("cal", "eps", "news-probe"))
    parser.add_argument("--ticker",
                        help="single ticker (required for --mode news-probe)")
    args = parser.parse_args(argv)

    if args.mode == "news-probe":
        if not args.ticker:
            parser.error("--ticker is required with --mode news-probe")
        return run_news_probe(args.ticker)

    db_path = os.environ.get("EARNINGS_EVAL_DB", _DB_PATH_DEFAULT)
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        try:
            summary = run_capture(args.mode, conn)
        finally:
            conn.close()
        logger.info("done: %s", json.dumps(summary))
    except Exception as exc:  # noqa: BLE001 — capture-only job never raises out
        logger.error("run FAILED: %s", _sanitize_error(f"{exc}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
