#!/usr/bin/env python3
"""Q2 earnings source evaluation — capture-only (R-spec 2026-08-24).

Standalone, NON-LIVE evaluation stack that captures earnings-calendar and
EPS-actuals data from two free providers (Finnhub + FMP) into its OWN
SQLite database, so the two sources can later be compared for calendar
accuracy and actuals latency.

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
``--mode cal``  (12:00 UTC timer) — forward calendar capture, one RANGE call
    per provider covering today .. today+30d, filtered to the universe.
``--mode eps``  (23:00 UTC timer) — actuals capture, one RANGE call per
    provider covering today-1 .. today, filtered to the universe.

Universe (sourced at runtime, never hard-coded)
-----------------------------------------------
* US watchlist: ``config/watchlist.yaml`` key ``us_tickers``.
* PEAD names:   ``config.settings.PEAD_TICKERS`` (15 names).
Rows whose symbol is not an exact match against the union are dropped.

Secrets discipline
------------------
API keys are read from env (``FINNHUB_API_KEY`` / ``FMP_API_KEY``).  The
Finnhub key travels in the ``X-Finnhub-Token`` header (never in the URL).
FMP requires the key as a query param, so NO full URL is ever logged and
every error string is passed through ``_sanitize_error`` (strips query
strings and masks ``apikey=``/``token=`` values) before logging or storage.

Retries
-------
Each provider call gets max 2 retries with exponential backoff after the
initial attempt (3 HTTP calls total), then the failure is logged to
``run_log`` and the run continues with the other provider.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
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

_CAL_FORWARD_DAYS = 30   # cal mode: today .. today+30d
_EPS_LOOKBACK_DAYS = 1   # eps mode: today-1 .. today

_MAX_RETRIES = 2         # retries after the initial attempt (3 calls total)
_BACKOFF_BASE_S = 2.0    # 2s, then 4s
_HTTP_TIMEOUT_S = 20.0

# Endpoint identifiers stored in run_log — path only, NEVER a full URL
# with query params (the FMP key rides in the query string).
_FINNHUB_ENDPOINT = "finnhub.io/api/v1/calendar/earnings"
_FMP_ENDPOINT = "financialmodelingprep.com/api/v3/earning_calendar"

_FINNHUB_URL = "https://finnhub.io/api/v1/calendar/earnings"
_FMP_URL = "https://financialmodelingprep.com/api/v3/earning_calendar"

_PROVIDERS = ("finnhub", "fmp")


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
    """Create the three eval tables if absent (exact R-spec columns)."""
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


def _default_fetchers() -> dict:
    """Build the live fetchers from env keys.

    A missing key raises inside the provider's callable so it lands in
    run_log as that provider's failure without touching the other.
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

    return {"finnhub": _finnhub, "fmp": _fmp}


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
    (bmo/amc/dmh), 'tentative' otherwise — neither free tier exposes an
    explicit confirmed/estimated flag, so session presence is the best
    available proxy.  provider_status_raw preserves the provider's raw
    timing/status field verbatim for later audit.
    """
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


# ── Capture body ─────────────────────────────────────────────────────

def run_capture(
    mode: str,
    conn: sqlite3.Connection,
    *,
    now: "datetime | None" = None,
    fetchers: "dict | None" = None,
    universe: "set[str] | None" = None,
    sleep_fn=time.sleep,
) -> dict:
    """Run one capture pass over both providers.

    Fail-soft: each provider is isolated in its own try/except; each
    outcome (success or final failure) gets a run_log row, and each
    provider's rows + run_log entry are committed independently so one
    provider's crash never loses the other's data.
    """
    if mode not in ("cal", "eps"):
        raise ValueError(f"unknown mode: {mode!r}")
    now = now or datetime.now(timezone.utc)
    today = now.date()
    capture_ts = now.isoformat()

    if mode == "cal":
        frm, to = today.isoformat(), (
            today + timedelta(days=_CAL_FORWARD_DAYS)).isoformat()
    else:
        frm, to = (
            today - timedelta(days=_EPS_LOOKBACK_DAYS)).isoformat(), today.isoformat()

    fetchers = fetchers or _default_fetchers()
    universe = universe if universe is not None else load_universe()
    endpoints = {"finnhub": _FINNHUB_ENDPOINT, "fmp": _FMP_ENDPOINT}

    ensure_schema(conn)
    summary: dict = {"mode": mode, "capture_ts": capture_ts, "providers": {}}

    for provider in _PROVIDERS:
        endpoint = endpoints[provider]
        try:
            rows, status, headers = call_with_retries(
                lambda: fetchers[provider](frm, to), sleep_fn=sleep_fn)
        except Exception as exc:  # noqa: BLE001 — fail-soft boundary
            status = getattr(getattr(exc, "response", None), "status_code", None)
            err = _sanitize_error(f"{type(exc).__name__}: {exc}")
            logger.error("%s %s capture FAILED: %s", provider, mode, err)
            conn.execute(
                "INSERT INTO run_log (run_ts, provider, endpoint, http_status,"
                " rows_returned, rate_limit_headers, error_text)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (capture_ts, provider, endpoint, status, None, None, err),
            )
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
                first_seen = None
                if norm["actual_eps"] is not None:
                    first_seen = _first_seen_ts(
                        conn, provider, norm["ticker"], norm["report_date"],
                        capture_ts)
                conn.execute(
                    "INSERT INTO eps_capture (capture_ts, provider, ticker,"
                    " report_date, estimate_eps, actual_eps, surprise_pct,"
                    " eps_method, available_same_day, first_seen_ts)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (capture_ts, provider, norm["ticker"], norm["report_date"],
                     norm["estimate_eps"], norm["actual_eps"],
                     norm["surprise_pct"], norm["eps_method"],
                     norm["available_same_day"], first_seen),
                )
            inserted += 1

        conn.execute(
            "INSERT INTO run_log (run_ts, provider, endpoint, http_status,"
            " rows_returned, rate_limit_headers, error_text)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (capture_ts, provider, endpoint, status, len(rows),
             _rate_limit_headers_json(headers), None),
        )
        conn.commit()
        logger.info(
            "%s %s capture OK: %d rows returned, %d in-universe inserted",
            provider, mode, len(rows), inserted,
        )
        summary["providers"][provider] = {
            "ok": True, "rows_returned": len(rows), "inserted": inserted}

    return summary


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: "list[str] | None" = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=("cal", "eps"))
    args = parser.parse_args(argv)

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
