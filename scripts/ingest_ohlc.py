#!/usr/bin/env python3
"""Daily OHLC ingest from Polygon.io into the news_trading.db `daily_ohlc` table.

Purpose: maintain a clean US daily price history. Since 2026-09-01 the live
path depends on it: RiskAgent computes Wilder ATR(14) — and therefore stops,
take-profits and sizing — from `daily_ohlc`. Staleness here means stale stops.

Universe: US-20 only — `config/watchlist.yaml::us_tickers` (11 names) plus
the US-only PEAD tickers in `config/settings.PEAD_TICKERS` (those without
a "." suffix; 9 names). Asserted to be exactly 20 uppercase tickers.

Modes
-----
--backfill     Fetch OHLC_BACKFILL_YEARS back to the window end for all 20
               tickers and upsert.
--incremental  Fetch a trailing 7-CALENDAR-DAY window for all 20 tickers
               and upsert. Idempotent; self-heals small gaps.

Window end / expected freshness
-------------------------------
The window ends on *today* (UTC) once the US session is over
(>= SAME_DAY_CUTOFF_UTC, 22:00 UTC — close is 20:00 UTC in DST, 21:00 UTC
in winter), otherwise on yesterday. The nightly timer fires 22:30 UTC, so
the production run always includes the just-closed session: after a run on
trading day T, MAX(date) in the store must be T.

Freshness gate: after upserting, every universe ticker's MAX(date) in the
store must equal the last US trading day <= window end
(data/market_calendar.py). Any stale ticker fails the run — a "success"
that wrote no new rows is no longer possible. Unscheduled market closures
(not in the calendar) cause one spurious failure; see market_calendar.py.

Hygiene
-------
Rows are FLAGGED, never dropped:
  - quality_flag='OHLC_INCONSISTENT' if internal bar shape is wrong
    (low>open, high<close, any non-positive value, etc.).
  - quality_flag='EXTREME_MOVE'      if |close/prev_close - 1| > 50%
    (using the previous bar for that ticker in the fetched series).
  - quality_flag='TICKER_RECYCLE'    identity break, NOT a corporate action:
    a recycled symbol whose pre-boundary bars are a different issuer (e.g.
    META < 2022-06-09). Set out-of-band by scripts/flag_meta_recycle.py; do
    NOT "clean up" — adj_close cannot bridge a symbol reassignment.
  - quality_flag=NULL                otherwise.

Exit codes:
    0 = ok (all tickers fetched AND store is fresh through the expected day)
    1 = aborted (universe mismatch / Polygon key missing / DB write failure /
        fetch failure / freshness gate: stale MAX(date) for >=1 ticker)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Make the repo root importable regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.settings import (  # noqa: E402
    OHLC_BACKFILL_YEARS,
    OHLC_EXTREME_MOVE_PCT,
    PEAD_TICKERS,
    POLYGON_API_KEY,
)
from data.market_calendar import last_us_trading_day  # noqa: E402
from data.polygon_feed import PolygonFeed  # noqa: E402
from storage.database import Database  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_ohlc")


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def build_us20_universe() -> list[str]:
    """Return the 20-ticker US universe, sorted, uppercase, no duplicates.

    Sources:
      - config/watchlist.yaml::us_tickers (11 names)
      - config/settings.PEAD_TICKERS minus any name containing "." (9 names)
    """
    path = _REPO_ROOT / "config" / "watchlist.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    us = cfg.get("us_tickers") or []
    if not isinstance(us, list):
        raise RuntimeError(
            f"watchlist.yaml::us_tickers must be a list, got {type(us).__name__}"
        )

    pead_us = [t for t in PEAD_TICKERS if "." not in t]
    universe = sorted({t.upper() for t in [*us, *pead_us]})
    return universe


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------

def _is_inconsistent(bar: dict) -> bool:
    """Internal bar-shape check. Returns True if the bar violates basic OHLC invariants."""
    try:
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
    except KeyError:
        return True
    if any(v is None for v in (o, h, l, c)):
        return True
    if any(v <= 0 for v in (o, h, l, c)):
        return True
    if l > o or l > c:
        return True
    if h < o or h < c:
        return True
    if h < l:
        return True
    return False


def flag_bars(bars: list[dict], extreme_pct: float) -> tuple[list[dict], list[tuple[str, str]]]:
    """Annotate each bar with a `quality_flag` (or None).

    Returns:
        (bars-with-flag, list-of-(ticker, date, flag) for the flagged ones)
    """
    flagged: list[tuple[str, str, str]] = []
    prev_close: float | None = None
    for b in bars:
        flag: str | None = None
        if _is_inconsistent(b):
            flag = "OHLC_INCONSISTENT"
        elif prev_close is not None and prev_close > 0:
            move = abs(b["close"] / prev_close - 1.0)
            if move > extreme_pct:
                flag = "EXTREME_MOVE"
        b["quality_flag"] = flag
        if flag:
            flagged.append((b.get("ticker", "?"), b["date"], flag))
        # Only advance prev_close from internally-consistent bars.
        if flag != "OHLC_INCONSISTENT":
            prev_close = b["close"]
    return bars, [(t, d) for (t, d, _f) in flagged]


# ---------------------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------------------

# US session close is 20:00 UTC (DST) / 21:00 UTC (winter). Past this cutoff
# the just-closed session's daily bar is fetched same-day; the nightly timer
# (22:30 UTC) is always past it. Earlier runs end the window on yesterday so
# an in-progress session can never be ingested as a partial bar.
SAME_DAY_CUTOFF_UTC = 22


def _window_end(now: datetime | None = None) -> date:
    now = now or datetime.now(timezone.utc)
    if now.hour >= SAME_DAY_CUTOFF_UTC:
        return now.date()
    return now.date() - timedelta(days=1)


def backfill_range(years: int = OHLC_BACKFILL_YEARS) -> tuple[str, str]:
    end = _window_end()
    start = end - timedelta(days=years * 365)
    return start.isoformat(), end.isoformat()


def incremental_range() -> tuple[str, str]:
    end = _window_end()
    start = end - timedelta(days=7)
    return start.isoformat(), end.isoformat()


# ---------------------------------------------------------------------------
# Failure alerting
# ---------------------------------------------------------------------------

def _alert_failure(message: str) -> None:
    """Best-effort Telegram alert on ingest failure. Never raises — the
    non-zero exit code remains the source of truth for the run status."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set — failure alert not sent")
        return
    try:
        from notifications.telegram_bot import TelegramNotifier
        TelegramNotifier(bot_token=token, chat_id=chat_id).send_error(
            f"nts-ohlc-ingest FAILED: {message}"
        )
    except Exception as exc:
        logger.warning("Telegram failure alert could not be sent: %s", exc)


# ---------------------------------------------------------------------------
# Freshness gate
# ---------------------------------------------------------------------------

def check_freshness(db, universe: list[str], end: str) -> tuple[str, list[tuple[str, str | None]]]:
    """Verify the store holds the expected latest session for every ticker.

    Returns (expected_date, stale) where `stale` lists (ticker, max_date)
    for every ticker whose stored MAX(date) is missing or older than the
    last US trading day <= `end`. Empty `stale` == fresh.
    """
    expected = last_us_trading_day(date.fromisoformat(end)).isoformat()
    max_dates = db.get_daily_ohlc_max_dates(universe)
    stale = [
        (t, max_dates.get(t))
        for t in universe
        if max_dates.get(t) is None or max_dates[t] < expected
    ]
    return expected, stale


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(mode: str, years: int | None = None) -> int:
    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY is not set — refusing to run")
        _alert_failure("POLYGON_API_KEY is not set")
        return 1

    universe = build_us20_universe()
    logger.info("Universe (%d): %s", len(universe), universe)
    if len(universe) != 20:
        logger.error(
            "Expected exactly 20 tickers in US-20 universe, got %d. Aborting.",
            len(universe),
        )
        _alert_failure(f"universe mismatch: expected 20 tickers, got {len(universe)}")
        return 1

    if mode == "backfill":
        start, end = backfill_range(years if years is not None else OHLC_BACKFILL_YEARS)
    elif mode == "incremental":
        start, end = incremental_range()
    else:
        logger.error("Unknown mode: %s", mode)
        return 1

    logger.info("Date range: %s..%s (mode=%s)", start, end, mode)

    feed = PolygonFeed()
    db = Database()

    total_rows = 0
    total_flagged = 0
    flagged_detail: list[tuple[str, str, str]] = []
    tickers_ok = 0
    tickers_failed: list[str] = []

    for ticker in universe:
        try:
            bars = feed.get_daily_aggs(ticker, start, end)
        except Exception as exc:
            logger.error("Polygon fetch failed for %s: %s", ticker, exc)
            tickers_failed.append(ticker)
            continue

        if not bars:
            logger.warning("Polygon returned 0 bars for %s [%s..%s]", ticker, start, end)
            tickers_ok += 1
            continue

        # Tag ticker on each bar so the flag log carries it.
        for b in bars:
            b["ticker"] = ticker
            b["source"] = "polygon"

        bars, _ = flag_bars(bars, OHLC_EXTREME_MOVE_PCT)
        flagged = [b for b in bars if b.get("quality_flag")]
        for b in flagged:
            flagged_detail.append((ticker, b["date"], b["quality_flag"]))

        n = db.upsert_daily_ohlc(bars)
        total_rows += n
        total_flagged += len(flagged)
        tickers_ok += 1
        logger.info(
            "%s: upserted %d rows, %d flagged",
            ticker, n, len(flagged),
        )

    logger.info(
        "Summary: tickers ok=%d failed=%d (failed=%s) | "
        "rows upserted=%d | flagged=%d",
        tickers_ok, len(tickers_failed),
        tickers_failed if tickers_failed else "none",
        total_rows, total_flagged,
    )
    if flagged_detail:
        logger.info("Flagged rows (ticker, date, flag):")
        for t, d, f in flagged_detail:
            logger.info("  %s %s %s", t, d, f)

    expected, stale = check_freshness(db, universe, end)
    if stale:
        stale_str = ", ".join(f"{t}={d or 'no rows'}" for t, d in stale)
        logger.error(
            "FRESHNESS GATE FAILED: expected MAX(date) >= %s (last US trading "
            "day <= window end %s), but %d ticker(s) are stale: %s",
            expected, end, len(stale), stale_str,
        )
        _alert_failure(
            f"freshness gate: {len(stale)} ticker(s) below expected {expected}: {stale_str}"
        )
        return 1
    logger.info("Freshness gate passed: all %d tickers at >= %s", len(universe), expected)

    if tickers_failed:
        _alert_failure(f"Polygon fetch failed for: {', '.join(tickers_failed)}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--backfill", action="store_true",
        help=f"Fetch {OHLC_BACKFILL_YEARS}yr of daily bars and upsert.",
    )
    group.add_argument(
        "--incremental", action="store_true",
        help="Fetch trailing 7-day window and upsert.",
    )
    parser.add_argument(
        "--years", type=int, default=None,
        help=(
            "Override backfill depth in years (backfill mode only). "
            f"Defaults to OHLC_BACKFILL_YEARS={OHLC_BACKFILL_YEARS}. "
            "Ignored in --incremental mode; the nightly path is unaffected."
        ),
    )
    args = parser.parse_args(argv)
    if args.years is not None and args.years <= 0:
        parser.error("--years must be a positive integer")
    if args.years is not None and args.incremental:
        parser.error("--years is only valid with --backfill")
    mode = "backfill" if args.backfill else "incremental"
    try:
        return run(mode, years=args.years)
    except Exception as exc:
        logger.exception("Ingest crashed")
        _alert_failure(f"crashed with {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
