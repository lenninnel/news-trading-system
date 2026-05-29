#!/usr/bin/env python3
"""Daily OHLC ingest from Polygon.io into the news_trading.db `daily_ohlc` table.

Purpose: feed Research (Q-003 sizing, Q-Regime Stage-4) a clean point-in-time
US daily price history. Fully decoupled from the live trading path — only
the `daily_ohlc` table is touched, and no other module reads from it yet.

Universe: US-20 only — `config/watchlist.yaml::us_tickers` (11 names) plus
the US-only PEAD tickers in `config/settings.PEAD_TICKERS` (those without
a "." suffix; 9 names). Asserted to be exactly 20 uppercase tickers.

Modes
-----
--backfill     Fetch OHLC_BACKFILL_YEARS back to yesterday for all 20
               tickers and upsert.
--incremental  Fetch a trailing 7-CALENDAR-DAY window for all 20 tickers
               and upsert. Idempotent; self-heals small gaps.

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
    0 = ok
    1 = aborted (universe mismatch / Polygon key missing / DB write failure)
"""
from __future__ import annotations

import argparse
import logging
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

def _yesterday_utc() -> date:
    return datetime.now(timezone.utc).date() - timedelta(days=1)


def backfill_range(years: int = OHLC_BACKFILL_YEARS) -> tuple[str, str]:
    end = _yesterday_utc()
    start = end - timedelta(days=years * 365)
    return start.isoformat(), end.isoformat()


def incremental_range() -> tuple[str, str]:
    end = _yesterday_utc()
    start = end - timedelta(days=7)
    return start.isoformat(), end.isoformat()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(mode: str, years: int | None = None) -> int:
    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY is not set — refusing to run")
        return 1

    universe = build_us20_universe()
    logger.info("Universe (%d): %s", len(universe), universe)
    if len(universe) != 20:
        logger.error(
            "Expected exactly 20 tickers in US-20 universe, got %d. Aborting.",
            len(universe),
        )
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

    return 0 if not tickers_failed else 1


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
    return run(mode, years=args.years)


if __name__ == "__main__":
    sys.exit(main())
