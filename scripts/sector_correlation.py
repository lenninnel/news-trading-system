#!/usr/bin/env python3
"""
Sector / correlation mapping for the trading universe (D4).

Three responsibilities, all backed by yfinance:

1. STATIC METADATA — fetched on first use and refreshed monthly
   (first Sunday of the month). Pulls sector / industry / market-cap
   bucket / region for each ticker in the full universe (18 core +
   15 PEAD = 33 tickers) and stores it in ``config/sector_map.json``.

2. ROLLING CORRELATION MATRIX — refreshed every Sunday. Downloads
   30 calendar days of daily close prices for the full universe and
   writes a Pearson correlation matrix to
   ``config/correlation_matrix.json`` with a timestamp.

3. PEER DETECTION — derived from the correlation matrix. For every
   ticker we keep the top 3 peers with correlation > 0.5 and embed
   them under a ``peers`` field in ``sector_map.json``. These peers
   are what the event-triggered session will scan when news hits.

Usage::

    python -m scripts.sector_correlation               # weekly run
    python -m scripts.sector_correlation --static      # force metadata refresh
    python -m scripts.sector_correlation --correlation # only the matrix
    python -m scripts.sector_correlation --dry-run     # don't write files
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

# Make ``import config.settings`` work when invoked as ``python scripts/...``
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.settings import PEAD_TICKERS  # noqa: E402

log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────
CONFIG_DIR = _REPO_ROOT / "config"
WATCHLIST_PATH = CONFIG_DIR / "watchlist.yaml"
SECTOR_MAP_PATH = CONFIG_DIR / "sector_map.json"
CORRELATION_MATRIX_PATH = CONFIG_DIR / "correlation_matrix.json"

# ── Tunables ─────────────────────────────────────────────────────────────
CORRELATION_LOOKBACK_DAYS = 30
PEER_CORRELATION_THRESHOLD = 0.5
PEER_TOP_N = 3


# ── Universe loading ─────────────────────────────────────────────────────

def load_full_universe() -> list[str]:
    """Return the full ticker universe (core watchlist ∪ PEAD universe)."""
    with open(WATCHLIST_PATH) as fh:
        cfg = yaml.safe_load(fh) or {}
    core = cfg.get("watchlist") or cfg.get("us_tickers") or []
    seen: set[str] = set()
    universe: list[str] = []
    for t in list(core) + list(PEAD_TICKERS):
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            universe.append(t)
    return universe


# ── Ticker symbol normalisation ──────────────────────────────────────────

def to_yf_symbol(ticker: str) -> str:
    """Convert internal ticker to yfinance symbol (XETRA → DE)."""
    if ticker.endswith(".XETRA"):
        return ticker.replace(".XETRA", ".DE")
    return ticker


# ── Static metadata ──────────────────────────────────────────────────────

def _market_cap_bucket(market_cap: float | int | None) -> str:
    """Bucket market cap. mega >$100B, large $10-100B, mid $2-10B, small <$2B."""
    if not market_cap:
        return "unknown"
    cap = float(market_cap)
    if cap >= 100_000_000_000:
        return "mega"
    if cap >= 10_000_000_000:
        return "large"
    if cap >= 2_000_000_000:
        return "mid"
    return "small"


_COUNTRY_TO_REGION: dict[str, str] = {
    "united states": "US",
    "usa": "US",
    "germany": "DE",
    "france": "FR",
    "united kingdom": "UK",
    "uk": "UK",
    "brazil": "BR",
    "netherlands": "NL",
    "switzerland": "CH",
    "italy": "IT",
    "spain": "ES",
    "ireland": "IE",
    "canada": "CA",
}


def _region(country: str | None, ticker: str) -> str:
    """Map country / ticker suffix to a coarse region label."""
    if ticker.endswith((".DE", ".XETRA")):
        return "DE"
    if ticker.endswith(".PA"):
        return "FR"
    if ticker.endswith(".L"):
        return "UK"
    if not country:
        return "unknown"
    c = country.strip().lower()
    if c in _COUNTRY_TO_REGION:
        return _COUNTRY_TO_REGION[c]
    for needle, code in _COUNTRY_TO_REGION.items():
        if needle in c:
            return code
    return "unknown"


def _slug(value: str | None) -> str:
    """Lower-case + underscore-separated slug. ``None`` → ``unknown``."""
    if not value:
        return "unknown"
    return value.strip().lower().replace(" ", "_").replace("-", "_").replace("&", "and")


def fetch_static_metadata(tickers: list[str]) -> dict[str, dict]:
    """Pull sector / industry / market-cap / region from yfinance."""
    out: dict[str, dict] = {}
    for ticker in tickers:
        symbol = to_yf_symbol(ticker)
        try:
            info = yf.Ticker(symbol).info or {}
        except Exception as exc:  # network / parsing failure on a single name
            log.warning("yfinance.info failed for %s (%s): %s", ticker, symbol, exc)
            info = {}

        record = {
            "sector": _slug(info.get("sector")),
            "industry": _slug(info.get("industry")),
            "market_cap_bucket": _market_cap_bucket(info.get("marketCap")),
            "region": _region(info.get("country"), ticker),
        }
        out[ticker] = record
        log.info("metadata %s → %s", ticker, record)
    return out


# ── Correlation matrix ───────────────────────────────────────────────────

def fetch_close_prices(tickers: list[str], lookback_days: int) -> pd.DataFrame:
    """Download daily close prices for *tickers* covering *lookback_days*."""
    yf_tickers = [to_yf_symbol(t) for t in tickers]
    # Pad lookback to absorb weekends/holidays so we still land 30 trading days.
    period = f"{max(lookback_days * 2, 60)}d"
    raw = yf.download(
        tickers=yf_tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    closes = pd.DataFrame(index=raw.index if hasattr(raw, "index") else None)
    for internal, symbol in zip(tickers, yf_tickers):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                series = raw[symbol]["Close"]
            else:
                # Single-ticker download collapses the MultiIndex.
                series = raw["Close"]
            closes[internal] = series
        except (KeyError, ValueError):
            log.warning("no price data for %s (%s)", internal, symbol)

    closes = closes.dropna(how="all")
    if len(closes) > lookback_days:
        closes = closes.tail(lookback_days)
    return closes


def compute_correlation_matrix(closes: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Pearson correlation on daily returns. Returns nested dict of floats."""
    if closes.empty or len(closes.columns) < 2:
        return {}
    returns = closes.pct_change(fill_method=None).dropna(how="all")
    matrix = returns.corr(method="pearson")
    out: dict[str, dict[str, float]] = {}
    for ticker in matrix.index:
        row = {}
        for other in matrix.columns:
            if other == ticker:
                continue
            value = matrix.loc[ticker, other]
            if pd.isna(value):
                continue
            row[other] = round(float(value), 4)
        out[ticker] = row
    return out


# ── Peer detection ───────────────────────────────────────────────────────

def detect_peers(
    matrix: dict[str, dict[str, float]],
    *,
    threshold: float = PEER_CORRELATION_THRESHOLD,
    top_n: int = PEER_TOP_N,
) -> dict[str, list[str]]:
    """For each ticker, return the top-N peers above *threshold*."""
    peers: dict[str, list[str]] = {}
    for ticker, row in matrix.items():
        ranked = sorted(row.items(), key=lambda kv: kv[1], reverse=True)
        peers[ticker] = [peer for peer, corr in ranked if corr > threshold][:top_n]
    return peers


# ── File IO ──────────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


def load_sector_map() -> dict[str, dict]:
    if not SECTOR_MAP_PATH.exists():
        return {}
    try:
        with open(SECTOR_MAP_PATH) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not parse sector_map.json: %s", exc)
        return {}


def save_sector_map(sector_map: dict[str, dict]) -> None:
    _atomic_write_json(SECTOR_MAP_PATH, sector_map)


def save_correlation_matrix(matrix: dict[str, dict[str, float]]) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "matrix": matrix,
    }
    _atomic_write_json(CORRELATION_MATRIX_PATH, payload)


# ── Refresh policy ───────────────────────────────────────────────────────

def is_first_sunday_of_month(dt: datetime | None = None) -> bool:
    dt = dt or datetime.now(timezone.utc)
    return dt.weekday() == 6 and dt.day <= 7


def should_refresh_static(force: bool = False, dt: datetime | None = None) -> bool:
    if force:
        return True
    if not SECTOR_MAP_PATH.exists():
        return True
    return is_first_sunday_of_month(dt)


# ── Top-level orchestration ──────────────────────────────────────────────

def run(
    *,
    static_only: bool = False,
    correlation_only: bool = False,
    force_static: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run the configured pipeline. Returns a small summary dict."""
    universe = load_full_universe()
    log.info("universe: %d tickers", len(universe))

    sector_map = load_sector_map()
    refreshed_static = False
    refreshed_matrix = False

    do_static = static_only or (not correlation_only and should_refresh_static(force=force_static))
    if do_static:
        log.info("refreshing static metadata for %d tickers", len(universe))
        new_meta = fetch_static_metadata(universe)
        for ticker, meta in new_meta.items():
            existing = sector_map.get(ticker, {})
            existing.update(meta)
            sector_map[ticker] = existing
        refreshed_static = True

    matrix: dict[str, dict[str, float]] = {}
    if not static_only:
        log.info(
            "fetching %d-day close prices for %d tickers",
            CORRELATION_LOOKBACK_DAYS, len(universe),
        )
        closes = fetch_close_prices(universe, CORRELATION_LOOKBACK_DAYS)
        log.info(
            "downloaded %d rows × %d tickers", len(closes), len(closes.columns),
        )
        matrix = compute_correlation_matrix(closes)
        if matrix:
            peers = detect_peers(matrix)
            for ticker, peer_list in peers.items():
                entry = sector_map.get(ticker, {})
                entry["peers"] = peer_list
                sector_map[ticker] = entry
            refreshed_matrix = True

    if dry_run:
        log.info("dry-run: no files written")
    else:
        if refreshed_static or refreshed_matrix:
            save_sector_map(sector_map)
            log.info("wrote %s (%d tickers)", SECTOR_MAP_PATH, len(sector_map))
        if matrix:
            save_correlation_matrix(matrix)
            log.info("wrote %s (%d tickers)", CORRELATION_MATRIX_PATH, len(matrix))

    return {
        "universe_size": len(universe),
        "static_refreshed": refreshed_static,
        "matrix_refreshed": refreshed_matrix,
        "sector_map_size": len(sector_map),
    }


# ── CLI ──────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sector / correlation refresher")
    parser.add_argument(
        "--static",
        action="store_true",
        help="Only refresh static metadata (skip correlation matrix).",
    )
    parser.add_argument(
        "--correlation",
        action="store_true",
        help="Only refresh the correlation matrix (skip static metadata).",
    )
    parser.add_argument(
        "--force-static",
        action="store_true",
        help="Force a static metadata refresh even if it's not the first Sunday.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute everything but do not touch the JSON files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    summary = run(
        static_only=args.static,
        correlation_only=args.correlation,
        force_static=args.force_static,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
