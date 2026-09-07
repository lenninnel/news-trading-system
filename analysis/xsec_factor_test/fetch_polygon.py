#!/usr/bin/env python3
"""Fetch and cache everything the factor test needs from Polygon (read-only).

Idempotent and resumable: every unit of work is one file under cache/ and is
skipped if it already exists. Safe to re-run after a rate-limit abort.

  --grouped     one grouped-daily call per weekday DATA_START..DATA_END
                -> cache/grouped/YYYY-MM-DD.parquet (empty file on holidays)
  --grouped_adj same, adjusted=true -> cache/grouped_adj/ (split factor source)
  --ref         all stock tickers, active + delisted, every type
                -> cache/ref/tickers_active.parquet / tickers_inactive.parquet
  --corp        splits + cash dividends with ex-date >= DATA_START
                -> cache/ref/splits.parquet / dividends.parquet
  --snapshots   point-in-time ticker snapshots (type=CS) at each month-end
                -> cache/snapshots/YYYY-MM-DD.parquet
  --all         everything, in the order ref, corp, snapshots, grouped
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

import pandas as pd

from common import (BASE, DATA_END, DATA_START, GROUPED_ADJ_DIR, GROUPED_DIR, REF_DIR,
                    SNAP_DIR, Polygon, log)

GROUPED_COLS = ["T", "v", "vw", "o", "c", "h", "l", "t", "n"]
SNAP_COLS = ["ticker", "name", "type", "primary_exchange", "active", "cik",
             "composite_figi", "share_class_figi", "delisted_utc", "last_updated_utc"]


def weekdays(a: date, b: date):
    d = a
    while d <= b:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def month_ends(a: date, b: date) -> list[date]:
    """Last calendar day of each month in [a, b] (snapshot 'date' param is
    calendar-based; the panel builder maps it to the last trading day)."""
    out = []
    d = date(a.year, a.month, 1)
    while d <= b:
        nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        me = nxt - timedelta(days=1)
        if a <= me <= b:
            out.append(me)
        d = nxt
    return out


def fetch_grouped(pg: Polygon, adjusted: bool = False) -> None:
    """adjusted=False -> raw bars (prices, dollar volume, $5 filter).
    adjusted=True  -> Polygon split-adjusted bars (as of fetch date); the
    per-cell ratio adjusted/raw is the split factor. Polygon's own adjustment
    follows a security across ticker renames, which the /v3/reference/splits
    feed does not (XSPA->XWEL, XL->SPRU have no split record at all)."""
    out_dir = GROUPED_ADJ_DIR if adjusted else GROUPED_DIR
    todo = [d for d in weekdays(DATA_START, DATA_END)
            if not (out_dir / f"{d.isoformat()}.parquet").exists()]
    log.info("grouped adjusted=%s: %d weekdays to fetch", adjusted, len(todo))
    for i, d in enumerate(todo):
        j = pg.get(f"{BASE}/v2/aggs/grouped/locale/us/market/stocks/{d.isoformat()}",
                   {"adjusted": "true" if adjusted else "false", "include_otc": "false"})
        if j.get("status") == "NOT_AUTHORIZED":
            log.error("grouped %s NOT_AUTHORIZED (entitlement boundary) -> stop", d)
            return
        res = j.get("results") or []
        df = pd.DataFrame(res, columns=GROUPED_COLS) if res else pd.DataFrame(columns=GROUPED_COLS)
        df.to_parquet(out_dir / f"{d.isoformat()}.parquet", index=False)
        if i % 25 == 0:
            log.info("grouped adj=%s %s rows=%d (%d/%d, calls=%d, pace=%.2fs)",
                     adjusted, d, len(df), i + 1, len(todo), pg.calls, pg.pace)


def _snap_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in SNAP_COLS:
        if c not in df.columns:
            df[c] = None
    return df[SNAP_COLS].astype({"active": "bool"})


def fetch_ref(pg: Polygon) -> None:
    for active in ("true", "false"):
        f = REF_DIR / f"tickers_{'active' if active == 'true' else 'inactive'}.parquet"
        if f.exists():
            continue
        rows = pg.paginate(f"{BASE}/v3/reference/tickers",
                           {"market": "stocks", "active": active, "limit": 1000})
        _snap_df(rows).to_parquet(f, index=False)
        log.info("ref tickers active=%s rows=%d", active, len(rows))


def fetch_corp(pg: Polygon) -> None:
    f = REF_DIR / "splits.parquet"
    if not f.exists():
        rows = pg.paginate(f"{BASE}/v3/reference/splits",
                           {"execution_date.gte": DATA_START.isoformat(),
                            "execution_date.lte": DATA_END.isoformat(), "limit": 1000})
        pd.DataFrame(rows).to_parquet(f, index=False)
        log.info("splits rows=%d", len(rows))
    f = REF_DIR / "dividends.parquet"
    if not f.exists():
        rows = pg.paginate(f"{BASE}/v3/reference/dividends",
                           {"ex_dividend_date.gte": DATA_START.isoformat(),
                            "ex_dividend_date.lte": DATA_END.isoformat(), "limit": 1000})
        pd.DataFrame(rows).to_parquet(f, index=False)
        log.info("dividends rows=%d", len(rows))


def fetch_snapshots(pg: Polygon) -> None:
    for me in month_ends(DATA_START, DATA_END):
        f = SNAP_DIR / f"{me.isoformat()}.parquet"
        if f.exists():
            continue
        rows = pg.paginate(f"{BASE}/v3/reference/tickers",
                           {"market": "stocks", "type": "CS", "date": me.isoformat(),
                            "limit": 1000})
        _snap_df(rows).to_parquet(f, index=False)
        log.info("snapshot %s CS rows=%d (calls=%d)", me, len(rows), pg.calls)


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("grouped", "grouped_adj", "ref", "corp", "snapshots", "all"):
        ap.add_argument(f"--{k}", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pg = Polygon()
    if a.ref or a.all:
        fetch_ref(pg)
    if a.corp or a.all:
        fetch_corp(pg)
    if a.snapshots or a.all:
        fetch_snapshots(pg)
    if a.grouped or a.all:
        fetch_grouped(pg, adjusted=False)
    if a.grouped_adj or a.all:
        fetch_grouped(pg, adjusted=True)
    log.info("done, %d API calls", pg.calls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
