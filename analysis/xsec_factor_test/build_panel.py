#!/usr/bin/env python3
"""Assemble the daily panel from the Polygon cache. Pure transformation, no API.

Output (cache/panel/):
  close.parquet, high.parquet, dollar_vol.parquet   raw (unadjusted), dates x entity
  tr.parquet                                        total-return index (splits + cash dividends)
  adj_close.parquet, adj_high.parquet               split-adjusted only (for 52w-high)
  entity_map.parquet                                (ticker, date_from, date_to) -> entity
  eligible.parquet                                  per (snapshot month-end, entity): CS, not SPAC,
                                                    listed exchange -> universe candidate flags

Entity key = Polygon composite FIGI (persists across ticker renames such as
FB->META, SQ->XYZ); fallback = ticker when no FIGI is known.

Corporate actions: daily gross return g_t = (adjc_t + D_t * S_t) / adjc_{t-1},
adjc_t = Polygon's split-adjusted close (adjusted=true grouped bars, as of the
fetch date), S_t = adjc_t / close_t the implied split factor. Cash dividends
(types CD, SC, from /v3/reference/dividends) enter on the ex-date. The
/v3/reference/splits feed is only used as a cross-check: it misses splits on
renamed tickers. Days without a bar are forward-filled (0 return) — a delisted name
is therefore liquidated at its last close and earns 0 afterwards.
"""
from __future__ import annotations

import logging
import re
import sys
from datetime import date

import numpy as np
import pandas as pd

from common import CACHE, GROUPED_ADJ_DIR, GROUPED_DIR, REF_DIR, SNAP_DIR, log

PANEL = CACHE / "panel"
PANEL.mkdir(exist_ok=True)

LISTED_EXCHANGES = {"XNYS", "XNAS", "XASE", "ARCX", "BATS"}
SPAC_RE = re.compile(r"acquisition", re.I)  # one rule, chosen once


def load_grouped(src=GROUPED_DIR) -> pd.DataFrame:
    files = sorted(src.glob("*.parquet"))
    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=["T", "v", "c", "h"])
        if len(df) == 0:
            continue
        df["date"] = pd.Timestamp(f.stem)
        parts.append(df)
    g = pd.concat(parts, ignore_index=True)
    g = g.rename(columns={"T": "ticker", "v": "vol", "c": "close", "h": "high"})
    log.info("grouped bars: %d rows, %d days, %s..%s", len(g), g.date.nunique(),
             g.date.min().date(), g.date.max().date())
    return g


def load_snapshots() -> pd.DataFrame:
    parts = []
    for f in sorted(SNAP_DIR.glob("*.parquet")):
        df = pd.read_parquet(f)
        df["snap"] = pd.Timestamp(f.stem)
        parts.append(df)
    s = pd.concat(parts, ignore_index=True)
    s["entity"] = s.composite_figi.fillna(s.ticker)
    s["spac"] = s.name.fillna("").str.contains(SPAC_RE)
    s["listed"] = s.primary_exchange.isin(LISTED_EXCHANGES)
    return s


def map_entities(g: pd.DataFrame, snaps: pd.DataFrame) -> pd.Series:
    """ticker on date -> entity, strictly point-in-time: the CS snapshot of
    that month-end first (covers IPOs and renames into the month), then the
    previous month-end snapshot (covers delistings and renames out of the
    month). Anything else is not a common stock at that date and is dropped.
    No fallback to the current reference tables: recycled symbols (ACH was an
    ADR in 2021 and is Owens & Minor's ticker in 2026) would be mis-mapped."""
    month_end = g.date + pd.offsets.MonthEnd(0)
    key_cur = pd.MultiIndex.from_arrays([g.ticker, month_end])
    key_prev = pd.MultiIndex.from_arrays([g.ticker, month_end - pd.offsets.MonthEnd(1)])
    snap_map = snaps.drop_duplicates(["ticker", "snap"]).set_index(["ticker", "snap"]).entity
    ent = pd.Series(snap_map.reindex(key_cur).values, index=g.index, dtype=object)
    miss = ent.isna()
    via_cur = 1 - miss.mean()
    ent[miss] = snap_map.reindex(key_prev[miss]).values
    miss = ent.isna()
    log.info("entity mapping: %.1f%% via month snapshot, %.1f%% via previous snapshot, "
             "%.1f%% not CS at that date (dropped)", 100 * via_cur,
             100 * (1 - via_cur - miss.mean()), 100 * miss.mean())
    return ent


def build() -> None:
    g = load_grouped()
    snaps = load_snapshots()
    g["entity"] = map_entities(g, snaps).values
    g = g.dropna(subset=["entity"])
    log.info("rows after CS-entity filter: %d, entities %d", len(g), g.entity.nunique())
    # one row per (date, entity): if two tickers map to one entity on one day
    # (rename day overlap), keep the one with the larger volume
    g = g.sort_values("vol").drop_duplicates(["date", "entity"], keep="last")

    close = g.pivot(index="date", columns="entity", values="close").sort_index()
    high = g.pivot(index="date", columns="entity", values="high").sort_index()
    vol = g.pivot(index="date", columns="entity", values="vol").sort_index()
    dollar = close * vol

    # split factor per cell from Polygon's own split-adjusted bars
    ga = load_grouped(GROUPED_ADJ_DIR)
    ga["entity"] = map_entities(ga, snaps).values
    ga = ga.dropna(subset=["entity"]).sort_values("vol").drop_duplicates(["date", "entity"], keep="last")
    pg_adj_close = ga.pivot(index="date", columns="entity", values="close").reindex(
        index=close.index, columns=close.columns)
    pg_adj_high = ga.pivot(index="date", columns="entity", values="high").reindex(
        index=close.index, columns=close.columns)
    S = (pg_adj_close / close)
    both = S.notna()
    log.info("cells with raw bar but no adjusted bar: %d", int((close.notna() & ~both).sum().sum()))
    S = S.ffill()
    # cross-check against the /v3/reference/splits feed (information only)
    sp = pd.read_parquet(REF_DIR / "splits.parquet")
    sp["date"] = pd.to_datetime(sp.execution_date).astype(close.index.dtype)
    tk2ent = g.drop_duplicates(["ticker", "date"]).set_index(["ticker", "date"]).entity
    jumps = (S / S.shift(1) - 1).abs() > 0.02
    jumps &= both
    n_jumps = int(jumps.sum().sum())
    sp_in = sp[sp.date.isin(close.index)]
    sp_in_ent = tk2ent.reindex(pd.MultiIndex.from_arrays([sp_in.ticker, sp_in.date]))
    matched = 0
    for (tk, d), e in sp_in_ent.dropna().items():
        if e in jumps.columns and bool(jumps.at[d, e]):
            matched += 1
    log.info("split factor jumps in panel: %d; splits-feed records on panel entities: %d, "
             "of which coincide with a jump: %d", n_jumps, int(sp_in_ent.notna().sum()), matched)

    # dividends
    dv = pd.read_parquet(REF_DIR / "dividends.parquet")
    dv = dv[dv.dividend_type.isin(["CD", "SC"])]
    dv = dv.drop_duplicates(["ticker", "ex_dividend_date", "cash_amount"])  # 249 exact dupes in feed
    dv["date"] = pd.to_datetime(dv.ex_dividend_date)
    dv["entity"] = tk2ent.reindex(pd.MultiIndex.from_arrays([dv.ticker, dv.date])).values
    miss = dv.entity.isna()
    if miss.any():
        tk_last = g.groupby("ticker").entity.last()
        dv.loc[miss, "entity"] = dv.ticker[miss].map(tk_last).values
    dv = dv.dropna(subset=["entity"])
    dv = dv[dv.entity.isin(close.columns) & dv.date.isin(close.index)]
    D = dv.groupby(["date", "entity"]).cash_amount.sum().unstack().reindex(
        index=close.index, columns=close.columns).fillna(0.0)
    log.info("cash dividends applied: %d ex-dates", int((D > 0).sum().sum()))

    adj_close = pg_adj_close.ffill()
    adj_high = pg_adj_high
    prev = adj_close.shift(1)
    gross = (adj_close + D * S) / prev
    gross = gross.where(prev.notna())
    # first observation of each entity: index starts at 1
    tr = gross.fillna(1.0).cumprod()
    tr = tr.where(adj_close.notna())

    # data-quality flags: absurd daily moves inside observed data
    obs = close.notna() & close.shift(1).notna()
    big = ((gross > 3.0) | (gross < 0.2)) & obs
    log.info("daily gross return >3x or <0.2x on observed bars: %d cells", int(big.sum().sum()))

    for name, df in [("close", close), ("high", high), ("dollar_vol", dollar), ("tr", tr),
                     ("adj_close", adj_close), ("adj_high", adj_high)]:
        df.astype("float64").to_parquet(PANEL / f"{name}.parquet")

    elig = snaps[snaps.type == "CS"].groupby(["snap", "entity"]).agg(
        spac=("spac", "any"), listed=("listed", "all"), name=("name", "first"),
        ticker=("ticker", "first")).reset_index()
    elig.to_parquet(PANEL / "eligible.parquet", index=False)
    big_cells = big.stack()
    big_cells = big_cells[big_cells]
    pd.DataFrame({"date": big_cells.index.get_level_values(0),
                  "entity": big_cells.index.get_level_values(1)}).to_parquet(
        PANEL / "big_moves.parquet", index=False)
    log.info("panel written: %d dates x %d entities", *close.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build()
