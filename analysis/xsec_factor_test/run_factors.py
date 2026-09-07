#!/usr/bin/env python3
"""Cross-sectional factor test on a point-in-time top-500 US universe.

Everything below is fixed in advance. There are no sweeps and no
parameter search; the only run is the one that writes out/REPORT.md.

Timeline per month m (strict one-day implementation lag, no lookahead):
  S = last trading day of month m-1  -> all signals use data through close(S)
  R = first trading day of month m   -> enter at close(R)
  R' = first trading day of month m+1 -> exit at close(R')  (== next R)
  monthly return of a name = TR[R'] / TR[R] - 1

Universe at S (fixed rules):
  * type CS in the point-in-time Polygon snapshot for that month-end,
    name does not contain "acquisition" (SPAC filter), listed exchange
  * raw close on S observed (traded that day) and >= 5 USD
  * >= 253 trading days of history (so every factor is defined)
  * 60-day median dollar volume (raw close x volume, >= 50 of 60 days observed)
  * top 500 by that median

Factors at S (higher = long side):
  mom_12_1 = TR[S-21] / TR[S-252] - 1
  st_rev   = -(TR[S] / TR[S-5] - 1)
  low_vol  = -std(daily TR returns, 60 days through S)
  hi52     = adj_close[S] / max(adj_high, 252 days through S)
  combo    = mean of the four percentile ranks (equal weight, no optimisation)

Portfolios: deciles by rank within the universe (D10 = top). Long-short =
D10 - D1 (each leg equal-weighted, 100% notional). Long-only = D10.
Benchmark = equal-weighted universe. Costs 10 bp per side on traded value:
cost_m = 0.001 * sum |w_target - w_drifted|, charged at each rebalance.
Turnover reported one-way = 0.5 * sum |w_target - w_drifted|, annualised x12.

Statistics on the monthly EXCESS series (long-only: strategy - benchmark;
long-short: the spread itself). Sharpe annualised sqrt(12). Deflated Sharpe
(Bailey & Lopez de Prado 2014) with N trials = 4 for the four factors and
N = 5 for the combination (it is a fifth pre-specified test), variance of
the trial Sharpes taken across the family, skew/kurtosis of the series.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from common import CACHE, OUT, log

PANEL = CACHE / "panel"
N_TOP = 500
MIN_PRICE = 5.0
MIN_HISTORY = 253
LIQ_WINDOW, LIQ_MIN_OBS = 60, 50
COST_PER_SIDE = 0.001
FACTORS = ["mom_12_1", "st_rev", "low_vol", "hi52"]
ALL = FACTORS + ["combo"]


# --------------------------------------------------------------------------- data
def load_panel():
    P = {k: pd.read_parquet(PANEL / f"{k}.parquet")
         for k in ("close", "high", "dollar_vol", "tr", "adj_close", "adj_high")}
    elig = pd.read_parquet(PANEL / "eligible.parquet")
    return P, elig


def month_boundaries(dates: pd.DatetimeIndex):
    """list of (S, R, R') triples covering complete months only."""
    df = pd.DataFrame({"d": dates})
    df["ym"] = df.d.dt.to_period("M")
    first = df.groupby("ym").d.first()
    last = df.groupby("ym").d.last()
    yms = list(first.index)
    out = []
    for i in range(1, len(yms) - 1):
        S, R, Rn = last[yms[i - 1]], first[yms[i]], first[yms[i + 1]]
        out.append((S, R, Rn))
    return out


# ------------------------------------------------------------------- universe + signals
def universe_and_signals(P, elig, S: pd.Timestamp):
    close, tr, dollar = P["close"], P["tr"], P["dollar_vol"]
    idx = close.index.get_loc(S)
    if idx < MIN_HISTORY:
        return None
    snap = S + pd.offsets.MonthEnd(0)
    e = elig[(elig.snap == snap) & (~elig.spac) & elig.listed]
    cand = close.columns.intersection(e.entity)
    px = close.loc[S, cand]
    cand = px.index[px.notna() & (px >= MIN_PRICE)]
    hist_ok = tr.iloc[idx - 252][cand].notna()  # first obs at least 252 days back
    cand = cand[hist_ok.values]
    liq_win = dollar.iloc[idx - LIQ_WINDOW + 1: idx + 1][cand]
    liq = liq_win.median()
    liq = liq[liq_win.notna().sum() >= LIQ_MIN_OBS]
    top = liq.sort_values(ascending=False).index[:N_TOP]
    if len(top) < N_TOP:
        log.warning("%s universe only %d names", S.date(), len(top))

    trw = tr[top]
    ret_d = trw.iloc[idx - 59: idx + 1].pct_change().iloc[1:]
    sig = pd.DataFrame({
        "mom_12_1": trw.iloc[idx - 21] / trw.iloc[idx - 252] - 1,
        "st_rev": -(trw.iloc[idx] / trw.iloc[idx - 5] - 1),
        "low_vol": -ret_d.std(),
        "hi52": P["adj_close"].loc[S, top] / P["adj_high"].iloc[idx - 251: idx + 1][top].max(),
    })
    ranks = sig.rank(pct=True)
    sig["combo"] = ranks.mean(axis=1)
    sig["liq"] = liq[top]
    sig["price"] = px[top]
    return sig


# ------------------------------------------------------------------------ portfolios
@dataclass
class Book:
    """Tracks drifted weights of one long-only leg to compute turnover."""
    w: pd.Series | None = None

    def rebalance(self, target: pd.Series, last_ret: pd.Series | None) -> float:
        if self.w is None:
            self.w = target
            return 1.0  # initial 100% buy: one-way turnover 1.0 (charged)
        drifted = self.w * (1 + last_ret.reindex(self.w.index).fillna(0))
        drifted = drifted / drifted.sum() if drifted.sum() > 0 else drifted
        allidx = drifted.index.union(target.index)
        delta = (target.reindex(allidx).fillna(0) - drifted.reindex(allidx).fillna(0)).abs().sum()
        self.w = target
        return 0.5 * delta  # one-way


def run(P, elig):
    tr = P["tr"]
    bounds = month_boundaries(tr.index)
    rows = []
    decile_rows = []
    books = {(f, side): Book() for f in ALL for side in ("L", "S")}
    books["bench"] = Book()
    last_ret = None
    members_by_month = {}
    holdings = []  # (R, Rn, factor, leg, entity)
    for S, R, Rn in bounds:
        sig = universe_and_signals(P, elig, S)
        if sig is None:
            continue
        uni = sig.index
        members_by_month[R] = uni
        r = (tr.loc[Rn, uni] / tr.loc[R, uni] - 1).fillna(0.0)  # missing R': liquidated at last close
        bench = r.mean()
        rec = {"S": S, "R": R, "Rn": Rn, "n_uni": len(uni), "bench": bench,
               "bench_to": books["bench"].rebalance(pd.Series(1 / len(uni), index=uni), last_ret)}
        for f in ALL:
            dec = pd.qcut(sig[f].rank(method="first"), 10, labels=False) + 1
            means = r.groupby(dec).mean()
            for d, v in means.items():
                decile_rows.append({"R": R, "factor": f, "decile": int(d), "ret": v, "bench": bench})
            longn, shortn = uni[dec == 10], uni[dec == 1]
            holdings += [(R, Rn, f, "L", e) for e in longn] + [(R, Rn, f, "S", e) for e in shortn]
            wl = pd.Series(1 / len(longn), index=longn)
            ws = pd.Series(1 / len(shortn), index=shortn)
            to_l = books[(f, "L")].rebalance(wl, last_ret)
            to_s = books[(f, "S")].rebalance(ws, last_ret)
            rl, rs = r[longn].mean(), r[shortn].mean()
            rec[f"{f}_L_gross"] = rl
            rec[f"{f}_L_net"] = rl - 2 * COST_PER_SIDE * to_l
            rec[f"{f}_L_to"] = to_l
            rec[f"{f}_LS_gross"] = rl - rs
            rec[f"{f}_LS_net"] = rl - rs - 2 * COST_PER_SIDE * (to_l + to_s)
            rec[f"{f}_LS_to"] = to_l + to_s
        rows.append(rec)
        last_ret = r
    m = pd.DataFrame(rows).set_index("R")
    dec = pd.DataFrame(decile_rows)
    hold = pd.DataFrame(holdings, columns=["R", "Rn", "factor", "leg", "entity"])
    return m, dec, members_by_month, hold


def big_moves_in_holdings(hold: pd.DataFrame, P, elig) -> None:
    """Data caveat list: holdings that had a daily gross return <0.2x or >3x
    during their holding month (corporate actions outside the splits and
    dividends feeds, e.g. cash takeovers, or genuine crashes)."""
    big = pd.read_parquet(PANEL / "big_moves.parquet")
    tr = P["tr"]
    names = elig.drop_duplicates("entity").set_index("entity")[["ticker", "name"]]
    rows = []
    for _, b in big.iterrows():
        h = hold[(hold.entity == b.entity) & (hold.R < b.date) & (hold.Rn >= b.date)]
        for _, x in h.iterrows():
            g = tr.at[b.date, b.entity] / tr[b.entity].shift(1).at[b.date]
            rows.append({"R": x.R.date(), "factor": x.factor, "leg": x.leg, "entity": b.entity,
                         "ticker": names.ticker.get(b.entity), "name": str(names.name.get(b.entity))[:30],
                         "date": b.date.date(), "gross_return": round(float(g), 3),
                         "leg_impact_pct": round(100 * (g - 1) / 50, 2)})
    df = pd.DataFrame(rows).sort_values(["R", "factor"]) if rows else pd.DataFrame(rows)
    df.to_csv(OUT / "big_moves_in_holdings.csv", index=False)
    log.info("big moves inside decile holdings: %d cells", len(df))


# ------------------------------------------------------------------------ statistics
def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(12)) if x.std(ddof=1) > 0 else np.nan


def max_drawdown(x: pd.Series) -> float:
    w = (1 + x).cumprod()
    return float((w / w.cummax() - 1).min())


def deflated_sharpe(x: pd.Series, trial_sharpes_monthly: list[float], n_trials: int) -> dict:
    """Bailey & Lopez de Prado (2014). Everything in monthly (non-annualised) units."""
    T = len(x)
    sr = float(x.mean() / x.std(ddof=1))
    sk = float(stats.skew(x, bias=False))
    ku = float(stats.kurtosis(x, bias=False, fisher=False))  # non-excess
    var_sr = float(np.var(trial_sharpes_monthly, ddof=1)) if len(trial_sharpes_monthly) > 1 else 0.0
    emc = 0.5772156649
    z = stats.norm.ppf
    sr0 = np.sqrt(var_sr) * ((1 - emc) * z(1 - 1 / n_trials) + emc * z(1 - 1 / (n_trials * np.e)))
    denom = np.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr ** 2, 1e-12))
    psr_stat = (sr - sr0) * np.sqrt(T - 1) / denom
    psr0_stat = sr * np.sqrt(T - 1) / denom
    return {"sr_monthly": sr, "sr0_monthly": float(sr0), "dsr": float(stats.norm.cdf(psr_stat)),
            "psr_vs_zero": float(stats.norm.cdf(psr0_stat)), "skew": sk, "kurt": ku, "T": T}


def yearly(x: pd.Series) -> pd.DataFrame:
    g = x.groupby(x.index.year)
    return pd.DataFrame({"excess_ret": g.apply(lambda s: (1 + s).prod() - 1),
                         "hit_rate": g.apply(lambda s: (s > 0).mean()), "months": g.size()})


def leave_one_year_out(x: pd.Series) -> pd.DataFrame:
    out = []
    for y in sorted(set(x.index.year)):
        s = x[x.index.year != y]
        if len(s) < 2:
            continue
        out.append({"excluded_year": y, "ann_excess": (1 + s).prod() ** (12 / len(s)) - 1,
                    "sharpe": sharpe(s)})
    return pd.DataFrame(out)


def concentration(ex: pd.Series) -> dict:
    """How much of the result hangs on a few months (diagnostic, not a variant)."""
    ann = lambda s: float((1 + s).prod() ** (12 / len(s)) - 1)
    srt = ex.sort_values(ascending=False)
    return {"ann_excess_drop_best1": ann(ex.drop(srt.index[:1])),
            "ann_excess_drop_best3": ann(ex.drop(srt.index[:3])),
            "sharpe_drop_best3": sharpe(ex.drop(srt.index[:3])),
            "best_month": str(srt.index[0].date()), "best_month_excess": float(srt.iloc[0]),
            "top3_share_of_log_excess": float(np.log1p(srt.iloc[:3]).sum() / np.log1p(ex).sum())
            if np.log1p(ex).sum() != 0 else np.nan}


def beta_to_bench(ex: pd.Series, bench: pd.Series) -> dict:
    """OLS of monthly excess on the benchmark return: is it alpha or beta?"""
    X = np.column_stack([np.ones(len(bench)), bench.values])
    b, res, *_ = np.linalg.lstsq(X, ex.values, rcond=None)
    resid = ex.values - X @ b
    s2 = resid @ resid / (len(ex) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return {"alpha_monthly": float(b[0]), "alpha_t": float(b[0] / np.sqrt(cov[0, 0])),
            "beta": float(b[1]), "beta_t": float(b[1] / np.sqrt(cov[1, 1])),
            "alpha_ann": float((1 + b[0]) ** 12 - 1)}


def evaluate(m: pd.DataFrame, dec: pd.DataFrame, members, P, elig) -> dict:
    res = {"months": len(m), "first_R": str(m.index[0].date()), "last_Rn": str(m.Rn.iloc[-1].date()),
           "bench_ann": float((1 + m.bench).prod() ** (12 / len(m)) - 1),
           "bench_sharpe_abs": sharpe(m.bench), "bench_turnover_ann": float(m.bench_to.mean() * 12),
           "portfolios": {}, "deciles": {}, "universe": {}}
    fam_sr = {}
    for side in ("L", "LS"):
        fam_sr[side] = {}
        for f in ALL:
            net = m[f"{f}_{side}_net"]
            ex = net - m.bench if side == "L" else net
            fam_sr[side][f] = float(ex.mean() / ex.std(ddof=1))
    for side in ("L", "LS"):
        for f in ALL:
            net = m[f"{f}_{side}_net"]
            gross = m[f"{f}_{side}_gross"]
            ex = net - m.bench if side == "L" else net
            ex_gross = gross - m.bench if side == "L" else gross
            trials = [fam_sr[side][k] for k in FACTORS]
            n_trials = 4 if f in FACTORS else 5
            if f == "combo":
                trials = trials + [fam_sr[side]["combo"]]
            d = deflated_sharpe(ex, trials, n_trials)
            res["portfolios"][f"{f}_{side}"] = {
                "ann_excess_net": float((1 + ex).prod() ** (12 / len(ex)) - 1),
                "ann_excess_gross": float((1 + ex_gross).prod() ** (12 / len(ex)) - 1),
                "mean_monthly_excess_net": float(ex.mean()),
                "sharpe_net": sharpe(ex),
                "sharpe_gross": sharpe(ex_gross),
                "dsr": d["dsr"], "psr_vs_zero": d["psr_vs_zero"], "sr0_monthly": d["sr0_monthly"],
                "skew": d["skew"], "kurt": d["kurt"],
                "max_dd_excess": max_drawdown(ex),
                "hit_rate": float((ex > 0).mean()),
                "turnover_oneway_ann": float(m[f"{f}_{side}_to"].mean() * 12),
                "cost_drag_ann": float(((1 + ex_gross).prod() ** (12 / len(ex)) - 1)
                                       - ((1 + ex).prod() ** (12 / len(ex)) - 1)),
                "t_stat": float(ex.mean() / ex.std(ddof=1) * np.sqrt(len(ex))),
                "concentration": concentration(ex),
                "beta": beta_to_bench(ex, m.bench),
                "yearly": yearly(ex).reset_index().rename(columns={"R": "year"}).to_dict("records"),
                "loyo": leave_one_year_out(ex).to_dict("records"),
            }
    # decile monotonicity (gross, excess over benchmark)
    dec["ex"] = dec.ret - dec.bench
    for f in ALL:
        dd = dec[dec.factor == f].groupby("decile").ex.mean()
        rho, p = stats.spearmanr(dd.index, dd.values)
        res["deciles"][f] = {"mean_monthly_excess_by_decile": {int(k): float(v) for k, v in dd.items()},
                             "spearman_rho": float(rho), "spearman_p": float(p),
                             "D10_minus_D1": float(dd[10] - dd[1]),
                             "D10_minus_D9": float(dd[10] - dd[9]), "D2_minus_D1": float(dd[2] - dd[1])}
    # universe diagnostics + delisting proof
    close = P["close"]
    last_obs = close.apply(lambda c: c.last_valid_index())
    end = close.index[-1]
    exits = {}
    ever = set()
    names_per_year = {}
    for R, uni in members.items():
        ever |= set(uni)
        names_per_year.setdefault(R.year, set()).update(uni)
    exit_dates = last_obs[list(ever)]
    exit_dates = exit_dates[exit_dates < end - pd.Timedelta(days=7)]
    for y, s in names_per_year.items():
        ex_y = exit_dates[exit_dates.index.isin(s) & (exit_dates.dt.year == y)]
        exits[int(y)] = {"members": len(s), "exited_panel_that_year": int(len(ex_y))}
    ref_inact = pd.read_parquet(CACHE / "ref" / "tickers_inactive.parquet")
    ref_inact = ref_inact[ref_inact.type == "CS"]
    ref_by_year = ref_inact.delisted_utc.str[:4].value_counts().sort_index()
    res["universe"] = {
        "n_per_month_min": int(m.n_uni.min()), "n_per_month_max": int(m.n_uni.max()),
        "distinct_entities_ever": len(ever),
        "exits_by_year": exits,
        "polygon_delisted_CS_by_year_all_market": {k: int(v) for k, v in ref_by_year.items()
                                                    if k >= "2021"},
        "median_liq_usd_last_month": None,
    }
    return res


# ---------------------------------------------------------------------------- report
def fmt_pct(x): return f"{100 * x:+.2f} %"


def write_report(res: dict, m: pd.DataFrame) -> None:
    L = []
    L.append("# Querschnitts-Faktortest, breites US-Universum\n")
    L.append(f"Zeitraum der Monatsrenditen: **{res['first_R']} bis {res['last_Rn']}** "
             f"({res['months']} Monate). Universum {res['universe']['n_per_month_min']}–"
             f"{res['universe']['n_per_month_max']} Namen je Monat, "
             f"{res['universe']['distinct_entities_ever']} verschiedene Entitäten insgesamt.\n")
    L.append(f"Benchmark (gleichgewichtetes Universum, ohne Kosten): {fmt_pct(res['bench_ann'])} p.a., "
             f"Sharpe absolut {res['bench_sharpe_abs']:.2f}, Umschlag {res['bench_turnover_ann']:.2f}x p.a.\n")
    for side, title in (("L", "Long-only, oberstes Dezil (Überschuss über Benchmark)"),
                        ("LS", "Long-Short, D10 minus D1")):
        L.append(f"\n## {title}\n")
        L.append("| Portfolio | Überschuss p.a. netto | brutto | Sharpe netto | DSR | PSR>0 | MaxDD | Trefferquote | Umschlag p.a. | Kosten p.a. | t |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for f in ALL:
            p = res["portfolios"][f"{f}_{side}"]
            L.append(f"| {f} | {fmt_pct(p['ann_excess_net'])} | {fmt_pct(p['ann_excess_gross'])} | "
                     f"{p['sharpe_net']:.2f} | {p['dsr']:.2f} | {p['psr_vs_zero']:.2f} | "
                     f"{fmt_pct(p['max_dd_excess'])} | {100 * p['hit_rate']:.0f} % | "
                     f"{p['turnover_oneway_ann']:.1f}x | {fmt_pct(-p['cost_drag_ann'])} | {p['t_stat']:.2f} |")
        L.append("\n### Diagnose: Konzentration und Beta (Überschuss netto)\n")
        L.append("| Portfolio | bester Monat | dessen Überschuss | p.a. ohne besten Monat | p.a. ohne beste 3 | Sharpe ohne beste 3 | Beta zur Benchmark (t) | Alpha p.a. (t) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for f in ALL:
            p = res["portfolios"][f"{f}_{side}"]; c = p["concentration"]; b = p["beta"]
            L.append(f"| {f} | {c['best_month'][:7]} | {fmt_pct(c['best_month_excess'])} | {fmt_pct(c['ann_excess_drop_best1'])} | "
                     f"{fmt_pct(c['ann_excess_drop_best3'])} | {c['sharpe_drop_best3']:.2f} | {b['beta']:+.2f} ({b['beta_t']:.1f}) | "
                     f"{fmt_pct(b['alpha_ann'])} ({b['alpha_t']:.1f}) |")
        L.append("\n### Verteilung über die Jahre (Überschuss netto, Trefferquote)\n")
        years = sorted({r["year"] for r in res["portfolios"][f"{ALL[0]}_{side}"]["yearly"]})
        L.append("| Portfolio | " + " | ".join(str(y) for y in years) + " |")
        L.append("|---|" + "---|" * len(years))
        for f in ALL:
            yr = {r["year"]: r for r in res["portfolios"][f"{f}_{side}"]["yearly"]}
            L.append(f"| {f} | " + " | ".join(
                f"{fmt_pct(yr[y]['excess_ret'])} ({100 * yr[y]['hit_rate']:.0f} %, n={yr[y]['months']})"
                if y in yr else "–" for y in years) + " |")
        L.append("\n### Leave-one-year-out (Überschuss p.a. netto / Sharpe ohne das Jahr)\n")
        L.append("| Portfolio | " + " | ".join(f"ohne {y}" for y in years) + " |")
        L.append("|---|" + "---|" * len(years))
        for f in ALL:
            lo = {r["excluded_year"]: r for r in res["portfolios"][f"{f}_{side}"]["loyo"]}
            L.append(f"| {f} | " + " | ".join(
                f"{fmt_pct(lo[y]['ann_excess'])} / {lo[y]['sharpe']:.2f}" for y in years) + " |")
    L.append("\n## Dezil-Monotonie (mittlerer Monats-Überschuss je Dezil, brutto)\n")
    L.append("| Faktor | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | Spearman ρ (p) | D10−D1 | D10−D9 | D2−D1 |")
    L.append("|---|" + "---|" * 14)
    for f in ALL:
        d = res["deciles"][f]
        cells = " | ".join(f"{100 * d['mean_monthly_excess_by_decile'][k]:+.2f}" for k in range(1, 11))
        L.append(f"| {f} | {cells} | {d['spearman_rho']:.2f} ({d['spearman_p']:.2f}) | "
                 f"{100 * d['D10_minus_D1']:+.2f} | {100 * d['D10_minus_D9']:+.2f} | {100 * d['D2_minus_D1']:+.2f} |")
    L.append("\n## Universum: Delistings sind enthalten\n")
    L.append("| Jahr | Universumsmitglieder im Jahr | davon im Jahr aus dem Panel ausgeschieden | Polygon: delistete CS gesamtmarkt |")
    L.append("|---|---|---|---|")
    pdl = res["universe"]["polygon_delisted_CS_by_year_all_market"]
    for y, v in sorted(res["universe"]["exits_by_year"].items()):
        L.append(f"| {y} | {v['members']} | {v['exited_panel_that_year']} | {pdl.get(str(y), '–')} |")
    OUT.joinpath("REPORT_TABLES.md").write_text("\n".join(L) + "\n")
    OUT.joinpath("results.json").write_text(json.dumps(res, indent=1, default=str))
    m.to_csv(OUT / "monthly_returns.csv")
    log.info("report tables written to %s", OUT)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    P, elig = load_panel()
    m, dec, members, hold = run(P, elig)
    dec.to_csv(OUT / "decile_returns.csv", index=False)
    hold.to_parquet(OUT / "holdings.parquet", index=False)
    big_moves_in_holdings(hold, P, elig)
    res = evaluate(m, dec, members, P, elig)
    write_report(res, m)
    print(json.dumps({k: v for k, v in res.items() if k in ("months", "first_R", "last_Rn", "bench_ann")}))


if __name__ == "__main__":
    main()
