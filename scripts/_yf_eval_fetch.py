#!/usr/bin/env python3
"""yfinance fetch helper for the Q2 earnings source eval (R spec 2026-08-26).

Executed as a SUBPROCESS by scripts/earnings_source_eval.py via the
isolated interpreter at $YF_EVAL_PYTHON (default
/home/trading/yfeval-venv/bin/python3) — NEVER imported by the main
script.  The eval job itself runs in the prod venv whose yfinance
(0.2.58) has a broken earnings_dates endpoint; this helper is the only
code that touches yfinance, inside its own venv.

Contract
--------
argv: --mode cal|eps --tickers T1,T2,...

stdout: exactly ONE json object:
    {"version": "<yfinance.__version__>",
     "rows": [...],
     "errors": [{"ticker": ..., "error": ...}, ...]}

cal rows:  {ticker, report_date (first "Earnings Date" element, ISO),
            estimate ("Earnings Average"), raw (whole calendar dict)}
eps rows:  {ticker, report_date, estimate_eps ("EPS Estimate"),
            actual_eps ("Reported EPS"), surprise_pct ("Surprise(%)"),
            report_time ("HH:MM" from the row timestamp when it carries
            a non-midnight time-of-day, else null)}

This process NEVER raises out: per-ticker failures land in errors[],
and even an import failure still prints the JSON envelope and exits 0.
The parent owns the timeout and treats bad JSON / non-zero exit as a
provider failure.
"""
from __future__ import annotations

import argparse
import json
import sys


def _f(value):
    """Coerce to float; NaN and unparseable values become None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _cal_rows(yf, tickers, rows, errors):
    for t in tickers:
        try:
            cal = yf.Ticker(t).calendar
            if not isinstance(cal, dict):
                raise TypeError(
                    f"unexpected calendar type: {type(cal).__name__}")
            earnings_dates = cal.get("Earnings Date") or []
            report_date = str(earnings_dates[0]) if earnings_dates else None
            rows.append({
                "ticker": t,
                "report_date": report_date,
                "estimate": _f(cal.get("Earnings Average")),
                "raw": {k: str(v) for k, v in cal.items()},
            })
        except Exception as exc:  # noqa: BLE001 — per-ticker fail-soft
            errors.append({"ticker": t,
                           "error": f"{type(exc).__name__}: {exc}"})


def _eps_rows(yf, tickers, rows, errors):
    for t in tickers:
        try:
            df = yf.Ticker(t).get_earnings_dates(limit=12)
            if df is None:
                continue
            for idx, row in df.iterrows():
                ts = idx.to_pydatetime()
                report_time = (ts.strftime("%H:%M")
                               if (ts.hour, ts.minute) != (0, 0) else None)
                rows.append({
                    "ticker": t,
                    "report_date": ts.date().isoformat(),
                    "estimate_eps": _f(row.get("EPS Estimate")),
                    "actual_eps": _f(row.get("Reported EPS")),
                    "surprise_pct": _f(row.get("Surprise(%)")),
                    "report_time": report_time,
                })
        except Exception as exc:  # noqa: BLE001 — per-ticker fail-soft
            errors.append({"ticker": t,
                           "error": f"{type(exc).__name__}: {exc}"})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("cal", "eps"))
    parser.add_argument("--tickers", required=True)
    args = parser.parse_args(argv)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    out = {"version": None, "rows": [], "errors": []}
    try:
        import yfinance as yf
        out["version"] = getattr(yf, "__version__", "unknown")
        if args.mode == "cal":
            _cal_rows(yf, tickers, out["rows"], out["errors"])
        else:
            _eps_rows(yf, tickers, out["rows"], out["errors"])
    except Exception as exc:  # noqa: BLE001 — envelope always printed
        out["errors"].append({"ticker": None,
                              "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
