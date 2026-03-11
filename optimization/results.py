"""
Persistence layer for walk-forward optimisation results.

Stores completed optimisation runs in the ``optimization_runs`` table and
provides a loader so past runs can be displayed on the dashboard.
"""

from __future__ import annotations

import json
from typing import Any

from storage.database import Database


def save_optimization_run(
    db: Database,
    ticker: str,
    start_date: str,
    end_date: str,
    best_params: dict[str, float],
    in_sample_sharpe: float,
    out_of_sample_sharpe: float,
    total_windows: int,
    equity_curve: list[float] | None = None,
) -> int:
    """
    Persist a completed optimisation run.

    Returns:
        The integer primary key of the newly inserted row.
    """
    return db.log_optimization_run(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        best_params=best_params,
        in_sample_sharpe=in_sample_sharpe,
        out_of_sample_sharpe=out_of_sample_sharpe,
        total_windows=total_windows,
        equity_curve=equity_curve,
    )


def load_optimization_runs(
    db: Database,
    ticker: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Load past optimisation runs, newest first.

    Args:
        db:     Database instance.
        ticker: Filter by ticker (None = all).
        limit:  Max rows to return.

    Returns:
        List of dicts with parsed ``best_params`` and ``equity_curve``.
    """
    rows = db.get_optimization_runs(ticker=ticker, limit=limit)
    for row in rows:
        if isinstance(row.get("best_params"), str):
            row["best_params"] = json.loads(row["best_params"])
        if isinstance(row.get("equity_curve"), str):
            row["equity_curve"] = json.loads(row["equity_curve"])
    return rows
