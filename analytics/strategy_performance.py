"""
Nightly per-strategy performance tracker.

Computes rolling 30-day metrics (Sharpe, win rate, signal count, avg
confidence) for each strategy and stores results in the
``strategy_perf_daily`` table.  Runs after the outcome tracker in
the EOD session.

The ``get_cluster_weights()`` method provides Sharpe-based weights
for the ClusterDetector — strategies with higher Sharpe get more vote
weight, with a floor at 0.1 to prevent zeroing out any strategy.

Usage::

    from analytics.strategy_performance import StrategyPerformanceTracker
    tracker = StrategyPerformanceTracker()
    metrics = tracker.compute()   # dict[str, StrategyMetrics]
    weights = tracker.get_cluster_weights()  # dict[str, float]
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from storage.database import Database

log = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS strategy_perf_daily (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name   TEXT    NOT NULL,
    run_date        TEXT    NOT NULL,
    sharpe_30d      REAL,
    win_rate_30d    REAL,
    avg_rr          REAL,
    signal_count    INTEGER,
    avg_confidence  REAL,
    UNIQUE(strategy_name, run_date)
)
"""

_KNOWN_STRATEGIES = ["Momentum", "Pullback", "NewsCatalyst", "PEAD", "Combined"]

_MIN_OUTCOMES_FOR_WEIGHT = 10
_WEIGHT_FLOOR = 0.1


@dataclass
class StrategyMetrics:
    """Rolling 30-day performance for a single strategy."""
    strategy: str
    sharpe_30d: float | None
    win_rate_30d: float | None
    avg_rr: float | None
    signal_count: int
    avg_confidence: float | None


class StrategyPerformanceTracker:
    """Computes and stores rolling per-strategy performance metrics."""

    def __init__(self, db: Database | None = None) -> None:
        self._db = db or Database()
        self._ensure_table()

    def _ensure_table(self) -> None:
        try:
            with self._db._connect() as conn:
                conn.executescript(_CREATE_TABLE)
        except Exception as exc:
            log.warning("strategy_perf_daily table creation failed: %s", exc)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self) -> dict[str, StrategyMetrics]:
        """
        Compute rolling 30-day metrics per strategy from signal_events.

        Returns:
            dict mapping strategy name → StrategyMetrics.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        results: dict[str, StrategyMetrics] = {}

        for strategy in _KNOWN_STRATEGIES:
            rows = self._query_strategy_outcomes(strategy, cutoff)
            metrics = self._compute_metrics(strategy, rows)
            results[strategy] = metrics
            self._store(metrics, today)

        return results

    def _query_strategy_outcomes(
        self, strategy: str, cutoff: str,
    ) -> list[dict]:
        """Query signal_events with outcomes for a specific strategy."""
        try:
            with self._db._connect() as conn:
                if strategy == "Combined":
                    # Combined = all strategies, or rows with no specific strategy
                    rows = conn.execute(
                        "SELECT signal, confidence, outcome_3d_pct, outcome_5d_pct, "
                        "       price_at_signal, outcome_correct "
                        "FROM signal_events "
                        "WHERE timestamp >= ? AND outcome_3d_pct IS NOT NULL",
                        (cutoff,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT signal, confidence, outcome_3d_pct, outcome_5d_pct, "
                        "       price_at_signal, outcome_correct "
                        "FROM signal_events "
                        "WHERE strategy = ? AND timestamp >= ? "
                        "AND outcome_3d_pct IS NOT NULL",
                        (strategy, cutoff),
                    ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:
            log.warning("Failed to query outcomes for %s: %s", strategy, exc)
            return []

    @staticmethod
    def _compute_metrics(strategy: str, rows: list[dict]) -> StrategyMetrics:
        """Compute Sharpe, win rate, avg R:R, count, avg confidence."""
        if not rows:
            return StrategyMetrics(
                strategy=strategy, sharpe_30d=None, win_rate_30d=None,
                avg_rr=None, signal_count=0, avg_confidence=None,
            )

        n = len(rows)

        # Win rate: signals where outcome_3d_pct > 0 for BUY, < 0 for SELL
        wins = sum(1 for r in rows if r.get("outcome_correct") == 1)
        win_rate = wins / n if n > 0 else None

        # Returns for Sharpe
        returns = [r["outcome_3d_pct"] / 100.0 for r in rows if r.get("outcome_3d_pct") is not None]
        if len(returns) >= 2:
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
            std_ret = math.sqrt(variance) if variance > 0 else 0
            sharpe = (mean_ret / std_ret * math.sqrt(252 / 3)) if std_ret > 0 else 0.0
        elif len(returns) == 1:
            sharpe = 0.0
        else:
            sharpe = None

        # Avg R:R (using outcome_5d / outcome_3d as proxy)
        rr_values = []
        for r in rows:
            o3 = r.get("outcome_3d_pct")
            o5 = r.get("outcome_5d_pct")
            if o3 is not None and o5 is not None and o3 != 0:
                rr_values.append(abs(o5) / abs(o3))
        avg_rr = sum(rr_values) / len(rr_values) if rr_values else None

        # Avg confidence
        confs = [r["confidence"] for r in rows if r.get("confidence") is not None]
        avg_conf = sum(confs) / len(confs) if confs else None

        return StrategyMetrics(
            strategy=strategy,
            sharpe_30d=round(sharpe, 4) if sharpe is not None else None,
            win_rate_30d=round(win_rate, 4) if win_rate is not None else None,
            avg_rr=round(avg_rr, 4) if avg_rr is not None else None,
            signal_count=n,
            avg_confidence=round(avg_conf, 4) if avg_conf is not None else None,
        )

    def _store(self, metrics: StrategyMetrics, date_str: str) -> None:
        """Upsert metrics into strategy_perf_daily table."""
        try:
            with self._db._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO strategy_perf_daily
                        (strategy_name, run_date, sharpe_30d, win_rate_30d, avg_rr,
                         signal_count, avg_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(strategy_name, run_date) DO UPDATE SET
                        sharpe_30d = excluded.sharpe_30d,
                        win_rate_30d = excluded.win_rate_30d,
                        avg_rr = excluded.avg_rr,
                        signal_count = excluded.signal_count,
                        avg_confidence = excluded.avg_confidence
                    """,
                    (
                        metrics.strategy, date_str, metrics.sharpe_30d,
                        metrics.win_rate_30d, metrics.avg_rr,
                        metrics.signal_count, metrics.avg_confidence,
                    ),
                )
        except Exception as exc:
            log.warning("Failed to store strategy_perf_daily for %s: %s",
                        metrics.strategy, exc)

    # ------------------------------------------------------------------
    # Cluster weights
    # ------------------------------------------------------------------

    def get_cluster_weights(self) -> dict[str, float]:
        """
        Return Sharpe-based weights for ClusterDetector vote fusion.

        Weight = max(0.1, sharpe).  Strategies with < 10 outcomes get
        equal weight.  Falls back to equal weights when no data exists.
        """
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            with self._db._connect() as conn:
                rows = conn.execute(
                    "SELECT strategy_name, sharpe_30d, signal_count "
                    "FROM strategy_perf_daily WHERE run_date = ?",
                    (today,),
                ).fetchall()
            rows = [dict(r) for r in rows]
        except Exception:
            rows = []

        if not rows:
            return {s: 1.0 for s in _KNOWN_STRATEGIES}

        weights: dict[str, float] = {}
        for row in rows:
            strategy = row["strategy_name"]
            count = row.get("signal_count", 0) or 0
            sharpe = row.get("sharpe_30d")

            if count < _MIN_OUTCOMES_FOR_WEIGHT or sharpe is None:
                weights[strategy] = 1.0
            else:
                weights[strategy] = max(_WEIGHT_FLOOR, sharpe)

        # Fill missing strategies with equal weight
        for s in _KNOWN_STRATEGIES:
            if s not in weights:
                weights[s] = 1.0

        return weights

    # ------------------------------------------------------------------
    # Read latest metrics
    # ------------------------------------------------------------------

    def get_latest(self) -> list[dict]:
        """Return the most recent row per strategy."""
        try:
            with self._db._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM strategy_perf_daily "
                    "WHERE run_date = (SELECT MAX(run_date) FROM strategy_perf_daily) "
                    "ORDER BY strategy_name",
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:
            log.warning("get_latest() failed: %s", exc)
            return []
