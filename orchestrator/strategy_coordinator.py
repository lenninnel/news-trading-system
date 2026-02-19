"""
StrategyCoordinator — runs all three strategy agents in parallel and
fuses their signals into a single ranked recommendation.

Pipeline
--------
1. Execute MomentumAgent, MeanReversionAgent, SwingAgent concurrently
   via ThreadPoolExecutor(max_workers=3).

2. Filter: discard HOLD signals and signals with confidence < 40%.

3. Consensus check:
       unanimous  — all surviving signals agree on direction
       majority   — 2 of 3 agree
       conflicting — even split (e.g. 1 BUY + 1 SELL)

4. In conflicting case: keep the majority direction; on exact tie keep the
   single highest-confidence signal and label as "conflicting".

5. Rank surviving signals by confidence descending.

6. Ensemble confidence = weighted average (weights: 1.0, 0.8, 0.6 for ranks
   1–3) with +5 bonus for unanimous consensus.

7. Call RiskAgent with combined_strategy_signal + ensemble_confidence.

8. Persist:
       strategy_signals     — one row per agent
       strategy_performance — one summary row per coordinator run
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from datetime import datetime, timezone
from typing import Any

from agents.momentum_agent import MomentumAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.swing_agent import SwingAgent
from agents.strategy_base import StrategySignal
from agents.risk_agent import RiskAgent
from storage.database import Database

log = logging.getLogger(__name__)

# Signals below this threshold are treated as noise
_MIN_CONFIDENCE: float = 40.0
# Confidence weights for rank-1, rank-2, rank-3
_RANK_WEIGHTS: list[float] = [1.0, 0.8, 0.6]


class StrategyCoordinator:
    """
    Parallel multi-strategy signal aggregator.

    Runs MomentumAgent, MeanReversionAgent, and SwingAgent in parallel,
    ranks their signals, and delegates position sizing to RiskAgent.

    Args:
        momentum_agent:       MomentumAgent instance (created if omitted).
        mean_reversion_agent: MeanReversionAgent instance (created if omitted).
        swing_agent:          SwingAgent instance (created if omitted).
        risk_agent:           RiskAgent instance (created if omitted).
        db:                   Database instance shared across all agents.
        max_workers:          ThreadPoolExecutor thread count. Default 3.
    """

    def __init__(
        self,
        momentum_agent: MomentumAgent | None = None,
        mean_reversion_agent: MeanReversionAgent | None = None,
        swing_agent: SwingAgent | None = None,
        risk_agent: RiskAgent | None = None,
        db: Database | None = None,
        max_workers: int = 3,
    ) -> None:
        self._db             = db or Database()
        self._momentum       = momentum_agent       or MomentumAgent(db=self._db)
        self._mean_reversion = mean_reversion_agent or MeanReversionAgent(db=self._db)
        self._swing          = swing_agent          or SwingAgent(db=self._db)
        self._risk           = risk_agent           or RiskAgent(db=self._db)
        self._max_workers    = max_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        ticker: str,
        account_balance: float = 10_000.0,
        verbose: bool = True,
    ) -> dict:
        """
        Run all three strategy agents in parallel and return a combined result.

        Args:
            ticker:          Stock ticker symbol.
            account_balance: Account value for RiskAgent position sizing.
            verbose:         Print progress to stdout.

        Returns:
            dict with keys:
                ticker                   (str)
                strategy_signals         (list[dict]) — all three agents' output
                ranked_signals           (list[dict]) — filtered + ranked subset
                combined_strategy_signal (str)        — "BUY" | "SELL" | "HOLD"
                ensemble_confidence      (float)      — 0.0 – 100.0
                consensus                (str)        — "unanimous" | "majority" |
                                                        "conflicting" | "none"
                risk                     (dict)       — RiskAgent output
                strategy_run_id          (int)        — strategy_performance PK
                errors                   (list[str])  — per-agent error messages
        """
        ticker = ticker.upper()
        run_at = datetime.now(timezone.utc).isoformat()

        if verbose:
            print(
                f"\n  Running {self._momentum.name}, "
                f"{self._mean_reversion.name}, "
                f"{self._swing.name} in parallel..."
            )

        # 1. Parallel execution
        signals, errors = self._run_all_agents(ticker)

        if verbose:
            for sig in signals:
                icon = {"BUY": "+", "SELL": "-", "HOLD": "~"}.get(sig.signal, "?")
                print(
                    f"  [{sig.strategy.upper():17s}] [{icon}] {sig.signal:<4}  "
                    f"conf {sig.confidence:.0f}%"
                )
            for err in errors:
                print(f"  [!] {err}")

        # 2. Rank + consensus
        ranked, combined_signal, ensemble_conf, consensus = self._rank_signals(signals)

        if verbose:
            print(
                f"\n  Combined strategy signal : {combined_signal}  "
                f"(ensemble confidence: {ensemble_conf:.1f}%,  "
                f"consensus: {consensus})"
            )

        # 3. Risk sizing
        price = self._extract_price(signals)
        risk = self._risk.run(
            ticker=ticker,
            signal=combined_signal,
            confidence=ensemble_conf,
            current_price=price or 0.0,
            account_balance=account_balance,
        )

        if verbose:
            if risk["skipped"]:
                print(f"  Risk: no position — {risk['skip_reason']}")
            else:
                print(
                    f"  Risk: ${risk['position_size_usd']:,.2f} "
                    f"({risk['shares']} shares)  "
                    f"SL ${risk['stop_loss']:.2f}  "
                    f"TP ${risk['take_profit']:.2f}"
                )

        # 4. Persist
        signal_ids = self._persist_signals(
            signals=signals,
            run_at=run_at,
            combined_signal=combined_signal,
            ensemble_conf=ensemble_conf,
            consensus=consensus,
            account_balance=account_balance,
            risk_calc_id=risk.get("calc_id"),
        )
        for i, sig in enumerate(signals):
            sig.signal_id = signal_ids[i] if i < len(signal_ids) else None

        perf_id = self._persist_performance(
            ticker=ticker,
            run_at=run_at,
            signals=signals,
            combined_signal=combined_signal,
            ensemble_conf=ensemble_conf,
            consensus=consensus,
            account_balance=account_balance,
            risk_calc_id=risk.get("calc_id"),
            errors=errors,
        )

        # Augment risk dict with context values for display/execution layers
        risk["current_price"]   = price or 0.0
        risk["account_balance"] = account_balance

        return {
            "ticker":                   ticker,
            "strategy_signals":         [self._signal_to_dict(s) for s in signals],
            "ranked_signals":           [self._signal_to_dict(s) for s in ranked],
            "combined_strategy_signal": combined_signal,
            "ensemble_confidence":      ensemble_conf,
            "consensus":                consensus,
            "risk":                     risk,
            "account_balance":          account_balance,
            "strategy_run_id":          perf_id,
            "errors":                   errors,
        }

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

    def _run_all_agents(
        self, ticker: str
    ) -> tuple[list[StrategySignal], list[str]]:
        """
        Execute all three agents concurrently.

        Uses as_completed() so a single agent failure does not cancel others.
        """
        agents = [self._momentum, self._mean_reversion, self._swing]
        results: list[StrategySignal] = []
        errors:  list[str]            = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            future_to_name = {
                executor.submit(agent.run, ticker): agent.name
                for agent in agents
            }
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    log.error("%s raised during parallel execution: %s", name, exc)
                    errors.append(f"{name}: {exc}")

        return results, errors

    # ------------------------------------------------------------------
    # Signal ranking + ensemble
    # ------------------------------------------------------------------

    def _rank_signals(
        self,
        signals: list[StrategySignal],
    ) -> tuple[list[StrategySignal], str, float, str]:
        """
        Filter, rank, and compute ensemble direction + confidence.

        Returns:
            (ranked_signals, combined_direction, ensemble_confidence, consensus_label)
        """
        # Filter: active signals only
        active = [
            s for s in signals
            if s.signal in ("BUY", "SELL") and s.confidence >= _MIN_CONFIDENCE
        ]

        if not active:
            return [], "HOLD", 25.0, "none"

        buys  = [s for s in active if s.signal == "BUY"]
        sells = [s for s in active if s.signal == "SELL"]

        if buys and sells:
            # Directional conflict — keep majority
            if len(buys) > len(sells):
                active, consensus, direction = buys, "majority", "BUY"
            elif len(sells) > len(buys):
                active, consensus, direction = sells, "majority", "SELL"
            else:
                # Exact tie — keep single highest-confidence signal
                best   = max(active, key=lambda s: s.confidence)
                active    = [best]
                consensus = "conflicting"
                direction = best.signal
        elif buys:
            direction = "BUY"
            consensus = "unanimous" if len(buys) == len(signals) else "majority"
        else:
            direction = "SELL"
            consensus = "unanimous" if len(sells) == len(signals) else "majority"

        ranked   = sorted(active, key=lambda s: s.confidence, reverse=True)
        ensemble = self._ensemble_confidence(ranked, consensus)

        return ranked, direction, ensemble, consensus

    @staticmethod
    def _ensemble_confidence(ranked: list[StrategySignal], consensus: str) -> float:
        if not ranked:
            return 25.0
        weights      = _RANK_WEIGHTS[: len(ranked)]
        total_w      = sum(weights)
        weighted_sum = sum(s.confidence * w for s, w in zip(ranked, weights))
        base         = weighted_sum / total_w
        if consensus == "unanimous":
            base += 5.0
        return round(min(max(base, 0.0), 100.0), 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_price(signals: list[StrategySignal]) -> float | None:
        """Extract the first valid price from any signal's indicators."""
        for sig in signals:
            p = sig.indicators.get("price")
            if p is not None and float(p) > 0:
                return float(p)
        return None

    @staticmethod
    def _signal_to_dict(sig: StrategySignal) -> dict:
        return {
            "ticker":     sig.ticker,
            "strategy":   sig.strategy,
            "signal":     sig.signal,
            "confidence": sig.confidence,
            "reasoning":  sig.reasoning,
            "indicators": sig.indicators,
            "timeframe":  sig.timeframe,
            "signal_id":  sig.signal_id,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_signals(
        self,
        signals: list[StrategySignal],
        run_at: str,
        combined_signal: str,
        ensemble_conf: float,
        consensus: str,
        account_balance: float,
        risk_calc_id: int | None,
    ) -> list[int]:
        ids: list[int] = []
        for sig in signals:
            try:
                sid = self._db.log_strategy_signal(
                    ticker=sig.ticker,
                    strategy=sig.strategy,
                    signal=sig.signal,
                    confidence=sig.confidence,
                    timeframe=sig.timeframe,
                    reasoning="; ".join(sig.reasoning),
                    indicators_json=json.dumps(sig.indicators),
                    ensemble_confidence=ensemble_conf,
                    combined_signal=combined_signal,
                    consensus=consensus,
                    account_balance=account_balance,
                    risk_calc_id=risk_calc_id,
                )
                ids.append(sid)
            except Exception as exc:
                log.warning("Could not persist signal for %s: %s", sig.strategy, exc)
                ids.append(-1)
        return ids

    def _persist_performance(
        self,
        ticker: str,
        run_at: str,
        signals: list[StrategySignal],
        combined_signal: str,
        ensemble_conf: float,
        consensus: str,
        account_balance: float,
        risk_calc_id: int | None,
        errors: list[str],
    ) -> int:
        sig_map = {s.strategy: s for s in signals}
        m  = sig_map.get("momentum")
        r  = sig_map.get("mean_reversion")
        sw = sig_map.get("swing")
        try:
            return self._db.log_strategy_performance(
                ticker=ticker,
                run_at=run_at,
                momentum_signal=m.signal if m else None,
                momentum_confidence=m.confidence if m else None,
                mean_reversion_signal=r.signal if r else None,
                mean_reversion_confidence=r.confidence if r else None,
                swing_signal=sw.signal if sw else None,
                swing_confidence=sw.confidence if sw else None,
                combined_signal=combined_signal,
                ensemble_confidence=ensemble_conf,
                consensus=consensus,
                risk_calc_id=risk_calc_id,
                account_balance=account_balance,
                errors_json=json.dumps(errors),
            )
        except Exception as exc:
            log.warning("Could not persist strategy performance: %s", exc)
            return -1
