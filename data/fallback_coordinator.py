"""
FallbackCoordinator — central registry for all fallback chains.

Tracks which data-source fallback level is currently active for each
service (news, price, anthropic, …).  Emits a warning when a service
has been running on a non-primary fallback for more than
``ALERT_THRESHOLD_HOURS`` hours, and provides a ``daily_health_check``
that probes each registered fallback handler.

Singleton design
----------------
All state is stored in class attributes so the coordinator is shared
across all modules without requiring explicit dependency injection.
Call ``FallbackCoordinator.set_db(db)`` once at startup if you want
fallback events written to the recovery_log table.

Usage
-----
    from data.fallback_coordinator import FallbackCoordinator

    # In any data-source module after a successful fetch:
    FallbackCoordinator.register("news", level=1, source="rss_yahoo", ticker="AAPL")

    # Periodic alerting (called from scheduler):
    alerts = FallbackCoordinator.check_and_alert()

    # Full health check (call once per day):
    report = FallbackCoordinator.daily_health_check()
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Any

log = logging.getLogger(__name__)

_ALERT_THRESHOLD_HOURS: int = 24   # alert when stuck on fallback > this long
_PRIMARY_LEVEL: int = 0            # level 0 is always the primary source


class FallbackCoordinator:
    """
    Class-level registry of active fallback levels across all services.

    Attributes (class-level, shared):
        _registry  : dict mapping service → latest registration metadata.
        _db        : Optional Database for writing recovery_log events.
        _lock      : Thread-lock protecting _registry mutations.
    """

    _registry: dict[str, dict] = {}   # {service: {level, source, since, ticker}}
    _db: "Any | None" = None
    _lock: threading.Lock = threading.Lock()

    # -- Setup ----------------------------------------------------------------

    @classmethod
    def set_db(cls, db: "Any") -> None:
        """Attach a Database instance for recovery_log writes."""
        cls._db = db

    # -- Registration ---------------------------------------------------------

    @classmethod
    def register(
        cls,
        service: str,
        level: int,
        source: str,
        ticker: str = "",
    ) -> None:
        """
        Record the current fallback level for *service*.

        Call this after every successful data fetch.  Level 0 = primary,
        level 1+ = fallback.  Consecutive calls at the same level update
        the ticker but do NOT reset the ``since`` timestamp, so alert
        thresholds accumulate correctly.

        Args:
            service: Logical service name, e.g. "news", "price", "anthropic".
            level:   Fallback level (0 = primary, 1+ = degraded).
            source:  Human-readable source name, e.g. "newsapi", "rss_yahoo".
            ticker:  Optional ticker for context in alerts.
        """
        with cls._lock:
            existing = cls._registry.get(service)
            if existing is None or existing["level"] != level:
                # Level changed (or first registration) — reset the timer.
                cls._registry[service] = {
                    "level":  level,
                    "source": source,
                    "since":  datetime.utcnow(),
                    "ticker": ticker,
                    "count":  1,
                }
                if level > _PRIMARY_LEVEL:
                    log.warning(
                        "[FALLBACK COORDINATOR] %s switched to L%d (%s) ticker=%s",
                        service, level, source, ticker or "—",
                    )
                else:
                    log.info(
                        "[FALLBACK COORDINATOR] %s restored to primary (L0)", service
                    )
            else:
                # Same level — just bump the counter and update ticker.
                existing["count"] += 1
                if ticker:
                    existing["ticker"] = ticker

    # -- Alert checking -------------------------------------------------------

    @classmethod
    def check_and_alert(cls) -> list[dict]:
        """
        Return a list of services that have been on a fallback >24 h.

        Side-effects:
            • Logs a WARNING for every affected service.
            • If a db is attached, writes a recovery_log entry.

        Returns:
            List of dicts: {service, level, source, since, hours_elapsed, ticker}
        """
        threshold = timedelta(hours=_ALERT_THRESHOLD_HOURS)
        now = datetime.utcnow()
        alerts: list[dict] = []

        with cls._lock:
            snapshot = dict(cls._registry)

        for service, info in snapshot.items():
            if info["level"] <= _PRIMARY_LEVEL:
                continue  # primary — no alert needed

            elapsed = now - info["since"]
            if elapsed >= threshold:
                hours = elapsed.total_seconds() / 3600
                alert = {
                    "service":       service,
                    "level":         info["level"],
                    "source":        info["source"],
                    "since":         info["since"].isoformat(),
                    "hours_elapsed": round(hours, 1),
                    "ticker":        info.get("ticker", ""),
                }
                alerts.append(alert)
                log.warning(
                    "[FALLBACK ALERT] %s has been on L%d (%s) for %.1f h (ticker=%s)",
                    service, info["level"], info["source"], hours,
                    info.get("ticker", "—"),
                )
                cls._log_alert(service, info, hours)

        return alerts

    # -- Health check ---------------------------------------------------------

    @classmethod
    def daily_health_check(cls) -> dict:
        """
        Probe each registered service's primary source to assess recovery.

        For each service currently on a fallback, this attempts a live call
        to the primary source.  If the primary is reachable again, a
        WARNING is emitted recommending the operator investigate why the
        system is still using a fallback.

        Returns:
            dict: {
                "timestamp": str,
                "services_checked": int,
                "services_on_fallback": int,
                "alerts": list,
                "probe_results": dict[service → bool],   # True = primary reachable
            }
        """
        now = datetime.utcnow()
        alerts = cls.check_and_alert()

        with cls._lock:
            snapshot = dict(cls._registry)

        on_fallback = [s for s, i in snapshot.items() if i["level"] > _PRIMARY_LEVEL]
        probe_results: dict[str, bool] = {}

        for service in on_fallback:
            reachable = cls._probe_primary(service)
            probe_results[service] = reachable
            if reachable:
                log.warning(
                    "[HEALTH CHECK] %s primary IS reachable but system is on L%d — "
                    "check circuit breaker or configuration.",
                    service, snapshot[service]["level"],
                )

        report = {
            "timestamp":            now.isoformat(),
            "services_checked":     len(snapshot),
            "services_on_fallback": len(on_fallback),
            "alerts":               alerts,
            "probe_results":        probe_results,
        }
        log.info("[HEALTH CHECK] %s", report)
        return report

    # -- Status ---------------------------------------------------------------

    @classmethod
    def get_status(cls) -> dict[str, dict]:
        """
        Return a copy of the current registry (all services).

        Returns:
            dict mapping service name → registration metadata.
        """
        with cls._lock:
            return {
                service: {**info, "since": info["since"].isoformat()}
                for service, info in cls._registry.items()
            }

    @classmethod
    def reset(cls, service: str | None = None) -> None:
        """
        Clear registry entries (primarily for testing).

        Args:
            service: If given, clears only that service.
                     If None, clears the entire registry.
        """
        with cls._lock:
            if service is None:
                cls._registry.clear()
            else:
                cls._registry.pop(service, None)

    # -- Internal helpers -----------------------------------------------------

    @classmethod
    def _log_alert(cls, service: str, info: dict, hours: float) -> None:
        """Write a fallback alert event to the DB (best-effort)."""
        if cls._db is None:
            return
        try:
            cls._db.log_recovery_event(
                service=service,
                event_type="fallback_alert",
                ticker=info.get("ticker", ""),
                error_msg=(
                    f"Stuck on L{info['level']} ({info['source']}) "
                    f"for {hours:.1f} h"
                ),
                recovery_action="check_primary_source",
                success=False,
            )
        except Exception:
            pass

    @classmethod
    def _probe_primary(cls, service: str) -> bool:
        """
        Best-effort probe of the primary source for *service*.

        Returns True if reachable, False otherwise.  Never raises.
        """
        try:
            if service == "news":
                from utils.network_recovery import get_cache as _gc  # noqa: F401
                import requests
                r = requests.get("https://newsapi.org", timeout=5)
                return r.status_code < 500
            if service == "price":
                import yfinance as yf
                info = yf.Ticker("SPY").fast_info
                return bool(info)
            if service == "anthropic":
                import os
                return bool(os.environ.get("ANTHROPIC_API_KEY"))
        except Exception:
            pass
        return False
