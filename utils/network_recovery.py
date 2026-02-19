"""
Network Recovery — outage detection, response caching, and degraded mode.

NetworkMonitor
--------------
Probes two reliable HTTP endpoints to determine connectivity.  When both
fail the system enters "degraded mode": all agents serve cached data and
live API calls are skipped.  When connectivity is restored an alert is
sent via the DB recovery_log and normal operation resumes.

ResponseCache
-------------
Thread-safe in-memory store for last-known-good API responses.  Keys are
(service, cache_key) pairs.  Entries older than max_age_seconds are
treated as cache misses.

Usage
-----
    from utils.network_recovery import NetworkMonitor, get_cache

    # Normal flow — store each successful response
    online = NetworkMonitor.check_and_update()
    if online:
        data = live_api_call(ticker)
        get_cache().set("newsapi", ticker, data)
    else:
        data, hit = get_cache().get("newsapi", ticker)
        if not hit:
            data = []   # nothing available at all

    # One-shot connectivity probe
    if NetworkMonitor.is_online():
        ...
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

import requests

log = logging.getLogger(__name__)

# ── Probe configuration ────────────────────────────────────────────────────────

# Probed in order; first success → online.
_PROBE_URLS    = [
    "https://httpbin.org/status/200",
    "https://www.google.com",
]
_PROBE_TIMEOUT = 5   # seconds per attempt


# ── Response Cache ─────────────────────────────────────────────────────────────

class ResponseCache:
    """
    Thread-safe TTL-based cache for last successful API responses.

    Args:
        max_age_seconds: Seconds before an entry is considered stale.
    """

    def __init__(self, max_age_seconds: float = 3600.0) -> None:
        self._max_age = max_age_seconds
        self._store:  dict[tuple, tuple[Any, datetime]] = {}
        self._lock    = threading.Lock()

    def set(self, service: str, key: str, value: Any) -> None:
        """Store *value* under *(service, key)* with the current timestamp."""
        with self._lock:
            self._store[(service, key)] = (value, datetime.now(timezone.utc))

    def get(self, service: str, key: str) -> tuple[Any, bool]:
        """
        Retrieve a cached value.

        Returns:
            (value, True)   when the entry exists and is within max_age.
            (None,  False)  when absent or stale.
        """
        with self._lock:
            entry = self._store.get((service, key))
            if entry is None:
                return None, False
            value, stored_at = entry
            age = (datetime.now(timezone.utc) - stored_at).total_seconds()
            if age > self._max_age:
                return None, False
            return value, True

    def has(self, service: str, key: str) -> bool:
        """Return True when a fresh entry exists for *(service, key)*."""
        _, hit = self.get(service, key)
        return hit

    def clear(self, service: "str | None" = None) -> None:
        """Delete all entries, or only those belonging to *service*."""
        with self._lock:
            if service is None:
                self._store.clear()
            else:
                self._store = {
                    k: v for k, v in self._store.items() if k[0] != service
                }

    def stats(self) -> dict:
        """Return summary counts for monitoring/status endpoints."""
        with self._lock:
            now   = datetime.now(timezone.utc)
            total = len(self._store)
            fresh = sum(
                1 for _, (_, ts) in self._store.items()
                if (now - ts).total_seconds() <= self._max_age
            )
            return {"total_entries": total, "fresh_entries": fresh, "stale_entries": total - fresh}


# Module-level singleton shared across all modules
_cache = ResponseCache()


def get_cache() -> ResponseCache:
    """Return the module-level ResponseCache singleton."""
    return _cache


# ── Network Monitor ────────────────────────────────────────────────────────────

class NetworkMonitor:
    """
    Detects network outages and manages degraded-mode operation.

    All state is class-level so it is shared across all code in the process.

    Degraded mode
    -------------
    When both probe URLs are unreachable the monitor sets _degraded=True.
    All callers that check ``NetworkMonitor.is_degraded()`` should skip
    live API calls and serve cached data instead.  When the network is
    restored the flag is cleared and an event is logged.

    Thread safety
    -------------
    All mutations are protected by a single class-level lock.
    """

    _lock:           threading.Lock       = threading.Lock()
    _degraded:       bool                 = False
    _offline_since:  "datetime | None"    = None
    _last_check_at:  "datetime | None"    = None
    _check_interval: float                = 60.0   # minimum seconds between probes
    _db:             Any                  = None

    # -- Setup -----------------------------------------------------------------

    @classmethod
    def set_db(cls, db: Any) -> None:
        """Attach a Database instance for recovery_log persistence."""
        cls._db = db

    # -- Probe -----------------------------------------------------------------

    @classmethod
    def is_online(cls) -> bool:
        """
        Probe external connectivity right now (ignores throttle interval).

        Returns:
            True  → at least one probe URL responded.
            False → all probe URLs failed.
        """
        for url in _PROBE_URLS:
            try:
                requests.head(url, timeout=_PROBE_TIMEOUT, allow_redirects=True)
                return True
            except Exception:
                continue
        return False

    # -- State management ------------------------------------------------------

    @classmethod
    def check_and_update(cls, force: bool = False) -> bool:
        """
        Probe connectivity, update the degraded-mode flag, and alert on restore.

        The probe is skipped if checked within *_check_interval* seconds
        unless *force=True*.

        Returns:
            True  → online (normal operation).
            False → offline (degraded mode).
        """
        with cls._lock:
            now = datetime.now(timezone.utc)

            # Throttle to avoid hammering probe servers
            if (
                not force
                and cls._last_check_at is not None
                and (now - cls._last_check_at).total_seconds() < cls._check_interval
            ):
                return not cls._degraded

            cls._last_check_at = now

        # Run the probe outside the lock (I/O should not hold lock)
        online = cls.is_online()

        with cls._lock:
            if not online and not cls._degraded:
                cls._degraded      = True
                cls._offline_since = datetime.now(timezone.utc)
                log.warning(
                    "Network outage detected — entering DEGRADED MODE "
                    "(cached data will be used for all API calls)"
                )
                cls._log_event(
                    "network_outage",
                    "Network connectivity lost — degraded mode active",
                    success=False,
                )

            elif online and cls._degraded:
                offline_since  = cls._offline_since
                cls._degraded      = False
                cls._offline_since = None
                duration = (
                    (datetime.now(timezone.utc) - offline_since).total_seconds()
                    if offline_since else 0.0
                )
                log.info(
                    "Network restored after %.0f s — exiting DEGRADED MODE", duration
                )
                cls._log_event(
                    "network_restored",
                    f"Network restored after {duration:.0f}s outage",
                    success=True,
                )

            return online

    @classmethod
    def is_degraded(cls) -> bool:
        """Return True when the system is currently in degraded mode."""
        with cls._lock:
            return cls._degraded

    @classmethod
    def offline_duration_s(cls) -> "float | None":
        """Seconds since the network went offline, or None if currently online."""
        with cls._lock:
            if cls._offline_since is None:
                return None
            return (datetime.now(timezone.utc) - cls._offline_since).total_seconds()

    @classmethod
    def status(cls) -> dict:
        """Return a status dict for health endpoints / monitoring dashboards."""
        with cls._lock:
            return {
                "degraded":      cls._degraded,
                "offline_since": (
                    cls._offline_since.isoformat() if cls._offline_since else None
                ),
                "last_check":    (
                    cls._last_check_at.isoformat() if cls._last_check_at else None
                ),
                "cache_stats":   _cache.stats(),
            }

    # -- Internal logging ------------------------------------------------------

    @classmethod
    def _log_event(cls, event_type: str, msg: str, success: bool = True) -> None:
        if cls._db is None:
            return
        try:
            cls._db.log_recovery_event(
                service="network",
                event_type=event_type,
                error_msg=msg if not success else None,
                recovery_action=msg if success else None,
                success=success,
            )
        except Exception:
            pass
