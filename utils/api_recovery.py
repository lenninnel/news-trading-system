"""
API Error Recovery — Circuit Breaker + Smart Retry.

Provides per-service retry logic and circuit-breaker protection for all
external API calls in the News Trading System.

Services and retry configs
--------------------------
newsapi   : max 3 retries, 60 s base delay
anthropic : max 2 retries, 30 s base delay
yfinance  : max 5 retries,  5 s base delay  (free API, often flaky)
database  : max 3 retries, 10 s base delay

Circuit breaker
---------------
After 5 consecutive failures → OPEN  (fail-fast for 5 minutes)
After 5-minute cooldown      → HALF_OPEN  (single probe call)
On probe success             → CLOSED  (normal operation)

HTTP error handling
-------------------
429 (Rate Limit)   → exponential backoff, longer waits
401 (Unauthorized) → raise UnauthorizedError immediately, no retry
502/503 (Gateway)  → shorter wait & retry
Timeout            → retry with progressively longer effective timeout
All others         → standard exponential backoff

All recovery events are logged to the recovery_log DB table when a
Database instance is attached via APIRecovery.set_db(db).

Usage
-----
    from utils.api_recovery import APIRecovery

    result = APIRecovery.call("newsapi", requests.get, url, params=p, ticker="AAPL")
    status = APIRecovery.get_status()   # dict of all circuit states
    APIRecovery.reset_circuit("newsapi")  # manual reset
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

log = logging.getLogger(__name__)

# ── Per-service retry configurations ──────────────────────────────────────────

SERVICE_CONFIGS: dict[str, dict] = {
    "newsapi": {
        "max_retries": 3,
        "base_delay":  60.0,
        "max_delay":   300.0,
    },
    "anthropic": {
        "max_retries": 2,
        "base_delay":  30.0,
        "max_delay":   120.0,
    },
    "yfinance": {
        "max_retries": 5,
        "base_delay":  5.0,
        "max_delay":   60.0,
    },
    "database": {
        "max_retries": 3,
        "base_delay":  10.0,
        "max_delay":   60.0,
    },
}

_DEFAULT_CONFIG = {"max_retries": 3, "base_delay": 10.0, "max_delay": 120.0}


# ── Exceptions ─────────────────────────────────────────────────────────────────

class CircuitOpenError(RuntimeError):
    """Raised when a circuit is OPEN and fast-failing calls."""

    def __init__(self, service: str, open_until: "datetime | None" = None) -> None:
        self.service    = service
        self.open_until = open_until
        remaining = (
            max(0.0, (open_until - datetime.now(timezone.utc)).total_seconds())
            if open_until else 0.0
        )
        super().__init__(
            f"Circuit OPEN for '{service}' — retry in {remaining:.0f}s"
        )


class APIError(RuntimeError):
    """Wraps an HTTP error with its numeric status code."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class UnauthorizedError(APIError):
    """Raised on HTTP 401 — API key invalid, do not retry."""

    def __init__(self, service: str) -> None:
        super().__init__(401, f"Unauthorized access to '{service}' — check API key")


# ── Circuit Breaker ────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Thread-safe three-state circuit breaker.

    States
    ------
    CLOSED    — normal; all calls pass through.
    OPEN      — fast-failing; waits reset_timeout before probing.
    HALF_OPEN — allows one probe call; closes on success, reopens on failure.

    Args:
        name:              Human-readable service name for logging.
        failure_threshold: Consecutive failures before opening.
        reset_timeout:     Seconds to wait in OPEN before allowing a probe.
    """

    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name:              str,
        failure_threshold: int   = 5,
        reset_timeout:     float = 300.0,
    ) -> None:
        self.name              = name
        self.failure_threshold = failure_threshold
        self.reset_timeout     = reset_timeout
        self._state            = self.CLOSED
        self._failures         = 0
        self._opened_at: "datetime | None" = None
        self._lock             = threading.Lock()

    # -- Properties ------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._check_state()

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    # -- Internal state machine ------------------------------------------------

    def _check_state(self) -> str:
        """Transition OPEN → HALF_OPEN when the reset timeout has elapsed."""
        if self._state == self.OPEN and self._opened_at is not None:
            elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
            if elapsed >= self.reset_timeout:
                self._state = self.HALF_OPEN
                log.info("Circuit '%s': OPEN → HALF_OPEN (probing)", self.name)
        return self._state

    # -- Record outcomes -------------------------------------------------------

    def record_success(self) -> None:
        """Record a successful call; resets failure count and closes circuit."""
        with self._lock:
            prev            = self._state
            self._failures  = 0
            self._state     = self.CLOSED
            self._opened_at = None
            if prev != self.CLOSED:
                log.info("Circuit '%s': %s → CLOSED", self.name, prev.upper())

    def record_failure(self) -> None:
        """Record a failed call; opens circuit after threshold is reached."""
        with self._lock:
            self._failures += 1
            if (
                self._failures >= self.failure_threshold
                or self._state == self.HALF_OPEN
            ):
                prev            = self._state
                self._state     = self.OPEN
                self._opened_at = datetime.now(timezone.utc)
                log.warning(
                    "Circuit '%s': %s → OPEN after %d failure(s)",
                    self.name, prev.upper(), self._failures,
                )

    # -- Gate ------------------------------------------------------------------

    def allow_call(self) -> bool:
        """Return True when a call is permitted (CLOSED or HALF_OPEN)."""
        with self._lock:
            s = self._check_state()
            return s in (self.CLOSED, self.HALF_OPEN)

    def open_until(self) -> "datetime | None":
        """Return when the circuit will next allow a probe, or None."""
        with self._lock:
            if self._state == self.OPEN and self._opened_at is not None:
                from datetime import timedelta
                return self._opened_at + timedelta(seconds=self.reset_timeout)
            return None

    def to_dict(self) -> dict:
        """Serialise current state to a plain dict (for status reports)."""
        with self._lock:
            return {
                "name":              self.name,
                "state":             self._check_state(),
                "failures":          self._failures,
                "failure_threshold": self.failure_threshold,
                "reset_timeout_s":   self.reset_timeout,
                "opened_at":         self._opened_at.isoformat() if self._opened_at else None,
            }


# ── APIRecovery ────────────────────────────────────────────────────────────────

class APIRecovery:
    """
    Central manager for all external API call retries and circuit breakers.

    All circuit breaker state is class-level and persists for the life of the
    process, so circuits opened by one module are visible to all others.

    Usage
    -----
        result = APIRecovery.call("newsapi", requests.get, url, params=p)

    Attach a Database instance once at startup for recovery event logging::

        APIRecovery.set_db(db)
    """

    _circuits:      dict[str, CircuitBreaker] = {}
    _circuits_lock: threading.Lock            = threading.Lock()
    _db:            Any                       = None   # optional Database reference

    # -- Setup -----------------------------------------------------------------

    @classmethod
    def set_db(cls, db: Any) -> None:
        """Attach a Database instance; enables recovery_log persistence."""
        cls._db = db

    @classmethod
    def get_circuit(cls, service: str) -> CircuitBreaker:
        """Return the named circuit breaker, creating it on first access."""
        with cls._circuits_lock:
            if service not in cls._circuits:
                cls._circuits[service] = CircuitBreaker(service)
            return cls._circuits[service]

    # -- Status ----------------------------------------------------------------

    @classmethod
    def get_status(cls) -> dict[str, dict]:
        """Return current state of all registered circuit breakers."""
        with cls._circuits_lock:
            return {name: cb.to_dict() for name, cb in cls._circuits.items()}

    @classmethod
    def reset_circuit(cls, service: str) -> None:
        """Manually close (reset) the circuit for *service*."""
        cls.get_circuit(service).record_success()
        log.info("Circuit '%s' manually reset to CLOSED", service)

    # -- HTTP helpers ----------------------------------------------------------

    @staticmethod
    def _status_code(exc: Exception) -> "int | None":
        """Extract an HTTP status code from various exception types."""
        # requests.HTTPError / anthropic SDK errors
        resp = getattr(exc, "response", None)
        if resp is not None:
            code = getattr(resp, "status_code", None)
            if isinstance(code, int):
                return code
        # anthropic SDK: RateLimitError etc. expose .status_code directly
        if hasattr(exc, "status_code") and isinstance(getattr(exc, "status_code"), int):
            return exc.status_code  # type: ignore[attr-defined]
        return None

    @staticmethod
    def _backoff(base: float, attempt: int, max_delay: float) -> float:
        """Exponential backoff: base * 2^(attempt-1), capped at max_delay."""
        return min(max_delay, base * (2 ** (attempt - 1)))

    # -- Core call method ------------------------------------------------------

    @classmethod
    def call(
        cls,
        service:  str,
        func:     Callable,
        *args:    Any,
        ticker:   "str | None" = None,
        **kwargs: Any,
    ) -> Any:
        """
        Call *func* with retry + circuit-breaker protection for *service*.

        Error handling by HTTP status
        ------------------------------
        429 → exponential backoff using full base_delay (rate-limit queue)
        401 → raise UnauthorizedError immediately, alert operator
        502/503 → shorter wait (gateway hiccup)
        Timeout → progressive shorter wait

        Args:
            service:  Service name: "newsapi" | "anthropic" | "yfinance" | "database"
            func:     Callable to invoke.
            *args:    Forwarded to func.
            ticker:   Optional ticker symbol for recovery log enrichment.
            **kwargs: Forwarded to func.

        Returns:
            Return value of func on success.

        Raises:
            CircuitOpenError:  Circuit is OPEN; caller should use fallback.
            UnauthorizedError: HTTP 401; caller must handle auth failure.
            Exception:         Last exception after all retries exhausted.
        """
        cfg     = SERVICE_CONFIGS.get(service, _DEFAULT_CONFIG)
        max_r   = cfg["max_retries"]
        base_d  = cfg["base_delay"]
        max_d   = cfg["max_delay"]
        circuit = cls.get_circuit(service)

        # Fast-fail when circuit is OPEN
        if not circuit.allow_call():
            until = circuit.open_until()
            log.warning("Circuit '%s' is OPEN — fast-failing call", service)
            cls._log_event(
                service, "circuit_open", ticker=ticker, success=False,
                error=f"Circuit OPEN until {until}",
            )
            raise CircuitOpenError(service, until)

        last_exc: "Exception | None" = None
        t_start = time.monotonic()

        for attempt in range(1, max_r + 1):
            try:
                result = func(*args, **kwargs)
                circuit.record_success()
                elapsed_ms = int((time.monotonic() - t_start) * 1000)
                if attempt > 1:
                    cls._log_event(
                        service, "retry", ticker=ticker, attempt=attempt,
                        success=True, duration_ms=elapsed_ms,
                        recovery_action=f"succeeded_on_attempt_{attempt}",
                    )
                return result

            except Exception as exc:
                last_exc = exc
                status   = cls._status_code(exc)

                # HTTP 401 — stop immediately, alert
                if status == 401:
                    log.error(
                        "Service '%s': HTTP 401 Unauthorized — "
                        "check your API key. No retries.", service,
                    )
                    circuit.record_failure()
                    cls._log_event(
                        service, "retry", ticker=ticker, attempt=attempt,
                        success=False, error="HTTP 401 Unauthorized",
                        recovery_action="stop_retries_auth_failed",
                    )
                    raise UnauthorizedError(service) from exc

                circuit.record_failure()

                if attempt == max_r:
                    elapsed_ms = int((time.monotonic() - t_start) * 1000)
                    cls._log_event(
                        service, "retry", ticker=ticker, attempt=attempt,
                        success=False, error=str(exc)[:500],
                        duration_ms=elapsed_ms,
                        recovery_action="all_retries_exhausted",
                    )
                    break

                # Determine wait strategy
                if status == 429:
                    delay  = cls._backoff(base_d, attempt, max_d)
                    action = f"rate_limited_backoff_{delay:.0f}s"
                elif status in (502, 503):
                    delay  = cls._backoff(base_d / 4, attempt, max_d / 4)
                    action = f"bad_gateway_wait_{delay:.0f}s"
                elif _is_timeout(exc):
                    delay  = cls._backoff(base_d / 6, attempt, max_d / 6)
                    action = f"timeout_retry_{delay:.0f}s"
                else:
                    delay  = cls._backoff(base_d / 10, attempt, max_d / 10)
                    action = f"generic_backoff_{delay:.0f}s"

                log.warning(
                    "Service '%s' attempt %d/%d failed: %s — "
                    "waiting %.1fs …",
                    service, attempt, max_r, exc, delay,
                )
                cls._log_event(
                    service, "retry", ticker=ticker, attempt=attempt,
                    success=False, error=str(exc)[:300],
                    recovery_action=action,
                )
                time.sleep(delay)

        assert last_exc is not None
        raise last_exc

    # -- Internal event logging ------------------------------------------------

    @classmethod
    def _log_event(
        cls,
        service:         str,
        event_type:      str,
        ticker:          "str | None" = None,
        attempt:         "int | None" = None,
        error:           "str | None" = None,
        recovery_action: "str | None" = None,
        duration_ms:     "int | None" = None,
        success:         bool         = True,
    ) -> None:
        """Persist a recovery event to the database (best-effort, never raises)."""
        if cls._db is None:
            return
        try:
            cls._db.log_recovery_event(
                service=service,
                event_type=event_type,
                ticker=ticker,
                attempt=attempt,
                error_msg=error,
                recovery_action=recovery_action,
                duration_ms=duration_ms,
                success=success,
            )
        except Exception:
            pass  # logging must never break the calling code


# ── Timeout helper ─────────────────────────────────────────────────────────────

def _is_timeout(exc: Exception) -> bool:
    """Return True when *exc* looks like a network/read timeout."""
    name = type(exc).__name__.lower()
    msg  = str(exc).lower()
    return (
        "timeout"       in name
        or "timed out"  in msg
        or "read timeout"    in msg
        or "connect timeout" in msg
        or "connectiontimeout" in name
    )
