"""
Thread-safe token-bucket rate limiter.

Used to enforce a maximum request rate across all threads / coroutines
hitting the same API.  Designed for the Anthropic API (40 req/min default)
but generic enough for any service.

Usage::

    limiter = TokenBucketRateLimiter(max_per_minute=40)
    limiter.acquire()          # blocks until a token is available
    response = client.call()   # guaranteed to stay under rate limit
"""

from __future__ import annotations

import threading
import time


class TokenBucketRateLimiter:
    """
    A simple token-bucket rate limiter backed by :mod:`threading`.

    Tokens refill continuously at *max_per_minute / 60* tokens per second.
    ``acquire()`` blocks (sleeps) until a token is available — it never raises.

    Args:
        max_per_minute: Maximum sustained requests per minute.
    """

    def __init__(self, max_per_minute: int = 40) -> None:
        self._rate = max_per_minute / 60.0  # tokens per second
        self._max_tokens = float(max_per_minute)
        self._tokens = float(max_per_minute)  # start full
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Add tokens accrued since the last refill (must hold lock)."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def acquire(self) -> None:
        """Block until a token is available, then consume it."""
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # How long until the next token?
                wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)
