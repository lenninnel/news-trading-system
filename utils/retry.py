"""
Auto-recovery utility: retry with exponential backoff.

Usage
-----
    from utils.retry import with_retry

    @with_retry(max_attempts=3, base_delay=2.0)
    def fetch_data():
        ...  # raises on failure

    # Or call directly (no decorator):
    result = retry_call(fetch_data, args=(arg1,), max_attempts=3)
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Type, Tuple

log = logging.getLogger(__name__)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    alert_fn: Callable[[str], None] | None = None,
) -> Callable:
    """
    Decorator: retry the wrapped function up to *max_attempts* times on failure.

    Waits base_delay * 2^(attempt-1) seconds between retries (1s, 2s, 4s, …).
    Re-raises the last exception if all attempts fail. Calls alert_fn(msg) on
    final failure if provided.

    Args:
        max_attempts: Maximum number of call attempts (including the first).
        base_delay:   Base delay in seconds (doubles each retry).
        exceptions:   Tuple of exception types to catch and retry.
        alert_fn:     Optional callable(message: str) called when all retries fail.

    Example::

        @with_retry(max_attempts=3, base_delay=2.0)
        def call_api():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        msg = (
                            f"{func.__qualname__} failed after {max_attempts} attempts. "
                            f"Last error: {exc}"
                        )
                        log.error(msg)
                        if alert_fn is not None:
                            try:
                                alert_fn(msg)
                            except Exception:
                                pass
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    log.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %.1fs …",
                        func.__qualname__, attempt, max_attempts, exc, delay,
                    )
                    time.sleep(delay)
            raise RuntimeError("unreachable")  # satisfies type checkers
        return wrapper
    return decorator


def retry_call(
    func: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    max_attempts: int = 3,
    base_delay: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    alert_fn: Callable[[str], None] | None = None,
) -> Any:
    """
    Functional version of the retry decorator — no decoration required.

    Args:
        func:         Callable to call.
        args:         Positional arguments to pass.
        kwargs:       Keyword arguments to pass.
        max_attempts: Maximum number of attempts.
        base_delay:   Base wait in seconds (doubles each retry).
        exceptions:   Exception types to catch.
        alert_fn:     Called with an error message on final failure.

    Returns:
        The return value of *func* on success.
    """
    kwargs = kwargs or {}
    decorated = with_retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        exceptions=exceptions,
        alert_fn=alert_fn,
    )(func)
    return decorated(*args, **kwargs)
