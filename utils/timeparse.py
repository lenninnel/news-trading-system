"""
UTC timestamp helpers for provider-supplied publication times.

Every news / social provider hands us its own flavour of timestamp
(NewsAPI ``2026-09-02T13:05:00Z``, Marketaux ``...00.000000Z``, EODHD
``...+00:00``, Reddit epoch floats, StockTwits ``Z``-suffixed).  Nothing
in the system parsed any of them before 2026-09-03; the publication
time was dropped at fetch and every headline was implicitly "now".

Contract:

* :func:`parse_utc` returns an aware UTC ``datetime`` or ``None``.  It
  never raises and never invents a value — an absent or unparseable
  input is ``None`` so it is visible AS missing downstream.
* Naive inputs are taken as UTC (every provider here documents UTC).
* :func:`to_iso_utc` renders for storage (ISO-8601 with ``+00:00``).
* :func:`age_minutes` is the evaluation primitive: minutes between a
  stored ISO timestamp and ``now`` (aware UTC).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def parse_utc(value: Any) -> datetime | None:
    """Parse a provider timestamp into an aware UTC datetime, or None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        # Epoch seconds (Reddit ``created_utc``).  Reject absurd values
        # rather than producing a 1970 timestamp that looks real.
        if value <= 0 or value > 4102444800:  # 2100-01-01
            return None
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        # RFC-2822 style dates (RSS ``pubDate``) are not ISO; try them last.
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(text)
            except (TypeError, ValueError, IndexError):
                return None
            if dt is None:
                return None
    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_iso_utc(dt: datetime | None) -> str | None:
    """Render an aware datetime as ISO-8601 UTC for storage; None → None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def normalise_published(value: Any) -> str | None:
    """Provider value → ISO-8601 UTC string, or None when absent/unparseable."""
    return to_iso_utc(parse_utc(value))


def age_minutes(published_iso: str | None, now: datetime | None = None) -> float | None:
    """Minutes from ``published_iso`` to ``now`` (default: current UTC).

    Returns None when the timestamp is missing or unparseable.  A
    negative value (published in the future by the provider's clock) is
    returned as-is so clock skew stays visible.
    """
    published = parse_utc(published_iso)
    if published is None:
        return None
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return round((ref - published).total_seconds() / 60.0, 2)
