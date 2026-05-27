"""
Polygon.io data feed for point-in-time daily OHLC aggregates.

Single purpose: feed the Research-side `daily_ohlc` table built by
`scripts/ingest_ohlc.py`. NOT wired into the live trading path — do not
import from agents/, risk/, or execution/.

Free-tier limits (as of 2026-05): 5 requests / minute, 2-year history.
This module paces calls at ~12s gaps and retries 429 / 5xx with
exponential backoff.

Usage::

    feed = PolygonFeed()
    bars = feed.get_daily_aggs("AAPL", "2025-05-27", "2026-05-26")
    # bars is a list of dicts:
    #   {date, open, high, low, close, adj_close, volume}
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import requests

from config.settings import POLYGON_API_KEY

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.polygon.io"

# Free-tier rate limit: 5 req/min. 12.5s gap leaves headroom; we also
# pair RAW + ADJUSTED calls per ticker (2 req each), so a 20-ticker
# backfill is ~40 calls = ~9 minutes wall time.
_MIN_GAP_SECONDS = 12.5
_MAX_RETRIES = 4
_BASE_BACKOFF = 4.0  # seconds; doubles each retry


class PolygonFeed:
    """Thin wrapper around Polygon.io's `/v2/aggs/ticker/.../range/1/day/...`.

    Single feed, single endpoint. Returns parsed dicts so callers don't
    need pandas. Designed for batch ingest, not realtime.
    """

    _last_call_at: float = 0.0
    _gap_lock = threading.Lock()

    def __init__(self, api_key: str = POLYGON_API_KEY) -> None:
        self.api_key = api_key

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_daily_aggs(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> list[dict]:
        """Fetch daily aggregates for `ticker` between `start` and `end`.

        Args:
            ticker: US equity symbol (e.g. "AAPL"). Must not contain ".".
            start:  ISO date "YYYY-MM-DD" (inclusive).
            end:    ISO date "YYYY-MM-DD" (inclusive).

        Returns:
            List of dicts with keys:
              date (str YYYY-MM-DD), open, high, low, close (RAW unadjusted),
              adj_close (split/dividend-adjusted close, may be None if the
              adjusted call fails), volume (int).

        Raises:
            RuntimeError on persistent fetch failure of the RAW series.
            (The adjusted series is best-effort and never raises.)
        """
        if not self.available:
            raise RuntimeError(
                "Polygon API key missing — set POLYGON_API_KEY in .env"
            )

        raw_bars = self._fetch_range(ticker, start, end, adjusted=False)
        if not raw_bars:
            return []

        # Best-effort adjusted close. Build a date→adj_close lookup;
        # missing dates fall through as None.
        adj_lookup: dict[str, float] = {}
        try:
            adj_bars = self._fetch_range(ticker, start, end, adjusted=True)
            for b in adj_bars:
                adj_lookup[b["date"]] = b["close"]
        except Exception as exc:
            logger.warning(
                "Polygon: adjusted fetch failed for %s (%s..%s), "
                "adj_close will be NULL: %s",
                ticker, start, end, exc,
            )

        for b in raw_bars:
            b["adj_close"] = adj_lookup.get(b["date"])
        return raw_bars

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_range(
        self, ticker: str, start: str, end: str, adjusted: bool,
    ) -> list[dict]:
        """Single paged fetch. Polygon returns up to 50k bars per call;
        a 2yr daily range is ~500 bars so one page suffices, but we
        follow `next_url` defensively.
        """
        url = (
            f"{_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start}/{end}"
        )
        params = {
            "adjusted": "true" if adjusted else "false",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }

        out: list[dict] = []
        page = 0
        while url:
            self._respect_rate_limit()
            data = self._request_with_retry(url, params)
            page += 1
            results = data.get("results") or []
            for r in results:
                # Polygon timestamp is ms since epoch UTC.
                ts_ms = r.get("t")
                if ts_ms is None:
                    continue
                d = time.strftime(
                    "%Y-%m-%d",
                    time.gmtime(ts_ms / 1000.0),
                )
                out.append({
                    "date": d,
                    "open": float(r["o"]),
                    "high": float(r["h"]),
                    "low": float(r["l"]),
                    "close": float(r["c"]),
                    "volume": int(r.get("v") or 0),
                })

            next_url = data.get("next_url")
            if not next_url:
                break
            url = next_url
            # next_url already carries query params; we only need to
            # re-append the API key.
            params = {"apiKey": self.api_key}

        logger.info(
            "Polygon: fetched %d bars for %s [%s..%s] adjusted=%s (pages=%d)",
            len(out), ticker, start, end, adjusted, page,
        )
        return out

    def _respect_rate_limit(self) -> None:
        """Sleep so consecutive Polygon calls are >=_MIN_GAP_SECONDS apart."""
        with self._gap_lock:
            now = time.monotonic()
            gap = now - PolygonFeed._last_call_at
            if gap < _MIN_GAP_SECONDS:
                time.sleep(_MIN_GAP_SECONDS - gap)
            PolygonFeed._last_call_at = time.monotonic()

    def _request_with_retry(self, url: str, params: dict) -> dict:
        """GET with exponential backoff on 429 / 5xx / network errors.

        Polygon returns 429 with a "You've exceeded the maximum requests
        per minute" body on free-tier overage. The Retry-After header is
        not always present; we fall back to a doubling sleep.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait = float(retry_after)
                    else:
                        wait = _BASE_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "Polygon HTTP %d (attempt %d/%d), sleeping %.1fs",
                        resp.status_code, attempt + 1, _MAX_RETRIES + 1, wait,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
            except requests.RequestException as exc:
                last_exc = exc
                wait = _BASE_BACKOFF * (2 ** attempt)
                logger.warning(
                    "Polygon network error (attempt %d/%d): %s — sleeping %.1fs",
                    attempt + 1, _MAX_RETRIES + 1, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"Polygon: exhausted {_MAX_RETRIES + 1} attempts for {url}: "
            f"{last_exc!r}"
        )
