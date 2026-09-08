"""
Alpaca Market Data feed for point-in-time daily OHLC bars.

Single purpose: feed the `daily_ohlc` table built by `scripts/ingest_ohlc.py`.
Replaces `data/polygon_feed.py` as the nightly source after the Polygon
Stocks Starter plan was cancelled (2026-09-08). NOT wired into the live
trading path — do not import from agents/, risk/, or execution/.

Contract (identical to PolygonFeed.get_daily_aggs so the ingest, the hygiene
flags and the freshness gate are untouched)::

    feed = AlpacaOHLCFeed()
    bars = feed.get_daily_aggs("AAPL", "2026-08-28", "2026-09-04")
    # -> [{date, open, high, low, close, adj_close, volume}, ...] ascending

Store convention (Fix D3): open/high/low/close/volume are the RAW,
unadjusted session bar (`adjustment=raw`). `adj_close` is the SPLIT-only
adjusted close (`adjustment=split`) — the same semantics the Polygon
`adjusted=true` series had (Polygon does not apply dividends; verified
2026-09-08: Alpaca split-adjusted vs stored adj_close agree within 0.01 %
on 99.9 % of 26 201 bars). Nothing on the live path reads adj_close.

Endpoint: GET https://data.alpaca.markets/v2/stocks/bars
  timeframe=1Day, feed=sip (consolidated tape — the IEX feed is a single
  venue and its high/low are NOT the session high/low), sort=asc,
  limit=10000 (one page covers > 39 years of daily bars).

Free-plan ("Basic") constraints, measured 2026-09-08:
  * 200 requests / minute (X-Ratelimit-Limit header). 20 tickers x 2 calls
    = 40 calls; a gentle 0.35 s gap keeps a run near 15 s.
  * `feed=sip` refuses any `end` inside the last 15 minutes with HTTP 403
    "subscription does not permit querying recent SIP data". A date-only
    `end` counts as end-of-day and is therefore refused for today. The
    feed clamps `end` to now - _RECENT_SIP_MARGIN_MINUTES; the nightly
    22:30 UTC run then asks for <= 22:10 UTC, well after the 20:00/21:00
    UTC close, so the just-closed session is included.
  * History back to 2016 (probed), i.e. far beyond OHLC_BACKFILL_YEARS.

Bar timestamps are the session date at 04:00/05:00 UTC (midnight
US/Eastern); the calendar date is taken from the first 10 characters.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://data.alpaca.markets"
_BARS_URL = f"{_BASE_URL}/v2/stocks/bars"

_MIN_GAP_SECONDS = 0.35          # 200 req/min allowed; ~170 req/min ceiling here
_MAX_RETRIES = 4
_BASE_BACKOFF = 4.0              # seconds; doubles each retry
_RECENT_SIP_MARGIN_MINUTES = 20  # Alpaca refuses end > now - 15 min; keep margin
_PAGE_LIMIT = 10000


def _api_credentials() -> tuple[str, str]:
    return (
        os.environ.get("ALPACA_API_KEY", ""),
        os.environ.get("ALPACA_SECRET_KEY", ""),
    )


class AlpacaOHLCFeed:
    """Thin wrapper around Alpaca's `/v2/stocks/bars` for daily bars.

    Single feed, single endpoint. Returns parsed dicts so callers don't
    need pandas. Designed for batch ingest, not realtime.
    """

    _last_call_at: float = 0.0
    _gap_lock = threading.Lock()

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        env_key, env_secret = _api_credentials()
        self.api_key = env_key if api_key is None else api_key
        self.secret_key = env_secret if secret_key is None else secret_key

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.secret_key)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_daily_aggs(self, ticker: str, start: str, end: str) -> list[dict]:
        """Fetch daily bars for `ticker` between `start` and `end` (inclusive).

        Args:
            ticker: US equity symbol (e.g. "AAPL"). Must not contain ".".
            start:  ISO date "YYYY-MM-DD" (inclusive).
            end:    ISO date "YYYY-MM-DD" (inclusive; clamped to the SIP
                    recency limit, see module docstring).

        Returns:
            List of dicts with keys:
              date (str YYYY-MM-DD), open, high, low, close (RAW unadjusted),
              adj_close (split-adjusted close, None if the split call fails),
              volume (int). Ascending by date.

        Raises:
            RuntimeError on persistent fetch failure of the RAW series.
            (The split-adjusted series is best-effort and never raises.)
        """
        if not self.available:
            raise RuntimeError(
                "Alpaca credentials missing — set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in .env"
            )

        raw_bars = self._fetch_range(ticker, start, end, adjustment="raw")
        if not raw_bars:
            return []

        adj_lookup: dict[str, float] = {}
        try:
            for b in self._fetch_range(ticker, start, end, adjustment="split"):
                adj_lookup[b["date"]] = b["close"]
        except Exception as exc:
            logger.warning(
                "Alpaca: split-adjusted fetch failed for %s (%s..%s), "
                "adj_close will be NULL: %s",
                ticker, start, end, exc,
            )

        for b in raw_bars:
            b["adj_close"] = adj_lookup.get(b["date"])
        return raw_bars

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_end(end: str, now: datetime | None = None) -> str:
        """RFC-3339 `end` for the request: end-of-day of the requested date,
        but never later than now - margin (free-plan SIP recency rule)."""
        now = now or datetime.now(timezone.utc)
        eod = datetime.fromisoformat(end).replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc,
        )
        latest = now - timedelta(minutes=_RECENT_SIP_MARGIN_MINUTES)
        return min(eod, latest).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    def _fetch_range(
        self, ticker: str, start: str, end: str, adjustment: str,
    ) -> list[dict]:
        """Single paged fetch. One page holds 10 000 bars (~39 years of
        daily data); `next_page_token` is followed defensively."""
        params: dict = {
            "symbols": ticker,
            "timeframe": "1Day",
            "start": start,
            "end": self._clamp_end(end),
            "adjustment": adjustment,
            "feed": "sip",
            "sort": "asc",
            "limit": _PAGE_LIMIT,
        }

        out: list[dict] = []
        page = 0
        while True:
            self._respect_rate_limit()
            data = self._request_with_retry(_BARS_URL, params)
            page += 1
            bars = (data.get("bars") or {}).get(ticker) or []
            for r in bars:
                ts = r.get("t")
                if not ts:
                    continue
                out.append({
                    "date": ts[:10],
                    "open": float(r["o"]),
                    "high": float(r["h"]),
                    "low": float(r["l"]),
                    "close": float(r["c"]),
                    "volume": int(r.get("v") or 0),
                })
            token = data.get("next_page_token")
            if not token:
                break
            params = {**params, "page_token": token}

        logger.info(
            "Alpaca: fetched %d bars for %s [%s..%s] adjustment=%s (pages=%d)",
            len(out), ticker, start, end, adjustment, page,
        )
        return out

    def _respect_rate_limit(self) -> None:
        """Sleep so consecutive Alpaca calls are >= _MIN_GAP_SECONDS apart."""
        with self._gap_lock:
            now = time.monotonic()
            gap = now - AlpacaOHLCFeed._last_call_at
            if gap < _MIN_GAP_SECONDS:
                time.sleep(_MIN_GAP_SECONDS - gap)
            AlpacaOHLCFeed._last_call_at = time.monotonic()

    def _request_with_retry(self, url: str, params: dict) -> dict:
        """GET with exponential backoff on 429 / 5xx / network errors.

        4xx other than 429 (bad credentials 401/403, the SIP recency 403,
        invalid symbol 422) are NOT retried — they will not fix themselves
        and the ingest must fail loudly instead of burning the run window.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    url, params=params, headers=self._headers(), timeout=30,
                )
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait = float(retry_after)
                    else:
                        wait = _BASE_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "Alpaca HTTP %d (attempt %d/%d), sleeping %.1fs",
                        resp.status_code, attempt + 1, _MAX_RETRIES + 1, wait,
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Alpaca HTTP {resp.status_code} for {url} "
                    f"(symbols={params.get('symbols')}): {resp.text[:200]}"
                )
            except requests.RequestException as exc:
                last_exc = exc
                wait = _BASE_BACKOFF * (2 ** attempt)
                logger.warning(
                    "Alpaca network error (attempt %d/%d): %s — sleeping %.1fs",
                    attempt + 1, _MAX_RETRIES + 1, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"Alpaca: exhausted {_MAX_RETRIES + 1} attempts for {url}: "
            f"{last_exc!r}"
        )
