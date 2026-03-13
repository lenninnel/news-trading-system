"""
Crypto Fear & Greed Index feed using the free Alternative.me API.

No API key required. Results are cached for 4 hours.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

_cache: dict | None = None
_cache_ts: float = 0.0
_CACHE_TTL = 4 * 3600  # 4 hours

_FNG_URL = "https://api.alternative.me/fng/?limit=1"


def get_fear_greed() -> dict | None:
    """
    Fetch the Crypto Fear & Greed Index.

    Returns:
        dict with keys: value (int 0-100), classification (str), timestamp (str)
        Returns None on failure.
    """
    global _cache, _cache_ts

    # Return cached result if still fresh
    if _cache is not None and (time.time() - _cache_ts) < _CACHE_TTL:
        logger.debug("Fear & Greed cache hit")
        return _cache

    try:
        resp = requests.get(_FNG_URL, timeout=10)
        resp.raise_for_status()
        payload = resp.json()

        data_list = payload.get("data")
        if not data_list or not isinstance(data_list, list):
            logger.warning("Fear & Greed API returned unexpected structure")
            return None

        entry = data_list[0]
        result = {
            "value": int(entry["value"]),
            "classification": entry["value_classification"],
            "timestamp": entry["timestamp"],
        }

        _cache = result
        _cache_ts = time.time()
        logger.info(
            "Fear & Greed Index: %d (%s)", result["value"], result["classification"]
        )
        return result

    except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
        logger.error("Fear & Greed fetch failed: %s", exc)
        return None


def clear_cache() -> None:
    """Clear the module-level cache."""
    global _cache, _cache_ts
    _cache = None
    _cache_ts = 0.0
