"""Tests for the Fear & Greed Index feed module."""

from unittest.mock import MagicMock, patch

from data.fear_greed_feed import clear_cache, get_fear_greed


class TestFearGreedFeed:
    """Tests for data.fear_greed_feed.get_fear_greed."""

    def setup_method(self) -> None:
        clear_cache()

    # ── Happy path ────────────────────────────────────────────────────────

    @patch("data.fear_greed_feed.requests.get")
    def test_happy_path(self, mock_get: MagicMock) -> None:
        """Successful API call returns a well-formed dict."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "value": "25",
                    "value_classification": "Extreme Fear",
                    "timestamp": "1710288000",
                }
            ]
        }
        mock_get.return_value = mock_resp

        result = get_fear_greed()

        assert result is not None
        assert result["value"] == 25
        assert isinstance(result["value"], int)
        assert result["classification"] == "Extreme Fear"
        assert result["timestamp"] == "1710288000"
        mock_get.assert_called_once()

    # ── Caching ───────────────────────────────────────────────────────────

    @patch("data.fear_greed_feed.requests.get")
    def test_caching_prevents_second_call(self, mock_get: MagicMock) -> None:
        """Second call within TTL should use cache, not hit API again."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "value": "50",
                    "value_classification": "Neutral",
                    "timestamp": "1710288000",
                }
            ]
        }
        mock_get.return_value = mock_resp

        first = get_fear_greed()
        second = get_fear_greed()

        assert first == second
        assert mock_get.call_count == 1

    # ── API error ─────────────────────────────────────────────────────────

    @patch("data.fear_greed_feed.requests.get")
    def test_api_error_returns_none(self, mock_get: MagicMock) -> None:
        """Network / HTTP errors should return None, not raise."""
        import requests

        mock_get.side_effect = requests.ConnectionError("timeout")

        result = get_fear_greed()
        assert result is None

    @patch("data.fear_greed_feed.requests.get")
    def test_http_error_returns_none(self, mock_get: MagicMock) -> None:
        """HTTP 500 should return None."""
        import requests

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        mock_get.return_value = mock_resp

        result = get_fear_greed()
        assert result is None

    # ── Malformed response ────────────────────────────────────────────────

    @patch("data.fear_greed_feed.requests.get")
    def test_malformed_response_missing_data(self, mock_get: MagicMock) -> None:
        """Response with no 'data' key returns None."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"name": "Fear and Greed Index"}
        mock_get.return_value = mock_resp

        result = get_fear_greed()
        assert result is None

    @patch("data.fear_greed_feed.requests.get")
    def test_malformed_response_empty_data(self, mock_get: MagicMock) -> None:
        """Response with empty data list returns None."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        result = get_fear_greed()
        assert result is None

    @patch("data.fear_greed_feed.requests.get")
    def test_malformed_response_missing_key(self, mock_get: MagicMock) -> None:
        """Response entry missing required keys returns None."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"value": "25"}]}
        mock_get.return_value = mock_resp

        result = get_fear_greed()
        assert result is None
