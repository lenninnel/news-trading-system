"""CORS origins of the read-only API come from NTS_CORS_ORIGINS (2026-11-14).

No server address is hard-coded any more: without the variable only the
two local dev origins on port 3000 are allowed and a WARNING is logged so
a mis-configured deployment is visible in the journal.

Run: python3 -m pytest tests/test_api_cors.py -v
"""
from __future__ import annotations

import logging
import os
import sys

from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import api.main as api_main  # noqa: E402


def test_cors_origins_from_env(monkeypatch, caplog):
    monkeypatch.setenv(
        "NTS_CORS_ORIGINS",
        "http://nts-vps:3000, http://100.118.106.64:3000 ,,",
    )
    with caplog.at_level(logging.WARNING, logger=api_main.logger.name):
        origins = api_main._cors_origins()
    assert origins == ["http://nts-vps:3000", "http://100.118.106.64:3000"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_cors_origins_default_without_env_and_warns(monkeypatch, caplog):
    monkeypatch.delenv("NTS_CORS_ORIGINS", raising=False)
    with caplog.at_level(logging.WARNING, logger=api_main.logger.name):
        origins = api_main._cors_origins()
    assert origins == ["http://localhost:3000", "http://127.0.0.1:3000"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "NTS_CORS_ORIGINS" in warnings[0].getMessage()


def test_cors_origins_blank_env_counts_as_unset(monkeypatch, caplog):
    monkeypatch.setenv("NTS_CORS_ORIGINS", " , ")
    with caplog.at_level(logging.WARNING, logger=api_main.logger.name):
        origins = api_main._cors_origins()
    assert origins == api_main.DEFAULT_CORS_ORIGINS
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_default_origins_contain_no_public_address():
    for origin in api_main.DEFAULT_CORS_ORIGINS:
        assert origin.startswith(("http://localhost:", "http://127.0.0.1:"))


def test_app_middleware_uses_resolved_origins():
    cors = [m for m in api_main.app.user_middleware if m.cls is CORSMiddleware]
    assert len(cors) == 1
    assert cors[0].kwargs["allow_origins"] == api_main._CORS_ORIGINS
    assert cors[0].kwargs["allow_methods"] == ["GET"]
