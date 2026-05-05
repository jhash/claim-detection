"""Integration tests that hit a real running service.

By default these are skipped. Run with `pytest -m integration` once the
docker stack is up:

    docker compose up -d api
    pytest -m integration

These tests intentionally fail when the service isn't reachable — that's
the failing-first state called out in the takehome.
"""

from __future__ import annotations

import os

import httpx
import pytest

API_URL = os.environ.get("API_URL", "http://localhost:8000")
pytestmark = pytest.mark.integration


def _live_check():
    try:
        return httpx.get(f"{API_URL}/healthz", timeout=2.0).status_code == 200
    except Exception:
        return False


def test_healthz_live():
    assert _live_check(), f"API not reachable at {API_URL} — start with `docker compose up`"


def test_predict_real_claim():
    if not _live_check():
        pytest.skip("API not reachable")
    r = httpx.post(
        f"{API_URL}/predict",
        json={"text": "The 9/11 Commission report said America is safer but not yet safe."},
        timeout=30.0,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["is_claim"] is True
    assert body["confidence"] > 0.7


def test_predict_real_opinion():
    if not _live_check():
        pytest.skip("API not reachable")
    r = httpx.post(
        f"{API_URL}/predict",
        json={"text": "I really love how the lighting feels in this room"},
        timeout=30.0,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["is_claim"] is False


def test_root_page_serves_html():
    if not _live_check():
        pytest.skip("API not reachable")
    r = httpx.get(f"{API_URL}/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_results_page_serves_html():
    if not _live_check():
        pytest.skip("API not reachable")
    r = httpx.get(f"{API_URL}/results")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Comparison" in r.text or "Bell" in r.text
