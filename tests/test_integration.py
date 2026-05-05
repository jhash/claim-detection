"""Integration tests that hit a real running service.

By default these are skipped. Run with `pytest -m integration` once the
docker stack is up:

    docker compose up -d
    pytest -m integration

These tests intentionally fail when the service isn't reachable — that's
the failing-first state called out in the takehome.

Env vars:
    APP_URL  base URL for the UI (default http://localhost:8000)
    API_URL  base URL for the API (default $APP_URL/api)
"""

from __future__ import annotations

import os

import httpx
import pytest

APP_URL = os.environ.get("APP_URL", "http://localhost:8000").rstrip("/")
API_URL = os.environ.get("API_URL", f"{APP_URL}/api").rstrip("/")
pytestmark = pytest.mark.integration


def _live_check():
    try:
        return httpx.get(f"{API_URL}/healthz", timeout=2.0).status_code == 200
    except Exception:
        return False


def test_healthz_live():
    assert _live_check(), f"API not reachable at {API_URL} — start with `docker compose up`"


def test_predict_sync_real_claim():
    if not _live_check():
        pytest.skip("API not reachable")
    r = httpx.post(
        f"{API_URL}/predict/sync",
        json={"text": "Octopuses can edit their own RNA in real time to adapt to cold water."},
        timeout=30.0,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["is_claim"] is True
    assert body["confidence"] > 0.7
    assert body["label"] == "claim"


def test_predict_sync_real_opinion():
    if not _live_check():
        pytest.skip("API not reachable")
    r = httpx.post(
        f"{API_URL}/predict/sync",
        json={"text": "I really love how the lighting feels in this room"},
        timeout=30.0,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["is_claim"] is False
    assert body["label"] == "not_claim"


def test_legacy_path_redirects_to_api():
    if not _live_check():
        pytest.skip("API not reachable")
    # Old /healthz should 308 to /api/healthz
    r = httpx.get(f"{APP_URL}/healthz", follow_redirects=False, timeout=5.0)
    assert r.status_code == 308
    assert r.headers["location"].endswith("/api/healthz")


def test_root_page_serves_html():
    if not _live_check():
        pytest.skip("API not reachable")
    r = httpx.get(f"{APP_URL}/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_results_page_serves_html():
    if not _live_check():
        pytest.skip("API not reachable")
    r = httpx.get(f"{APP_URL}/results")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Comparison" in r.text or "Bell" in r.text


def test_swagger_docs_serve():
    if not _live_check():
        pytest.skip("API not reachable")
    r = httpx.get(f"{APP_URL}/docs")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "swagger" in r.text.lower()


def test_openapi_schema_lists_predict_sync_first():
    if not _live_check():
        pytest.skip("API not reachable")
    r = httpx.get(f"{APP_URL}/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert "/api/predict/sync" in schema["paths"]
    assert "/api/predict" in schema["paths"]
