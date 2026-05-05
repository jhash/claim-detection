"""Tests for APP_URL / API_URL env-var configuration and legacy redirects."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


def _reload_with_env(monkeypatch, **env):
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import app.main as app_main_mod

    return importlib.reload(app_main_mod)


def test_default_api_url_is_app_url_plus_api(monkeypatch):
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    assert mod.APP_URL == "http://localhost:8000"
    assert mod.API_URL == "http://localhost:8000/api"


def test_app_url_can_be_overridden(monkeypatch):
    mod = _reload_with_env(monkeypatch, APP_URL="https://claims.jakehash.com", API_URL=None)
    assert mod.APP_URL == "https://claims.jakehash.com"
    assert mod.API_URL == "https://claims.jakehash.com/api"


def test_api_url_can_be_set_independently(monkeypatch):
    """Useful if you split api + ui across subdomains."""
    mod = _reload_with_env(
        monkeypatch,
        APP_URL="https://claims.jakehash.com",
        API_URL="https://api.claims.jakehash.com",
    )
    assert mod.APP_URL == "https://claims.jakehash.com"
    assert mod.API_URL == "https://api.claims.jakehash.com"


def test_curl_example_uses_api_url(monkeypatch):
    mod = _reload_with_env(monkeypatch, APP_URL="https://claims.jakehash.com", API_URL=None)
    assert "https://claims.jakehash.com/api/predict/sync" in mod.CURL_EXAMPLE


def test_legacy_predict_path_redirects(monkeypatch):
    """Old /predict (POST) → 308 to /api/predict, body preserved."""
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.post("/predict", json={"text": "hello"}, follow_redirects=False)
    assert r.status_code == 308
    assert r.headers["location"].endswith("/api/predict")


def test_legacy_healthz_redirects(monkeypatch):
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/healthz", follow_redirects=False)
    assert r.status_code == 308
    assert r.headers["location"].endswith("/api/healthz")


def test_legacy_predict_sync_redirects(monkeypatch):
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.post("/predict/sync", json={"text": "hello"}, follow_redirects=False)
    assert r.status_code == 308
    assert r.headers["location"].endswith("/api/predict/sync")


def test_legacy_stream_path_redirects(monkeypatch):
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/predict/some-id/stream", follow_redirects=False)
    assert r.status_code == 308
    assert r.headers["location"].endswith("/api/predict/some-id/stream")


def test_docs_page_wraps_swagger_in_base_layout(monkeypatch):
    """/docs returns our wrapped page (nav, footer, iframe to /docs-raw)
    so the Swagger UI inherits the site's container + header."""
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/docs")
    assert r.status_code == 200
    assert "iframe" in r.text  # wrapper page embeds Swagger
    assert "/docs-raw" in r.text  # iframe src
    assert 'href="/"' in r.text  # nav still present


def test_docs_raw_serves_swagger(monkeypatch):
    """The bare Swagger renderer remains reachable at /docs-raw for the
    iframe to load (and for direct linking)."""
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/docs-raw")
    assert r.status_code == 200
    assert "swagger" in r.text.lower()


def test_api_docs_redirects_to_docs(monkeypatch):
    """Friendly /api-docs alias."""
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/api-docs", follow_redirects=False)
    assert r.status_code in (302, 307)  # RedirectResponse default
    assert r.headers["location"] == "/docs"


def test_openapi_schema_includes_sync_endpoint(monkeypatch):
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/openapi.json")
    schema = r.json()
    assert "/api/predict/sync" in schema["paths"]
    assert "post" in schema["paths"]["/api/predict/sync"]
    assert "/api/predict" in schema["paths"]
    assert "/api/healthz" in schema["paths"]
