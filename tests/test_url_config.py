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


def test_favicon_redirects_to_static(monkeypatch):
    """Browsers request /favicon.ico by convention; we 301 to the
    /static/favicon.ico that StaticFiles serves."""
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/favicon.ico", follow_redirects=False)
    assert r.status_code == 301
    assert r.headers["location"].endswith("/static/favicon.ico")


def test_base_template_includes_favicon_links(monkeypatch):
    """The <link rel="icon"> tags exist so browsers don't fall back to
    the default."""
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/")
    body = r.text
    assert 'rel="icon"' in body
    assert "favicon.svg" in body
    assert "favicon.ico" in body
    assert "apple-touch-icon" in body


def test_nav_links_to_github(monkeypatch):
    """Header has an outbound GitHub link."""
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/")
    body = r.text
    assert "https://github.com/jhash/claim-detection" in body
    # opens in a new tab + secured against referrer-leak
    assert 'target="_blank"' in body
    assert 'rel="noopener"' in body


def test_health_link_not_in_nav(monkeypatch):
    """Healthz is for ops/monitoring, not human visitors. Don't surface it."""
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/")
    body = r.text
    # The endpoint still exists (other tests cover that). Just isn't a nav link.
    assert '>Health<' not in body
    assert 'href="/api/healthz">' not in body


def test_index_heading_is_claim_detection(monkeypatch):
    """The h1 reads 'Claim detection' (not the old 'Type a sentence')."""
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/")
    body = r.text
    assert "<h1>Claim detection</h1>" in body
    assert "Type a sentence" not in body  # old heading is gone


def test_index_links_to_huggingface_model(monkeypatch):
    """The intro paragraph links the headline model to its Hugging Face page."""
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/")
    body = r.text
    assert "https://huggingface.co/jhu-clsp/ettin-encoder-150m" in body
    # outbound link styling
    assert 'target="_blank"' in body and 'rel="noopener"' in body


def test_index_cites_bell_paper_with_link(monkeypatch):
    """Bell (FEVER 2025) is cited like a real paper reference, with a link."""
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/")
    body = r.text
    assert "Bell" in body
    # ACL Anthology canonical URL for the FEVER 2025 paper.
    assert "https://aclanthology.org/2025.fever-1.6" in body


def test_footer_carries_full_citation(monkeypatch):
    """The footer has the full Bell citation in journal-style format."""
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/")
    body = r.text
    assert "Less Can Be More" in body
    assert "FEVER" in body
    assert "ACL 2025" in body or "2025" in body


def test_index_auto_submits_on_multi_line_paste(monkeypatch):
    """Pasting multi-line text triggers requestSubmit() automatically.
    Single-line paste should NOT trigger submit (the user can keep
    typing). The handler checks the clipboard data for newlines before
    deciding."""
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/")
    body = r.text
    # The textarea has an onpaste handler.
    assert "onpaste" in body
    # It checks for newlines in the pasted clipboard data.
    assert "clipboardData" in body
    assert "\\r?\\n" in body or "/\\n/" in body or "/\r?\n/" in body
    # On match it calls requestSubmit (after a tick so the paste settles).
    assert "requestSubmit" in body
    # Helper text mentions the paste behavior so users know.
    assert "paste" in body.lower()


def test_index_page_explains_what_it_does(monkeypatch):
    """The 'Try it' page tells the user what they're trying. The
    'Server is running in queued mode' line is gone."""
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/")
    body = r.text
    assert "claim detection" in body.lower()
    # Intro mentions the actual subject of the page.
    assert "check-worthy" in body or "factual claim" in body
    # The phrase we removed:
    assert "Server is running in" not in body


def test_openapi_schema_includes_sync_endpoint(monkeypatch):
    mod = _reload_with_env(monkeypatch, APP_URL=None, API_URL=None)
    with TestClient(mod.app) as c:
        r = c.get("/openapi.json")
    schema = r.json()
    assert "/api/predict/sync" in schema["paths"]
    assert "post" in schema["paths"]["/api/predict/sync"]
    assert "/api/predict" in schema["paths"]
    assert "/api/healthz" in schema["paths"]
