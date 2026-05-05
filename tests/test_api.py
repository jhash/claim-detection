"""Unit tests for the FastAPI predict endpoint, using a mock predictor."""

from __future__ import annotations

import pytest


def test_healthz(client):
    r = client.get("/api/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["queue"] is False


def test_predict_returns_claim_for_factual_sentence(client):
    r = client.post("/api/predict", json={"text": "Inflation hit 9.1% in June 2022."})
    assert r.status_code == 200
    body = r.json()
    assert body["is_claim"] is True
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["label"] in ("claim", "not_claim")


def test_predict_returns_not_claim_for_opinion(client):
    r = client.post("/api/predict", json={"text": "I love this weather"})
    assert r.status_code == 200
    body = r.json()
    assert body["is_claim"] is False
    assert body["confidence"] > 0.5  # confidence should reflect predicted class


def test_predict_rejects_empty_text(client):
    r = client.post("/api/predict", json={"text": ""})
    assert r.status_code == 422  # pydantic min_length=1


def test_predict_rejects_missing_field(client):
    r = client.post("/api/predict", json={})
    assert r.status_code == 422


def test_predict_response_shape(client):
    r = client.post("/api/predict", json={"text": "The report cited 3 sources."})
    body = r.json()
    assert set(body.keys()) == {"is_claim", "confidence", "label"}
    assert isinstance(body["is_claim"], bool)
    assert isinstance(body["confidence"], float)
    assert isinstance(body["label"], str)


def test_index_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "claim" in r.text.lower()


def test_ui_predict_returns_a_table_row_in_sync_mode(client):
    """Sync mode: response is a fully-resolved <tr> ready to append to the
    table (no SSE setup needed because result is already known)."""
    r = client.post("/ui/predict", data={"text": "The 2024 budget passed yesterday."})
    assert r.status_code == 200
    body = r.text
    # Top-level element is a <tr>, not a <div> — table-row pattern.
    assert body.lstrip().startswith("<tr"), f"expected <tr fragment, got: {body[:80]!r}"
    # Sentence shows up in its own cell so the user can read it back.
    assert "The 2024 budget passed yesterday." in body
    # Verdict + confidence both rendered.
    assert "CLAIM" in body
    assert "%" in body  # confidence percentage
    # No SSE machinery in sync mode.
    assert "sse-connect" not in body


def test_ui_predict_returns_streaming_row_in_queued_mode(client, monkeypatch):
    """Queued mode: row with sse-connect attribute pointing at the job's
    stream URL. The result cell is a placeholder until SSE fires."""
    from app import main as app_main

    class _FakeJob:
        def __init__(self, jid):
            self.id = jid

    enq_calls = []

    def fake_enqueue(text):
        enq_calls.append(text)
        return _FakeJob("test-job-abc")

    monkeypatch.setattr("app.queue.enqueue_predict", fake_enqueue, raising=False)
    monkeypatch.setattr(app_main, "USE_QUEUE", True)
    r = client.post("/ui/predict", data={"text": "Inflation is up."})
    assert r.status_code == 200
    body = r.text
    assert body.lstrip().startswith("<tr")
    assert "Inflation is up." in body
    assert 'sse-connect="/api/predict/test-job-abc/stream"' in body
    assert "spinner" in body  # loading indicator class
    assert "queued" in body.lower()
    assert enq_calls == ["Inflation is up."]
    # Out-of-band swap removes the placeholder row on the first enqueue.
    assert 'id="placeholder-row"' in body
    assert 'hx-swap-oob="delete"' in body


def test_ui_predict_blank_returns_error_row(client):
    """Blank input → 400 + a single-row error fragment that fits the table."""
    r = client.post("/ui/predict", data={"text": "   "})
    assert r.status_code == 400
    assert r.text.lstrip().startswith("<tr")
    assert "Please enter" in r.text
    assert "status-failed" in r.text


def test_index_page_has_streaming_table(client):
    """Form posts append to a tbody; tbody starts with a placeholder row."""
    r = client.get("/")
    body = r.text
    assert "results-tbody" in body
    assert 'hx-target="#results-tbody"' in body
    assert 'hx-swap="afterbegin"' in body
    # Re-focus textarea after each submit.
    assert "after-request" in body
    assert ".focus()" in body
    # Placeholder row, with id matching what the OOB swap targets.
    assert 'id="placeholder-row"' in body
    assert "No predictions yet" in body


def test_index_page_supports_enter_to_submit(client):
    """Pressing Enter in the textarea should submit the form, with
    Shift+Enter reserved for newlines."""
    r = client.get("/")
    body = r.text
    # The onkeydown handler checks Enter and !shiftKey.
    assert "Enter" in body
    assert "shiftKey" in body
    assert "requestSubmit" in body
