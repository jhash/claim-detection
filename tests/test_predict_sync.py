"""Unit tests for the new /api/predict/sync endpoint."""

from __future__ import annotations

import pytest


def test_predict_sync_runs_in_process_when_queue_disabled(client):
    """Default test fixture has QUEUE=0, so this hits the predictor path
    directly — no Redis involvement, returns the prediction body
    immediately."""
    r = client.post("/api/predict/sync", json={"text": "Inflation hit 9.1% in 2022."})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"is_claim", "confidence", "label"}
    assert body["is_claim"] is True
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_sync_rejects_blank(client):
    r = client.post("/api/predict/sync", json={"text": ""})
    assert r.status_code == 422


def test_predict_sync_for_opinion(client):
    r = client.post("/api/predict/sync", json={"text": "I love this weather"})
    assert r.status_code == 200
    body = r.json()
    assert body["is_claim"] is False
    assert body["label"] == "not_claim"


def test_predict_sync_in_queued_mode_polls_until_finished(client, monkeypatch):
    """When QUEUE=1, predict_sync enqueues + polls job_status until
    'finished', then returns the same JSON shape."""
    from app import main as app_main

    class _FakeJob:
        def __init__(self, jid):
            self.id = jid

    states = iter([
        ("queued", None),
        ("started", None),
        ("finished", {"is_claim": True, "confidence": 0.91, "label": "claim"}),
    ])

    def fake_status(_job_id):
        return next(states, ("finished", {"is_claim": True, "confidence": 0.91, "label": "claim"}))

    def fake_enqueue(_text):
        return _FakeJob("test-sync-job")

    monkeypatch.setattr("app.queue.enqueue_predict", fake_enqueue, raising=False)
    monkeypatch.setattr("app.queue.job_status", fake_status, raising=False)
    monkeypatch.setattr(app_main, "USE_QUEUE", True)
    monkeypatch.setattr(app_main, "POLL_INTERVAL_SEC", 0.001)

    r = client.post("/api/predict/sync", json={"text": "Some sentence."})
    assert r.status_code == 200
    body = r.json()
    assert body == {"is_claim": True, "confidence": 0.91, "label": "claim"}


def test_predict_sync_returns_504_on_timeout(client, monkeypatch):
    """If the worker never finishes within POLL_TIMEOUT_SEC, the endpoint
    returns 504 instead of hanging."""
    from app import main as app_main

    class _FakeJob:
        def __init__(self, jid):
            self.id = jid

    monkeypatch.setattr(
        "app.queue.enqueue_predict", lambda _t: _FakeJob("hung"), raising=False
    )
    monkeypatch.setattr(
        "app.queue.job_status", lambda _jid: ("started", None), raising=False
    )
    monkeypatch.setattr(app_main, "USE_QUEUE", True)
    monkeypatch.setattr(app_main, "POLL_INTERVAL_SEC", 0.001)
    monkeypatch.setattr(app_main, "POLL_TIMEOUT_SEC", 0.05)

    r = client.post("/api/predict/sync", json={"text": "Some sentence."})
    assert r.status_code == 504
    assert "timeout" in r.text.lower()


def test_predict_sync_returns_500_on_failed_job(client, monkeypatch):
    from app import main as app_main

    class _FakeJob:
        def __init__(self, jid):
            self.id = jid

    monkeypatch.setattr(
        "app.queue.enqueue_predict", lambda _t: _FakeJob("doomed"), raising=False
    )
    monkeypatch.setattr(
        "app.queue.job_status",
        lambda _jid: ("failed", {"error": "model exploded"}),
        raising=False,
    )
    monkeypatch.setattr(app_main, "USE_QUEUE", True)
    monkeypatch.setattr(app_main, "POLL_INTERVAL_SEC", 0.001)

    r = client.post("/api/predict/sync", json={"text": "Some sentence."})
    assert r.status_code == 500
    assert "model exploded" in r.text
