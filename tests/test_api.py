"""Unit tests for the FastAPI predict endpoint, using a mock predictor."""

from __future__ import annotations

import pytest


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["queue"] is False


def test_predict_returns_claim_for_factual_sentence(client):
    r = client.post("/predict", json={"text": "Inflation hit 9.1% in June 2022."})
    assert r.status_code == 200
    body = r.json()
    assert body["is_claim"] is True
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["label"] in ("claim", "not_claim")


def test_predict_returns_not_claim_for_opinion(client):
    r = client.post("/predict", json={"text": "I love this weather"})
    assert r.status_code == 200
    body = r.json()
    assert body["is_claim"] is False
    assert body["confidence"] > 0.5  # confidence should reflect predicted class


def test_predict_rejects_empty_text(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422  # pydantic min_length=1


def test_predict_rejects_missing_field(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_response_shape(client):
    r = client.post("/predict", json={"text": "The report cited 3 sources."})
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


def test_ui_predict_renders_result_html(client):
    r = client.post("/ui/predict", data={"text": "The 2024 budget passed yesterday."})
    assert r.status_code == 200
    assert "CLAIM" in r.text  # uppercase verdict in template
    assert "confidence" in r.text


def test_ui_predict_handles_blank(client):
    r = client.post("/ui/predict", data={"text": "   "})
    assert r.status_code == 400
    assert "Please enter" in r.text
