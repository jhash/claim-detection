"""Pytest fixtures shared across tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.predictor import Prediction


@dataclass
class FakePredictor:
    """Drop-in stand-in for app.predictor.Predictor.

    Returns deterministic output keyed off the input text, so unit tests
    can assert on specific shapes without touching real model weights."""

    def predict(self, text: str, max_length: int = 256) -> Prediction:
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        # Heuristic: anything containing a digit or "report" is a "claim"
        # in fake-land. Keeps tests intuitive without being random.
        is_claim = any(c.isdigit() for c in text) or "report" in text.lower()
        confidence = 0.93 if is_claim else 0.71
        return Prediction(
            is_claim=is_claim,
            confidence=confidence,
            label="claim" if is_claim else "not_claim",
            logits=[0.1, 0.9] if is_claim else [0.7, 0.3],
        )


@pytest.fixture
def fake_predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def client(fake_predictor) -> Iterator[TestClient]:
    """FastAPI test client wired to a fake predictor."""
    app_main.set_predictor(fake_predictor)
    with TestClient(app_main.app) as c:
        yield c
    app_main.set_predictor(None)
