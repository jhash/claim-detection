"""Unit tests for the Predictor wrapper that don't require model weights."""

from __future__ import annotations

import pytest

from app.predictor import Prediction


def test_prediction_to_dict_excludes_logits():
    p = Prediction(is_claim=True, confidence=0.9, label="claim", logits=[0.1, 0.9])
    d = p.to_dict()
    assert d == {"is_claim": True, "confidence": 0.9, "label": "claim"}
    # Logits intentionally hidden from the API surface.
    assert "logits" not in d


def test_fake_predictor_rejects_blank(fake_predictor):
    with pytest.raises(ValueError):
        fake_predictor.predict("")
    with pytest.raises(ValueError):
        fake_predictor.predict("   ")
