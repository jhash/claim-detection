"""Unit tests for the /results page view-model builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.results_view import build_view


def test_build_view_handles_empty_results(tmp_path, monkeypatch):
    """When no results exist, every model is 'pending', no errors."""
    monkeypatch.setattr("app.results_view.RESULTS_DIR", tmp_path)
    view = build_view()
    assert view["ours_sorted"] == []
    assert len(view["pending"]) > 0
    assert all(r["pending"] for r in view["pending"])


def test_build_view_sorts_by_f1_desc(tmp_path, monkeypatch):
    """Two fake results get sorted highest-F1 first."""
    monkeypatch.setattr("app.results_view.RESULTS_DIR", tmp_path)
    (tmp_path / "ettin-150m-ft.json").write_text(json.dumps({
        "slug": "ettin-150m-ft", "accuracy": 0.92, "precision": 0.91, "recall": 0.91, "f1": 0.91,
    }))
    (tmp_path / "ettin-150m-pretrained.json").write_text(json.dumps({
        "slug": "ettin-150m-pretrained", "accuracy": 0.88, "precision": 0.88, "recall": 0.86, "f1": 0.87,
    }))
    view = build_view()
    sorted_rows = view["ours_sorted"]
    assert len(sorted_rows) == 2
    assert sorted_rows[0]["slug"] == "ettin-150m-ft"
    assert sorted_rows[0]["f1"] >= sorted_rows[1]["f1"]


def test_build_view_includes_bell_reference():
    view = build_view()
    bell = view["bell"]
    assert any(r["label"].startswith("BERT") for r in bell)
    assert all(r["f1"] is not None for r in bell)
