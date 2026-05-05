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


def test_build_view_loads_ood_sections_from_disk(tmp_path, monkeypatch):
    """OOD section is dynamically populated per-request from
    results/ood/*.json. No restart needed when new files land."""
    monkeypatch.setattr("app.results_view.RESULTS_DIR", tmp_path / "in")
    monkeypatch.setattr("app.results_view.OOD_RESULTS_DIR", tmp_path / "ood")
    (tmp_path / "in").mkdir()
    (tmp_path / "ood").mkdir()
    # In-domain F1 so the delta column has a baseline.
    (tmp_path / "in" / "ettin-150m-ft.json").write_text(json.dumps({
        "slug": "ettin-150m-ft", "accuracy": 0.92, "precision": 0.92, "recall": 0.91, "f1": 0.917,
    }))
    # OOD result for same slug across two datasets.
    (tmp_path / "ood" / "ettin-150m-ft.json").write_text(json.dumps({
        "slug": "ettin-150m-ft",
        "per_dataset": {
            "claimify": {"n": 6490, "accuracy": 0.6, "precision": 0.7, "recall": 0.5, "f1": 0.55},
            "ct22":     {"n":  911, "accuracy": 0.7, "precision": 0.8, "recall": 0.6, "f1": 0.69},
        },
    }))
    view = build_view()
    sections = view["ood_sections"]
    assert {s["name"] for s in sections} == {"claimify", "ct22"}
    claimify = next(s for s in sections if s["name"] == "claimify")
    assert claimify["rows"][0]["slug"] == "ettin-150m-ft"
    assert claimify["rows"][0]["f1"] == 0.55
    assert claimify["rows"][0]["in_f1"] == 0.917
    assert claimify["rows"][0]["delta"] == pytest.approx(0.55 - 0.917)


def test_build_view_no_ood_section_when_disk_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("app.results_view.OOD_RESULTS_DIR", tmp_path / "missing")
    view = build_view()
    assert view["ood_sections"] == []


def test_build_view_marks_column_bests_in_main_table(tmp_path, monkeypatch):
    """`*_best` flags drive the bold rendering — exactly one True per
    metric column."""
    monkeypatch.setattr("app.results_view.RESULTS_DIR", tmp_path)
    (tmp_path / "ettin-150m-ft.json").write_text(json.dumps({
        "slug": "ettin-150m-ft", "accuracy": 0.92, "precision": 0.91, "recall": 0.91, "f1": 0.917,
    }))
    (tmp_path / "modernbert-base-ft.json").write_text(json.dumps({
        "slug": "modernbert-base-ft", "accuracy": 0.917, "precision": 0.92, "recall": 0.91, "f1": 0.916,
    }))
    view = build_view()
    rows = view["ours_sorted"]
    # F1 best: ettin-150m-ft (0.917)
    f1_bests = [r for r in rows if r.get("f1_best")]
    assert len(f1_bests) == 1
    assert f1_bests[0]["slug"] == "ettin-150m-ft"
    # Precision best: modernbert-base-ft (0.92)
    p_bests = [r for r in rows if r.get("precision_best")]
    assert len(p_bests) == 1
    assert p_bests[0]["slug"] == "modernbert-base-ft"
    # Recall ties between the two — both flagged True (max-equality)
    r_bests = [r for r in rows if r.get("recall_best")]
    assert len(r_bests) == 2


def test_build_view_marks_bell_column_bests():
    """Bell reference table also gets *_best flags — uses the same helper."""
    view = build_view()
    bell = view["bell"]
    f1_bests = [r for r in bell if r.get("f1_best")]
    assert len(f1_bests) == 1
    # Bell's BERT row has the highest F1 (0.916).
    assert f1_bests[0]["label"].startswith("BERT")
