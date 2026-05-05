"""Tests for src/compare.py table generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.compare as cmp


def test_fmt_handles_none():
    assert cmp.fmt(None) == "—"


def test_fmt_formats_float_to_4dp():
    assert cmp.fmt(0.91667) == "0.9167"


def test_render_with_no_results_says_so():
    md = cmp.render([])
    assert "No results yet" in md


def test_render_with_one_result_includes_bell_table(tmp_path, monkeypatch):
    """Even with one model, the Bell reference + side-by-side sections render."""
    from src.models import get
    spec = get("ettin-150m-ft")
    results = [{
        "spec": spec,
        "slug": spec.slug,
        "hf_id": spec.hf_id,
        "accuracy": 0.92,
        "precision": 0.91,
        "recall": 0.91,
        "f1": 0.917,
    }]
    md = cmp.render(results)
    assert "Bell (FEVER 2025) reference" in md
    assert "BERT (Finetuned)" in md
    assert "ettin-150m-ft" in md
    # Side-by-side delta column shows our F1 vs Bell's best
    assert "Side-by-side" in md


def test_load_results_skips_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cmp, "RESULTS_DIR", tmp_path)
    # No files written; loader should just return [].
    assert cmp.load_results() == []
