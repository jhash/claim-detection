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


def test_fmt_best_bolds_when_flagged():
    assert cmp.fmt_best(0.92, True) == "**0.9200**"
    assert cmp.fmt_best(0.92, False) == "0.9200"
    # None values aren't bolded even if the flag is True (no value to compare).
    assert cmp.fmt_best(None, True) == "—"


def test_column_maxes_handles_missing():
    rows = [
        {"f1": 0.91, "accuracy": None},
        {"f1": 0.93, "accuracy": 0.94},
        {"f1": None, "accuracy": 0.92},
    ]
    bests = cmp.column_maxes(rows, ["f1", "accuracy", "missing_key"])
    assert bests == {"f1": 0.93, "accuracy": 0.94}
    # Missing key entirely: not in dict (callers can use .get).
    assert "missing_key" not in bests


def test_section_our_results_bolds_one_per_column():
    """Each metric column gets exactly one bold value (the column max)."""
    from src.models import get
    rows = [
        {"spec": get("ettin-150m-ft"), "slug": "ettin-150m-ft",
         "hf_id": "x", "accuracy": 0.92, "precision": 0.91, "recall": 0.91, "f1": 0.917},
        {"spec": get("bert-base-ft"), "slug": "bert-base-ft",
         "hf_id": "y", "accuracy": 0.917, "precision": 0.92, "recall": 0.89, "f1": 0.910},
    ]
    md = cmp.section_our_results(rows)
    # F1 best is on ettin-150m-ft (0.917)
    assert "**0.9170**" in md
    # Precision best is on bert-base-ft (0.92) — the 4dp form
    assert "**0.9200**" in md
    # Non-bold form should also appear (the loser's value)
    assert "0.9100" in md  # bert F1 (not bolded since 0.917 wins)


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
