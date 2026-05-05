"""Unit tests for the static model registry (src/models.py)."""

from __future__ import annotations

import pytest

from src.models import MODELS, get


def test_registry_has_no_duplicate_slugs():
    slugs = [m.slug for m in MODELS]
    assert len(slugs) == len(set(slugs)), f"duplicate slugs: {slugs}"


def test_every_entry_has_required_fields():
    for m in MODELS:
        assert m.slug, f"empty slug on {m}"
        assert m.hf_id, f"empty hf_id on {m.slug}"
        assert isinstance(m.finetune, bool)


def test_get_unknown_slug_raises():
    with pytest.raises(KeyError):
        get("not-a-real-slug")


def test_get_returns_matching_spec():
    m = get("ettin-150m-ft")
    assert m.hf_id == "jhu-clsp/ettin-encoder-150m"
    assert m.finetune is True


def test_ettin_appears_first_so_headline_runs_first():
    """Run-order matters for the demo: Ettin fine-tune is the headline."""
    assert MODELS[0].slug == "ettin-150m-ft"


def test_each_finetune_has_a_pretrained_sibling():
    """Bell-style table needs both rows for each architecture we test."""
    by_hf = {}
    for m in MODELS:
        by_hf.setdefault(m.hf_id, set()).add(m.finetune)
    for hf, modes in by_hf.items():
        assert modes == {True, False}, f"{hf} missing one of finetune/pretrained"
