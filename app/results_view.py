"""Build a structured view-model for the /results page from results/*.json."""

from __future__ import annotations

import json
from pathlib import Path

from src.compare import BELL_REFERENCE, SLUG_TO_BELL
from src.models import MODELS

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def build_view() -> dict:
    rows = []
    for spec in MODELS:
        fp = RESULTS_DIR / f"{spec.slug}.json"
        if not fp.exists():
            rows.append({"slug": spec.slug, "hf_id": spec.hf_id, "finetune": spec.finetune, "pending": True})
            continue
        d = json.loads(fp.read_text())
        rows.append({
            "slug": spec.slug,
            "hf_id": spec.hf_id,
            "finetune": spec.finetune,
            "accuracy": d.get("accuracy"),
            "precision": d.get("precision"),
            "recall": d.get("recall"),
            "f1": d.get("f1"),
            "pending": False,
        })

    sorted_rows = sorted(
        [r for r in rows if not r["pending"]],
        key=lambda r: r["f1"] or -1,
        reverse=True,
    )
    pending = [r for r in rows if r["pending"]]

    bell = [
        {"label": label, "accuracy": a, "precision": p, "recall": r, "f1": f1}
        for (label, a, p, r, f1) in BELL_REFERENCE
    ]

    by_slug = {r["slug"]: r for r in rows if not r["pending"]}
    bell_by_label = {b["label"]: b for b in bell}
    sxs = []
    for slug, bell_label in SLUG_TO_BELL.items():
        bell_row = bell_by_label[bell_label]
        ours = by_slug.get(slug)
        if ours is None:
            sxs.append({"bell_label": bell_label, "bell_f1": bell_row["f1"], "slug": slug, "our_f1": None, "delta": None})
            continue
        delta = ours["f1"] - bell_row["f1"]
        sxs.append({
            "bell_label": bell_label,
            "bell_f1": bell_row["f1"],
            "slug": slug,
            "our_f1": ours["f1"],
            "delta": delta,
        })
    ettin_ft = by_slug.get("ettin-150m-ft")
    if ettin_ft:
        bell_top = max(b["f1"] for b in bell)
        sxs.append({
            "bell_label": "(best Bell encoder)",
            "bell_f1": bell_top,
            "slug": "ettin-150m-ft",
            "our_f1": ettin_ft["f1"],
            "delta": ettin_ft["f1"] - bell_top,
        })

    return {
        "ours_sorted": sorted_rows,
        "pending": pending,
        "bell": bell,
        "sxs": sxs,
    }
