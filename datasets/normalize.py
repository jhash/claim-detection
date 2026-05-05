"""Normalize per-dataset raw files into a consistent text,label CSV.

Output schema (binary classification, Bell-style):
    text,label,source

`label` is 0/1; 1 = claim-like / check-worthy / subjective / contains a
factual claim, depending on the source's native semantics. Per-source
mapping is documented inline below and in each dataset's README.

Generation-style datasets (FEVERFact, CheckThat-2025 Task 2) are NOT
normalized — they are one-to-many or generation tasks that don't fit the
binary schema. Their raw files are preserved instead.

Run from repo root:
    python datasets/normalize.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
csv.field_size_limit(sys.maxsize)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows):>6} rows -> {path.relative_to(ROOT.parent)}")


def normalize_checkthat_task1() -> None:
    """SUBJ -> 1, OBJ -> 0. English splits only for default training; the raw
    multilingual TSVs are kept under raw/ for OOD ablations."""
    print("CheckThat 2025 Task 1 (Subjectivity)")
    base = ROOT / "checkthat-2025" / "task1-subjectivity"
    raw = base / "raw" / "english"
    splits = {
        "train": raw / "train_en.tsv",
        "dev": raw / "dev_en.tsv",
        "test": raw / "test_en_labeled.tsv",
    }
    for split, fp in splits.items():
        rows = []
        with fp.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                lbl = r["label"].strip().upper()
                if lbl not in {"SUBJ", "OBJ"}:
                    continue
                rows.append({
                    "text": r["sentence"],
                    "label": 1 if lbl == "SUBJ" else 0,
                    "source": "checkthat-2025-task1-subjectivity",
                })
        write_csv(base / f"normalized_{split}.csv", rows)


def normalize_claimify() -> None:
    """contains_factual_claim True -> 1, False -> 0.
    Claimify ships a single unsplit data.csv (6,490 sentences from BingCheck)."""
    print("Claimify (microsoft/claimify-dataset)")
    base = ROOT / "claimify"
    raw = base / "raw" / "data.csv"
    rows = []
    with raw.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            v = r["contains_factual_claim"].strip().lower()
            if v not in {"true", "false"}:
                continue
            rows.append({
                "text": r["sentence"],
                "label": 1 if v == "true" else 0,
                "source": "claimify",
            })
    write_csv(base / "normalized.csv", rows)


def build_aggregate() -> None:
    """Concatenate every binary-normalized file with verita-composite/ours
    into a single aggregate CSV for one-shot training experiments."""
    print("Aggregate (verita-composite + checkthat-2025-task1 + claimify)")
    out = ROOT / "all_binary.csv"
    sources = [
        (ROOT / "verita-composite" / "ours" / "train.csv", "verita-composite-ours-train", "label_to_int"),
        (ROOT / "verita-composite" / "ours" / "test.csv", "verita-composite-ours-test", "label_to_int"),
        (ROOT / "checkthat-2025" / "task1-subjectivity" / "normalized_train.csv", None, "passthrough"),
        (ROOT / "checkthat-2025" / "task1-subjectivity" / "normalized_dev.csv", None, "passthrough"),
        (ROOT / "checkthat-2025" / "task1-subjectivity" / "normalized_test.csv", None, "passthrough"),
        (ROOT / "claimify" / "normalized.csv", None, "passthrough"),
    ]
    rows = []
    for fp, src_label, mode in sources:
        if not fp.exists():
            print(f"  skip (missing) {fp.relative_to(ROOT.parent)}")
            continue
        with fp.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if mode == "passthrough":
                    rows.append({"text": r["text"], "label": int(r["label"]), "source": r["source"]})
                else:  # label_to_int
                    rows.append({"text": r["text"], "label": int(r["label"]), "source": src_label})
    write_csv(out, rows)


if __name__ == "__main__":
    normalize_checkthat_task1()
    normalize_claimify()
    build_aggregate()
