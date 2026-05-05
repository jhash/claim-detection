"""Build a custom 80/20 train/test split that broadens the training
distribution beyond `verita-composite/ours/`.

Sources mixed (binary-fit only — generation tasks like FEVERFact and
CheckThat-2025-Task2 are excluded by design):

  * verita-composite/ours/{train,test}.csv  (13,032 sentences)
    Bell composite: ClaimBuster + PoliClaim + AVeriTeC.

  * claimify/normalized.csv                 (6,490 sentences)
    Microsoft's BingCheck → contains_factual_claim ∈ {0, 1}.
    Note: this used to be an OOD test set for verita-trained models;
    once it's in training, the OOD-vs-Claimify comparison no longer
    holds for `*-combined` slugs.

  * checkthat-2025/task1-subjectivity/normalized_{train,dev,test}.csv  (1,592 EN sentences)
    SUBJ → 1, OBJ → 0. CAVEAT: subjectivity is correlated with
    check-worthiness but not the same target. Including it broadens
    the distribution but introduces some label-semantics drift; a
    sentence can be subjective without being a check-worthy claim.

Output:
  datasets/combined-v1/train.csv  ~ 80% (shuffled, deduped)
  datasets/combined-v1/test.csv   ~ 20%

Reproducibility: fixed seed (42), so re-running yields the same files.

Run from repo root:
    python -m src.build_combined_split
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
OUT_DIR = DATASETS / "combined-v1"
SEED = 42
TEST_FRACTION = 0.20

csv.field_size_limit(sys.maxsize)


def _read_csv(path: Path, source_label: str, text_col: str = "text", label_col: str = "label") -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = (r.get(text_col) or "").strip()
            l = (r.get(label_col) or "").strip()
            if not t or l not in {"0", "1"}:
                continue
            rows.append({"text": t, "label": int(l), "source": source_label})
    return rows


def gather() -> list[dict]:
    """Concat every binary-fit corpus we vendor. Each row carries
    its `source` so a downstream filter could re-derive the original
    composition without re-reading these files."""
    rows: list[dict] = []
    rows += _read_csv(DATASETS / "verita-composite/ours/train.csv", "verita-composite-ours-train")
    rows += _read_csv(DATASETS / "verita-composite/ours/test.csv", "verita-composite-ours-test")
    rows += _read_csv(DATASETS / "claimify/normalized.csv", "claimify")
    for split in ("train", "dev", "test"):
        fp = DATASETS / "checkthat-2025/task1-subjectivity" / f"normalized_{split}.csv"
        if fp.exists():
            rows += _read_csv(fp, f"checkthat-2025-task1-{split}")
    return rows


def dedup(rows: list[dict]) -> tuple[list[dict], int]:
    """Drop exact-duplicate sentences (text, lowercased + stripped). For
    sentences that appear with conflicting labels across sources, keep
    the first occurrence — comes up rarely (<0.5%) but worth being
    explicit."""
    seen: dict[str, dict] = {}
    conflicts = 0
    for r in rows:
        key = r["text"].strip().lower()
        if key in seen:
            if seen[key]["label"] != r["label"]:
                conflicts += 1
            continue
        seen[key] = r
    return list(seen.values()), conflicts


def split_80_20(rows: list[dict], seed: int = SEED, test_fraction: float = TEST_FRACTION) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - test_fraction))
    return shuffled[:cut], shuffled[cut:]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)


def label_balance(rows: list[dict]) -> tuple[int, int]:
    pos = sum(1 for r in rows if r["label"] == 1)
    return pos, len(rows) - pos


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    return p.parse_args()


def main():
    args = parse_args()
    print("gathering source rows...")
    raw = gather()
    print(f"  raw rows (with overlap): {len(raw):,}")
    by_source: dict[str, int] = {}
    for r in raw:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    for src, n in sorted(by_source.items()):
        print(f"    {n:>6,}  {src}")

    deduped, conflicts = dedup(raw)
    dropped = len(raw) - len(deduped)
    print(f"  deduped: {len(deduped):,}  ({dropped} duplicates dropped, {conflicts} of those had conflicting labels)")

    train, test = split_80_20(deduped, args.seed, args.test_fraction)
    train_pos, train_neg = label_balance(train)
    test_pos, test_neg = label_balance(test)

    write(OUT_DIR / "train.csv", train)
    write(OUT_DIR / "test.csv", test)

    print(f"\nwrote {OUT_DIR.relative_to(ROOT)}/train.csv: {len(train):,} rows ({train_pos:,} pos / {train_neg:,} neg)")
    print(f"wrote {OUT_DIR.relative_to(ROOT)}/test.csv:  {len(test):,} rows ({test_pos:,} pos / {test_neg:,} neg)")
    print(f"\nseed={args.seed}  test_fraction={args.test_fraction}  ratio={len(train)/len(deduped):.4f}/{len(test)/len(deduped):.4f}")


if __name__ == "__main__":
    main()
