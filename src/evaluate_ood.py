"""Out-of-domain evaluation of every trained checkpoint.

The in-domain test set in `datasets/verita-composite/ours/test.csv`
shares its source distribution with the training set (political
debates, state-of-state speeches, fact-check articles). A model can
look great on it while still being useless on real-world text from
other domains.

This script measures **transfer** to two OOD sets:

  * Claimify — 6,490 sentences extracted from BingCheck (search-
    assistant LLM answers). Domain: LLM-generated long-form text.
    Label: `contains_factual_claim` (close enough to "claim-worthy").
    [microsoft/claimify-dataset]
  * CheckThat 2022 (CT22) — 911 tweets, COVID/political. Domain:
    short social-media text. Label: `class_label` (1 = check-worthy).

For each model with a checkpoint at runs/<slug>/final/, predicts on
both OOD sets and writes results/ood/<slug>.json with per-dataset
accuracy / precision / recall / F1.

Usage:
    python -m src.evaluate_ood                         # all checkpoints
    python -m src.evaluate_ood --model ettin-150m-ft   # one
    python -m src.evaluate_ood --force                 # re-run even if results exist
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models import MODELS, ModelSpec, get

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "datasets"
RUNS_DIR = ROOT / "runs"
OOD_RESULTS_DIR = ROOT / "results" / "ood"
csv.field_size_limit(sys.maxsize)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_claimify() -> tuple[list[str], list[int]]:
    """Pre-normalized binary CSV: text,label,source."""
    texts, labels = [], []
    with (DATA_ROOT / "claimify" / "normalized.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = (row.get("text") or "").strip()
            l = row.get("label", "").strip()
            if not t or l not in {"0", "1"}:
                continue
            texts.append(t)
            labels.append(int(l))
    return texts, labels


def load_ct22() -> tuple[list[str], list[int]]:
    """CheckThat 2022 task 1B — tweets with class_label ∈ {0, 1}."""
    texts, labels = [], []
    fp = DATA_ROOT / "verita-composite" / "CheckThat" / "CT22_english_1B_claim_dev_test.tsv"
    with fp.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t = (row.get("tweet_text") or "").strip()
            l = row.get("class_label", "").strip()
            if not t or l not in {"0", "1"}:
                continue
            texts.append(t)
            labels.append(int(l))
    return texts, labels


OOD_LOADERS = {
    "claimify": load_claimify,
    "ct22": load_ct22,
}


def predict_batched(model, tok, texts: list[str], device: str, batch_size: int, max_length: int) -> np.ndarray:
    model.eval()
    model.to(device)
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            out.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(out)


def evaluate_one(spec: ModelSpec, args, device: str) -> dict | None:
    out_path = OOD_RESULTS_DIR / f"{spec.slug}.json"
    if out_path.exists() and not args.force:
        print(f"  skip {spec.slug} — {out_path.relative_to(ROOT)} already exists (--force to redo)")
        return json.loads(out_path.read_text())

    ckpt = RUNS_DIR / spec.slug / "final"
    if not ckpt.exists():
        print(f"  skip {spec.slug} — no checkpoint at {ckpt.relative_to(ROOT)}")
        return None

    print(f"\n=== ood eval {spec.slug} ===")
    started = time.time()
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))

    per_dataset = {}
    for name, loader in OOD_LOADERS.items():
        if args.dataset and args.dataset != name:
            continue
        texts, y_true = loader()
        y_pred = predict_batched(model, tok, texts, device, args.batch_size, args.max_length)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        per_dataset[name] = {
            "n": len(y_true),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
        }
        print(f"  {name:>10s}: F1 {per_dataset[name]['f1']:.4f}  Acc {per_dataset[name]['accuracy']:.4f}  (n={len(y_true)})")

    result = {
        "slug": spec.slug,
        "hf_id": spec.hf_id,
        "finetune": spec.finetune,
        "device": device,
        "elapsed_sec": round(time.time() - started, 1),
        "per_dataset": per_dataset,
    }
    OOD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Eval only this slug.")
    p.add_argument("--dataset", choices=list(OOD_LOADERS), help="Eval only this OOD dataset.")
    p.add_argument("--force", action="store_true", help="Re-run even if results exist.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device()
    print(f"device: {device}")
    targets = [get(args.model)] if args.model else list(MODELS)
    for spec in targets:
        try:
            evaluate_one(spec, args, device)
        except Exception as e:
            print(f"!! {spec.slug} FAILED: {e!r}")


if __name__ == "__main__":
    main()
