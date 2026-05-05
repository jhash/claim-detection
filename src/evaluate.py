"""Per-model evaluation against the verita-composite test split.

Loads a saved checkpoint from runs/<slug>/final/ and writes
results/<slug>.json with accuracy, precision, recall, F1 (binary, claim=1).

Skips re-running if results/<slug>.json already exists, unless --force.

Usage:
    python -m src.evaluate                      # evaluate every model in registry
    python -m src.evaluate --model ettin-150m-ft
    python -m src.evaluate --force --model ettin-150m-ft
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
DEFAULT_DATA_DIR = ROOT / "datasets" / "verita-composite" / "ours"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
csv.field_size_limit(sys.maxsize)


def data_dir_for(spec: ModelSpec) -> Path:
    if spec.data_dir:
        return ROOT / spec.data_dir
    return DEFAULT_DATA_DIR


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_test_split(data_dir: Path | None = None) -> tuple[list[str], list[int]]:
    data_dir = data_dir or DEFAULT_DATA_DIR
    texts, labels = [], []
    with (data_dir / "test.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = (row.get("text") or "").strip()
            l = row.get("label", "").strip()
            if not t or l not in {"0", "1"}:
                continue
            texts.append(t)
            labels.append(int(l))
    return texts, labels


def predict(model, tok, texts: list[str], device: str, batch_size: int, max_length: int) -> np.ndarray:
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


def evaluate_one(spec: ModelSpec, args, device: str) -> dict:
    out_path = RESULTS_DIR / f"{spec.slug}.json"
    if out_path.exists() and not args.force:
        print(f"  skip {spec.slug} — {out_path.relative_to(ROOT)} already exists (use --force to redo)")
        return json.loads(out_path.read_text())

    ckpt = RUNS_DIR / spec.slug / "final"
    if not ckpt.exists():
        print(f"  skip {spec.slug} — no checkpoint at {ckpt.relative_to(ROOT)} (run pipeline first)")
        return {"slug": spec.slug, "error": "no_checkpoint"}

    print(f"\n=== eval {spec.slug} ({'finetune' if spec.finetune else 'frozen'}) ===")
    started = time.time()
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))

    texts, y_true = read_test_split(data_dir_for(spec))
    y_pred = predict(model, tok, texts, device, args.batch_size, args.max_length)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    elapsed = time.time() - started

    result = {
        "slug": spec.slug,
        "hf_id": spec.hf_id,
        "finetune": spec.finetune,
        "device": device,
        "n_test": len(y_true),
        "data_dir": str(data_dir_for(spec).relative_to(ROOT)),
        "elapsed_sec": round(elapsed, 1),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Eval only this slug.")
    p.add_argument("--force", action="store_true", help="Re-run eval even if results/<slug>.json exists.")
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
