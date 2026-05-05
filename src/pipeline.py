"""Sequential fine-tune pipeline for sentence-level claim detection.

Trains (or frozen-probes) every model in `src.models.MODELS` one at a time,
saves the best checkpoint plus held-out metrics under `runs/<slug>/`.
Designed to be resource-conservative: one model in memory at a time, MPS
on Apple Silicon when available, CPU fallback otherwise.

Usage:
    python -m src.pipeline                      # all models, in order
    python -m src.pipeline --list               # show registry, exit
    python -m src.pipeline --model ettin-150m-ft  # one-off, then stop
    python -m src.pipeline --epochs 1 --max-train-rows 500 --model ettin-150m-ft  # smoke test
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.models import MODELS, ModelSpec, get

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "verita-composite" / "ours"
RUNS_DIR = ROOT / "runs"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_splits(max_train_rows: int | None, max_eval_rows: int | None):
    files = {"train": str(DATA_DIR / "train.csv"), "test": str(DATA_DIR / "test.csv")}
    ds = load_dataset("csv", data_files=files)
    ds = ds.filter(lambda r: r["text"] is not None and r["label"] in (0, 1))
    if max_train_rows:
        ds["train"] = ds["train"].shuffle(seed=42).select(range(min(max_train_rows, len(ds["train"]))))
    if max_eval_rows:
        ds["test"] = ds["test"].select(range(min(max_eval_rows, len(ds["test"]))))
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def run_one(spec: ModelSpec, args, device: str) -> dict:
    out_dir = RUNS_DIR / spec.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {spec.slug} ({'finetune' if spec.finetune else 'frozen'}) — {spec.hf_id} ===")
    started = time.time()

    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.hf_id, num_labels=2, id2label={0: "not_claim", 1: "claim"}, label2id={"not_claim": 0, "claim": 1}
    )

    if not spec.finetune:
        # Frozen encoder, train only the classification head.
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False

    ds = load_splits(args.max_train_rows, args.max_eval_rows)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    ds_tok = ds.map(tokenize, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in {"label"}])

    fp16 = device == "cuda"
    training_args = TrainingArguments(
        output_dir=str(out_dir / "ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=5e-5 if spec.finetune else 1e-3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        report_to=[],
        fp16=fp16,
        seed=42,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tokenizer=tok),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    elapsed = time.time() - started

    # Save final model + tokenizer for serving.
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))

    summary = {
        "slug": spec.slug,
        "hf_id": spec.hf_id,
        "finetune": spec.finetune,
        "device": device,
        "elapsed_sec": round(elapsed, 1),
        "train_rows": len(ds_tok["train"]),
        "eval_rows": len(ds_tok["test"]),
        "epochs": args.epochs,
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Free memory between runs.
    del trainer, model, tok, ds, ds_tok
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Sequential claim-detection fine-tune pipeline.")
    p.add_argument("--model", help="Train only this model slug (see --list), then exit.")
    p.add_argument("--list", action="store_true", help="Print the model registry and exit.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-train-rows", type=int, default=None, help="Cap training rows (smoke testing).")
    p.add_argument("--max-eval-rows", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        for m in MODELS:
            mode = "finetune" if m.finetune else "frozen   "
            print(f"  {m.slug:<32s}  {mode}  {m.hf_id}")
            if m.note:
                print(f"      └─ {m.note}")
        return

    device = pick_device()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"device: {device}")
    RUNS_DIR.mkdir(exist_ok=True)

    targets = [get(args.model)] if args.model else list(MODELS)
    print(f"will run {len(targets)} model(s):")
    for m in targets:
        print(f"  - {m.slug}")

    summaries = []
    for spec in targets:
        try:
            summaries.append(run_one(spec, args, device))
        except Exception as e:
            print(f"!! {spec.slug} FAILED: {e!r}")
            summaries.append({"slug": spec.slug, "error": repr(e)})

    with (RUNS_DIR / "all_summaries.json").open("w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nwrote {RUNS_DIR / 'all_summaries.json'}")


if __name__ == "__main__":
    main()
