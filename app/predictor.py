"""Single-sentence claim-detection predictor.

Loads a fine-tuned checkpoint once at startup and exposes a `predict`
function returning {is_claim: bool, confidence: float, label: str}.

`confidence` is the softmax probability of the predicted class — the same
value Bell-style classifiers use as a calibration proxy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_CHECKPOINT = Path(
    os.environ.get("MODEL_PATH", "runs/ettin-150m-ft/final")
).resolve()


@dataclass
class Prediction:
    is_claim: bool
    confidence: float
    label: str
    logits: list[float]

    def to_dict(self) -> dict:
        return {
            "is_claim": self.is_claim,
            "confidence": self.confidence,
            "label": self.label,
        }


class Predictor:
    """Loads model+tokenizer once, predicts on demand. Thread-safe for
    inference (PyTorch eval-mode forward is reentrant)."""

    def __init__(self, checkpoint_path: Path | str | None = None, device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"checkpoint not found at {self.checkpoint_path} — train a model first via "
                "`python -m src.pipeline --model ettin-150m-ft` or set MODEL_PATH"
            )
        self.device = device or self._pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.checkpoint_path))
        self.model.to(self.device)
        self.model.eval()
        # id2label is saved in config.json; fall back if missing.
        self.id2label = getattr(self.model.config, "id2label", None) or {0: "not_claim", 1: "claim"}

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @torch.inference_mode()
    def predict(self, text: str, max_length: int = 256) -> Prediction:
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
        label = str(self.id2label[pred_idx])
        return Prediction(
            is_claim=(pred_idx == 1),
            confidence=confidence,
            label=label,
            logits=[float(x) for x in logits.tolist()],
        )
