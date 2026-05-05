"""Static registry of models to compare for sentence-level claim detection.

The set mirrors the comparison table in Bell, "Less Can be More" (FEVER 2025) —
the encoder rows that appear with and without "(Finetuned)" — plus our chosen
upgrade (Ettin-encoder) and a few same-family alternatives left commented out.

Each entry says whether to fine-tune (`finetune=True`) or evaluate the
pretrained model in a zero-shot / frozen-classifier setting (`finetune=False`).
The Bell table reports BOTH a pretrained and a fine-tuned row for the
encoder family — the finetune flag is what distinguishes them.

CLI usage (see pipeline.py):
    python -m src.pipeline                      # all enabled, in order
    python -m src.pipeline --list               # show registry
    python -m src.pipeline --model ettin-150m-ft  # one-off
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    hf_id: str
    finetune: bool
    note: str = ""


MODELS: list[ModelSpec] = [
    # ----- Run order rationale -----
    # 1) Primary fine-tune (Ettin) — our headline number, run first.
    # 2) All frozen-probe baselines (cheap, ~12-15 min each) — quick comparison
    #    points so we have something to look at while bigger trains run.
    # 3) Bell-paper fine-tunes (BERT, ModernBERT, RoBERTa) — slowest, last.
    # ---------------------------------

    ModelSpec(
        slug="ettin-150m-ft",
        hf_id="jhu-clsp/ettin-encoder-150m",
        finetune=True,
        note="Primary candidate. ModernBERT-architecture encoder, post-dates Bell.",
    ),
    ModelSpec(
        slug="ettin-150m-pretrained",
        hf_id="jhu-clsp/ettin-encoder-150m",
        finetune=False,
        note="Pretrained baseline (frozen encoder + linear probe).",
    ),
    ModelSpec(
        slug="bert-base-pretrained",
        hf_id="google-bert/bert-base-uncased",
        finetune=False,
        note="Bell row: BERT (pretrained / frozen).",
    ),
    ModelSpec(
        slug="modernbert-base-pretrained",
        hf_id="answerdotai/ModernBERT-base",
        finetune=False,
        note="Bell row: ModernBERT (pretrained / frozen).",
    ),
    ModelSpec(
        slug="roberta-base-pretrained",
        hf_id="FacebookAI/roberta-base",
        finetune=False,
        note="Bell row: RoBERTa / AFaCTA (pretrained / frozen).",
    ),
    ModelSpec(
        slug="bert-base-ft",
        hf_id="google-bert/bert-base-uncased",
        finetune=True,
        note="Bell row: BERT (Finetuned). Reference baseline.",
    ),
    ModelSpec(
        slug="modernbert-base-ft",
        hf_id="answerdotai/ModernBERT-base",
        finetune=True,
        note="Bell row: ModernBERT (Finetuned).",
    ),
    ModelSpec(
        slug="roberta-base-ft",
        hf_id="FacebookAI/roberta-base",
        finetune=True,
        note="Bell row: RoBERTa / AFaCTA (Finetuned).",
    ),

    # --- Same-family alternatives, left disabled by default ---
    # ModelSpec("ettin-400m-ft", "jhu-clsp/ettin-encoder-400m", True, "Larger Ettin if 150m headroom is exhausted."),
    # ModelSpec("neobert-base-ft", "chandar-lab/NeoBERT", True, "Another 2025 encoder; mentioned in Ettin paper."),
    # ModelSpec("deberta-v3-base-ft", "microsoft/deberta-v3-base", True, "Strong pre-Bell encoder, useful sanity check."),

    # --- LLM track: Bell shows decoders aren't worth the latency in-domain.
    #     Left commented to keep the takehome scope focused, easy to re-enable.
    # ModelSpec("llama-3.2-1b-instruct-ft", "meta-llama/Llama-3.2-1B-Instruct", True, "Bell row: Llama-3.2-1B-Instruct (Finetuned)."),
]


def get(slug: str) -> ModelSpec:
    for m in MODELS:
        if m.slug == slug:
            return m
    raise KeyError(f"unknown model slug: {slug!r}. known: {[m.slug for m in MODELS]}")
