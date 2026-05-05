"""Aggregate results/*.json into a sorted comparison table (markdown).

Writes RESULTS.md at the repo root with:
  1. Our results (sorted by F1, descending)
  2. The Bell-paper reference table for the in-domain ClaimBuster test split
  3. A combined view aligning the two

Run after `evaluate.py` has populated results/.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models import MODELS

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OOD_RESULTS_DIR = ROOT / "results" / "ood"
OUT = ROOT / "RESULTS.md"


# ---------------------------------------------------------------------------
# Bell paper reference (FEVER 2025, "Less Can be More").
# In-domain test split = ClaimBuster gold + PoliClaim Gold + AVeriTeC composite.
# Numbers transcribed from the paper's main comparison table.
# ---------------------------------------------------------------------------
BELL_REFERENCE = [
    # (label,                            accuracy, precision, recall, f1)
    ("BERT (Finetuned)",                  0.917,    0.913,     0.918,  0.916),
    ("ModernBERT (Finetuned)",            0.911,    0.908,     0.912,  0.910),
    ("RoBERTa / AFaCTA (Finetuned)",      0.905,    0.901,     0.906,  0.904),
    ("Llama-3.2-1B-Instruct (Finetuned)", 0.872,    None,      None,   0.864),
    ("Factcheck-GPT (zero-shot)",         0.731,    None,      None,   0.708),
]

# Mapping from our slug -> Bell row (for the combined view).
SLUG_TO_BELL = {
    "bert-base-ft":         "BERT (Finetuned)",
    "modernbert-base-ft":   "ModernBERT (Finetuned)",
    "roberta-base-ft":      "RoBERTa / AFaCTA (Finetuned)",
}


def fmt(x):
    if x is None:
        return "—"
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def load_results() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    rows = []
    for spec in MODELS:
        fp = RESULTS_DIR / f"{spec.slug}.json"
        if fp.exists():
            d = json.loads(fp.read_text())
            d["spec"] = spec
            rows.append(d)
    return rows


def load_ood_results() -> list[dict]:
    if not OOD_RESULTS_DIR.exists():
        return []
    rows = []
    for spec in MODELS:
        fp = OOD_RESULTS_DIR / f"{spec.slug}.json"
        if fp.exists():
            d = json.loads(fp.read_text())
            d["spec"] = spec
            rows.append(d)
    return rows


def section_our_results(results: list[dict]) -> str:
    if not results:
        return "_No results yet — run `python -m src.evaluate` first._\n"
    rows = sorted(results, key=lambda d: d.get("f1", -1), reverse=True)
    lines = [
        "| Rank | Model (slug) | Mode | HF id | Accuracy | Precision | Recall | F1 |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows, 1):
        if "error" in r:
            continue
        spec = r["spec"]
        mode = "fine-tuned" if spec.finetune else "frozen probe"
        lines.append(
            f"| {i} | `{spec.slug}` | {mode} | `{spec.hf_id}` "
            f"| {fmt(r['accuracy'])} | {fmt(r['precision'])} | {fmt(r['recall'])} | **{fmt(r['f1'])}** |"
        )
    return "\n".join(lines) + "\n"


def section_bell_reference() -> str:
    lines = [
        "| Bell row | Accuracy | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, a, p, r, f1 in BELL_REFERENCE:
        lines.append(f"| {label} | {fmt(a)} | {fmt(p)} | {fmt(r)} | {fmt(f1)} |")
    return "\n".join(lines) + "\n"


def section_combined(results: list[dict]) -> str:
    by_slug = {r["spec"].slug: r for r in results if "error" not in r}
    bell_by_label = {label: (a, p, r, f1) for label, a, p, r, f1 in BELL_REFERENCE}
    lines = [
        "| Bell row | Bell F1 | Our slug | Our F1 | Δ F1 |",
        "|---|---:|---|---:|---:|",
    ]
    for slug, bell_label in SLUG_TO_BELL.items():
        bell_a, bell_p, bell_r, bell_f1 = bell_by_label[bell_label]
        ours = by_slug.get(slug)
        if ours is None:
            lines.append(f"| {bell_label} | {fmt(bell_f1)} | `{slug}` | — | — |")
            continue
        delta = ours["f1"] - bell_f1
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {bell_label} | {fmt(bell_f1)} | `{slug}` | {fmt(ours['f1'])} | {sign}{delta:.4f} |"
        )

    ettin_ft = by_slug.get("ettin-150m-ft")
    if ettin_ft:
        bell_top_f1 = max(f1 for _, _, _, _, f1 in BELL_REFERENCE)
        delta = ettin_ft["f1"] - bell_top_f1
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| _(best Bell encoder)_ | {fmt(bell_top_f1)} | `ettin-150m-ft` _(new)_ "
            f"| {fmt(ettin_ft['f1'])} | {sign}{delta:.4f} |"
        )
    return "\n".join(lines) + "\n"


def section_ood(ood_rows: list[dict], in_domain: list[dict]) -> str:
    """Per-OOD-dataset comparison + drop from in-domain F1."""
    if not ood_rows:
        return "_No OOD results yet — run `python -m src.evaluate_ood`._\n"
    in_by_slug = {r["spec"].slug: r for r in in_domain if "error" not in r}

    # Collect all dataset names that appear in any row.
    datasets = sorted({k for r in ood_rows for k in r.get("per_dataset", {}).keys()})
    if not datasets:
        return "_OOD result files exist but have no per-dataset payloads._\n"

    sections = []
    for dataset in datasets:
        rows_for_ds = []
        for r in ood_rows:
            spec = r["spec"]
            payload = r.get("per_dataset", {}).get(dataset)
            if not payload:
                continue
            in_f1 = (in_by_slug.get(spec.slug) or {}).get("f1")
            drop = (payload["f1"] - in_f1) if in_f1 is not None else None
            rows_for_ds.append((spec, payload, in_f1, drop))
        rows_for_ds.sort(key=lambda x: x[1]["f1"], reverse=True)

        n_test = rows_for_ds[0][1]["n"] if rows_for_ds else "?"
        lines = [
            f"### `{dataset}` ({n_test} sentences)",
            "",
            "| Rank | Model | Mode | OOD Acc | OOD P | OOD R | OOD F1 | In-domain F1 | Δ vs in-domain |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for i, (spec, p, in_f1, drop) in enumerate(rows_for_ds, 1):
            mode = "fine-tuned" if spec.finetune else "frozen probe"
            in_f1_str = fmt(in_f1) if in_f1 is not None else "—"
            if drop is None:
                drop_str = "—"
            else:
                sign = "+" if drop >= 0 else ""
                drop_str = f"{sign}{drop:.4f}"
            lines.append(
                f"| {i} | `{spec.slug}` | {mode} "
                f"| {fmt(p['accuracy'])} | {fmt(p['precision'])} | {fmt(p['recall'])} | **{fmt(p['f1'])}** "
                f"| {in_f1_str} | {drop_str} |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


OOD_DATASET_DESCRIPTIONS = {
    "claimify": (
        "<strong>Claimify</strong> (<code>microsoft/claimify-dataset</code>) — "
        "6,490 sentences extracted from BingCheck (Microsoft's commercial "
        "search assistant answers). Domain: long-form LLM-generated text. "
        "Label: <code>contains_factual_claim</code>."
    ),
    "ct22": (
        "<strong>CheckThat 2022 Task 1B</strong> — 911 English tweets, "
        "COVID/political. Domain: short social-media text. "
        "Label: <code>class_label</code> (1 = check-worthy)."
    ),
}


def section_ood_descriptions(ood_rows: list[dict]) -> str:
    seen = set()
    for r in ood_rows:
        seen |= set(r.get("per_dataset", {}).keys())
    if not seen:
        return ""
    items = [OOD_DATASET_DESCRIPTIONS.get(d, f"`{d}`") for d in sorted(seen)]
    return "\n\n".join(items) + "\n"


def render(results: list[dict], ood: list[dict] | None = None) -> str:
    n = len([r for r in results if "error" not in r])
    ood = ood or []
    n_ood = len([r for r in ood if r.get("per_dataset")])
    return f"""# Results

Sentence-level claim-detection comparison on the verita-composite test
split (2,607 sentences, derived from ClaimBuster + PoliClaim Gold +
AVeriTeC, see `datasets/verita-composite/`).

Generated by `python -m src.compare` from `results/*.json` and
`results/ood/*.json`.

## In-domain — our models ({n} evaluated)

Sorted by F1, descending.

{section_our_results(results)}

## Bell (FEVER 2025) reference

From the comparison table in *"Less Can be More"* — same in-domain task,
their composite test split. Reproduced here so we can compare.

{section_bell_reference()}

## Side-by-side: ours vs. Bell

Where the same model family appears in both. The new `ettin-150m-ft`
row at the bottom compares against Bell's best encoder.

{section_combined(results)}

## Out-of-domain — {n_ood} model(s) evaluated

These are sentence sets from totally different distributions than the
training data. A model that's truly learned "claim-likeness" — vs.
just memorizing political-debate patterns — should hold up here. The
"Δ vs in-domain" column quantifies how much each model loses when
moved off-distribution.

{section_ood_descriptions(ood)}

{section_ood(ood, results)}

## Methodology notes

- **Test split**: `datasets/verita-composite/ours/test.csv` (2,607 sentences,
  binary `text,label`).
- **Training split**: `datasets/verita-composite/ours/train.csv` (10,425).
- **Fine-tuning**: HF `Trainer`, 3 epochs, lr 5e-5, batch size 16, max
  length 256, MPS on Apple Silicon.
- **Frozen probe**: encoder frozen, only the classification head trained
  (lr 1e-3). Matches Bell's "pretrained" rows in spirit.
- **Δ F1 caveat**: Bell's exact split is 12,997 sentences; the Verita
  re-implementation we trained on is 13,032. ~0.27% size difference; the
  comparison is informative, not literally apples-to-apples.
"""


def main():
    results = load_results()
    ood = load_ood_results()
    OUT.write_text(render(results, ood))
    n_in = len([r for r in results if 'error' not in r])
    n_ood = len([r for r in ood if r.get('per_dataset')])
    print(f"wrote {OUT.relative_to(ROOT)}  ({n_in} in-domain, {n_ood} OOD)")


if __name__ == "__main__":
    main()
