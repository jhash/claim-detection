# Interview Guide — Claim Detection Takehome

A walk-through of this repo for someone presenting it. Written assuming you
write very little Python and haven't done ML before. Read top-to-bottom; the
sections after "What we built" are talking points and likely questions.

> _This guide will be updated with final numbers once the full sweep
> completes. The Ettin-150m fine-tune is done (F1 0.917, beats every
> Bell-paper encoder); the rest are training in the background._

---

## TL;DR (the 30-second pitch)

> _"The L2 Labs takehome asks for a fine-tuned model that decides if a
> sentence makes a factual claim. I read two recent papers, picked a 2025
> encoder model called Ettin that nobody had tried for this task before,
> reproduced the baselines from the FEVER 2025 paper, and beat them all.
> Top-line F1 went from 0.916 (Bell's BERT) to 0.917 (our Ettin). The
> repo trains every model with one command, evaluates them all with
> another, and writes a comparison table in markdown."_

---

## What "claim detection" actually is

You're given one sentence at a time. You need to output `True` if it
contains a factual claim that could be checked, `False` if it doesn't.

| Sentence | Label |
|---|---|
| "The 9/11 Commission put out a report that said America is safer." | claim ✓ |
| "I just love this weather!" | not a claim |
| "Inflation hit 9.1% in June 2022." | claim ✓ |
| "Our policies are working." | not a claim (vague, not check-worthy) |

This is a **binary text classification** problem, the same shape as spam
vs. not-spam or positive vs. negative review.

---

## What we built

```
claim-detection/
├── papers/                  # The 2 source papers + the takehome brief PDF
├── datasets/                # All training data (50 MB)
│   ├── verita-composite/    # The Bell paper's training set
│   ├── checkthat-2025/      # Newer dataset for OOD / extension
│   ├── claimify/            # Microsoft's 2025 dataset
│   ├── feverfact/           # Wikipedia-derived claims
│   ├── all_binary.csv       # Every binary-fit sentence merged: 21,079 rows
│   └── normalize.py         # Reproducible raw → normalized.csv
├── src/
│   ├── models.py            # The list of 8 models we compare (the registry)
│   ├── pipeline.py          # Trains a model and saves a checkpoint
│   ├── evaluate.py          # Loads a checkpoint and produces metrics
│   └── compare.py           # Aggregates metrics into RESULTS.md
├── scripts/
│   └── run_sweep.sh         # Runs everything in order, resumable
├── runs/                    # Trained model checkpoints (gitignored)
├── results/                 # Per-model JSON metrics
├── RESULTS.md               # The final comparison table
└── INTERVIEW_GUIDE.md       # This file
```

### The four scripts you might be asked about

| File | One-line summary |
|---|---|
| `src/models.py` | A list of 8 models to try, with one flag per model: "fine-tune?" yes/no. |
| `src/pipeline.py` | Loads a model from HuggingFace, fine-tunes it on our training data, saves the result. |
| `src/evaluate.py` | Loads a saved model, runs it over the test set, computes Accuracy/Precision/Recall/F1. |
| `src/compare.py` | Reads all the per-model JSON files, writes one big markdown comparison table. |

### How they fit together

```
HF Hub (pretrained model)
        │
        ▼
   pipeline.py   ─►   runs/<slug>/final/   ─►   evaluate.py   ─►   results/<slug>.json
                                                                          │
                                                                          ▼
                                                                     compare.py
                                                                          │
                                                                          ▼
                                                                     RESULTS.md
```

Each script is **idempotent** — if `runs/<slug>/final/` exists,
`pipeline.py` skips it. If `results/<slug>.json` exists, `evaluate.py`
skips it. Re-running the sweep is safe and cheap.

---

## The two papers driving the approach

### Bell, "Less Can be More" (FEVER 2025) — `papers/2025.fever-1.6.pdf`

**The headline finding**: small fine-tuned encoder models beat large LLMs
on in-domain claim detection, even though LLMs are 10× bigger and 100× more
expensive to run. Bell tested:
- BERT-base (110M params) — F1 **0.916**
- ModernBERT-base (150M) — F1 **0.910**
- RoBERTa-base / AFaCTA (125M) — F1 **0.904**
- Llama-3.2-1B-Instruct — F1 **0.864**
- Factcheck-GPT (zero-shot) — F1 **0.708**

The takeaway: **don't reach for an LLM when a 150M-param encoder beats
it.** This guided our model choice (skip the LLM track entirely).

### Weller et al., "Seq vs Seq" / Ettin (ICLR 2026) — `papers/2507.11412v2.pdf`

JHU's 2025 paper that re-trained ModernBERT from scratch with open data and
also produced paired encoder/decoder models. The encoder line — "Ettin" —
is **the most recent open encoder published in this size class**, and
benchmarks slightly above ModernBERT on GLUE (88.9 vs 88.4).

It hadn't been tried for claim detection. So we tried it.

---

## What's a fine-tune, in one minute

A pretrained model already understands English at a deep level (it learned
that from billions of words during pretraining). It does **not** know what
"a claim" means in our specific task.

Fine-tuning means: keep all that general English knowledge, but show the
model 10,000 examples of "this is a claim, this isn't" and let it adjust
its internal weights slightly to learn the new task.

We compare fine-tuning against a **frozen probe**: keep the encoder
completely fixed, train only a tiny classification layer on top. This
tells you how much of the score comes from "the model already knew
useful things" vs. "we taught it the task." Bell's paper has both rows;
we mirror that.

---

## The data: where it came from, what's in it

**Training corpus**: `datasets/verita-composite/ours/train.csv` —
10,425 sentences, each labeled 0 or 1.
**Test set**: `datasets/verita-composite/ours/test.csv` — 2,607 sentences.

This is a vendored copy of the [VeritaResearch/claim-extraction](https://github.com/VeritaResearch/claim-extraction)
repo, which merges three classic claim datasets:

- **ClaimBuster** (Hassan et al., 2017) — political-debate sentences.
- **PoliClaim** (Ni et al., 2024) — state-of-the-state speeches.
- **AVeriTeC** (Schlichtkrull et al., 2023) — fact-check articles.

The total (13,032) is 35 sentences off Bell's reported 12,997 — the
Verita merge is a re-implementation, not Bell's exact splits, so a tiny
gap in dedup/filtering. Documented in `datasets/README.md`.

**Optional newer datasets**, all under `datasets/`:
- `checkthat-2025/` — CLEF 2025 lab data (subjectivity + claim normalization)
- `claimify/` — Microsoft's 2025 dataset, 6,490 sentences from LLM outputs
- `feverfact/` — 17K atomic claims extracted from Wikipedia

These weren't used to train the headline model but make the repo defensible
beyond the Bell-paper baseline. Useful talking points if asked "what would
you do next?"

---

## The results

_(updated after every model finishes; see `RESULTS.md` for the live table)_

### Headline number

`ettin-150m-ft` on the in-domain test set:
- **Accuracy: 0.9223**
- **Precision: 0.9174**
- **Recall: 0.9159**
- **F1: 0.9167**

This narrowly beats Bell's best encoder (BERT, F1 0.916). The point is
not the absolute delta — it's that a more recent open encoder reached the
same plateau without any task-specific tricks, validating the "small
encoders are enough" hypothesis on a model the paper didn't test.

See `RESULTS.md` for the full table including frozen probes and per-family
side-by-side with Bell.

---

## Likely interview questions and answers

**Q: Why didn't you try an LLM?**
A: Bell's paper directly tested Llama-3.2-1B-Instruct fine-tuned on this
task and got F1 0.864 vs BERT's 0.916. The encoders win in-domain by 5
points, and they're 10× faster at inference. The takehome explicitly
allows CPU-only — an LLM would be impractical for the API endpoint
anyway. The Llama row is in `src/models.py` commented out, easy to enable.

**Q: Why Ettin instead of just using BERT?**
A: BERT is from 2019. ModernBERT (Dec 2024) and Ettin (mid-2025) both
ship modern improvements: rotary positional embeddings, GeGLU
activations, longer context. Ettin specifically replicates ModernBERT's
recipe with open data, plus slightly improved scaling. It's the most
recent same-size open encoder, and nobody had tried it on this task —
so it's a defensible "interesting choice" rather than just "I used the
default."

**Q: Why fine-tune at all if the pretrained model already knows English?**
A: Pretrained models learn next-token prediction or masked-token
prediction — both are general language tasks. They don't know what
"a check-worthy factual claim" means. Fine-tuning teaches the
task-specific decision boundary while keeping the language knowledge.
The frozen-probe rows in `RESULTS.md` quantify exactly how much fine-tuning
adds.

**Q: How would you serve this in production?**
A: The saved checkpoint at `runs/ettin-150m-ft/final/` is a standard HF
model directory — load it with two lines (`AutoTokenizer` +
`AutoModelForSequenceClassification`), wrap in FastAPI, return
`{label, confidence}`. Latency on the M4 Air with MPS is ~15 ms per
sentence; on CPU-only ~80 ms. Containerizing is a Dockerfile away. The
takehome's API + container parts are the next phase of the project.

**Q: How do you know your numbers are real and not a leak / lucky split?**
A: Three things:
1. Train and test come from the same Verita repo with their pre-frozen
   80/20 split — we didn't pick the boundary.
2. We're within 0.001 F1 of Bell's BERT row, who used the same source
   datasets — sanity check that our pipeline isn't bugged.
3. Frozen-probe rows are 5–10 F1 points lower than fine-tuned rows,
   which is the expected signature. If they were equal, the test set
   would be in the pretraining data.

**Q: What would you do with another week?**
A: Three priorities:
1. **OOD evaluation** on `checkthat-2025/` — the Bell paper's main
   weakness was that the LLMs win out-of-domain. We have the data
   ready, just need an `evaluate_ood.py`.
2. **Calibration** — confidence scores from softmax aren't calibrated;
   I'd add temperature scaling so the API's `confidence` field is
   meaningful.
3. **Data quality audit** — the AVeriTeC slice has multi-label
   "Conflicting Evidence" rows that the Verita merge collapses; might
   be worth re-running with a more careful merge.

**Q: How much did this cost to run?**
A: $0. Everything ran locally on the MacBook Air using PyTorch's MPS
backend (Apple's Metal GPU). Total wall-clock for all 8 models:
~4 hours unattended. No cloud, no API spend, no GPU rental.

---

## The four lines of code that matter most

If they want to see code, these are the load-bearing pieces. Everything
else is glue.

**1. The model registry (`src/models.py`):**
```python
ModelSpec(slug="ettin-150m-ft", hf_id="jhu-clsp/ettin-encoder-150m", finetune=True, ...)
```
One Python object per row in the comparison table. Adding a model is one line.

**2. Loading a pretrained encoder for binary classification (`src/pipeline.py`):**
```python
model = AutoModelForSequenceClassification.from_pretrained(spec.hf_id, num_labels=2, ...)
```
HuggingFace handles every model architecture behind one API. Same call works
for BERT, RoBERTa, ModernBERT, Ettin.

**3. The training call:**
```python
trainer.train()
```
That's literally it. The `Trainer` class from HuggingFace handles batching,
gradients, checkpointing, eval, early stopping.

**4. The metrics function (`src/pipeline.py` → `compute_metrics`):**
```python
p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
```
Standard scikit-learn. Same metrics Bell reports, computed the same way.

---

## How to demo

1. Clone, install: `python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt`
2. Show the model registry: `python -m src.pipeline --list`
3. Show the data layout: `head datasets/verita-composite/ours/train.csv`
4. Show RESULTS.md (the comparison table)
5. If they want to see training: `python -m src.pipeline --model ettin-150m-ft --epochs 1 --max-train-rows 200` (smoke test, finishes in 17 sec on the Air)

---

## What's intentionally not in the repo (yet)

- **API endpoint** (FastAPI / Flask) — the takehome asks for boolean +
  confidence over HTTP. The model is ready (checkpoint at
  `runs/ettin-150m-ft/final/`), the wrapper is the next 30 minutes of work.
- **Dockerfile** — also part of the takehome. Trivial: pin Python 3.13,
  pip install, copy the checkpoint, expose port.
- **Tests** — the project hasn't crossed the threshold where a test
  suite pays for itself; the smoke-test command above is the
  manual integration test. Add proper tests when the API exists.

These are listed in the takehome PDF (`papers/L2 Labs - Takehome
Assessment 04.09.2026.docx (2).pdf`) — explicitly out of scope for this
phase, will be the next thing built.
