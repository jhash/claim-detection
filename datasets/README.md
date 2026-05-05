# Datasets

```
datasets/
├── all_binary.csv                          # 21,079 sentences, all binary-fit data merged
├── normalize.py                            # raw → normalized.csv reproducer
├── verita-composite/                       # Bell-paper sources (Claimbuster, PoliClaim, AVeriTeC, +extras)
├── checkthat-2025/                         # CLEF CheckThat! 2025 — Tasks 1 & 2
├── claimify/                               # microsoft/claimify-dataset (Metropolitansky & Larson, ACL 2025)
└── feverfact/                              # FEVERFact (Ullrich et al., 2025)
```

## Consistent format

Per-dataset normalized files use:

| column | type | meaning |
|---|---|---|
| `text` | str | the input sentence |
| `label` | int (0 / 1) | 1 = claim-like / check-worthy / subjective / contains a factual claim (per-source semantics, see each README) |
| `source` | str | dataset slug |

`datasets/all_binary.csv` is the concatenation of every binary-fit normalized
file plus `verita-composite/ours/{train,test}.csv`. Drop-in for HF
`datasets.load_dataset("csv", data_files="datasets/all_binary.csv")`.

Run `python datasets/normalize.py` to rebuild any time the raw files change.

| source slug | rows | task |
|---|---:|---|
| `verita-composite-ours-train` | 10,425 | check-worthy claim (Bell composite) |
| `verita-composite-ours-test` | 2,607 | check-worthy claim (Bell composite) |
| `checkthat-2025-task1-subjectivity` (en train+dev+test) | 1,592 | subjectivity (SUBJ=1, OBJ=0) |
| `claimify` | 6,490 | sentence contains verifiable factual claim |
| **aggregate `all_binary.csv`** | **21,079** | mixed (use `source` column to filter / weight) |

`label` semantics differ per source — they are **not strictly the same target**.
Subjectivity (CheckThat Task 1) and check-worthiness (Bell, Claimify) are
correlated but distinct. Train mixed only if the cross-task signal is wanted;
otherwise filter on `source`.

Two of the four newer datasets are generation/extraction tasks and don't fit
the binary schema — see `claimify/`, `feverfact/`, and
`checkthat-2025/task2-claim-normalization/` READMEs for what's preserved
in raw form.

### What flows into training (and what doesn't)

**Only `verita-composite/ours/`** feeds the fine-tuning pipeline —
`train.csv` (10,425 rows) for training, `test.csv` (2,607 rows) for
the in-domain test. That's a frozen 80/20 split shipped pre-divided
by Verita; we don't shuffle or re-split. See `src/pipeline.py:DATA_DIR`.

The other folders here (`checkthat-2025/`, `claimify/`, `feverfact/`)
are **not** in the training mix today. They serve two purposes:

- **Out-of-domain evaluation** — Claimify (LLM-generated text) and
  CT22 (tweets, inside `verita-composite/CheckThat/`) are read by
  `src/evaluate_ood.py` to test transfer outside the political-debate
  distribution.
- **Future training-mix candidates** — pre-normalized to
  `text,label,source` so they can be concatenated into the training
  set later if we broaden the distribution.

---

## `verita-composite/`

Vendored copy of the `data/` tree from
[VeritaResearch/claim-extraction](https://github.com/VeritaResearch/claim-extraction)
@ commit `4fd0cbe0f74fb08d3caf76d77f6757fc9207ebe9` (2025-05-19). Default
training corpus for the Bell-paper replication.

| Path | Bell paper? | Format | Notes |
|---|---|---|---|
| `ours/train.csv`, `ours/test.csv` | — (Verita's own merge) | CSV: `text,label` | 10,425 train + 2,607 test = **13,032 sentences**. Bell reports **12,997** for Claimbuster + PoliClaim Gold + AVeriTeC; this is a re-implementation, not Bell's exact split. |
| `Claimbuster/full.json`, `train.json`, `test.json` | yes | JSON array of `{sentence_id, label, text}` | `full.json` is the source; `train`/`test` are minified single-line arrays (~37k sentences total). |
| `PoliClaim/*.xlsx` | yes (Gold subset) | XLSX, one file per state-year | Eight files (AL2003, CT2014, DE1999, DE2021, IN2001, IN2011, KY2018, US2016). |
| `AVeriTeC/train.json` | yes | JSON array, claim-level with rich metadata | Different schema — claim-level, not sentence-level. |
| `CheckThat/CT22_english_1B_claim_dev_test.tsv` | no | TSV | CLEF CheckThat 2022 Task 1B. Useful as OOD eval. |
| `congressional_records/session_apr10_2025.csv` | no | CSV | Congressional Record extract — unlabeled real-world. |
| `subj/train.tsv` | no | TSV | Subjectivity — possible auxiliary signal. |

---

## `checkthat-2025/`

[CLEF CheckThat! Lab 2025](https://checkthat.gitlab.io/clef2025/)
([overview paper, arXiv:2503.14828](https://arxiv.org/abs/2503.14828)).
Source: [GitLab repo](https://gitlab.com/checkthat_lab/clef2025-checkthat-lab).

### `task1-subjectivity/`

Binary classification: SUBJ vs OBJ at sentence level. Successor to the
`subj/` data already in `verita-composite/`.

- `raw/` — full multilingual TSVs (English, Arabic, Bulgarian, German, Greek,
  Italian, Polish, Romanian, Ukrainian, plus a multilingual blend) with the
  original `sentence_id, sentence, label, solved_conflict` schema.
- `normalized_{train,dev,test}.csv` — English splits only, mapped to
  `text,label,source` with **SUBJ → 1, OBJ → 0**.
- License: CC BY-NC-SA 4.0 (per-language `Licenses_*.txt` files included).

### `task2-claim-normalization/`

Generation task — given a noisy social-media post, produce a normalized
fact-checkable claim. Doesn't fit the binary `text,label` schema.

- `raw/{train,dev,test}/` — **English split only** to keep size sane (full
  data covers 20 languages, ~35 MB). Re-clone the GitLab repo for other
  languages.
- Format: CSV with at minimum `post,normalized_claim` (see `baseline.ipynb`
  upstream).

---

## `claimify/`

[`microsoft/claimify-dataset`](https://huggingface.co/datasets/microsoft/claimify-dataset)
on HF — **6,490 sentences** from 396 [BingCheck](https://github.com/HuangOwen/BingCheck)
answers, each annotated with `contains_factual_claim` ∈ {True, False}.
Companion to [arXiv:2502.10855](https://arxiv.org/abs/2502.10855)
(Metropolitansky & Larson, ACL 2025) — *"Towards Effective Extraction and
Evaluation of Factual Claims"*.

- `raw/data.csv` — untouched HF download. Schema:
  `answer_id, question, sentence_id, sentence, contains_factual_claim`.
- `normalized.csv` — `text,label,source` with **True → 1, False → 0**.
- License: CDLA-Permissive-2.0 (per HF dataset card).
- Notable: this is **LLM-generated text** (search-assistant answers),
  unlike the political / news domain of Claimbuster + PoliClaim. Useful for
  showing OOD robustness on AI-generated text.

---

## `feverfact/`

[FEVERFact](https://github.com/aic-factcheck/claim_extraction) — Ullrich et al.,
*"Claim Extraction for Fact-Checking: Data, Models, and Automated Metrics"*,
[arXiv:2502.04955](https://arxiv.org/abs/2502.04955).
**17K atomic factual claims extracted from ~4K contextualised Wikipedia
sentences**, derived from FEVER.

- `raw/{train,test,validation}.jsonl` — 3,530 / 444 / 445 input sentences
  respectively.
- Schema per row: `{source, sentence, sentence_id, claims: [str, ...],
  claim_ids: [int, ...], sentence_context: [str, ...], source_text}`.
- One-to-many generation task — each input sentence maps to **multiple**
  atomic claims. Not directly normalized to binary `text,label`. Useful as
  a **claim-extraction** training set if we extend the API beyond pure
  classification.

---

## Considered but not vendored

- **AVeriTeC-DCE** (Deng et al., ACL 2024,
  [arXiv:2406.03239](https://arxiv.org/abs/2406.03239),
  [github.com/Tswings/AVeriTeC-DCE](https://github.com/Tswings/AVeriTeC-DCE))
  — document-level claim extraction with decontextualisation. The repo
  ships pipeline scripts and AVeriTeC URL→fulltext joins (~12 MB), but the
  underlying claim-level data is the same AVeriTeC we already have under
  `verita-composite/AVeriTeC/`. Re-clone if extending to document-level
  extraction.

## Source papers

- Bell, *"Less Can be More: Comparing LLMs to Smaller Encoder-Only Models for Claim Detection,"* FEVER 2025 (`../papers/2025.fever-1.6.pdf`).
- Weller et al., *"Seq vs Seq: An Open Suite of Paired Encoders and Decoders,"* ICLR 2026 (`../papers/2507.11412v2.pdf`) — model paper, no claim-detection corpus introduced.

## License

No single license covers everything in this directory. Per-source:

- **Claimbuster** — research use, citation required (Hassan et al., 2017).
- **PoliClaim** — Ni et al., 2024.
- **AVeriTeC** — CC BY-SA 4.0 (Schlichtkrull et al., NeurIPS 2023).
- **CheckThat 2022 / 2025 Task 1** — CC BY-NC-SA 4.0.
- **CheckThat 2025 Task 2** — see upstream lab terms.
- **Claimify** — CDLA-Permissive-2.0.
- **FEVERFact** — derived from FEVER (CC BY-SA 3.0).

Verify upstream terms before any redistribution.
