# Datasets

## `verita-composite/`

Vendored copy of the `data/` tree from
[VeritaResearch/claim-extraction](https://github.com/VeritaResearch/claim-extraction)
at commit `4fd0cbe0f74fb08d3caf76d77f6757fc9207ebe9` (2025-05-19).

The Verita repo ships the source datasets used by Bell (FEVER 2025, "Less Can
be More") plus a few extras useful for transfer / OOD evaluation.

### What's inside

| Path | Bell paper? | Format | Notes |
|---|---|---|---|
| `ours/train.csv`, `ours/test.csv` | — (Verita's own merge) | CSV: `text,label` | 10,425 train + 2,607 test = **13,032 sentences**. Bell reports **12,997** for Claimbuster + PoliClaim Gold + AVeriTeC; these splits are close but not identical. |
| `Claimbuster/full.json`, `train.json`, `test.json` | yes | JSON array of `{sentence_id, label, text}` | `full.json` is the source; `train`/`test` are minified single-line arrays (~37k sentences total). |
| `PoliClaim/*.xlsx` | yes (Gold subset) | XLSX, one file per state-year | Eight files (AL2003, CT2014, DE1999, DE2021, IN2001, IN2011, KY2018, US2016). |
| `AVeriTeC/train.json` | yes | JSON array, claim-level with rich metadata (`claim`, `label` ∈ {Supported, Refuted, ...}, `speaker`, `claim_date`, `fact_checking_article`, ...) | Different schema from the others — claim-level, not sentence-level. |
| `CheckThat/CT22_english_1B_claim_dev_test.tsv` | no | TSV | CLEF CheckThat 2022 Task 1B. Useful as an OOD eval set. |
| `congressional_records/session_apr10_2025.csv` | no | CSV | Congressional Record extract — useful as an unlabeled real-world test. |
| `subj/train.tsv` | no | TSV | Subjectivity dataset — possible auxiliary signal. |

Total size: ~16 MB.

### Default training data

Plan to train + evaluate on `verita-composite/ours/train.csv` and
`verita-composite/ours/test.csv` (binary `text,label`). Drop-in for HF
`datasets.load_dataset("csv", ...)`. Conversion to JSONL is a one-liner if
needed later.

### License

The Verita repo has no `LICENSE` file. Upstream license terms apply per source:

- **Claimbuster** — released by UT Arlington, [claimbuster.org](https://idir.uta.edu/claimbuster/) — research use, citation required (Hassan et al., 2017).
- **PoliClaim** — Ni et al., 2024.
- **AVeriTeC** — Schlichtkrull et al., NeurIPS 2023; CC BY-SA 4.0 per the official release.
- **CheckThat** — CLEF shared-task data, research use.

Verify upstream terms before any redistribution.

### Newer datasets to consider (post-Bell, optional)

The Bell paper (FEVER 2025) was finalized before these landed. Worth a look if
we want to (a) cite recency, (b) demonstrate OOD robustness, or (c) handle
LLM-generated text:

- **CLEF CheckThat! 2025** — successor to the CT22 set already in
  `verita-composite/CheckThat/`. Task 1 (Subjectivity, multilingual), Task 2
  (Claim Extraction & Normalization — CLAN dataset, 6,388 post-claim pairs,
  13 languages with full splits + 7 zero-shot), Task 3 (numerical claims),
  Task 4 (scientific discourse). Lab overview:
  [arXiv:2503.14828](https://arxiv.org/abs/2503.14828).
- **Claimify** (Microsoft Research, 2025) — pipeline + eval set for extracting
  high-quality, decontextualized claims *from LLM outputs*. Good differentiator
  if we want to show robustness on AI-generated text.
- **"Claim Extraction for Fact-Checking: Data, Models, and Automated Metrics"**
  ([arXiv:2502.04955](https://arxiv.org/abs/2502.04955), Feb 2025) — sentence-
  to-claim dataset with new automated eval metrics.
- **Document-level Claim Extraction & Decontextualisation**
  ([arXiv:2406.03239](https://arxiv.org/abs/2406.03239)) — argues some
  sentences only become check-worthy after decontextualisation; relevant if we
  want to push beyond pure sentence-level classification.

### Source papers

- Bell, "Less Can be More: Comparing LLMs to Smaller Encoder-Only Models for Claim Detection," FEVER 2025 (`../papers/2025.fever-1.6.pdf`).
- Weller et al., "Seq vs Seq: An Open Suite of Paired Encoders and Decoders," ICLR 2026 (`../papers/2507.11412v2.pdf`) — model paper, no claim-detection corpus introduced.
