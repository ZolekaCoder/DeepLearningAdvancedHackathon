# Multilingual Health QA — Retrieval Solution (IndabaX × Zindi)

A **retrieve-and-copy** system for the IndabaX South Africa Multilingual Health
Question Answering hackathon. Fine-tuned multilingual dense retrievers, chosen and
tuned entirely on held-out validation.

**Best public score: 0.6567** (peaked **1st place** during the competition).

---

## 1. The problem

Answer **maternal / sexual / reproductive-health** questions (HIV, STIs,
contraception, pregnancy, puberty, GBV, …) **in the same language as the question**,
across **8 language-country subsets**:

| Subset | Language | Country | Script |
|---|---|---|---|
| Eng_Uga · Eng_Gha · Eng_Ken · Eng_Eth | English | Uganda / Ghana / Kenya / Ethiopia | Latin |
| Aka_Gha | Akan | Ghana | Latin |
| Lug_Uga | Luganda | Uganda | Latin |
| Swa_Ken | Swahili | Kenya | Latin |
| Amh_Eth | Amharic | Ethiopia | **Ge'ez** |

**Data:** Train 29,815 · Val 6,686 · Test 2,618 rows (`ID, input, output, subset`).

**Metric:** `0.37·ROUGE-1 F1 + 0.37·ROUGE-L F1 + 0.26·LLM-judge`. ~74% is lexical
overlap with the hidden gold answer. The `rouge-score` tokenizer strips non-ASCII,
so **Amharic (Ge'ez) ROUGE ≈ 0** regardless of answer quality. Submission = `ID` +
three identical answer columns; public LB = 30% of test, private = 70% (decides it).

**The structural fact that drives everything (from the EDA):** answers are
**canonical, not individually written** — ~39% are exact duplicates and many
distinct questions map to the same answer. The gold answers effectively **live in
the train/val answer pool.**

## 2. The solution

For each test question, **retrieve the most similar train/val question and copy its
answer verbatim.** No generation (generation loses to copying on a ROUGE-dominated,
canonical-answer task — see `SOLUTION.md`).

The winning system is a **4-way ensemble** of dense retrievers:

- `fthn`, `fthn2` — **bge-m3** fine-tuned on **question→question same-answer pairs +
  hard negatives** (round-1 and an intensified round-2 mined with the round-1 model).
- `fte`, `fte2` — **multilingual-e5-large** fine-tuned the same way.
- Ensemble by averaged cosine (weights `0.3 / 0.2 / 0.2 / 0.3`), **within-subset**
  (language-safe) retrieval, over the **train + val** answer pool.

**Key idea:** the *question→question same-answer* contrastive objective teaches the
embedder exactly what inference needs — *which stored question shares my answer* —
rather than generic relevance. That drove the biggest gains.

## 3. Results

| Stage | Public LB |
|---|---|
| Off-the-shelf e5-base retrieval | 0.560 |
| e5-large + train+val corpus | 0.578 |
| q→q fine-tuned bge-m3 | 0.628 |
| + hard negatives + bge/e5 ensemble | 0.6375 (**1st**) |
| + intensified round-2 (`fthn2`,`fte2`), 4-way | **0.6567** |

Every step was validated on held-out **Val** (never the public LB); EDA showed no
distribution shift, so Val ≈ the private set. Full analysis of what the ensemble
thrives at vs fails at (e.g. Akan/Eng_Ghana answers are ~0–2% in the pool → copy is
structurally capped there) is in `SOLUTION.md`.

## 4. Repository layout

```
src/health_eda/         # reusable EDA package (io, text stats, retrieval, topics, viz, ...)
scripts/                # milestone runners + notebook builders + submission scripts
  run_all.py            #   full EDA pipeline (CI-friendly)
  build_winning_notebook.py   # emits notebooks/winning_solution.ipynb
notebooks/
  health_qa_eda.ipynb          # narrated end-to-end EDA
  winning_solution.ipynb       # ← reproduces the 0.6567 4-way ensemble (Colab/GPU)
  colab_finetune_retriever.ipynb        # modelling dev notebook (Parts 0-9)
  colab_advanced_reranker_qwen3.ipynb   # rejected-lever experiments (reranker, Qwen3)
outputs/figures, outputs/tables         # EDA deliverables
DATASET_ANALYSIS_REPORT.md   # full EDA report
MODELLING_DECISIONS.md       # evidence-linked modelling decisions (the blueprint)
SOLUTION.md                  # detailed solution write-up (strengths, failures, rejected ideas)
requirements.txt
```

Not committed (see `.gitignore`): the competition CSVs (obtain from Zindi), the
`.venv`, cached embeddings/checkpoints in `outputs/artifacts/` (regenerable), and
fine-tuned model weights (kept on Google Drive; too large for git).

## 5. Reproduce

**Data:** download `Train.csv`, `Val.csv`, `Test.csv`, `SampleSubmission.csv` from
the Zindi competition page into the repo root.

**EDA (local, CPU/MPS):**
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
PYTHONPATH=. .venv/bin/python scripts/run_all.py     # or run notebooks/health_qa_eda.ipynb
```

**Winning model (Colab, GPU/A100 — fine-tuning needs a GPU):**
open `notebooks/winning_solution.ipynb` in Google Colab, set an A100 runtime,
Run All, upload the 4 CSVs when prompted. It fine-tunes the four retrievers,
verifies the 4-way ensemble on Val (~0.4192 `rouge_wtd`), and writes
`submission_winning.csv`. Fixed seeds; ~50–70 min end-to-end.

## 6. Documentation

- **`SOLUTION.md`** — the winning solution in depth (architecture, what it thrives
  at / fails at, every rejected alternative and why, future work).
- **`DATASET_ANALYSIS_REPORT.md`** — full exploratory data analysis.
- **`MODELLING_DECISIONS.md`** — the evidence-linked decision log that guided modelling.
