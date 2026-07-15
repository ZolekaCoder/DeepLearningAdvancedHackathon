"""Section 5 — Duplicate analysis (drives the retrieval-feasibility verdict).

We quantify exact and semantic duplication of questions and answers, and the
crucial many-to-one mapping: how often do *different* questions share the *same*
answer? A high rate means a small canonical answer bank exists and retrieval can
win. We also measure how often identical questions have *different* answers,
which caps achievable accuracy for any system.
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def exact_duplicate_counts(df: pd.DataFrame) -> pd.DataFrame:
    q = df[C.INPUT_COL]
    a = df[C.OUTPUT_COL]
    n = len(df)
    recs = [
        ("unique_questions", q.nunique(), round(q.nunique() / n * 100, 2)),
        ("duplicated_questions", int(q.duplicated().sum()),
         round(q.duplicated().mean() * 100, 2)),
        ("unique_answers", a.nunique(), round(a.nunique() / n * 100, 2)),
        ("duplicated_answers", int(a.duplicated().sum()),
         round(a.duplicated().mean() * 100, 2)),
        ("unique_qa_pairs", df[[C.INPUT_COL, C.OUTPUT_COL]].drop_duplicates().shape[0],
         round(df[[C.INPUT_COL, C.OUTPUT_COL]].drop_duplicates().shape[0] / n * 100, 2)),
        ("duplicated_qa_pairs", int(df[[C.INPUT_COL, C.OUTPUT_COL]].duplicated().sum()),
         round(df[[C.INPUT_COL, C.OUTPUT_COL]].duplicated().mean() * 100, 2)),
    ]
    return pd.DataFrame(recs, columns=["metric", "value", "pct_of_rows"]).set_index("metric")


def many_questions_one_answer(df: pd.DataFrame, top: int = 15) -> tuple[dict, pd.DataFrame]:
    """How often multiple distinct questions map to the same answer."""
    g = (df.dropna(subset=[C.OUTPUT_COL])
           .groupby(C.OUTPUT_COL)[C.INPUT_COL]
           .nunique())
    multi = g[g > 1]
    summary = {
        "n_answers_with_multiple_distinct_questions": int(len(multi)),
        "pct_answers_shared_by_multiple_questions": round(len(multi) / len(g) * 100, 2) if len(g) else 0,
        "max_distinct_questions_per_answer": int(g.max()) if len(g) else 0,
        "mean_distinct_questions_per_answer": round(float(g.mean()), 3) if len(g) else 0,
    }
    top_tbl = (multi.sort_values(ascending=False).head(top)
               .rename("n_distinct_questions").reset_index())
    top_tbl[C.OUTPUT_COL] = top_tbl[C.OUTPUT_COL].map(
        lambda s: (s[:140] + " …") if isinstance(s, str) and len(s) > 140 else s)
    return summary, top_tbl


def one_question_many_answers(df: pd.DataFrame, top: int = 15) -> tuple[dict, pd.DataFrame]:
    """Identical questions with DIFFERENT answers -> ceiling on determinism."""
    g = (df.dropna(subset=[C.INPUT_COL])
           .groupby(C.INPUT_COL)[C.OUTPUT_COL]
           .nunique())
    multi = g[g > 1]
    summary = {
        "n_questions_with_multiple_distinct_answers": int(len(multi)),
        "pct_questions_with_ambiguous_answers": round(len(multi) / len(g) * 100, 2) if len(g) else 0,
        "max_distinct_answers_per_question": int(g.max()) if len(g) else 0,
    }
    top_tbl = (multi.sort_values(ascending=False).head(top)
               .rename("n_distinct_answers").reset_index())
    top_tbl[C.INPUT_COL] = top_tbl[C.INPUT_COL].map(
        lambda s: (s[:140] + " …") if isinstance(s, str) and len(s) > 140 else s)
    return summary, top_tbl
