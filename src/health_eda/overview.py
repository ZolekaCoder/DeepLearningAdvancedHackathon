"""Section 1 — Dataset Overview.

Answers: what are we actually working with? Shapes, dtypes, missingness,
duplicates, memory, and the target/submission structure. These facts frame
every later decision (e.g. missing answers => can't use those rows for
supervised generation; duplicate Q/A => retrieval leakage risk).
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def split_summary(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per split: rows, cols, memory, duplicates, missing per column."""
    rows = []
    for name, df in dfs.items():
        rec = {
            "split": name,
            "rows": len(df),
            "cols": df.shape[1],
            "memory_MB": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            "dup_rows": int(df.duplicated().sum()),
        }
        for col in (C.INPUT_COL, C.OUTPUT_COL, C.SUBSET_COL):
            if col in df.columns:
                rec[f"missing_{col}"] = int(df[col].isna().sum())
        rows.append(rec)
    return pd.DataFrame(rows).set_index("split")


def column_dtypes(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Column names + dtypes per split (transposed for readability)."""
    recs = {}
    for name, df in dfs.items():
        recs[name] = {c: str(t) for c, t in df.dtypes.items()}
    return pd.DataFrame(recs)


def duplicate_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Exact-match duplicate counts for questions, answers and Q/A pairs."""
    recs = []
    q = df[C.INPUT_COL]
    recs.append(("duplicated_questions", int(q.duplicated().sum()),
                 round(q.duplicated().mean() * 100, 2)))
    if C.OUTPUT_COL in df.columns:
        a = df[C.OUTPUT_COL]
        recs.append(("duplicated_answers", int(a.duplicated().sum()),
                     round(a.duplicated().mean() * 100, 2)))
        pair = df[[C.INPUT_COL, C.OUTPUT_COL]].duplicated()
        recs.append(("duplicated_qa_pairs", int(pair.sum()),
                     round(pair.mean() * 100, 2)))
    return pd.DataFrame(recs, columns=["kind", "count", "pct_of_rows"]).set_index("kind")


def example_records(df: pd.DataFrame, n: int = 3, per_subset: bool = True,
                    max_chars: int = 300) -> pd.DataFrame:
    """A few readable example rows (optionally one per subset)."""
    def _clip(s):
        return s if not isinstance(s, str) or len(s) <= max_chars else s[:max_chars] + " …"

    if per_subset and C.SUBSET_COL in df.columns:
        sample = df.groupby(C.SUBSET_COL, group_keys=False).head(1)
    else:
        sample = df.head(n)
    cols = [c for c in (C.SUBSET_COL, C.INPUT_COL, C.OUTPUT_COL) if c in df.columns]
    out = sample[cols].copy()
    for c in (C.INPUT_COL, C.OUTPUT_COL):
        if c in out.columns:
            out[c] = out[c].map(_clip)
    return out.reset_index(drop=True)


def submission_summary(sample_sub: pd.DataFrame) -> pd.DataFrame:
    """Describe the submission's target columns (the evaluation surface)."""
    recs = []
    for col in sample_sub.columns:
        if col == C.ID_COL:
            continue
        recs.append({
            "target_column": col,
            "n_rows": len(sample_sub),
            "n_unique_values": sample_sub[col].nunique(dropna=True),
        })
    return pd.DataFrame(recs).set_index("target_column")
