"""Section 6 — Cross-language analysis.

Are English examples translations of the African-language examples? We test two
independent signals and do NOT assume an answer:
  1. ID structure: the ID hash suffix may be shared across subsets (a strong,
     dataset-level indicator that the same item exists in multiple languages).
  2. Embedding alignment: for each African-language question, the cosine to its
     nearest English question, calibrated against the English-English baseline.
Conclusions are reported with the evidence for each.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C


def id_suffix(id_str: str) -> str | None:
    """Extract the trailing hash of an ID like ID_TR_Aka_Gha_A3B1799D."""
    m = re.search(r"([0-9A-Fa-f]{8})$", str(id_str))
    return m.group(1) if m else None


def shared_id_analysis(df: pd.DataFrame) -> dict:
    """Do the same ID-suffixes recur across different subsets?"""
    tmp = df.copy()
    tmp["suffix"] = tmp[C.ID_COL].map(id_suffix)
    tmp = tmp.dropna(subset=["suffix"])
    per_suffix_subsets = tmp.groupby("suffix")[C.SUBSET_COL].nunique()
    multi = per_suffix_subsets[per_suffix_subsets > 1]
    return {
        "n_unique_suffixes": int(tmp["suffix"].nunique()),
        "n_suffixes_in_multiple_subsets": int(len(multi)),
        "pct_suffixes_shared": round(len(multi) / tmp["suffix"].nunique() * 100, 2),
        "max_subsets_per_suffix": int(per_suffix_subsets.max()),
        "mean_subsets_per_shared_suffix": round(float(multi.mean()), 2) if len(multi) else 0,
    }


def aligned_examples_by_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Show all rows sharing a suffix, side by side (evidence of parallelism)."""
    tmp = df.copy()
    tmp["suffix"] = tmp[C.ID_COL].map(id_suffix)
    cols = [c for c in (C.SUBSET_COL, C.INPUT_COL, C.OUTPUT_COL) if c in tmp.columns]
    return tmp[tmp["suffix"] == suffix][cols]


def embedding_alignment(emb: np.ndarray, subsets: np.ndarray,
                        english_label_fn=lambda s: s.startswith("Eng_"),
                        sample: int = 2000, seed: int = C.SEED) -> pd.DataFrame:
    """For each non-English subset, distribution of max cosine to any English
    question, vs the English->English nearest baseline."""
    rng = np.random.default_rng(seed)
    is_eng = np.array([english_label_fn(s) for s in subsets])
    eng_emb = emb[is_eng]

    rows = []
    # English->English baseline (exclude self by masking the diagonal-ish top1).
    for subset in sorted(set(subsets)):
        idx = np.where(subsets == subset)[0]
        if len(idx) > sample:
            idx = rng.choice(idx, sample, replace=False)
        q = emb[idx]
        sims = q @ eng_emb.T                       # (n, n_eng)
        if english_label_fn(subset):
            # remove self-match (each row's own vector sits in eng_emb)
            np.fill_diagonal(sims[:, :0], 0)  # no-op guard
            top = np.sort(sims, axis=1)[:, -2]     # 2nd best = nearest other-English
        else:
            top = sims.max(axis=1)                 # nearest English
        rows.append({
            "subset": subset,
            "is_english": english_label_fn(subset),
            "mean_max_cos_to_english": round(float(top.mean()), 4),
            "median_max_cos_to_english": round(float(np.median(top)), 4),
            "p90_max_cos_to_english": round(float(np.quantile(top, 0.9)), 4),
        })
    return pd.DataFrame(rows)
