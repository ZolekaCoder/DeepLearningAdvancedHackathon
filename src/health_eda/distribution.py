"""Sections 14 & 15 — Validation and Test distribution vs Train.

Do Val/Test come from the same distribution as Train? We compare, per subset:
length distributions (KS test), language mix, vocabulary OOV, and — using
embeddings — how close each Val/Test question is to its nearest Train question
(a distribution-shift proxy). Test has no answers, so only questions are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import text_stats as ts


def language_mix(train: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    a = train[C.SUBSET_COL].value_counts(normalize=True).mul(100).round(2)
    b = other[C.SUBSET_COL].value_counts(normalize=True).mul(100).round(2)
    out = pd.concat([a.rename("train_pct"), b.rename("other_pct")], axis=1).fillna(0)
    out["delta_pp"] = (out["other_pct"] - out["train_pct"]).round(2)
    return out


def length_ks(train: pd.DataFrame, other: pd.DataFrame, col: str) -> pd.DataFrame:
    """Kolmogorov-Smirnov test of word-length distributions per subset."""
    from scipy.stats import ks_2samp

    rows = []
    for subset in sorted(set(train[C.SUBSET_COL]) & set(other[C.SUBSET_COL])):
        tr = train.loc[train[C.SUBSET_COL] == subset, col].dropna().map(ts.n_words)
        ot = other.loc[other[C.SUBSET_COL] == subset, col].dropna().map(ts.n_words)
        if len(tr) < 10 or len(ot) < 10:
            continue
        ks = ks_2samp(tr, ot)
        rows.append({
            "subset": subset,
            "train_mean": round(float(tr.mean()), 1),
            "other_mean": round(float(ot.mean()), 1),
            "ks_stat": round(float(ks.statistic), 4),
            "ks_pvalue": float(f"{ks.pvalue:.2e}"),
            "same_dist_at_0.05": bool(ks.pvalue > 0.05),
        })
    return pd.DataFrame(rows)


def nearest_train_similarity(other_emb: np.ndarray, train_emb: np.ndarray,
                             other_subset: np.ndarray, batch: int = 512) -> pd.DataFrame:
    """Per subset: distribution of each other-question's max cosine to Train.

    Low similarity => distribution shift / novel questions retrieval can't cover.
    """
    tops = np.empty(len(other_emb), dtype=np.float32)
    for s in range(0, len(other_emb), batch):
        q = other_emb[s:s + batch]
        tops[s:s + batch] = (q @ train_emb.T).max(axis=1)
    df = pd.DataFrame({"subset": other_subset, "max_cos_to_train": tops})
    summ = df.groupby("subset")["max_cos_to_train"].agg(
        mean="mean", median="median",
        p10=lambda x: x.quantile(0.10), p90=lambda x: x.quantile(0.90)).round(4)
    overall = df["max_cos_to_train"].agg(["mean", "median"]).round(4)
    summ.loc["OVERALL"] = [overall["mean"], overall["median"],
                           round(df["max_cos_to_train"].quantile(0.1), 4),
                           round(df["max_cos_to_train"].quantile(0.9), 4)]
    return summ.reset_index()
