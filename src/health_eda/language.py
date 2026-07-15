"""Section 2 — Language / subset analysis.

For each language-country subset we quantify representation (counts, %),
question/answer length distributions, vocabulary size, lexical diversity and
character-script profile. The point is to surface imbalance and script issues
that will bias a naively trained model toward the majority (English) subsets.
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from . import text_stats as ts


def subset_table(df: pd.DataFrame, has_answers: bool = True) -> pd.DataFrame:
    """Per-subset representation + length + vocabulary summary."""
    total = len(df)
    rows = []
    for subset, sub in df.groupby(C.SUBSET_COL):
        meta = C.SUBSET_META.get(subset, {})
        q = sub[C.INPUT_COL]
        ql = ts.describe_lengths(q, "words")
        rec = {
            "subset": subset,
            "language": meta.get("language", "?"),
            "country": meta.get("country", "?"),
            "script": meta.get("script", "?"),
            "n": len(sub),
            "pct_of_total": round(len(sub) / total * 100, 2),
            "q_words_mean": round(ql.get("mean", 0), 1),
            "q_words_median": ql.get("median", 0),
            "q_words_min": ql.get("min", 0),
            "q_words_max": ql.get("max", 0),
            "q_vocab": ts.vocab_size(q),
            "q_ttr": round(ts.type_token_ratio(q), 4),
        }
        if has_answers and C.OUTPUT_COL in sub.columns:
            a = sub[C.OUTPUT_COL]
            al = ts.describe_lengths(a, "words")
            rec.update({
                "a_words_mean": round(al.get("mean", 0), 1),
                "a_words_median": al.get("median", 0),
                "a_words_min": al.get("min", 0),
                "a_words_max": al.get("max", 0),
                "a_vocab": ts.vocab_size(a),
                "a_ttr": round(ts.type_token_ratio(a), 4),
            })
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    return out


def imbalance_metrics(subset_tab: pd.DataFrame) -> dict:
    """Quantify class imbalance across subsets (ratio + entropy-based balance)."""
    import numpy as np

    n = subset_tab["n"].to_numpy(dtype=float)
    p = n / n.sum()
    entropy = -(p * np.log(p)).sum()
    max_entropy = np.log(len(p))
    return {
        "n_subsets": len(n),
        "largest_subset": subset_tab.iloc[0]["subset"],
        "smallest_subset": subset_tab.iloc[-1]["subset"],
        "imbalance_ratio_max_min": round(n.max() / n.min(), 2),
        "normalized_entropy_balance": round(float(entropy / max_entropy), 4),
    }


def add_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach q/a word & char length columns (used for plots & downstream)."""
    out = df.copy()
    out["q_words"] = out[C.INPUT_COL].map(ts.n_words)
    out["q_chars"] = out[C.INPUT_COL].map(ts.n_chars)
    if C.OUTPUT_COL in out.columns:
        out["a_words"] = out[C.OUTPUT_COL].map(ts.n_words)
        out["a_chars"] = out[C.OUTPUT_COL].map(ts.n_chars)
    return out


def script_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Character-script composition per subset (share of Ethiopic/Latin/etc.)."""
    rows = []
    for subset, sub in df.groupby(C.SUBSET_COL):
        prof = ts.script_profile_series(sub[col])
        tot = sum(prof.values()) or 1
        rec = {"subset": subset}
        for bucket in ("ASCII_Latin", "Latin_ext", "Ethiopic", "Digit", "Punct/Symbol"):
            rec[bucket] = round(prof.get(bucket, 0) / tot * 100, 1)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("subset")
