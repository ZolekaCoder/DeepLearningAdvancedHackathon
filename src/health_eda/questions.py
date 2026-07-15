"""Section 3 — Question analysis.

Goal: is the input space templated or diverse? Templating (few intents, many
paraphrases) favours retrieval/canonical answering; diversity favours
generation. We measure prefixes, starting phrases, n-grams and a rough
template signature, then (in topics.py) cluster intents semantically.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from . import config as C
from . import text_stats as ts


def starting_ngrams(series: pd.Series, n: int = 3, top: int = 30) -> pd.DataFrame:
    """Most common opening n-grams (the 'question template' signal)."""
    c: Counter = Counter()
    for txt in series.dropna():
        toks = ts.tokenize(txt)
        if len(toks) >= n:
            c[" ".join(toks[:n])] += 1
    total = sum(c.values()) or 1
    rows = [(" ".join(k) if isinstance(k, tuple) else k, v, round(v / total * 100, 2))
            for k, v in c.most_common(top)]
    return pd.DataFrame(rows, columns=["prefix", "count", "pct"])


def first_word_distribution(series: pd.Series, top: int = 25) -> pd.DataFrame:
    c: Counter = Counter()
    for txt in series.dropna():
        toks = ts.tokenize(txt)
        if toks:
            c[toks[0]] += 1
    total = sum(c.values()) or 1
    rows = [(w, v, round(v / total * 100, 2)) for w, v in c.most_common(top)]
    return pd.DataFrame(rows, columns=["first_word", "count", "pct"])


def top_ngrams(series: pd.Series, n: int, top: int = 30) -> pd.DataFrame:
    c = ts.ngram_counts(series, n=n, top=top)
    total = sum(ts.ngram_counts(series, n=n).values()) or 1
    rows = [(" ".join(k), v, round(v / total * 100, 3)) for k, v in c.most_common(top)]
    return pd.DataFrame(rows, columns=[f"{n}gram", "count", "pct"])


# English interrogative templates (only meaningful for the English subsets).
_TEMPLATES = {
    "what": r"^\s*what\b",
    "how": r"^\s*how\b",
    "why": r"^\s*why\b",
    "can/could i": r"^\s*(can|could)\s+(i|we)\b",
    "is it safe/ok": r"^\s*is\s+it\s+(safe|ok|okay|normal)\b",
    "is/are": r"^\s*(is|are)\b",
    "do/does": r"^\s*(do|does)\b",
    "when": r"^\s*when\b",
    "where": r"^\s*where\b",
    "should": r"^\s*should\b",
    "which": r"^\s*which\b",
    "who": r"^\s*who\b",
}


def template_coverage(series: pd.Series) -> pd.DataFrame:
    """Fraction of (English) questions matching each interrogative template."""
    compiled = {k: re.compile(v, re.I) for k, v in _TEMPLATES.items()}
    counts = Counter()
    n = 0
    for txt in series.dropna():
        n += 1
        for name, rx in compiled.items():
            if rx.search(txt):
                counts[name] += 1
                break  # first-match template
        else:
            counts["<other>"] += 1
    rows = [(k, v, round(v / (n or 1) * 100, 2))
            for k, v in counts.most_common()]
    return pd.DataFrame(rows, columns=["template", "count", "pct"])


def diversity_signature(series: pd.Series) -> dict:
    """Cheap templating signal: how concentrated are opening phrases?

    High share captured by the top few 3-word prefixes => templated.
    """
    pre = starting_ngrams(series, n=3, top=10000)
    total = pre["count"].sum() or 1
    top5 = pre.head(5)["count"].sum()
    top20 = pre.head(20)["count"].sum()
    return {
        "n_questions": int(total),
        "n_unique_3word_prefixes": int(len(pre)),
        "unique_prefix_ratio": round(len(pre) / total, 4),
        "pct_covered_by_top5_prefixes": round(top5 / total * 100, 2),
        "pct_covered_by_top20_prefixes": round(top20 / total * 100, 2),
    }
