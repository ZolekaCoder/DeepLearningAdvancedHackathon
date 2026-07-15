"""Section 13 — Language complexity + Unicode/character analysis.

Per-language sentence length, lexical/token diversity, readability (English),
special-symbol usage and Unicode block coverage. These flags matter for
tokenizer choice and for spotting encoding/normalization issues (e.g. Ge'ez,
stray control characters) that would silently degrade retrieval & generation.
"""
from __future__ import annotations

import unicodedata
from collections import Counter

import pandas as pd

from . import config as C
from . import text_stats as ts


def complexity_by_subset(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for subset, sub in df.groupby(C.SUBSET_COL):
        texts = sub[col].dropna()
        words_per_sent = []
        for t in texts:
            w, s = ts.n_words(t), ts.n_sentences(t)
            if s:
                words_per_sent.append(w / s)
        c = ts.vocab(texts)
        total = sum(c.values()) or 1
        rows.append({
            "subset": subset,
            "mean_words_per_sentence": round(sum(words_per_sent) / len(words_per_sent), 2)
            if words_per_sent else 0,
            "ttr": round(len(c) / total, 4),
            "vocab_size": len(c),
        })
    return pd.DataFrame(rows).sort_values("mean_words_per_sentence", ascending=False)


def unicode_blocks(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-subset Unicode block coverage + special-symbol / control-char counts."""
    rows = []
    for subset, sub in df.groupby(C.SUBSET_COL):
        blocks: Counter = Counter()
        specials = 0
        controls = 0
        total = 0
        for t in sub[col].dropna():
            for ch in t:
                total += 1
                cat = unicodedata.category(ch)
                if cat.startswith("C"):
                    controls += 1
                if cat.startswith("S") or (cat.startswith("P") and not ch.isascii()):
                    specials += 1
                try:
                    name = unicodedata.name(ch)
                    block = name.split(" ")[0]
                except ValueError:
                    block = "UNNAMED"
                blocks[block] += 1
        top = ", ".join(f"{b}:{round(n/(total or 1)*100,1)}%"
                        for b, n in blocks.most_common(4))
        rows.append({
            "subset": subset,
            "chars": total,
            "top_unicode_blocks": top,
            "control_chars": controls,
            "nonascii_symbols": specials,
        })
    return pd.DataFrame(rows)
