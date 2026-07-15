"""Language-agnostic text statistics.

The dataset mixes Latin-script languages (English, Akan, Luganda, Swahili) and
Ge'ez script (Amharic). We therefore avoid English-only tokenizers and use a
Unicode-aware word regex so counts are comparable across scripts.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

# \w in the `regex`/`re` unicode sense captures letters across scripts incl.
# Ge'ez; we treat runs of word characters as tokens.
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_SENT_RE = re.compile(r"[.!?።፧፨]+|\n+")  # incl. Amharic full stop (።) & question (፧)


def tokenize(text: str) -> list[str]:
    """Unicode-aware whitespace/word tokenizer, lowercased."""
    if not isinstance(text, str):
        return []
    return _WORD_RE.findall(text.lower())


def n_words(text: str) -> int:
    return len(tokenize(text))


def n_chars(text: str) -> int:
    return len(text) if isinstance(text, str) else 0


def n_sentences(text: str) -> int:
    """Rough sentence count using multi-script terminators."""
    if not isinstance(text, str) or not text.strip():
        return 0
    parts = [p for p in _SENT_RE.split(text) if p.strip()]
    return max(1, len(parts))


def describe_lengths(series: pd.Series, unit: str = "words") -> dict:
    """Summary stats (mean/median/min/max/std/percentiles) for a text column."""
    fn = n_words if unit == "words" else n_chars
    lens = series.dropna().map(fn)
    if len(lens) == 0:
        return {}
    return {
        "count": int(lens.count()),
        "mean": float(lens.mean()),
        "median": float(lens.median()),
        "std": float(lens.std()),
        "min": int(lens.min()),
        "p25": float(lens.quantile(0.25)),
        "p75": float(lens.quantile(0.75)),
        "p95": float(lens.quantile(0.95)),
        "max": int(lens.max()),
    }


def vocab(series: pd.Series) -> Counter:
    """Token-frequency Counter over a text column."""
    c: Counter = Counter()
    for txt in series.dropna():
        c.update(tokenize(txt))
    return c


def vocab_size(series: pd.Series) -> int:
    return len(vocab(series))


def type_token_ratio(series: pd.Series) -> float:
    """Lexical diversity = unique tokens / total tokens (0..1)."""
    c = vocab(series)
    total = sum(c.values())
    return len(c) / total if total else 0.0


def ngram_counts(series: pd.Series, n: int = 2, top: int | None = None) -> Counter:
    """Word n-gram frequencies across a text column."""
    c: Counter = Counter()
    for txt in series.dropna():
        toks = tokenize(txt)
        c.update(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))
    if top:
        return Counter(dict(c.most_common(top)))
    return c


def char_script_profile(text: str) -> Counter:
    """Count characters by Unicode script block name (Latin, Ethiopic, ...)."""
    prof: Counter = Counter()
    if not isinstance(text, str):
        return prof
    for ch in text:
        if ch.isspace():
            continue
        try:
            name = unicodedata.name(ch).split(" ")[0]
        except ValueError:
            name = "UNKNOWN"
        # Bucket into coarse categories.
        if "ETHIOPIC" in unicodedata.name(ch, ""):
            bucket = "Ethiopic"
        elif ch.isascii() and ch.isalpha():
            bucket = "ASCII_Latin"
        elif ch.isalpha():
            bucket = "Latin_ext"
        elif ch.isdigit():
            bucket = "Digit"
        elif not ch.isalnum():
            bucket = "Punct/Symbol"
        else:
            bucket = name
        prof[bucket] += 1
    return prof


def script_profile_series(series: pd.Series) -> Counter:
    prof: Counter = Counter()
    for txt in series.dropna():
        prof.update(char_script_profile(txt))
    return prof
