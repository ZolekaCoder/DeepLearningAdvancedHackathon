"""Section 7 — Vocabulary analysis, per language.

Per-subset vocabulary size, hapax (rare-word) rate, train/val OOV, and the
frequency of a curated health/medical lexicon. The medical lexicon is a
transparent seed list (English + a few cognates) used only to *measure*
terminology density, not to label topics (topics are discovered in topics.py).
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from . import config as C
from . import text_stats as ts

# Transparent, auditable health/SRH lexicon (lowercase). Extend freely.
MEDICAL_LEXICON = {
    "hiv", "aids", "sti", "stis", "std", "herpes", "syphilis", "hpv", "chlamydia",
    "gonorrhea", "gonorrhoea", "trichomoniasis", "hepatitis", "condom", "condoms",
    "contraception", "contraceptive", "prep", "pep", "vaccine", "vaccination",
    "pregnancy", "pregnant", "breastfeeding", "menstruation", "menstrual", "period",
    "periods", "fertility", "infertility", "abortion", "antibiotics", "antiretroviral",
    "art", "infection", "infections", "symptoms", "diagnosis", "treatment", "screening",
    "testing", "transmission", "prevention", "cervical", "cancer", "uterus", "vagina",
    "penis", "genital", "genitals", "ovulation", "sperm", "semen", "hormone", "hormonal",
    "iud", "implant", "injection", "pill", "pills", "emergency", "family", "planning",
    "prenatal", "antenatal", "postnatal", "midwife", "clinic", "immunization",
}


def per_language_vocab(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for subset, sub in df.groupby(C.SUBSET_COL):
        c = ts.vocab(sub[col])
        total = sum(c.values()) or 1
        hapax = sum(1 for _, v in c.items() if v == 1)
        med = sum(c[w] for w in MEDICAL_LEXICON if w in c)
        rows.append({
            "subset": subset,
            "tokens_total": total,
            "vocab_size": len(c),
            "ttr": round(len(c) / total, 4),
            "hapax": hapax,
            "hapax_pct_of_vocab": round(hapax / len(c) * 100, 1) if c else 0,
            "medical_token_pct": round(med / total * 100, 3),
        })
    return pd.DataFrame(rows).sort_values("vocab_size", ascending=False)


def top_tokens(df: pd.DataFrame, col: str, subset: str, top: int = 25,
               drop_stop: bool = True) -> pd.DataFrame:
    """Most frequent tokens for a subset (crude English stop-word filter)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    sub = df[df[C.SUBSET_COL] == subset]
    c = ts.vocab(sub[col])
    if drop_stop:
        for w in list(c):
            if w in ENGLISH_STOP_WORDS or len(w) <= 2:
                del c[w]
    rows = [(w, n) for w, n in c.most_common(top)]
    return pd.DataFrame(rows, columns=["token", "count"])


def oov_between_splits(train: pd.DataFrame, other: pd.DataFrame, col: str) -> pd.DataFrame:
    """For each subset, fraction of `other` tokens unseen in Train (OOV)."""
    rows = []
    for subset in sorted(set(train[C.SUBSET_COL]) & set(other[C.SUBSET_COL])):
        tr_vocab = set(ts.vocab(train[train[C.SUBSET_COL] == subset][col]))
        o_counter = ts.vocab(other[other[C.SUBSET_COL] == subset][col])
        o_total = sum(o_counter.values()) or 1
        oov_types = [w for w in o_counter if w not in tr_vocab]
        oov_tokens = sum(o_counter[w] for w in oov_types)
        rows.append({
            "subset": subset,
            "other_vocab": len(o_counter),
            "oov_types": len(oov_types),
            "oov_type_pct": round(len(oov_types) / len(o_counter) * 100, 2) if o_counter else 0,
            "oov_token_pct": round(oov_tokens / o_total * 100, 2),
        })
    return pd.DataFrame(rows)
