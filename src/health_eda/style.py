"""Section 12 — Answer style analysis.

Should a generator imitate the answers' style? We measure sentence counts,
readability (English only), bullet/list formatting, common disclaimers and
closings. Consistent style => a fine-tuned or style-primed generator should
copy it; ROUGE rewards matching these boilerplate spans.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from . import config as C
from . import text_stats as ts

_BULLET_RE = re.compile(r"(^|\n)\s*([-*•]|\d+\.)\s+")
_DISCLAIMER_PATTERNS = {
    "consult_professional": r"consult (a|your)?\s*(doctor|healthcare|health care|"
                            r"provider|professional|clinician|nurse)",
    "seek_medical": r"seek (medical|professional|immediate) (help|care|attention|advice)",
    "not_medical_advice": r"not (a substitute|intended).{0,30}medical advice",
    "see_a_doctor": r"see (a|your) (doctor|healthcare provider|clinician)",
    "talk_to": r"talk to (a|your) (doctor|healthcare|provider|nurse)",
}


def _flesch_reading_ease(text: str) -> float | None:
    """Flesch Reading Ease (English heuristic; syllables ~ vowel groups)."""
    words = ts.tokenize(text)
    sents = ts.n_sentences(text)
    if not words or not sents:
        return None
    syll = 0
    for w in words:
        groups = re.findall(r"[aeiouy]+", w)
        syll += max(1, len(groups))
    W, S = len(words), sents
    return round(206.835 - 1.015 * (W / S) - 84.6 * (syll / W), 1)


def style_by_subset(df: pd.DataFrame, col: str = C.OUTPUT_COL) -> pd.DataFrame:
    from . import config as C

    rows = []
    for subset, sub in df.groupby(C.SUBSET_COL):
        texts = sub[col].dropna()
        n = len(texts) or 1
        sent_counts = texts.map(ts.n_sentences)
        bullets = texts.map(lambda t: bool(_BULLET_RE.search(t))).mean()
        is_english = C.SUBSET_META.get(subset, {}).get("language") == "English"
        flesch = None
        if is_english:
            fl = texts.map(_flesch_reading_ease).dropna()
            flesch = round(float(fl.mean()), 1) if len(fl) else None
        rows.append({
            "subset": subset,
            "mean_sentences": round(float(sent_counts.mean()), 2),
            "median_sentences": int(sent_counts.median()),
            "pct_with_bullets": round(bullets * 100, 2),
            "flesch_reading_ease": flesch,
        })
    return pd.DataFrame(rows).sort_values("mean_sentences", ascending=False)


def disclaimer_frequency(df: pd.DataFrame, col: str = C.OUTPUT_COL) -> pd.DataFrame:
    compiled = {k: re.compile(v, re.I) for k, v in _DISCLAIMER_PATTERNS.items()}
    texts = df[col].dropna()
    n = len(texts) or 1
    rows = []
    for name, rx in compiled.items():
        hits = texts.map(lambda t: bool(rx.search(t))).sum()
        rows.append({"disclaimer": name, "count": int(hits),
                     "pct_of_answers": round(hits / n * 100, 2)})
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def common_openings(df: pd.DataFrame, col: str = C.OUTPUT_COL, top: int = 15) -> pd.DataFrame:
    """Most common first sentences of answers (style templates)."""
    c: Counter = Counter()
    for txt in df[col].dropna():
        parts = [p.strip() for p in ts._SENT_RE.split(txt) if p.strip()]
        if parts:
            c[parts[0]] += 1
    rows = [(cnt, (s[:150] + " …") if len(s) > 150 else s) for s, cnt in c.most_common(top)]
    return pd.DataFrame(rows, columns=["count", "opening_sentence"])
