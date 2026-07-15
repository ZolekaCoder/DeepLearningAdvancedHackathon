"""ROUGE-1 / ROUGE-L F1, implemented Unicode-aware to match the competition.

The leaderboard targets (TargetR1F1, TargetRLF1) are ROUGE-1 and ROUGE-L F1.
We reuse our multilingual tokenizer (text_stats.tokenize) so Amharic/Ge'ez and
African-language answers are scored consistently rather than with an
English-only tokenizer.
"""
from __future__ import annotations

from . import text_stats as ts


def _lcs(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (for ROUGE-L)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _f1(match: int, n_pred: int, n_ref: int) -> float:
    if n_pred == 0 or n_ref == 0 or match == 0:
        return 0.0
    p = match / n_pred
    r = match / n_ref
    return 2 * p * r / (p + r)


def rouge1_f1(pred: str, ref: str) -> float:
    """Unigram-overlap F1 (multiset overlap)."""
    from collections import Counter

    p_tok, r_tok = ts.tokenize(pred), ts.tokenize(ref)
    overlap = sum((Counter(p_tok) & Counter(r_tok)).values())
    return _f1(overlap, len(p_tok), len(r_tok))


def rougeL_f1(pred: str, ref: str) -> float:
    """LCS-based F1 (ROUGE-L)."""
    p_tok, r_tok = ts.tokenize(pred), ts.tokenize(ref)
    lcs = _lcs(p_tok, r_tok)
    return _f1(lcs, len(p_tok), len(r_tok))


def score_pair(pred: str, ref: str) -> dict:
    return {
        "rouge1_f1": rouge1_f1(pred, ref),
        "rougeL_f1": rougeL_f1(pred, ref),
        "exact": float(str(pred).strip() == str(ref).strip()),
    }
