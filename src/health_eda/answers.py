"""Section 4 — Answer analysis (a very high-priority section).

Central question: are answers *individually written* or *selected from a small
set of canonical responses*? If canonical, a retrieval/copy system can score
extremely well on ROUGE and we should lean retrieval-first. We measure exact &
near-duplicate answers, repeated closings/disclaimers, and reuse distribution.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

import pandas as pd

from . import config as C
from . import text_stats as ts


def exact_reuse(series: pd.Series) -> dict:
    """Distribution of exact answer reuse."""
    vc = series.dropna().value_counts()
    n = int(vc.sum())
    return {
        "n_answers": n,
        "n_unique_answers": int(len(vc)),
        "unique_ratio": round(len(vc) / n, 4) if n else 0,
        "pct_answers_in_reused_group": round(
            int(vc[vc > 1].sum()) / n * 100, 2) if n else 0,
        "max_reuse_count": int(vc.iloc[0]) if len(vc) else 0,
        "n_answers_reused_2plus": int((vc > 1).sum()),
    }


def reuse_distribution(series: pd.Series, top: int = 15) -> pd.DataFrame:
    """The most-reused exact answers (candidate canonical responses)."""
    vc = series.dropna().value_counts().head(top)
    rows = [(i + 1, int(cnt), (txt[:160] + " …") if len(txt) > 160 else txt)
            for i, (txt, cnt) in enumerate(vc.items())]
    return pd.DataFrame(rows, columns=["rank", "times_used", "answer_preview"])


def _shingle_hash(text: str, k: int = 8) -> set[int]:
    """Hashed word-shingles for cheap near-duplicate detection (Jaccard)."""
    toks = ts.tokenize(text)
    if len(toks) < k:
        return {hash(tuple(toks))} if toks else set()
    return {hash(tuple(toks[i:i + k])) for i in range(len(toks) - k + 1)}


def near_duplicate_estimate(series: pd.Series, sample: int = 4000,
                            k: int = 8, threshold: float = 0.8,
                            seed: int = C.SEED) -> dict:
    """MinHash-free approximate near-duplicate rate on a random sample.

    We compare each sampled answer to others sharing at least one shingle-band
    signature; report the share with Jaccard >= threshold to some other answer.
    This is an *estimate* (sampled) to keep it tractable on CPU.
    """
    import random

    ans = series.dropna().tolist()
    rng = random.Random(seed)
    if len(ans) > sample:
        ans = rng.sample(ans, sample)
    shingles = [(_shingle_hash(a, k), a) for a in ans]

    # Inverted index: shingle -> list of doc indices, to avoid O(n^2).
    from collections import defaultdict
    inv = defaultdict(list)
    for i, (sh, _) in enumerate(shingles):
        for s in list(sh)[:64]:  # cap bands per doc
            inv[s].append(i)

    near = 0
    for i, (sh_i, _) in enumerate(shingles):
        if not sh_i:
            continue
        cands = set()
        for s in list(sh_i)[:64]:
            cands.update(inv[s])
        cands.discard(i)
        best = 0.0
        for j in cands:
            sh_j = shingles[j][0]
            if not sh_j:
                continue
            jac = len(sh_i & sh_j) / len(sh_i | sh_j)
            if jac > best:
                best = jac
                if best >= 0.999:
                    break
        if best >= threshold:
            near += 1
    return {
        "sampled": len(ans),
        "shingle_k": k,
        "jaccard_threshold": threshold,
        "pct_near_duplicate": round(near / len(ans) * 100, 2) if ans else 0,
    }


def repeated_sentences(series: pd.Series, top: int = 20, min_len: int = 4) -> pd.DataFrame:
    """Most frequently repeated sentences (closings, disclaimers, boilerplate)."""
    c: Counter = Counter()
    for txt in series.dropna():
        for sent in ts._SENT_RE.split(txt):
            s = sent.strip()
            if len(ts.tokenize(s)) >= min_len:
                c[s] += 1
    rows = [(cnt, (s[:180] + " …") if len(s) > 180 else s)
            for s, cnt in c.most_common(top)]
    return pd.DataFrame(rows, columns=["count", "sentence"])


def closing_lines(series: pd.Series, top: int = 15) -> pd.DataFrame:
    """Most common final sentences (repeated closing statements / advice)."""
    c: Counter = Counter()
    for txt in series.dropna():
        parts = [p.strip() for p in ts._SENT_RE.split(txt) if p.strip()]
        if parts:
            c[parts[-1]] += 1
    rows = [(cnt, (s[:180] + " …") if len(s) > 180 else s)
            for s, cnt in c.most_common(top)]
    return pd.DataFrame(rows, columns=["count", "closing_sentence"])
