"""Section 11 (Similarity Analysis) — subset-level similarity structure.

Full pairwise similarity over ~30k items is 900M cells; instead we compute
interpretable summaries: subset centroid similarity matrices (question-question
and answer-answer), the question<->answer coupling, and how "dense" each
subset's embedding cloud is (mean intra-subset similarity => redundancy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _centroids(emb: np.ndarray, subsets: np.ndarray) -> tuple[list[str], np.ndarray]:
    labels = sorted(set(subsets))
    cents = []
    for s in labels:
        v = emb[subsets == s].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-9)
        cents.append(v)
    return labels, np.vstack(cents)


def centroid_similarity(emb: np.ndarray, subsets: np.ndarray) -> pd.DataFrame:
    """Cosine similarity between per-subset embedding centroids."""
    labels, cents = _centroids(emb, subsets)
    mat = cents @ cents.T
    return pd.DataFrame(np.round(mat, 3), index=labels, columns=labels)


def intra_subset_density(emb: np.ndarray, subsets: np.ndarray,
                         sample: int = 1500, seed: int = C.SEED) -> pd.DataFrame:
    """Mean pairwise cosine within each subset (higher => more redundant/dense)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in sorted(set(subsets)):
        idx = np.where(subsets == s)[0]
        if len(idx) > sample:
            idx = rng.choice(idx, sample, replace=False)
        v = emb[idx]
        sims = v @ v.T
        n = len(idx)
        off = (sims.sum() - np.trace(sims)) / (n * (n - 1)) if n > 1 else 0
        rows.append({"subset": s, "n_sampled": n,
                     "mean_intra_cosine": round(float(off), 4)})
    return pd.DataFrame(rows).sort_values("mean_intra_cosine", ascending=False)


def question_answer_coupling(q_emb: np.ndarray, a_emb: np.ndarray,
                             sample: int = 4000, seed: int = C.SEED) -> dict:
    """How well does a question's embedding predict its own answer's embedding?

    Compares the true Q-A cosine against a shuffled baseline; a large gap means
    questions strongly determine answers (good for retrieval).
    """
    rng = np.random.default_rng(seed)
    n = len(q_emb)
    idx = rng.choice(n, min(sample, n), replace=False)
    q, a = q_emb[idx], a_emb[idx]
    true_sim = (q * a).sum(axis=1)
    perm = rng.permutation(len(idx))
    shuf_sim = (q * a[perm]).sum(axis=1)
    return {
        "mean_true_qa_cosine": round(float(true_sim.mean()), 4),
        "mean_shuffled_qa_cosine": round(float(shuf_sim.mean()), 4),
        "coupling_gap": round(float(true_sim.mean() - shuf_sim.mean()), 4),
    }
