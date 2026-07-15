"""Phase 2 Stage 2a — canonical-answer selection strategies.

Instead of blindly copying the top-1 retrieved answer, exploit the canonical
answer bank: among the top-k retrieved train answers, pick the most
*representative* one. Strategies:
  * top1    : nearest question's answer (baseline)
  * majority: most frequent exact answer among top-k (canonical voting)
  * medoid  : answer whose embedding is most central among top-k answers
  * hybrid  : if top-1 similarity is very high -> top1 (near-duplicate question);
              else -> medoid (robust to a single noisy neighbour)

All operate on already-retrieved top-k indices; no model download needed.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from . import config as C


def select_answers(top_idx: np.ndarray, top_sim: np.ndarray,
                   train_answers: list[str], train_ans_emb: np.ndarray,
                   strategy: str = "top1", hybrid_thresh: float = 0.95) -> list[str]:
    out = []
    for i in range(len(top_idx)):
        cand = top_idx[i]
        if strategy == "top1":
            out.append(train_answers[cand[0]])
        elif strategy == "majority":
            texts = [train_answers[j] for j in cand]
            out.append(Counter(texts).most_common(1)[0][0])
        elif strategy == "medoid":
            out.append(_medoid(cand, train_answers, train_ans_emb))
        elif strategy == "hybrid":
            if top_sim[i, 0] >= hybrid_thresh:
                out.append(train_answers[cand[0]])
            else:
                out.append(_medoid(cand, train_answers, train_ans_emb))
        else:
            raise ValueError(strategy)
    return out


def _medoid(cand: np.ndarray, train_answers: list[str],
            train_ans_emb: np.ndarray) -> str:
    """Answer maximising mean cosine to the other candidate answers."""
    embs = train_ans_emb[cand]                      # (k, d), normalized
    sims = embs @ embs.T                            # (k, k)
    mean_sim = (sims.sum(axis=1) - 1.0) / (len(cand) - 1)
    return train_answers[cand[int(np.argmax(mean_sim))]]
