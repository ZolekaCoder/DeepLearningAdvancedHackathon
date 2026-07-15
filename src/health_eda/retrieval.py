"""Section 10 — Retrieval feasibility.

We retrieve, for each query question, the nearest TRAIN question and copy its
answer. Three retrievers:
  * BM25   (lexical, rank_bm25)
  * dense  (multilingual-e5-base cosine over question embeddings)
  * hybrid (Reciprocal Rank Fusion of the two)  <- headline

No answer generation happens here; we only measure how good a retrieve-and-copy
system could be (ROUGE-1/L F1 vs the gold answer, exact-match, topic-match).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config as C
from . import metrics
from . import text_stats as ts


# --------------------------------------------------------------------------- #
# Individual retrievers: return top-k train indices per query.
# --------------------------------------------------------------------------- #
class BM25Retriever:
    def __init__(self, train_questions: list[str]):
        from rank_bm25 import BM25Okapi

        self.corpus_tokens = [ts.tokenize(q) for q in train_questions]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def topk(self, queries: list[str], k: int = 5) -> np.ndarray:
        out = np.empty((len(queries), k), dtype=np.int64)
        for i, q in enumerate(tqdm(queries, desc="BM25 retrieve")):
            scores = self.bm25.get_scores(ts.tokenize(q))
            out[i] = np.argsort(scores)[::-1][:k]
        return out


class DenseRetriever:
    """Cosine top-k via matrix product (embeddings are L2-normalized)."""

    def __init__(self, train_emb: np.ndarray):
        self.train_emb = train_emb  # (N, d), normalized

    def topk(self, query_emb: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        idx = np.empty((len(query_emb), k), dtype=np.int64)
        sim = np.empty((len(query_emb), k), dtype=np.float32)
        bs = 512
        for s in tqdm(range(0, len(query_emb), bs), desc="dense retrieve"):
            q = query_emb[s:s + bs]
            scores = q @ self.train_emb.T           # cosine
            part = np.argpartition(-scores, k, axis=1)[:, :k]
            for r in range(len(q)):
                order = part[r][np.argsort(-scores[r, part[r]])]
                idx[s + r] = order
                sim[s + r] = scores[r, order]
        return idx, sim


def rrf_fuse(bm25_topk: np.ndarray, dense_topk: np.ndarray, k: int = 5,
             c: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion of two ranked lists -> fused top-1..k indices."""
    n = bm25_topk.shape[0]
    fused = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        scores: dict[int, float] = {}
        for rank, idx in enumerate(bm25_topk[i]):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (c + rank)
        for rank, idx in enumerate(dense_topk[i]):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (c + rank)
        ranked = sorted(scores, key=scores.get, reverse=True)[:k]
        # pad if fewer than k unique
        while len(ranked) < k:
            ranked.append(ranked[-1] if ranked else 0)
        fused[i] = ranked
    return fused


# --------------------------------------------------------------------------- #
# Evaluation: copy retrieved train answer, score vs gold query answer.
# --------------------------------------------------------------------------- #
def evaluate(top1_train_idx: np.ndarray,
             train_answers: list[str],
             query_answers: list[str],
             train_topics: np.ndarray | None = None,
             query_topics: np.ndarray | None = None,
             train_subset: list[str] | None = None,
             query_subset: list[str] | None = None,
             dense_sim: np.ndarray | None = None) -> pd.DataFrame:
    """Per-query scores for a retrieve-and-copy system (top-1)."""
    recs = []
    for i, ti in enumerate(top1_train_idx):
        pred = train_answers[ti]
        gold = query_answers[i]
        sc = metrics.score_pair(pred, gold)
        rec = {"rouge1_f1": sc["rouge1_f1"], "rougeL_f1": sc["rougeL_f1"],
               "exact": sc["exact"]}
        if train_topics is not None and query_topics is not None:
            rec["topic_match"] = float(train_topics[ti] == query_topics[i])
        if train_subset is not None and query_subset is not None:
            rec["subset"] = query_subset[i]
            rec["retrieved_subset"] = train_subset[ti]
            rec["subset_match"] = float(train_subset[ti] == query_subset[i])
        if dense_sim is not None:
            rec["retrieval_sim"] = float(dense_sim[i])
        recs.append(rec)
    return pd.DataFrame(recs)


def summarize(eval_df: pd.DataFrame, by_subset: bool = True) -> pd.DataFrame:
    metric_cols = [c for c in ["rouge1_f1", "rougeL_f1", "exact", "topic_match",
                               "subset_match", "retrieval_sim"] if c in eval_df]
    overall = eval_df[metric_cols].mean().to_frame("overall").T
    if by_subset and "subset" in eval_df:
        per = eval_df.groupby("subset")[metric_cols].mean()
        return pd.concat([overall, per])
    return overall
