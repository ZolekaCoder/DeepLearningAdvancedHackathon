"""Phase 2 Stage 2b — multilingual cross-encoder reranking.

A bi-encoder (e5) ranks candidates cheaply; a cross-encoder re-scores the
top-k query/candidate pairs jointly and usually picks a better top-1. We use
BAAI/bge-reranker-v2-m3 (open, multilingual, covers all our languages incl.
Ge'ez). We can rerank the query against each candidate's QUESTION or its ANSWER;
both are evaluated on Val and the better one is kept.

Scores are cached so re-runs are fast.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from . import io_utils as io

# Fast multilingual cross-encoder (118M, MS MARCO multilingual). ~10x faster on
# MPS than bge-reranker-v2-m3; good enough to test the reranking hypothesis.
RERANKER_FAST = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_STRONG = "BAAI/bge-reranker-v2-m3"
RERANKER = RERANKER_FAST
_MODELS: dict = {}


def _device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_reranker(model_name: str = RERANKER):
    if model_name not in _MODELS:
        from sentence_transformers import CrossEncoder
        dev = _device()
        print(f"[rerank] loading {model_name} on {dev} ...", flush=True)
        _MODELS[model_name] = CrossEncoder(model_name, device=dev, max_length=512)
    return _MODELS[model_name]


def rerank_topk(query_texts: list[str], top_idx: np.ndarray,
                cand_texts: list[str], cache_name: str | None = None,
                batch_size: int = 64, model_name: str = RERANKER) -> np.ndarray:
    """Return reranked index array (same shape as top_idx), best first.

    cand_texts: the pool text (train questions OR train answers) indexed like the
    corpus, so cand_texts[top_idx[i, r]] is candidate r for query i.
    """
    if cache_name:
        cached = io.load_artifact(f"rerank_{cache_name}")
        if cached is not None and cached.shape == top_idx.shape:
            print(f"[rerank] cache hit: rerank_{cache_name}")
            return cached

    model = get_reranker(model_name)
    n, k = top_idx.shape
    # Build all (query, candidate) pairs.
    pairs, owner = [], []
    for i in range(n):
        for r in range(k):
            pairs.append([query_texts[i], cand_texts[top_idx[i, r]]])
            owner.append((i, r))
    with io.timer(f"cross-encoder scoring {len(pairs):,} pairs"):
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)

    score_mat = np.full(top_idx.shape, -1e9, dtype=np.float32)
    for (i, r), s in zip(owner, scores):
        score_mat[i, r] = s
    order = np.argsort(-score_mat, axis=1)                 # best first per row
    reranked = np.take_along_axis(top_idx, order, axis=1)
    if cache_name:
        io.save_artifact(reranked, f"rerank_{cache_name}")
    return reranked
