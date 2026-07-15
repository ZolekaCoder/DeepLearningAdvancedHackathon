"""Phase 2 — production prediction pipeline (Stage 1: retrieval baseline).

Given a query set (Val or Test), retrieve the nearest Train question with dense
e5 embeddings and copy its answer. Supports a language-safe variant that
restricts candidates to the query's own subset (avoids returning a wrong-language
answer, which is catastrophic for both ROUGE and the LLM judge).

Later stages (rerank, routing, generation) plug into `predict()` via the
`rerank_fn` / `generate_fn` hooks without changing callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import config as C
from . import embeddings as E
from . import metrics


SUB_DIR = C.ROOT / "submissions"
SUB_DIR.mkdir(exist_ok=True)


@dataclass
class RetrievalResult:
    top_idx: np.ndarray      # (n_query, k) indices into the train corpus
    top_sim: np.ndarray      # (n_query, k) cosine similarities
    answers: list[str]       # top-1 answer per query (post any rerank)
    chosen_rank: np.ndarray  # which of the k was chosen (0 unless reranked)


def _dense_topk(query_emb: np.ndarray, train_emb: np.ndarray, k: int,
                candidate_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cosine top-k; optional per-query boolean mask over train candidates."""
    idx = np.empty((len(query_emb), k), dtype=np.int64)
    sim = np.empty((len(query_emb), k), dtype=np.float32)
    bs = 256
    for s in range(0, len(query_emb), bs):
        q = query_emb[s:s + bs]
        # macOS Accelerate BLAS raises spurious FP-exception flags on matmul even
        # for finite inputs/outputs (verified: embeddings have no nan/inf); ignore.
        with np.errstate(all="ignore"):
            scores = q @ train_emb.T                  # (b, N)
        if candidate_mask is not None:
            scores = np.where(candidate_mask[s:s + bs], scores, -1.0)
        kk = min(k, scores.shape[1])
        part = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
        for r in range(len(q)):
            order = part[r][np.argsort(-scores[r, part[r]])]
            idx[s + r, :kk] = order
            sim[s + r, :kk] = scores[r, order]
    return idx, sim


def _lang_of(subset: str) -> str:
    return C.SUBSET_META.get(subset, {}).get("language", subset)


def predict(query_df: pd.DataFrame, train_df: pd.DataFrame,
            query_cache: str, train_cache: str = "train_q",
            k: int = 5, restrict_subset: bool = False,
            pool: str | None = None,
            embed_model: str = C.EMBED_MODEL,
            rerank_fn: Callable | None = None,
            generate_fn: Callable | None = None,
            route_threshold: float | None = None) -> RetrievalResult:
    """Retrieve (and optionally rerank / generate) answers for query_df.

    pool: candidate restriction — 'subset' (same language-country), 'language'
        (same language across countries), or 'global'/None (all train).
        `restrict_subset=True` is shorthand for pool='subset'.
    embed_model: which SentenceTransformer to use for both corpus and queries.
    rerank_fn(query_text, cand_texts, cand_answers) -> chosen_rank:int
    generate_fn(query_row, retrieved_answer, top_sim) -> str
    """
    if restrict_subset and pool is None:
        pool = "subset"
    # encode() adds a per-model tag + legacy fallback, so pass plain cache names.
    train_emb = E.encode(train_df[C.INPUT_COL].tolist(), "query",
                         cache_name=train_cache, model_name=embed_model)
    query_emb = E.encode(query_df[C.INPUT_COL].tolist(), "query",
                         cache_name=query_cache, model_name=embed_model)

    mask = None
    if pool == "subset":
        tr = train_df[C.SUBSET_COL].to_numpy(); q = query_df[C.SUBSET_COL].to_numpy()
        mask = (tr[None, :] == q[:, None])
    elif pool == "language":
        tr = np.array([_lang_of(s) for s in train_df[C.SUBSET_COL]])
        q = np.array([_lang_of(s) for s in query_df[C.SUBSET_COL]])
        mask = (tr[None, :] == q[:, None])

    top_idx, top_sim = _dense_topk(query_emb, train_emb, k, mask)

    tr_answers = train_df[C.OUTPUT_COL].tolist()
    tr_questions = train_df[C.INPUT_COL].tolist()
    q_texts = query_df[C.INPUT_COL].tolist()

    answers, chosen = [], np.zeros(len(query_df), dtype=int)
    for i in range(len(query_df)):
        cand_idx = top_idx[i]
        rank = 0
        if rerank_fn is not None:
            rank = rerank_fn(q_texts[i],
                             [tr_questions[j] for j in cand_idx],
                             [tr_answers[j] for j in cand_idx])
        chosen[i] = rank
        ans = tr_answers[cand_idx[rank]]
        if (generate_fn is not None and route_threshold is not None
                and top_sim[i, 0] < route_threshold):
            ans = generate_fn(query_df.iloc[i], ans, float(top_sim[i, 0]))
        answers.append(ans)

    return RetrievalResult(top_idx, top_sim, answers, chosen)


# --------------------------------------------------------------------------- #
# Evaluation + submission
# --------------------------------------------------------------------------- #
def evaluate(pred_answers: list[str], gold_answers: list[str],
             subsets: list[str]) -> pd.DataFrame:
    """Per-subset + overall ROUGE-1/L F1 and exact match."""
    rows = []
    for p, g, s in zip(pred_answers, gold_answers, subsets):
        sc = metrics.score_pair(p, g)
        rows.append({"subset": s, **sc})
    df = pd.DataFrame(rows)
    cols = ["rouge1_f1", "rougeL_f1", "exact"]
    overall = df[cols].mean().to_frame("overall").T
    per = df.groupby("subset")[cols].mean()
    return pd.concat([overall, per]).round(4)


def write_submission(ids: list[str], answers: list[str], name: str,
                     columns=("TargetRLF1", "TargetR1F1", "TargetLLM")) -> str:
    """Write a SampleSubmission-format CSV. NOTE: same answer in all 3 target
    columns (safe default; confirm competition semantics before final submit)."""
    out = pd.DataFrame({C.ID_COL: ids})
    for col in columns:
        out[col] = answers
    path = SUB_DIR / f"{name}.csv"
    out.to_csv(path, index=False)
    return str(path)
