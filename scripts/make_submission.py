"""Generate a Test submission from the best retrieval config.

- Validates the config on Val with TRAIN-ONLY corpus (honest, no leakage) using
  the official rouge-score, so we only ship configs that beat the current best.
- Builds the Test submission using the COMBINED train+val corpus (Val has gold
  answers; more canonical candidates => better coverage; no distribution shift).
- Subset-pooled top-1 dense retrieval, answer copied verbatim (best for ROUGE).

Usage: python scripts/make_submission.py <model_key> <out_name>
  model_key in {e5-base, e5-large, bge-m3}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from health_eda import config as C, io_utils as io, embeddings as E, predict as P
from health_eda import eval_official as EO

C.set_seed()
MODELS = {"e5-base": "intfloat/multilingual-e5-base",
          "e5-large": "intfloat/multilingual-e5-large",
          "bge-m3": "BAAI/bge-m3"}


def subset_topk(val_emb, corpus_emb, corpus_sub, query_sub, k=1):
    n = len(val_emb); top = np.empty(n, dtype=np.int64)
    for s in range(0, n, 128):
        with np.errstate(all="ignore"):
            sc = val_emb[s:s+128] @ corpus_emb.T
        m = (corpus_sub[None, :] == query_sub[s:s+128][:, None])
        sc = np.where(m, sc, -1e9)
        top[s:s+128] = sc.argmax(axis=1)
    return top


def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "e5-large"
    out_name = sys.argv[2] if len(sys.argv) > 2 else f"submission_{model_key}_trainval"
    mid = MODELS[model_key]
    print(f"Model: {model_key} ({mid}) -> {out_name}")

    train = io.load_split("train").reset_index(drop=True)
    val = io.load_split("val").reset_index(drop=True)
    test = io.load_split("test").reset_index(drop=True)

    tr_emb = E.encode(train[C.INPUT_COL].tolist(), "query", cache_name="train_q", model_name=mid)
    va_emb = E.encode(val[C.INPUT_COL].tolist(), "query", cache_name="val_q", model_name=mid)
    te_emb = E.encode(test[C.INPUT_COL].tolist(), "query", cache_name="test_q", model_name=mid)

    # --- Honest Val check: TRAIN-only corpus --- #
    top = subset_topk(va_emb, tr_emb, train[C.SUBSET_COL].to_numpy(),
                      val[C.SUBSET_COL].to_numpy())
    tr_ans = train[C.OUTPUT_COL].tolist()
    val_preds = [tr_ans[j] for j in top]
    sf = EO.score_frame(val_preds, val[C.OUTPUT_COL].tolist(), val[C.SUBSET_COL].tolist())
    rw = sf.loc["overall", "rouge_weighted"]
    print("\n=== Val (train-only corpus) official ROUGE ===")
    print(sf.loc[["overall"]].to_string())
    print(f"ROUGE-weighted={rw:.4f}  (baseline e5-base=0.3534)  "
          f"est LB @LLM=0.71: {rw + 0.26*0.71:.4f}")

    # --- Test submission: COMBINED train+val corpus --- #
    corpus_emb = np.vstack([tr_emb, va_emb])
    corpus_ans = train[C.OUTPUT_COL].tolist() + val[C.OUTPUT_COL].tolist()
    corpus_sub = np.concatenate([train[C.SUBSET_COL].to_numpy(),
                                 val[C.SUBSET_COL].to_numpy()])
    top_t = subset_topk(te_emb, corpus_emb, corpus_sub, test[C.SUBSET_COL].to_numpy())
    test_ans = [corpus_ans[j] for j in top_t]
    path = P.write_submission(test[C.ID_COL].tolist(), test_ans, name=out_name)
    print(f"\nWrote submission -> {path}")
    print(f"Corpus size (train+val): {len(corpus_ans):,}. "
          f"Test answers drawn from val: {int((top_t >= len(train)).sum())}/{len(test)}")


if __name__ == "__main__":
    main()
