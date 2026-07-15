"""Stage 2d — ensemble retrievers + answer-space retrieval on full Val.

- Ensemble: average per-model cosine (e5-base/e5-large/bge-m3) over the same
  subset-restricted candidate pool, pick top-1. Ensembling often moves top-1
  closer to the oracle than any single model.
- Answer-space: retrieve the nearest train ANSWER directly (query vs passage),
  instead of going through the nearest question.

All use cached embeddings; ranked by official ROUGE-weighted.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from health_eda import config as C, io_utils as io, embeddings as E
from health_eda import eval_official as EO

pd.set_option("display.width", 160)
C.set_seed()

MODELS = {"e5-base": "intfloat/multilingual-e5-base",
          "e5-large": "intfloat/multilingual-e5-large",
          "bge-m3": "BAAI/bge-m3"}


def subset_mask(train_sub, val_sub):
    return (train_sub[None, :] == val_sub[:, None])


def topk_ensemble(val_embs, train_embs, weights, mask, k=1):
    """Average cosine across models over masked candidates -> top-1 indices."""
    n = val_embs[0].shape[0]
    top = np.empty(n, dtype=np.int64)
    bs = 128
    for s in range(0, n, bs):
        acc = None
        for emb_v, emb_t, w in zip(val_embs, train_embs, weights):
            with np.errstate(all="ignore"):
                sc = emb_v[s:s+bs] @ emb_t.T
            acc = w*sc if acc is None else acc + w*sc
        acc = np.where(mask[s:s+bs], acc, -1e9)
        top[s:s+bs] = acc.argmax(axis=1)
    return top


def main():
    train = io.load_split("train").reset_index(drop=True)
    val = io.load_split("val").reset_index(drop=True)
    tr_ans = train[C.OUTPUT_COL].tolist(); gold = val[C.OUTPUT_COL].tolist()
    subs = val[C.SUBSET_COL].tolist()
    mask = subset_mask(train[C.SUBSET_COL].to_numpy(), val[C.SUBSET_COL].to_numpy())

    # load cached question embeddings per model
    trE, vaE = {}, {}
    for name, mid in MODELS.items():
        trE[name] = E.encode(train[C.INPUT_COL].tolist(), "query", cache_name="train_q", model_name=mid)
        vaE[name] = E.encode(val[C.INPUT_COL].tolist(), "query", cache_name="val_q", model_name=mid)

    rows = {}
    def record(tag, top_idx):
        preds = [tr_ans[j] for j in top_idx]
        rows[tag] = EO.score_frame(preds, gold, subs).loc["overall"]
        print(f"{tag:32s} rw={rows[tag]['rouge_weighted']:.4f} "
              f"R1={rows[tag]['rouge1_f1']:.4f} RL={rows[tag]['rougeL_f1']:.4f}")

    # ensembles
    record("ens:large+bge", topk_ensemble([vaE["e5-large"], vaE["bge-m3"]],
           [trE["e5-large"], trE["bge-m3"]], [0.5, 0.5], mask))
    record("ens:base+large+bge", topk_ensemble(
           [vaE["e5-base"], vaE["e5-large"], vaE["bge-m3"]],
           [trE["e5-base"], trE["e5-large"], trE["bge-m3"]], [1/3]*3, mask))
    record("ens:large+bge(0.6/0.4)", topk_ensemble([vaE["e5-large"], vaE["bge-m3"]],
           [trE["e5-large"], trE["bge-m3"]], [0.6, 0.4], mask))

    # answer-space retrieval with e5-base (cached train_a + val_q)
    tr_a = E.encode(tr_ans, "passage", cache_name="train_a", model_name=MODELS["e5-base"])
    top = topk_ensemble([vaE["e5-base"]], [tr_a], [1.0], mask)
    record("answer-space:e5-base", top)

    comp = pd.DataFrame(rows).T.sort_values("rouge_weighted")
    io.save_table(comp, "stage2d_ensemble_comparison")
    print("\n=== HEADLINE ===")
    print(comp[["rouge1_f1", "rougeL_f1", "rouge_weighted"]].to_string())
    print("\nRef: e5-base/subset=0.3534, e5-large/subset=0.3603, oracle@10=0.4573")


if __name__ == "__main__":
    main()
