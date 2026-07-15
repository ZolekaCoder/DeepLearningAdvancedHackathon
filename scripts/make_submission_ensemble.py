"""Generate the ensemble Test submission (bge-m3 + e5-large, equal weights).

Best validated config: Val ROUGE-weighted 0.3705 (> bge 0.3665 > e5-large 0.3607
> e5-base 0.3534). Subset-pooled top-1 over the COMBINED train+val corpus; answer
copied verbatim. All embeddings are cached.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from health_eda import config as C, io_utils as io, embeddings as E, predict as P
from health_eda import eval_official as EO

C.set_seed()
MODELS = {"e5-large": "intfloat/multilingual-e5-large", "bge-m3": "BAAI/bge-m3"}
WEIGHTS = {"e5-large": 0.5, "bge-m3": 0.5}


def ens_topk(q_embs, c_embs, weights, c_sub, q_sub):
    n = len(next(iter(q_embs.values()))); top = np.empty(n, dtype=np.int64)
    for s in range(0, n, 128):
        acc = None
        for k in q_embs:
            with np.errstate(all="ignore"):
                sc = q_embs[k][s:s+128] @ c_embs[k].T
            acc = weights[k]*sc if acc is None else acc + weights[k]*sc
        m = (c_sub[None, :] == q_sub[s:s+128][:, None])
        acc = np.where(m, acc, -1e9)
        top[s:s+128] = acc.argmax(axis=1)
    return top


def main():
    train = io.load_split("train").reset_index(drop=True)
    val = io.load_split("val").reset_index(drop=True)
    test = io.load_split("test").reset_index(drop=True)

    trE, vaE, teE = {}, {}, {}
    for k, mid in MODELS.items():
        trE[k] = E.encode(train[C.INPUT_COL].tolist(), "query", cache_name="train_q", model_name=mid)
        vaE[k] = E.encode(val[C.INPUT_COL].tolist(), "query", cache_name="val_q", model_name=mid)
        teE[k] = E.encode(test[C.INPUT_COL].tolist(), "query", cache_name="test_q", model_name=mid)

    # Honest Val check (train-only corpus)
    top = ens_topk(vaE, trE, WEIGHTS, train[C.SUBSET_COL].to_numpy(), val[C.SUBSET_COL].to_numpy())
    tr_ans = train[C.OUTPUT_COL].tolist()
    sf = EO.score_frame([tr_ans[j] for j in top], val[C.OUTPUT_COL].tolist(), val[C.SUBSET_COL].tolist())
    print("=== Val (train-only) ensemble official ROUGE ==="); print(sf.to_string())
    print(f"ROUGE-weighted={sf.loc['overall','rouge_weighted']:.4f}")

    # Test submission (train+val corpus)
    cE = {k: np.vstack([trE[k], vaE[k]]) for k in MODELS}
    c_ans = train[C.OUTPUT_COL].tolist() + val[C.OUTPUT_COL].tolist()
    c_sub = np.concatenate([train[C.SUBSET_COL].to_numpy(), val[C.SUBSET_COL].to_numpy()])
    top_t = ens_topk(teE, cE, WEIGHTS, c_sub, test[C.SUBSET_COL].to_numpy())
    path = P.write_submission(test[C.ID_COL].tolist(), [c_ans[j] for j in top_t],
                              name="submission_stage2_ensemble_trainval")
    print(f"\nWrote -> {path}  (val-sourced test answers: {int((top_t>=len(train)).sum())}/{len(test)})")


if __name__ == "__main__":
    main()
