"""Stage 2b — evaluate cross-encoder reranking on Val (official ROUGE).

Reranks the top-10 dense candidates by (a) query vs candidate QUESTION and
(b) query vs candidate ANSWER; compares top-1 of each against the baseline and
the oracle ceiling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from health_eda import config as C, io_utils as io, predict as P
from health_eda import rerank as RR, eval_official as EO

pd.set_option("display.width", 160)
C.set_seed()
K = 10


def main():
    train = io.load_split("train").reset_index(drop=True)
    val = io.load_split("val").reset_index(drop=True)
    tr_q = train[C.INPUT_COL].tolist(); tr_ans = train[C.OUTPUT_COL].tolist()
    va_q = val[C.INPUT_COL].tolist(); gold = val[C.OUTPUT_COL].tolist()
    subs = val[C.SUBSET_COL].tolist()

    r = P.predict(val, train, query_cache="val_q", k=K, restrict_subset=True)

    base = EO.score_frame([tr_ans[j] for j in r.top_idx[:, 0]], gold, subs)
    print("=== baseline top1 ==="); print(base.loc[["overall"]].to_string())

    results = {"top1_baseline": base.loc["overall"]}
    for pool_name, pool in [("question", tr_q), ("answer", tr_ans)]:
        reranked = RR.rerank_topk(va_q, r.top_idx, pool, cache_name=f"val_{pool_name}")
        preds = [tr_ans[j] for j in reranked[:, 0]]
        sf = EO.score_frame(preds, gold, subs)
        results[f"rerank_{pool_name}"] = sf.loc["overall"]
        print(f"\n=== rerank by {pool_name} ==="); print(sf.to_string())

    comp = pd.DataFrame(results).T
    io.save_table(comp, "stage2b_rerank_comparison")
    print("\n=== HEADLINE (overall) ===")
    print(comp[["rouge1_f1", "rougeL_f1", "rouge_weighted"]].sort_values("rouge_weighted").to_string())
    best = comp["rouge_weighted"].idxmax()
    print(f"\nBest: {best} (ROUGE-weighted {comp.loc[best,'rouge_weighted']:.4f}); "
          f"baseline {comp.loc['top1_baseline','rouge_weighted']:.4f}. "
          f"Est LB (LLM~0.79): {comp.loc[best,'rouge_weighted']+0.26*0.79:.4f}")


if __name__ == "__main__":
    main()
