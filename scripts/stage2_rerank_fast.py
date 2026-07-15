"""Stage 2b (fast) — test reranking hypothesis on a stratified Val sample.

Uses the fast multilingual cross-encoder on a per-subset stratified sample so we
learn whether reranking helps in minutes, not hours. If it beats baseline toward
the oracle, we apply it to the full Test set.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from health_eda import config as C, io_utils as io, predict as P
from health_eda import rerank as RR, eval_official as EO

pd.set_option("display.width", 160)
C.set_seed()
K = 8
PER_SUBSET = 250   # stratified sample size per subset


def main():
    train = io.load_split("train").reset_index(drop=True)
    val_full = io.load_split("val").reset_index(drop=True)

    # stratified Val sample
    rng = np.random.default_rng(C.SEED)
    idx = []
    for s, grp in val_full.groupby(C.SUBSET_COL):
        take = min(PER_SUBSET, len(grp))
        idx += list(rng.choice(grp.index.to_numpy(), take, replace=False))
    val = val_full.loc[idx].reset_index(drop=True)
    print(f"Val sample: {len(val)} rows across {val[C.SUBSET_COL].nunique()} subsets")

    tr_q = train[C.INPUT_COL].tolist(); tr_ans = train[C.OUTPUT_COL].tolist()
    va_q = val[C.INPUT_COL].tolist(); gold = val[C.OUTPUT_COL].tolist()
    subs = val[C.SUBSET_COL].tolist()

    r = P.predict(val, train, query_cache="val_sample_q", k=K, restrict_subset=True)

    base = EO.score_frame([tr_ans[j] for j in r.top_idx[:, 0]], gold, subs)
    results = {"top1_baseline": base.loc["overall"]}

    for pool_name, pool in [("question", tr_q), ("answer", tr_ans)]:
        reranked = RR.rerank_topk(va_q, r.top_idx, pool, batch_size=64)
        sf = EO.score_frame([tr_ans[j] for j in reranked[:, 0]], gold, subs)
        results[f"rerank_{pool_name}"] = sf.loc["overall"]
        print(f"\n=== rerank by {pool_name} ==="); print(sf.to_string())

    # oracle on the sample
    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    orc = []
    for i in range(len(val)):
        best, bs = tr_ans[r.top_idx[i, 0]], -1
        for j in r.top_idx[i]:
            o = sc.score(gold[i], tr_ans[j]); v = o["rouge1"].fmeasure + o["rougeL"].fmeasure
            if v > bs: bs, best = v, tr_ans[j]
        orc.append(best)
    results["ORACLE"] = EO.score_frame(orc, gold, subs).loc["overall"]

    comp = pd.DataFrame(results).T
    io.save_table(comp, "stage2b_rerank_fast_comparison")
    print("\n=== HEADLINE (Val sample, overall) ===")
    print(comp[["rouge1_f1", "rougeL_f1", "rouge_weighted"]].sort_values("rouge_weighted").to_string())


if __name__ == "__main__":
    main()
