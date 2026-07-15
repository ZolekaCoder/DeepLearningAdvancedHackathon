"""Stage 2 bake-off + oracle ceiling on Val (official rouge-score).

Compares answer-selection strategies AND computes the oracle: the best possible
ROUGE from copying a retrieved candidate (upper bound for any retrieve-and-copy
system). Grounds how high a retrieval approach can realistically go.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from health_eda import config as C, io_utils as io, predict as P
from health_eda import embeddings as E, select as S, eval_official as EO

pd.set_option("display.width", 160)
C.set_seed()
K = 10


def main():
    train = io.load_split("train").reset_index(drop=True)
    val = io.load_split("val").reset_index(drop=True)
    tr_ans = train[C.OUTPUT_COL].tolist()
    tr_ans_emb = E.encode(tr_ans, "passage", cache_name="train_a")
    gold = val[C.OUTPUT_COL].tolist(); subs = val[C.SUBSET_COL].tolist()

    with io.timer(f"Val top-{K} retrieval (within-subset)"):
        r = P.predict(val, train, query_cache="val_q", k=K, restrict_subset=True)

    # --- selection strategies --- #
    print("\n=== Selection strategy bake-off (official ROUGE weighted) ===")
    summary = {}
    for strat in ("top1", "majority", "medoid", "hybrid"):
        preds = S.select_answers(r.top_idx, r.top_sim, tr_ans, tr_ans_emb, strategy=strat)
        sf = EO.score_frame(preds, gold, subs)
        summary[strat] = sf.loc["overall"]
        print(f"\n--- {strat} ---")
        print(sf.to_string())

    # --- ORACLE ceiling: best candidate among top-k by ROUGE vs gold --- #
    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    oracle_preds = []
    for i in range(len(val)):
        best, best_s = tr_ans[r.top_idx[i, 0]], -1.0
        for j in r.top_idx[i]:
            o = sc.score(gold[i], tr_ans[j])
            s = o["rouge1"].fmeasure + o["rougeL"].fmeasure
            if s > best_s:
                best_s, best = s, tr_ans[j]
        oracle_preds.append(best)
    oracle_sf = EO.score_frame(oracle_preds, gold, subs)
    print("\n=== ORACLE (best of top-%d, retrieval-selection ceiling) ===" % K)
    print(oracle_sf.to_string())

    # --- headline comparison --- #
    comp = pd.DataFrame(summary).T
    comp.loc["ORACLE_topk"] = oracle_sf.loc["overall"]
    io.save_table(comp, "stage2_strategy_comparison")
    print("\n=== HEADLINE: overall ROUGE-weighted term (0.37*R1+0.37*RL) ===")
    print(comp[["rouge1_f1", "rougeL_f1", "rouge_weighted"]].sort_values("rouge_weighted").to_string())

    print("\nNote: leaderboard = rouge_weighted + 0.26*LLM_judge. "
          "At LB 0.560 with ROUGE-weighted ~0.354, implied LLM-judge ~0.79.")


if __name__ == "__main__":
    main()
