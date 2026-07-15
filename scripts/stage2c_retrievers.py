"""Stage 2c — stronger retriever + candidate-pool comparison on full Val.

The leaderboard breakdown shows the gap to #1 is ~87% ROUGE, and every top team
has a similar LLM-judge (~0.71-0.74) => everyone is retrieving canonical answers;
the differentiator is retrieval quality. So we test better bi-encoders and pool
modes, ranked by official ROUGE-weighted on the full Val set.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from health_eda import config as C, io_utils as io, predict as P
from health_eda import eval_official as EO

pd.set_option("display.width", 160)
C.set_seed()

MODELS = {
    "e5-base":  "intfloat/multilingual-e5-base",     # current baseline
    "e5-large": "intfloat/multilingual-e5-large",
    "bge-m3":   "BAAI/bge-m3",
}


def main():
    train = io.load_split("train").reset_index(drop=True)
    val = io.load_split("val").reset_index(drop=True)
    tr_ans = train[C.OUTPUT_COL].tolist()
    gold = val[C.OUTPUT_COL].tolist(); subs = val[C.SUBSET_COL].tolist()

    rows = {}
    for mname, mid in MODELS.items():
        for pool in ("subset", "language"):
            tag = f"{mname}/{pool}"
            with io.timer(f"Val retrieval {tag}"):
                r = P.predict(val, train, query_cache="val_q", train_cache="train_q",
                              k=5, pool=pool, embed_model=mid)
            preds = [tr_ans[j] for j in r.top_idx[:, 0]]
            sf = EO.score_frame(preds, gold, subs)
            rows[tag] = sf.loc["overall"]
            print(f"\n=== {tag} ===")
            print(sf.loc[["overall"]].to_string())
            io.save_table(sf, f"stage2c_{mname}_{pool}")

    comp = pd.DataFrame(rows).T.sort_values("rouge_weighted")
    io.save_table(comp, "stage2c_retriever_comparison")
    print("\n=== HEADLINE (full Val, overall) ===")
    print(comp[["rouge1_f1", "rougeL_f1", "rouge_weighted"]].to_string())
    best = comp["rouge_weighted"].idxmax()
    print(f"\nBEST: {best} rouge_weighted={comp.loc[best,'rouge_weighted']:.4f} "
          f"(baseline e5-base/subset=0.3532). "
          f"Est LB @LLM=0.71: {comp.loc[best,'rouge_weighted']+0.26*0.71:.4f}")


if __name__ == "__main__":
    main()
