"""Phase 2 Stage 1 — dense-retrieval baseline.

Evaluates global vs language-safe (within-subset) retrieval on Val, picks the
better, and writes a Test submission. Uses cached e5 embeddings from the EDA.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from health_eda import config as C, io_utils as io
from health_eda import predict as P

pd.set_option("display.width", 160)


def main():
    print("=" * 70, "\nSTAGE 1 — DENSE RETRIEVAL BASELINE\n", "=" * 70)
    train = io.load_split("train").reset_index(drop=True)
    val = io.load_split("val").reset_index(drop=True)
    test = io.load_split("test").reset_index(drop=True)

    # --- Val: compare global vs within-subset retrieval --------------- #
    results = {}
    for restrict in (False, True):
        tag = "within_subset" if restrict else "global"
        with io.timer(f"Val retrieval ({tag})"):
            r = P.predict(val, train, query_cache="val_q", k=5, restrict_subset=restrict)
        ev = P.evaluate(r.answers, val[C.OUTPUT_COL].tolist(), val[C.SUBSET_COL].tolist())
        results[tag] = (r, ev)
        io.save_table(ev, f"stage1_val_{tag}")
        print(f"\n--- Val: {tag} ---"); print(ev.to_string())

    # pick the better strategy by overall ROUGE-L
    best = max(results, key=lambda t: results[t][1].loc["overall", "rougeL_f1"])
    print(f"\n>>> Best Val strategy by ROUGE-L: {best} "
          f"(ROUGE-L={results[best][1].loc['overall','rougeL_f1']:.4f})")

    # --- Test submission using the best strategy ---------------------- #
    restrict = best == "within_subset"
    with io.timer("Test retrieval"):
        rt = P.predict(test, train, query_cache="test_q", k=5, restrict_subset=restrict)
    path = P.write_submission(test[C.ID_COL].tolist(), rt.answers,
                              name="submission_stage1_retrieval")
    print(f"\nWrote submission -> {path}")
    print(f"(All 3 target columns filled identically — confirm rules before final submit.)")

    # sanity: retrieval confidence distribution on test (for later routing)
    import numpy as np
    sims = rt.top_sim[:, 0]
    print(f"\nTest top-1 similarity: mean={sims.mean():.3f} "
          f"p10={np.quantile(sims,0.1):.3f} min={sims.min():.3f}")
    io.save_artifact({"test_top_sim": rt.top_sim, "test_top_idx": rt.top_idx},
                     "stage1_test_retrieval")


if __name__ == "__main__":
    main()
