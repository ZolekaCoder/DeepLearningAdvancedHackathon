"""Offline evaluation with Zindi's official metric so Val matches the leaderboard.

Weighted score = 0.37*ROUGE-1 F1 + 0.37*ROUGE-L F1 + 0.26*LLM-judge.
We can compute the two ROUGE terms exactly (rouge-score library, default
tokenizer — note it strips non-ASCII, so Amharic ROUGE ~0). The LLM-judge cannot
be run locally under the competition's open-source-only rules, so we report the
ROUGE-only weighted term and treat any LLM-judge gain separately.
"""
from __future__ import annotations

import pandas as pd

from . import config as C

W_R1, W_RL, W_LLM = 0.37, 0.37, 0.26


def score_frame(preds: list[str], golds: list[str], subsets: list[str]) -> pd.DataFrame:
    """Per-subset + overall official ROUGE-1/L F1 and the ROUGE weighted term."""
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    rows = []
    for p, g, s in zip(preds, golds, subsets):
        o = sc.score(g, p)                      # (reference, prediction)
        rows.append({"subset": s,
                     "rouge1_f1": o["rouge1"].fmeasure,
                     "rougeL_f1": o["rougeL"].fmeasure})
    df = pd.DataFrame(rows)
    overall = df[["rouge1_f1", "rougeL_f1"]].mean().to_frame("overall").T
    per = df.groupby("subset")[["rouge1_f1", "rougeL_f1"]].mean()
    out = pd.concat([overall, per])
    out["rouge_weighted"] = W_R1 * out["rouge1_f1"] + W_RL * out["rougeL_f1"]
    return out.round(4)


def leaderboard_estimate(rouge_weighted_overall: float, llm_judge: float) -> float:
    """Combine the measured ROUGE term with an assumed LLM-judge value."""
    return round(rouge_weighted_overall + W_LLM * llm_judge, 4)
