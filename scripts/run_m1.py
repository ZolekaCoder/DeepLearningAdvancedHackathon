"""Milestone 1 runner: Dataset Overview + Language Analysis.

Run standalone to compute + persist all M1 tables/figures and print an
interpretable summary. The notebook calls the same helper functions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from health_eda import config as C
from health_eda import io_utils as io
from health_eda import overview, language
from health_eda import decisions as dec
from health_eda.decisions import Decision

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

C.set_seed()


def main():
    print("=" * 80, "\nMILESTONE 1 — DATASET OVERVIEW + LANGUAGE ANALYSIS\n", "=" * 80)
    dec.reset()  # fresh decisions log for a clean run

    with io.timer("load all splits"):
        dfs = io.load_all()

    # ------------------------------------------------------------------ #
    # 1. Dataset Overview
    # ------------------------------------------------------------------ #
    print("\n### Split summary")
    ss = overview.split_summary(dfs)
    io.save_table(ss, "01_split_summary")
    print(ss)

    print("\n### Column dtypes")
    dt = overview.column_dtypes(dfs)
    io.save_table(dt, "01_column_dtypes")
    print(dt)

    print("\n### Duplicate breakdown (Train)")
    dup = overview.duplicate_breakdown(dfs["train"])
    io.save_table(dup, "01_train_duplicates")
    print(dup)

    print("\n### Submission target structure")
    sub = overview.submission_summary(dfs["sample_sub"])
    io.save_table(sub, "01_submission_targets")
    print(sub)

    print("\n### Example records (one per subset)")
    ex = overview.example_records(dfs["train"])
    io.save_table(ex, "01_examples", index=False)
    for _, r in ex.iterrows():
        print(f"\n[{r[C.SUBSET_COL]}]")
        print("  Q:", str(r[C.INPUT_COL])[:200])
        print("  A:", str(r[C.OUTPUT_COL])[:200])

    # ------------------------------------------------------------------ #
    # 2. Language / subset analysis
    # ------------------------------------------------------------------ #
    print("\n\n### Per-subset language table (Train)")
    with io.timer("subset table (train)"):
        lang_tab = language.subset_table(dfs["train"], has_answers=True)
    io.save_table(lang_tab, "02_subset_summary_train", index=False)
    print(lang_tab.to_string(index=False))

    imb = language.imbalance_metrics(lang_tab)
    print("\n### Imbalance metrics:", imb)

    print("\n### Script composition (questions, Train)")
    scr = language.script_table(dfs["train"], C.INPUT_COL)
    io.save_table(scr, "02_script_profile_questions_train")
    print(scr)

    # Length columns for plots
    with io.timer("length columns + plots"):
        train_len = language.add_length_columns(dfs["train"])
        io.save_artifact(train_len[[C.ID_COL, C.SUBSET_COL, "q_words", "q_chars",
                                    "a_words", "a_chars"]], "train_lengths")
        from health_eda import viz
        f1 = viz.bar(lang_tab.set_index("subset")["n"], "Examples per subset (Train)",
                     "subset", "count", color_by_subset=True)
        io.save_fig(f1, "02_counts_per_subset")
        f2 = viz.box_by_group(train_len, "q_words", C.SUBSET_COL,
                              "Question length (words) by subset", "words")
        io.save_fig(f2, "02_qlen_box_by_subset")
        f3 = viz.box_by_group(train_len, "a_words", C.SUBSET_COL,
                              "Answer length (words) by subset", "words")
        io.save_fig(f3, "02_alen_box_by_subset")

    # ------------------------------------------------------------------ #
    # Record modelling decisions discovered in M1
    # ------------------------------------------------------------------ #
    dec.record(Decision(
        id="language-imbalance",
        section="Language Analysis",
        observation=(f"Subsets are imbalanced: largest={imb['largest_subset']}, "
                     f"smallest={imb['smallest_subset']}, "
                     f"max/min ratio={imb['imbalance_ratio_max_min']}x, "
                     f"normalized balance={imb['normalized_entropy_balance']}."),
        evidence="outputs/tables/02_subset_summary_train.csv; figure 02_counts_per_subset.png",
        impact=("A model trained on the pooled data will be dominated by English "
                "subsets; low-resource languages (Amharic, Swahili, Luganda, Akan) "
                "risk poor quality yet count equally in per-language evaluation."),
        recommendation=("Report metrics per subset; consider subset-balanced "
                        "sampling / loss weighting during fine-tuning; ensure "
                        "retrieval corpus is not English-dominated per query."),
        priority="High",
        tags=["fine-tuning", "evaluation", "retrieval"],
    ))

    amh = lang_tab.loc[lang_tab["subset"] == "Amh_Eth"]
    if len(amh):
        dec.record(Decision(
            id="amharic-script",
            section="Language Analysis",
            observation="Amharic (Amh_Eth) uses Ge'ez script, unlike all other subsets (Latin).",
            evidence="outputs/tables/02_script_profile_questions_train.csv",
            impact=("Tokenizers/embedding models with weak Ge'ez coverage will "
                    "fragment Amharic badly, hurting both retrieval and generation; "
                    "ROUGE token overlap also behaves differently for Ge'ez."),
            recommendation=("Verify the chosen embedding/LLM tokenizers cover Ge'ez; "
                            "evaluate Amharic separately; consider script-aware "
                            "normalization."),
            priority="High",
            tags=["retrieval", "fine-tuning", "evaluation", "tokenization"],
        ))

    print("\nMILESTONE 1 COMPLETE. Tables in outputs/tables, figures in outputs/figures.")


if __name__ == "__main__":
    main()
