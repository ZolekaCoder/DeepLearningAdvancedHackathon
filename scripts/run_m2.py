"""Milestone 2 runner: Question + Answer + Duplicate analysis (lexical).

Semantic clustering of intents / near-duplicate *semantic* questions is done in
Milestone 3 (needs embeddings). Here we cover everything computable without a
neural model: templating signals, answer reuse, near-duplicate (shingle) rate,
repeated closings, and the many-to-one Q->A mapping.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from health_eda import config as C
from health_eda import io_utils as io
from health_eda import questions as Q
from health_eda import answers as A
from health_eda import duplicates as D
from health_eda import viz
from health_eda import decisions as dec
from health_eda.decisions import Decision

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)
C.set_seed()

ENGLISH = ["Eng_Uga", "Eng_Gha", "Eng_Eth", "Eng_Ken"]


def main():
    print("=" * 80, "\nMILESTONE 2 — QUESTION / ANSWER / DUPLICATE ANALYSIS\n", "=" * 80)
    train = io.load_split("train")
    q, a = train[C.INPUT_COL], train[C.OUTPUT_COL]

    # ================================================================== #
    # QUESTION ANALYSIS
    # ================================================================== #
    print("\n########## QUESTION ANALYSIS ##########")

    sig = Q.diversity_signature(q)
    print("\nDiversity signature (all questions):", sig)
    io.save_table(pd.DataFrame([sig]).T.rename(columns={0: "value"}),
                  "03_question_diversity_signature")

    # Per-subset diversity signature (templating differs by language).
    rows = []
    for s, sub in train.groupby(C.SUBSET_COL):
        d = Q.diversity_signature(sub[C.INPUT_COL]); d["subset"] = s
        rows.append(d)
    subset_sig = pd.DataFrame(rows).set_index("subset")
    io.save_table(subset_sig, "03_question_diversity_by_subset")
    print("\nPer-subset templating (higher top20% => more templated):")
    print(subset_sig[["n_questions", "unique_prefix_ratio",
                      "pct_covered_by_top20_prefixes"]])

    fw = Q.first_word_distribution(q, top=25)
    io.save_table(fw, "03_question_first_words", index=False)
    print("\nTop first words:\n", fw.head(12).to_string(index=False))

    pre3 = Q.starting_ngrams(q, n=3, top=30)
    io.save_table(pre3, "03_question_prefixes_3gram", index=False)
    print("\nTop opening 3-grams:\n", pre3.head(15).to_string(index=False))

    for n in (2, 3, 4):
        ng = Q.top_ngrams(q, n=n, top=30)
        io.save_table(ng, f"03_question_{n}grams", index=False)

    # English interrogative templates (only meaningful for Eng_* subsets).
    eng_q = train.loc[train[C.SUBSET_COL].isin(ENGLISH), C.INPUT_COL]
    tmpl = Q.template_coverage(eng_q)
    io.save_table(tmpl, "03_question_templates_english", index=False)
    print("\nEnglish interrogative templates:\n", tmpl.to_string(index=False))

    f = viz.bar(tmpl.set_index("template")["pct"].head(12),
                "English question templates (% of English questions)",
                "template", "%")
    io.save_fig(f, "03_question_templates_english")

    # ================================================================== #
    # ANSWER ANALYSIS
    # ================================================================== #
    print("\n\n########## ANSWER ANALYSIS ##########")

    reuse = A.exact_reuse(a)
    print("\nExact answer reuse:", reuse)
    io.save_table(pd.DataFrame([reuse]).T.rename(columns={0: "value"}),
                  "04_answer_exact_reuse")

    top_reuse = A.reuse_distribution(a, top=15)
    io.save_table(top_reuse, "04_answer_most_reused", index=False)
    print("\nMost-reused exact answers:\n",
          top_reuse[["rank", "times_used"]].to_string(index=False))
    print("Example most-reused answer:\n  ", top_reuse.iloc[0]["answer_preview"])

    with io.timer("near-duplicate answer estimate (sampled)"):
        nd = A.near_duplicate_estimate(a, sample=5000, k=8, threshold=0.8)
    print("\nNear-duplicate answers:", nd)
    io.save_table(pd.DataFrame([nd]).T.rename(columns={0: "value"}),
                  "04_answer_near_duplicates")

    closings = A.closing_lines(a, top=15)
    io.save_table(closings, "04_answer_closings", index=False)
    print("\nMost common closing sentences:\n", closings.to_string(index=False))

    rep = A.repeated_sentences(a, top=20)
    io.save_table(rep, "04_answer_repeated_sentences", index=False)
    print("\nMost repeated sentences (boilerplate/advice):\n",
          rep.head(10).to_string(index=False))

    # Reuse distribution histogram (how many answers used k times).
    vc = a.dropna().value_counts()
    f = viz.bar(vc.value_counts().sort_index().head(15).rename_axis("times_used"),
                "Answer reuse distribution (#answers used k times)",
                "times an answer is reused (k)", "# distinct answers")
    io.save_fig(f, "04_answer_reuse_distribution")

    # ================================================================== #
    # DUPLICATE ANALYSIS
    # ================================================================== #
    print("\n\n########## DUPLICATE ANALYSIS ##########")
    edc = D.exact_duplicate_counts(train)
    io.save_table(edc, "05_exact_duplicate_counts")
    print("\nExact duplicate counts:\n", edc)

    m2o, m2o_tbl = D.many_questions_one_answer(train, top=15)
    print("\nMany-questions -> one-answer:", m2o)
    io.save_table(pd.DataFrame([m2o]).T.rename(columns={0: "value"}),
                  "05_many_questions_one_answer_summary")
    io.save_table(m2o_tbl, "05_many_questions_one_answer_top", index=False)

    o2m, o2m_tbl = D.one_question_many_answers(train, top=15)
    print("\nOne-question -> many-answers (answer ambiguity ceiling):", o2m)
    io.save_table(pd.DataFrame([o2m]).T.rename(columns={0: "value"}),
                  "05_one_question_many_answers_summary")
    io.save_table(o2m_tbl, "05_one_question_many_answers_top", index=False)

    # ================================================================== #
    # DECISIONS
    # ================================================================== #
    dec.record(Decision(
        id="canonical-answer-bank-size",
        section="Answer / Duplicate Analysis",
        observation=(f"{reuse['n_unique_answers']} unique answers for "
                     f"{reuse['n_answers']} rows (unique ratio "
                     f"{reuse['unique_ratio']}); "
                     f"{reuse['pct_answers_in_reused_group']}% of rows reuse an "
                     f"answer used >1x; most-reused answer appears "
                     f"{reuse['max_reuse_count']}x. Near-duplicate (Jaccard>=0.8) "
                     f"answers ~{nd['pct_near_duplicate']}% (sampled)."),
        evidence="outputs/tables/04_answer_exact_reuse, 04_answer_near_duplicates, 05_*",
        impact=("Confirms/quantifies a canonical answer bank. If the bank is small, "
                "a retrieve-and-copy system can match many gold answers verbatim "
                "(high ROUGE)."),
        recommendation=("Build the retrieval corpus from UNIQUE answers; treat the "
                        "task partly as answer classification/selection; measure "
                        "oracle ROUGE of nearest-answer copy in Section Retrieval."),
        priority="High",
        tags=["retrieval", "reranking", "post-processing"],
    ))

    dec.record(Decision(
        id="question-templating",
        section="Question Analysis",
        observation=(f"Opening 3-gram unique ratio={sig['unique_prefix_ratio']}, "
                     f"top-20 prefixes cover "
                     f"{sig['pct_covered_by_top5_prefixes']}%/"
                     f"{sig['pct_covered_by_top20_prefixes']}% (top5/top20). "
                     "English is templated ('what/how/can I...'); "
                     "Akan/Luganda/Amharic questions are longer & more diverse."),
        evidence="outputs/tables/03_question_diversity_by_subset, 03_question_templates_english",
        impact=("Templated English questions cluster into few intents -> retrieval "
                "and few-shot prompting work well; diverse African-language "
                "questions need semantic (not lexical) matching."),
        recommendation=("Use dense/semantic retrieval (not just BM25) especially for "
                        "African languages; consider intent clustering to build "
                        "few-shot prompt exemplars."),
        priority="Medium",
        tags=["retrieval", "prompting"],
    ))

    if o2m["pct_questions_with_ambiguous_answers"] > 1:
        dec.record(Decision(
            id="answer-ambiguity-ceiling",
            section="Duplicate Analysis",
            observation=(f"{o2m['n_questions_with_multiple_distinct_answers']} "
                         f"({o2m['pct_questions_with_ambiguous_answers']}%) identical "
                         f"questions have >1 distinct gold answer (max "
                         f"{o2m['max_distinct_answers_per_question']})."),
            evidence="outputs/tables/05_one_question_many_answers_top",
            impact=("There is irreducible answer variance: no system can match all "
                    "gold answers for these questions -> caps achievable ROUGE and "
                    "means the LLM judge (semantic) may be fairer than ROUGE."),
            recommendation=("Prefer semantic evaluation where possible; for retrieval, "
                            "pick the most frequent/representative answer among "
                            "variants."),
            priority="Medium",
            tags=["evaluation", "retrieval"],
        ))

    print("\nMILESTONE 2 COMPLETE.")


if __name__ == "__main__":
    main()
