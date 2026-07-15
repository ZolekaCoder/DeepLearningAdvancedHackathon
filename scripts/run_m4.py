"""Milestone 4 runner: Cross-language, Vocabulary, Style, Complexity,
Similarity, and Validation/Test distribution.

Reuses cached question embeddings from M3; additionally encodes Train answers
(as e5 'passage') for answer-side similarity, cached to artifacts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from health_eda import config as C
from health_eda import io_utils as io
from health_eda import embeddings as E
from health_eda import vocabulary as V
from health_eda import style as ST
from health_eda import complexity as CX
from health_eda import crosslingual as XL
from health_eda import distribution as DIST
from health_eda import similarity as SIM
from health_eda import viz
from health_eda import decisions as dec
from health_eda.decisions import Decision
from health_eda.topics import parse_eng_eth_topics  # reuse parser

pd.set_option("display.width", 190); pd.set_option("display.max_columns", 60)
C.set_seed()


def main():
    print("=" * 80, "\nMILESTONE 4 — CROSS-LINGUAL / VOCAB / STYLE / COMPLEXITY / DIST\n", "=" * 80)
    train = io.load_split("train").reset_index(drop=True)
    val = io.load_split("val").reset_index(drop=True)
    test = io.load_split("test").reset_index(drop=True)
    train["gold_topic"] = parse_eng_eth_topics(train)

    # cached embeddings (cache hit)
    tr_emb = E.encode(train[C.INPUT_COL].tolist(), kind="query", cache_name="train_q")
    va_emb = E.encode(val[C.INPUT_COL].tolist(), kind="query", cache_name="val_q")
    te_emb = E.encode(test[C.INPUT_COL].tolist(), kind="query", cache_name="test_q")
    tr_sub = train[C.SUBSET_COL].to_numpy()

    # ============================ VOCABULARY ============================ #
    print("\n########## VOCABULARY ##########")
    vq = V.per_language_vocab(train, C.INPUT_COL); io.save_table(vq, "07_vocab_questions", index=False)
    va_ = V.per_language_vocab(train, C.OUTPUT_COL); io.save_table(va_, "07_vocab_answers", index=False)
    print("\nQuestion vocab per language:\n", vq.to_string(index=False))
    oov = V.oov_between_splits(train, val, C.INPUT_COL); io.save_table(oov, "07_oov_train_val_questions", index=False)
    print("\nTrain->Val question OOV:\n", oov.to_string(index=False))
    for s in ["Eng_Uga", "Amh_Eth", "Swa_Ken", "Lug_Uga", "Aka_Gha"]:
        io.save_table(V.top_tokens(train, C.INPUT_COL, s), f"07_top_tokens_{s}", index=False)

    # ============================== STYLE =============================== #
    print("\n########## ANSWER STYLE ##########")
    sty = ST.style_by_subset(train); io.save_table(sty, "12_style_by_subset", index=False)
    print(sty.to_string(index=False))
    disc = ST.disclaimer_frequency(train); io.save_table(disc, "12_disclaimers", index=False)
    print("\nDisclaimer frequency:\n", disc.to_string(index=False))
    io.save_table(ST.common_openings(train), "12_common_openings", index=False)

    # =========================== COMPLEXITY ============================= #
    print("\n########## COMPLEXITY / UNICODE ##########")
    cxq = CX.complexity_by_subset(train, C.INPUT_COL); io.save_table(cxq, "13_complexity_questions", index=False)
    cxa = CX.complexity_by_subset(train, C.OUTPUT_COL); io.save_table(cxa, "13_complexity_answers", index=False)
    uni = CX.unicode_blocks(train, C.OUTPUT_COL); io.save_table(uni, "13_unicode_blocks_answers", index=False)
    print("Answer complexity:\n", cxa.to_string(index=False))
    print("\nUnicode blocks (answers):\n", uni.to_string(index=False))

    # ========================= CROSS-LINGUAL ============================ #
    print("\n########## CROSS-LINGUAL ##########")
    sid = XL.shared_id_analysis(train)
    print("Shared ID-suffix analysis:", sid)
    io.save_table(pd.DataFrame([sid]).T.rename(columns={0: "value"}), "06_shared_id_analysis")
    # show one aligned group as evidence, if any
    if sid["n_suffixes_in_multiple_subsets"] > 0:
        tmp = train.copy(); tmp["suffix"] = tmp[C.ID_COL].map(XL.id_suffix)
        shared = tmp.groupby("suffix")[C.SUBSET_COL].nunique()
        example_suffix = shared[shared == shared.max()].index[0]
        io.save_table(XL.aligned_examples_by_suffix(train, example_suffix),
                      "06_aligned_example", index=False)
    align = XL.embedding_alignment(tr_emb, tr_sub)
    io.save_table(align, "06_embedding_alignment", index=False)
    print("\nEmbedding alignment to English:\n", align.to_string(index=False))

    # =========================== SIMILARITY ============================= #
    print("\n########## SIMILARITY ##########")
    tr_a_emb = E.encode(train[C.OUTPUT_COL].tolist(), kind="passage", cache_name="train_a")
    qq = SIM.centroid_similarity(tr_emb, tr_sub); io.save_table(qq, "11_centroid_sim_questions")
    aa = SIM.centroid_similarity(tr_a_emb, tr_sub); io.save_table(aa, "11_centroid_sim_answers")
    dens = SIM.intra_subset_density(tr_emb, tr_sub); io.save_table(dens, "11_intra_subset_density", index=False)
    coup = SIM.question_answer_coupling(tr_emb, tr_a_emb)
    io.save_table(pd.DataFrame([coup]).T.rename(columns={0: "value"}), "11_qa_coupling")
    print("Intra-subset density (question redundancy):\n", dens.to_string(index=False))
    print("\nQ-A coupling:", coup)
    io.save_fig(viz.heatmap(qq.to_numpy(), list(qq.index),
                "Question centroid cosine similarity (subset x subset)"),
                "11_centroid_sim_questions")

    # ======================= DISTRIBUTION (VAL) ========================= #
    print("\n########## VALIDATION vs TRAIN ##########")
    lm = DIST.language_mix(train, val); io.save_table(lm, "14_language_mix_val")
    print("Language mix (train vs val):\n", lm.to_string())
    ks_q = DIST.length_ks(train, val, C.INPUT_COL); io.save_table(ks_q, "14_length_ks_questions_val", index=False)
    ks_a = DIST.length_ks(train, val, C.OUTPUT_COL); io.save_table(ks_a, "14_length_ks_answers_val", index=False)
    print("\nQuestion length KS (train vs val):\n", ks_q.to_string(index=False))
    nts_val = DIST.nearest_train_similarity(va_emb, tr_emb, val[C.SUBSET_COL].to_numpy())
    io.save_table(nts_val, "14_val_nearest_train_sim", index=False)
    print("\nVal nearest-train similarity:\n", nts_val.to_string(index=False))

    # ======================= DISTRIBUTION (TEST) ======================== #
    print("\n########## TEST vs TRAIN ##########")
    lm_t = DIST.language_mix(train, test); io.save_table(lm_t, "15_language_mix_test")
    ks_t = DIST.length_ks(train, test, C.INPUT_COL); io.save_table(ks_t, "15_length_ks_questions_test", index=False)
    nts_test = DIST.nearest_train_similarity(te_emb, tr_emb, test[C.SUBSET_COL].to_numpy())
    io.save_table(nts_test, "15_test_nearest_train_sim", index=False)
    print("Test nearest-train similarity:\n", nts_test.to_string(index=False))
    print("\nTest language mix:\n", lm_t.to_string())

    # ============================ DECISIONS ============================= #
    shared_pct = sid["pct_suffixes_shared"]
    dec.record(Decision(
        id="cross-lingual-parallelism",
        section="Cross-language Analysis",
        observation=(f"{shared_pct}% of ID-suffixes appear in >1 subset "
                     f"(max {sid['max_subsets_per_suffix']} subsets/suffix). "
                     f"Nearest-English embedding cosine for non-English subsets: "
                     f"see 06_embedding_alignment."),
        evidence="outputs/tables/06_shared_id_analysis, 06_aligned_example, 06_embedding_alignment",
        impact=("If items are parallel across languages, cross-lingual retrieval "
                "and translate-then-answer become viable, and English answers can "
                "seed low-resource generation."),
        recommendation=("If parallel: exploit cross-lingual retrieval (query in any "
                        "language against the full corpus); consider pivoting through "
                        "English. Verify with aligned examples before relying on it."),
        priority="High",
        tags=["retrieval", "architecture", "data"],
    ))
    ov = nts_test[nts_test["subset"] == "OVERALL"].iloc[0]
    dec.record(Decision(
        id="test-distribution-shift",
        section="Test Distribution",
        observation=(f"Test questions' nearest-Train cosine: mean={ov['mean']}, "
                     f"median={ov['median']}, p10={ov['p10']}. Language mix vs train "
                     f"in 15_language_mix_test."),
        evidence="outputs/tables/15_test_nearest_train_sim, 15_language_mix_test",
        impact=("High nearest-train similarity => retrieval-first will generalise to "
                "Test; a low-similarity tail marks questions needing genuine "
                "generation."),
        recommendation=("Use retrieval confidence (nearest-train cosine) to ROUTE: "
                        "high-sim -> copy/rerank; low-sim -> generate. Tune the "
                        "threshold on Val."),
        priority="High",
        tags=["architecture", "retrieval", "post-processing"],
    ))
    dec.record(Decision(
        id="qa-coupling",
        section="Similarity Analysis",
        observation=(f"True Q-A cosine {coup['mean_true_qa_cosine']} vs shuffled "
                     f"{coup['mean_shuffled_qa_cosine']} (gap {coup['coupling_gap']})."),
        evidence="outputs/tables/11_qa_coupling.csv",
        impact="A positive gap confirms questions strongly determine answers -> retrieval signal is real.",
        recommendation="Safe to rely on question-based retrieval; optionally add answer-space reranking.",
        priority="Medium",
        tags=["retrieval", "reranking"],
    ))
    print("\nMILESTONE 4 COMPLETE.")


if __name__ == "__main__":
    main()
