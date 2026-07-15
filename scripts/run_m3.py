"""Milestone 3 runner: Embeddings + Topic modelling + Retrieval feasibility.

Heaviest step. Embeddings are cached to outputs/artifacts so re-runs are fast.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from health_eda import config as C
from health_eda import io_utils as io
from health_eda import embeddings as E
from health_eda import topics as T
from health_eda import retrieval as R
from health_eda import viz
from health_eda import decisions as dec
from health_eda.decisions import Decision

from health_eda.topics import parse_eng_eth_topics

pd.set_option("display.width", 180); pd.set_option("display.max_columns", 50)
C.set_seed()


def main():
    print("=" * 80, "\nMILESTONE 3 — EMBEDDINGS / TOPICS / RETRIEVAL\n", "=" * 80)
    train = io.load_split("train").reset_index(drop=True)
    val = io.load_split("val").reset_index(drop=True)
    test = io.load_split("test").reset_index(drop=True)

    train["gold_topic"] = parse_eng_eth_topics(train)

    # ------------------------------------------------------------------ #
    # 1. Embeddings (questions as e5 'query')
    # ------------------------------------------------------------------ #
    tr_q = train[C.INPUT_COL].tolist()
    va_q = val[C.INPUT_COL].tolist()
    te_q = test[C.INPUT_COL].tolist()
    tr_emb = E.encode(tr_q, kind="query", cache_name="train_q")
    va_emb = E.encode(va_q, kind="query", cache_name="val_q")
    te_emb = E.encode(te_q, kind="query", cache_name="test_q")
    print("embeddings:", tr_emb.shape, va_emb.shape, te_emb.shape)

    # ------------------------------------------------------------------ #
    # 2. Topic modelling (automatic, BERTopic over train question embeddings)
    # ------------------------------------------------------------------ #
    with io.timer("BERTopic fit (train questions)"):
        tmodel, tr_topics, info = T.run_bertopic(tr_q, tr_emb, min_topic_size=60)
    n_topics = int((info["Topic"] != -1).sum())
    print(f"\nDiscovered {n_topics} topics (excluding outliers).")

    dist = T.topic_distribution(tr_topics, train[C.SUBSET_COL].tolist(), tmodel)
    io.save_table(dist, "08_topic_distribution", index=False)
    print("\nTop topics:\n", dist.head(20).to_string(index=False))

    xtab = T.topic_by_subset(tr_topics, train[C.SUBSET_COL].tolist())
    io.save_table(xtab, "08_topic_by_subset")

    gold_val = T.validate_against_gold(tr_topics, train["gold_topic"])
    print("\nTopic validation vs Eng_Eth gold labels:", gold_val)
    io.save_table(pd.DataFrame([gold_val]).T.rename(columns={0: "value"}),
                  "08_topic_validation")

    io.save_artifact({"train_topics": tr_topics}, "train_topics")

    f = viz.bar(dist[dist.topic_id != -1].set_index("label")["pct"].head(15),
                "Discovered health topics (% of Train questions)", "topic", "%",
                rotate=75, figsize=(11, 5))
    io.save_fig(f, "08_topic_distribution")

    # Transform val questions to topics (for retrieval topic-match).
    with io.timer("BERTopic transform (val)"):
        va_topics, _ = tmodel.transform(va_q, embeddings=va_emb)
    va_topics = np.asarray(va_topics)

    # ------------------------------------------------------------------ #
    # 3. Embedding exploration — UMAP 2-D (sampled for a readable plot)
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(C.SEED)
    n_plot = min(7000, len(train))
    sel = rng.choice(len(train), n_plot, replace=False)
    with io.timer(f"UMAP 2-D ({n_plot} pts)"):
        xy = T.umap_2d(tr_emb[sel])
    io.save_artifact({"xy": xy, "sel": sel}, "umap_train_q")
    f = viz.scatter_2d(xy, train[C.SUBSET_COL].to_numpy()[sel],
                       "UMAP of question embeddings — coloured by subset/language")
    io.save_fig(f, "09_umap_by_subset")
    # colour by topic (top clusters only for legibility)
    tp = tr_topics[sel].astype(str)
    f = viz.scatter_2d(xy, tp, "UMAP of question embeddings — coloured by topic",
                       palette={t: None for t in np.unique(tp)})
    io.save_fig(f, "09_umap_by_topic")

    # ------------------------------------------------------------------ #
    # 4. Retrieval feasibility (BM25 + dense + hybrid), evaluate on Val
    # ------------------------------------------------------------------ #
    print("\n########## RETRIEVAL FEASIBILITY (Val -> nearest Train) ##########")
    tr_ans = train[C.OUTPUT_COL].tolist()
    va_ans = val[C.OUTPUT_COL].tolist()
    tr_sub = train[C.SUBSET_COL].tolist()
    va_sub = val[C.SUBSET_COL].tolist()

    with io.timer("BM25 build+retrieve"):
        bm25 = R.BM25Retriever(tr_q)
        bm25_topk = bm25.topk(va_q, k=5)
    with io.timer("dense retrieve"):
        dense = R.DenseRetriever(tr_emb)
        dense_topk, dense_sim = dense.topk(va_emb, k=5)
    hybrid_topk = R.rrf_fuse(bm25_topk, dense_topk, k=5)

    results = {}
    for name, topk, sim in [("BM25", bm25_topk, None),
                            ("dense_e5", dense_topk, dense_sim[:, 0]),
                            ("hybrid", hybrid_topk, None)]:
        ev = R.evaluate(topk[:, 0], tr_ans, va_ans,
                        train_topics=tr_topics, query_topics=va_topics,
                        train_subset=tr_sub, query_subset=va_sub,
                        dense_sim=sim)
        summ = R.summarize(ev)
        results[name] = summ
        io.save_table(summ, f"10_retrieval_{name}_summary")
        io.save_table(ev.describe(), f"10_retrieval_{name}_detail_stats")
        print(f"\n--- {name} (overall + per subset) ---")
        print(summ.round(4).to_string())

    # Combined headline table
    headline = pd.concat({k: v.loc[["overall"]] for k, v in results.items()})
    headline.index = headline.index.droplevel(1)
    io.save_table(headline, "10_retrieval_headline")
    print("\n=== HEADLINE (overall) ===\n", headline.round(4).to_string())

    io.save_artifact({"bm25_topk": bm25_topk, "dense_topk": dense_topk,
                      "hybrid_topk": hybrid_topk, "dense_sim": dense_sim,
                      "val_topics": va_topics}, "retrieval_val")

    # ------------------------------------------------------------------ #
    # DECISIONS
    # ------------------------------------------------------------------ #
    hb = results["hybrid"].loc["overall"]
    dec.record(Decision(
        id="retrieval-feasibility",
        section="Retrieval Feasibility",
        observation=(f"Retrieve-and-copy nearest Train answer (hybrid BM25+e5) on "
                     f"Val: ROUGE-L={hb['rougeL_f1']:.3f}, ROUGE-1={hb['rouge1_f1']:.3f}, "
                     f"exact={hb['exact']:.3f}, topic_match={hb.get('topic_match', float('nan')):.3f}, "
                     f"subset_match={hb.get('subset_match', float('nan')):.3f}."),
        evidence="outputs/tables/10_retrieval_headline.csv and per-method summaries",
        impact=("Sets the retrieval-only ceiling with ZERO training. If ROUGE is "
                "already high, retrieval-first is a very strong baseline and "
                "generation only needs to close the gap / smooth style."),
        recommendation=("Adopt hybrid retrieval as baseline; use retrieved answer as "
                        "context for a generator (RAG); rerank top-k; evaluate whether "
                        "generation beats copy per subset."),
        priority="High",
        tags=["retrieval", "reranking", "architecture"],
    ))
    dec.record(Decision(
        id="topic-inventory",
        section="Topic Analysis",
        observation=(f"BERTopic discovered {n_topics} data-driven health topics; "
                     f"validation vs Eng_Eth gold labels mean cluster purity="
                     f"{gold_val.get('mean_cluster_purity')}."),
        evidence="outputs/tables/08_topic_distribution.csv, 08_topic_validation.csv",
        impact=("Clean topical structure enables topic-routed retrieval, topic-"
                "balanced evaluation, and few-shot exemplar selection by topic."),
        recommendation=("Store topic ids as metadata; consider a topic filter in "
                        "retrieval; report metrics per topic as well as per subset."),
        priority="Medium",
        tags=["retrieval", "topic-modelling", "evaluation"],
    ))

    print("\nMILESTONE 3 COMPLETE.")


if __name__ == "__main__":
    main()
