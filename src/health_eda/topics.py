"""Sections 8 & 9 — Topic modelling and embedding exploration.

We discover health topics *automatically* (no manual labels) with BERTopic over
precomputed multilingual-e5 question embeddings, and project embeddings to 2-D
with UMAP for visual inspection (do languages / topics cluster?).

Everything accepts precomputed embeddings so we never re-encode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def parse_eng_eth_topics(df: pd.DataFrame) -> pd.Series:
    """Extract the gold topic label embedded in Eng_Eth answers (weak labels).

    Eng_Eth answers uniquely begin 'This is a question about, <TOPIC>.' — we lift
    that <TOPIC> out as a near-free gold label (see MODELLING_DECISIONS
    eng-eth-topic-prefix). Returns NaN for non-Eng_Eth rows.
    """
    import numpy as np

    lab = df[C.OUTPUT_COL].str.extract(
        r"^This is a question about,?\s*([A-Za-z/ ]+?)\.")[0].str.strip()
    lab[df[C.SUBSET_COL] != "Eng_Eth"] = np.nan
    return lab


def umap_2d(emb: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1,
            seed: int = C.SEED, metric: str = "cosine") -> np.ndarray:
    """Project embeddings to 2-D for plotting (seeded for reproducibility)."""
    import umap

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, metric=metric, random_state=seed)
    return reducer.fit_transform(emb)


def run_bertopic(docs: list[str], emb: np.ndarray, min_topic_size: int = 60,
                 seed: int = C.SEED):
    """Fit BERTopic on precomputed embeddings; return (model, topics, info).

    We supply our own UMAP + a multilingual-agnostic vectorizer so topic
    keywords are readable. Topics are data-driven (not manually assigned).
    """
    import umap as umap_mod
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from hdbscan import HDBSCAN

    umap_model = umap_mod.UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                              metric="cosine", random_state=seed)
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean",
                            cluster_selection_method="eom", prediction_data=True)
    # Keep English stop-words out of topic keywords; harmless for other langs.
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                                 min_df=5)
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=emb)
    info = topic_model.get_topic_info()
    return topic_model, np.asarray(topics), info


def topic_distribution(topics: np.ndarray, subsets: list[str],
                       topic_model) -> pd.DataFrame:
    """Proportion of each discovered topic, with a readable label."""
    s = pd.Series(topics)
    counts = s.value_counts().sort_values(ascending=False)
    rows = []
    for tid, cnt in counts.items():
        if tid == -1:
            label = "<outlier/noise>"
        else:
            words = [w for w, _ in topic_model.get_topic(tid)[:5]]
            label = ", ".join(words)
        rows.append({"topic_id": int(tid), "label": label, "count": int(cnt),
                     "pct": round(cnt / len(topics) * 100, 2)})
    return pd.DataFrame(rows)


def topic_by_subset(topics: np.ndarray, subsets: list[str]) -> pd.DataFrame:
    """Cross-tab of topic id vs subset (are topics language-specific?)."""
    df = pd.DataFrame({"topic": topics, "subset": subsets})
    return pd.crosstab(df["topic"], df["subset"])


def validate_against_gold(topics: np.ndarray, gold_labels: pd.Series) -> dict:
    """Cluster-purity style check vs the Eng_Eth embedded topic labels.

    For each discovered cluster, how consistent is the gold label of its members?
    Reported as mean max-purity over clusters that contain gold-labelled docs.
    """
    df = pd.DataFrame({"topic": topics}).join(gold_labels.rename("gold"))
    df = df.dropna(subset=["gold"])
    if df.empty:
        return {"note": "no gold labels overlap"}
    purities = []
    for tid, grp in df.groupby("topic"):
        if tid == -1 or len(grp) < 5:
            continue
        purities.append(grp["gold"].value_counts(normalize=True).iloc[0])
    return {
        "n_gold_docs": int(len(df)),
        "n_clusters_evaluated": len(purities),
        "mean_cluster_purity": round(float(np.mean(purities)), 4) if purities else None,
    }
