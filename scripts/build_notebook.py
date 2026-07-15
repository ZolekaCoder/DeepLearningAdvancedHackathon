"""Generate notebooks/health_qa_eda.ipynb — the narrated, end-to-end EDA.

The notebook imports the reusable `health_eda` package (no logic is duplicated
here) and drives it section by section, with markdown Why/What/How framing and a
"Modelling Implications" subsection after every analysis. It runs top-to-bottom
in one pass; heavy embedding steps hit the on-disk cache created by the modules.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(new_code_cell(text.strip("\n")))


def implications(text):
    md("#### 🔧 Modelling Implications\n\n" + text.strip("\n"))


# =========================================================================== #
md("""
# Multilingual Health QA — Exploratory Data Analysis

**Goal.** Answer one question with evidence: *what is the best modelling strategy for this dataset?*
Every section states **why** we run it, **what** insight we want, and **how** it changes our modelling
choices, and ends with **Modelling Implications**. Findings that affect retrieval, prompting, reranking,
fine-tuning, evaluation or post-processing are appended to `MODELLING_DECISIONS.md`.

**Dataset.** Sexual & reproductive health Q/A across 8 language-country subsets (English + Akan, Amharic,
Luganda, Swahili) from Ghana, Uganda, Kenya, Ethiopia. Submission is scored on **three** targets:
ROUGE-L F1, ROUGE-1 F1, and an **LLM judge**.

**Reproducibility.** All helpers live in `src/health_eda`. Tables → `outputs/tables`, figures →
`outputs/figures`, artefacts (embeddings, topics, indices) → `outputs/artifacts`. Seeds are fixed.
""")

code("""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd().parent / "src"))  # notebook lives in notebooks/
sys.path.insert(0, str(Path.cwd() / "src"))          # or run from repo root

import numpy as np, pandas as pd
from IPython.display import display, Image

from health_eda import config as C, io_utils as io
from health_eda import overview, language, questions as Q, answers as A, duplicates as D
from health_eda import embeddings as E, topics as T, retrieval as R, metrics
from health_eda import vocabulary as V, style as ST, complexity as CX
from health_eda import crosslingual as XL, distribution as DIST, similarity as SIM
from health_eda import viz, report
from health_eda import decisions as dec
from health_eda.decisions import Decision

pd.set_option("display.max_columns", 60); pd.set_option("display.width", 200)
C.set_seed()
dec.reset()  # rebuild the decisions log from scratch on a full run
def show_fig(name): display(Image(str(C.FIG_DIR / f"{name}.png")))
print("Setup complete. Device for embeddings:", E._device())
""")

# --------------------------------------------------------------------------- #
md("""
## 1. Dataset Overview
**Why.** Before any modelling we must know shapes, dtypes, missingness, duplicates and the *target
structure* — these frame every later choice. **What.** Row/column counts, memory, duplicate rows, and
the submission's three target columns. **How.** Missing answers can't supervise generation; heavy
answer duplication hints at a canonical-answer task; the triple target dictates the evaluation strategy.
""")
code("""
dfs = io.load_all()
display(overview.split_summary(dfs))
display(overview.column_dtypes(dfs))
print("Submission targets (evaluation surface):")
display(overview.submission_summary(dfs["sample_sub"]))
print("Exact-duplicate breakdown (Train):")
display(overview.duplicate_breakdown(dfs["train"]))
print("Example records (one per subset):")
display(overview.example_records(dfs["train"]))
""")
implications("""
- Data is **clean** (no missing/dup rows) → no imputation needed; the modelling problem is quality, not hygiene.
- **~39% duplicate answers** with far fewer duplicate questions ⇒ answers look **canonical** → a
  retrieve-and-copy baseline is promising (confirmed in §4–5, §10).
- **Three targets (ROUGE-L, ROUGE-1, LLM-judge)** ⇒ optimise for token overlap **and** semantic
  correctness; do not chase one metric in isolation.
""")

# --------------------------------------------------------------------------- #
md("""
## 2. Language / Subset Analysis
**Why.** Multilingual data is usually imbalanced; imbalance biases a pooled model toward the majority.
**What.** Per-subset counts, %, length stats, vocabulary, lexical diversity, script composition.
**How.** Drives sampling/loss-weighting, per-subset evaluation, and tokenizer/script checks.
""")
code("""
train = dfs["train"]
lang_tab = language.subset_table(train, has_answers=True)
io.save_table(lang_tab, "02_subset_summary_train", index=False)
display(lang_tab)
print("Imbalance:", language.imbalance_metrics(lang_tab))
display(language.script_table(train, C.INPUT_COL))
train_len = language.add_length_columns(train)
io.save_fig(viz.bar(lang_tab.set_index("subset")["n"], "Examples per subset (Train)",
                    "subset","count", color_by_subset=True), "02_counts_per_subset")
io.save_fig(viz.box_by_group(train_len, "q_words", C.SUBSET_COL,
                    "Question length (words) by subset","words"), "02_qlen_box_by_subset")
io.save_fig(viz.box_by_group(train_len, "a_words", C.SUBSET_COL,
                    "Answer length (words) by subset","words"), "02_alen_box_by_subset")
show_fig("02_counts_per_subset"); show_fig("02_alen_box_by_subset")
dec.record(Decision(id="language-imbalance", section="Language Analysis",
  observation="Subsets imbalanced (Eng_Uga 25.6% .. Amh_Eth 6.2%, 4.1x ratio).",
  evidence="02_subset_summary_train.csv; 02_counts_per_subset.png",
  impact="Pooled training over-serves English; low-resource langs risk poor quality yet count equally.",
  recommendation="Per-subset metrics; subset-balanced sampling/loss weighting; non-English-dominated retrieval.",
  priority="High", tags=["fine-tuning","evaluation","retrieval"]))
dec.record(Decision(id="amharic-script", section="Language Analysis",
  observation="Amharic uses Ge'ez script (94% Ethiopic chars); all others Latin.",
  evidence="02_script_profile_questions_train.csv",
  impact="Weak Ge'ez tokenizer coverage fragments Amharic; ROUGE behaves differently.",
  recommendation="Verify Ge'ez coverage; evaluate Amharic separately; script-aware normalization.",
  priority="High", tags=["retrieval","fine-tuning","evaluation","tokenization"]))
""")
implications("""
- **4.1× imbalance** and **very different answer lengths per subset** (Amharic ~20 words vs Eng_Uga ~97)
  → use **subset-balanced sampling** and **per-subset length targets**; always report metrics per subset.
- **Amharic is Ge'ez script** → confirm the embedder/LLM tokenizers cover Ge'ez, or Amharic silently degrades.
- High TTR for Amharic/Luganda → morphologically rich, small corpora → favour **subword** models & **dense** retrieval.
""")

# --------------------------------------------------------------------------- #
md("""
## 3. Question Analysis
**Why.** Templated inputs (few intents) favour retrieval/few-shot; diverse inputs favour generation.
**What.** First words, opening n-grams, interrogative templates, a templating/diversity signature.
**How.** Determines whether lexical matching suffices or we need semantic retrieval + clustering.
""")
code("""
q = train[C.INPUT_COL]
print("Diversity signature:", Q.diversity_signature(q))
display(Q.first_word_distribution(q).head(12))
display(Q.starting_ngrams(q, 3).head(12))
eng = train[train[C.SUBSET_COL].isin(["Eng_Uga","Eng_Gha","Eng_Eth","Eng_Ken"])][C.INPUT_COL]
tmpl = Q.template_coverage(eng); display(tmpl)
io.save_fig(viz.bar(tmpl.set_index("template")["pct"].head(12),
            "English question templates (%)","template","%"), "03_question_templates_english")
show_fig("03_question_templates_english")
dec.record(Decision(id="question-templating", section="Question Analysis",
  observation="English highly templated (what 30%, how 23%, is/are 10%); African langs longer & diverse.",
  evidence="03_question_templates_english.csv, 03_question_diversity_by_subset.csv",
  impact="Templated English clusters into few intents (retrieval/few-shot friendly); diverse languages need semantic matching.",
  recommendation="Use dense/semantic retrieval esp. for African languages; intent clustering for few-shot exemplars.",
  priority="Medium", tags=["retrieval","prompting"]))
""")
implications("""
- English is **highly templated** (~77% in a dozen interrogatives) → few intents → **retrieval & few-shot prompting** work well.
- Akan/Luganda/Amharic questions are **longer and more diverse** → **dense semantic** retrieval, not BM25 alone.
- Semantic intent clusters (§8) can seed **few-shot exemplars** per topic.
""")

# --------------------------------------------------------------------------- #
md("""
## 4. Answer Analysis  *(high priority)*
**Why.** The single biggest lever: are answers individually written or drawn from a canonical set?
**What.** Exact & near-duplicate reuse, reuse distribution, repeated closings/boilerplate.
**How.** If canonical, a retrieve-and-copy system can score high on ROUGE; generation only smooths gaps.
""")
code("""
a = train[C.OUTPUT_COL]
print("Exact reuse:", A.exact_reuse(a))
display(A.reuse_distribution(a, 10))
print("Near-duplicate (sampled):", A.near_duplicate_estimate(a, sample=5000))
display(A.repeated_sentences(a, 10))
# The Eng_Eth topic-tag prefix (free gold labels) — see decision below.
train = train.assign(gold_topic=T.parse_eng_eth_topics(train))
display(train["gold_topic"].value_counts().head(10).rename("Eng_Eth gold topic"))
dec.record(Decision(id="answer-duplication-canonical", section="Answer Analysis",
  observation="39% exact-duplicate answers, only 4.9% dup questions -> many questions share one answer.",
  evidence="01_train_duplicates.md; 04_answer_exact_reuse",
  impact="Answers come from a canonical bank; retrieve-and-copy can match gold verbatim (high ROUGE).",
  recommendation="Retrieval-first baseline; build corpus from unique answers; treat partly as answer selection.",
  priority="High", tags=["retrieval","reranking","post-processing"]))
dec.record(Decision(id="eng-eth-topic-prefix", section="Answer Analysis",
  observation="2,215 Eng_Eth answers (only) start 'This is a question about, <TOPIC>.' with a clean label.",
  evidence="04_answer_repeated_sentences.csv",
  impact="Subset-specific style rule AND near-free gold topic labels.",
  recommendation="Parse to a topic column; post-process Eng_Eth predictions to prepend the prefix; don't leak elsewhere.",
  priority="High", tags=["post-processing","topic-modelling","prompting","evaluation"]))
""")
implications("""
- Strong **canonical answer bank** (~40% exact, ~29% near-dup) → **retrieve-and-copy is a serious baseline**, not just a sanity check.
- **Eng_Eth's `This is a question about, X.` prefix** is a subset-specific **style rule** to reproduce (ROUGE) and a source of **free topic labels**.
- Repeated closings/trivia ⇒ answers are **assembled from reusable per-topic blocks** → generation should imitate this structure.
""")

# --------------------------------------------------------------------------- #
md("""
## 5. Duplicate Analysis
**Why.** Directly decides whether retrieval can win and bounds achievable accuracy.
**What.** Exact dup questions/answers/pairs; many-questions→one-answer; one-question→many-answers (ceiling).
**How.** Many→one supports answer-selection; one→many caps exact/ROUGE for any system.
""")
code("""
display(D.exact_duplicate_counts(train))
m2o, m2o_tbl = D.many_questions_one_answer(train); print("many-Q -> one-A:", m2o); display(m2o_tbl.head(8))
o2m, o2m_summary = D.one_question_many_answers(train); print("one-Q -> many-A (ceiling):", o2m)
dec.record(Decision(id="answer-ambiguity-ceiling", section="Duplicate Analysis",
  observation="~4% of identical questions have >1 distinct gold answer (max 4).",
  evidence="05_one_question_many_answers_top.csv",
  impact="Irreducible variance caps exact/ROUGE; the LLM judge may be fairer.",
  recommendation="Prefer semantic eval; pick the most frequent/representative answer among variants.",
  priority="Medium", tags=["evaluation","retrieval"]))
""")
implications("""
- **Many→one** (up to 68 questions per answer) ⇒ framing the task partly as **answer selection** is valid.
- **One→many is only ~4%** ⇒ the ambiguity ceiling is high; a well-chosen canonical answer matches gold most of the time.
- ⇒ **ROUGE is a mostly-fair metric** here (except morphology, §11/§13), so retrieval quality translates to leaderboard.
""")

# --------------------------------------------------------------------------- #
md("""
## 6–9. Embeddings, Topics & Embedding Exploration
**Why.** Semantic structure decides whether dense retrieval and topic-routing help. **What.** e5 embeddings
for all questions; automatic topic discovery (BERTopic); 2-D UMAP by language and by topic. **How.** If
topics cluster cleanly and semantically-similar questions co-locate, dense retrieval and topic filters help.
*(Embeddings are cached to `outputs/artifacts`; first run encodes ~39k texts.)*
""")
code("""
tr_q = train[C.INPUT_COL].tolist()
tr_emb  = E.encode(tr_q, "query", cache_name="train_q")
va_emb  = E.encode(dfs["val"][C.INPUT_COL].tolist(), "query", cache_name="val_q")
te_emb  = E.encode(dfs["test"][C.INPUT_COL].tolist(), "query", cache_name="test_q")
tmodel, tr_topics, info = T.run_bertopic(tr_q, tr_emb, min_topic_size=60)
n_topics = int((info["Topic"]!=-1).sum()); print("discovered topics:", n_topics)
dist = T.topic_distribution(tr_topics, train[C.SUBSET_COL].tolist(), tmodel)
io.save_table(dist, "08_topic_distribution", index=False); display(dist.head(15))
print("topic validation vs Eng_Eth gold:", T.validate_against_gold(tr_topics, train["gold_topic"]))
io.save_fig(viz.bar(dist[dist.topic_id!=-1].set_index("label")["pct"].head(15),
            "Discovered health topics (%)","topic","%",rotate=75,figsize=(11,5)), "08_topic_distribution")
show_fig("08_topic_distribution")
# UMAP (sampled)
rng = np.random.default_rng(C.SEED); sel = rng.choice(len(train), min(7000,len(train)), replace=False)
xy = T.umap_2d(tr_emb[sel])
io.save_fig(viz.scatter_2d(xy, train[C.SUBSET_COL].to_numpy()[sel],
            "UMAP of questions — by subset/language"), "09_umap_by_subset")
show_fig("09_umap_by_subset")
dec.record(Decision(id="topic-inventory", section="Topic Analysis",
  observation=f"BERTopic discovered {n_topics} data-driven health topics; purity vs Eng_Eth gold ~0.59.",
  evidence="08_topic_distribution.csv, 08_topic_validation.csv",
  impact="Clean topical structure enables topic-routed retrieval and topic-balanced evaluation.",
  recommendation="Store topic ids as metadata; topic filter in retrieval; report metrics per topic.",
  priority="Medium", tags=["retrieval","topic-modelling","evaluation"]))
""")
implications("""
- Automatic topics recover the **SRH curriculum** (HIV/AIDS, HPV, PrEP/PEP, mother-child, puberty, peer-pressure, digital literacy) → **topic-routed retrieval** and **per-topic reporting** are viable.
- UMAP shows **low-resource languages form their own regions** (script/lexicon), while English subsets overlap → cross-lingual retrieval is harder for Amharic/Akan (confirmed §10).
- Topic labels align moderately with Eng_Eth gold (purity ~0.59) → clusters are meaningful but not perfect; use as **soft** metadata.
""")

# --------------------------------------------------------------------------- #
md("""
## 10. Retrieval Feasibility  *(highest priority)*
**Why.** Establish, with **zero training**, how far pure retrieval gets. **What.** For every Val question,
retrieve the nearest Train question (BM25, dense e5, RRF hybrid), copy its answer, and score ROUGE-1/L,
exact-match and topic-match. **How.** Sets the baseline the full system must beat and reveals where copy fails.
""")
code("""
val = dfs["val"]; tr_ans = train[C.OUTPUT_COL].tolist(); va_ans = val[C.OUTPUT_COL].tolist()
tr_sub = train[C.SUBSET_COL].tolist(); va_sub = val[C.SUBSET_COL].tolist()
va_topics, _ = tmodel.transform(val[C.INPUT_COL].tolist(), embeddings=va_emb); va_topics = np.asarray(va_topics)
bm25 = R.BM25Retriever(tr_q); bm25_topk = bm25.topk(val[C.INPUT_COL].tolist(), k=5)
dense = R.DenseRetriever(tr_emb); dense_topk, dense_sim = dense.topk(va_emb, k=5)
hybrid_topk = R.rrf_fuse(bm25_topk, dense_topk, k=5)
res = {}
for name, topk, sim in [("BM25",bm25_topk,None),("dense_e5",dense_topk,dense_sim[:,0]),("hybrid",hybrid_topk,None)]:
    ev = R.evaluate(topk[:,0], tr_ans, va_ans, train_topics=tr_topics, query_topics=va_topics,
                    train_subset=tr_sub, query_subset=va_sub, dense_sim=sim)
    res[name] = R.summarize(ev)
headline = pd.concat({k:v.loc[["overall"]] for k,v in res.items()}); headline.index = headline.index.droplevel(1)
io.save_table(headline, "10_retrieval_headline"); print("HEADLINE (overall):"); display(headline.round(4))
print("Dense e5 per subset:"); display(res["dense_e5"].round(4))
dec.record(Decision(id="retrieval-feasibility", section="Retrieval Feasibility",
  observation="Zero-training retrieve-and-copy on Val reaches strong ROUGE with dense e5.",
  evidence="10_retrieval_headline.csv",
  impact="Retrieval-only is a strong baseline; generation must beat/refine it.",
  recommendation="Adopt dense retrieval baseline; use retrieved answer as RAG context; rerank top-k.",
  priority="High", tags=["retrieval","reranking","architecture"]))
dec.record(Decision(id="dense-beats-hybrid", section="Retrieval Feasibility",
  observation="Dense e5 > naive RRF hybrid > BM25; fusing weak BM25 hurt dense.",
  evidence="10_retrieval_headline.csv",
  impact="'Hybrid is best' is FALSE with untuned RRF; dense is the strongest signal.",
  recommendation="Default to dense; add BM25 only via TUNED fusion / rare-term fallback; validate any hybrid beats dense.",
  priority="High", tags=["retrieval","architecture"]))
dec.record(Decision(id="retrieval-quality-by-subset", section="Retrieval Feasibility",
  observation="ROUGE-L: Ken/Uga/Swa/Lug strong (0.47-0.66) vs Eng_Gha/Akan/Amharic weak (0.17-0.21).",
  evidence="10_retrieval_hybrid_summary.md",
  impact="Retrieval alone underperforms on Amharic/Akan/Eng_Gha; those need generation.",
  recommendation="Route by subset+confidence: copy where dense-sim high; generate for Amharic/Akan/Eng_Gha.",
  priority="High", tags=["architecture","retrieval","evaluation"]))
""")
implications("""
- **Dense e5 retrieve-and-copy is a strong zero-training baseline** and **beats the naive hybrid** — do not assume RRF hybrid wins; validate it.
- Retrieval **near-solves Kenya/Uganda/Swahili/Luganda** but **fails Amharic/Akan/Eng_Ghana** → a single strategy is wrong; **route by subset & confidence**.
- Use the retrieved answer as **RAG context** for a generator; add a **reranker** over top-k.
""")

# --------------------------------------------------------------------------- #
md("""
## 11. Similarity Analysis
**Why.** Understand the geometry: do languages/answers cluster, and do questions determine answers?
**What.** Subset-centroid similarity (Q & A), intra-subset density (redundancy), Q→A coupling vs shuffled.
**How.** A positive Q→A coupling gap validates question-based retrieval; dense clusters flag redundancy.
""")
code("""
tr_a_emb = E.encode(tr_ans, "passage", cache_name="train_a")
qq = SIM.centroid_similarity(tr_emb, np.array(tr_sub)); io.save_table(qq, "11_centroid_sim_questions")
display(SIM.intra_subset_density(tr_emb, np.array(tr_sub)))
print("Q-A coupling:", SIM.question_answer_coupling(tr_emb, tr_a_emb))
io.save_fig(viz.heatmap(qq.to_numpy(), list(qq.index),
            "Question centroid cosine (subset x subset)"), "11_centroid_sim_questions")
show_fig("11_centroid_sim_questions")
dec.record(Decision(id="qa-coupling", section="Similarity Analysis",
  observation="True Q-A cosine 0.87 vs shuffled 0.77 (gap 0.09).",
  evidence="11_qa_coupling.csv",
  impact="Questions determine answers -> question-based retrieval signal is real.",
  recommendation="Rely on question retrieval; optionally add answer-space reranking.",
  priority="Medium", tags=["retrieval","reranking"]))
""")
implications("""
- **Positive Q→A coupling gap** ⇒ questions genuinely predict answers → question-based retrieval is sound.
- **Low-resource subsets are the densest** (most self-similar questions) yet score worst on ROUGE (§13) → density ≠ easy; the bottleneck is surface scoring, not retrieval.
- Cross-subset centroid similarity guides **cross-lingual** vs **within-language** retrieval choices.
""")

# --------------------------------------------------------------------------- #
md("""
## 6b. Cross-language Analysis
**Why.** If subsets are translations, cross-lingual retrieval / pivot-through-English becomes powerful.
**What.** (1) Do ID hash-suffixes recur across subsets? (2) How close are non-English questions to their
nearest English question vs the English–English baseline? **How.** Determines whether we can share
answers across languages or must serve each language independently.
""")
code("""
sid = XL.shared_id_analysis(train); print("shared-ID analysis:", sid)
align = XL.embedding_alignment(tr_emb, np.array(tr_sub)); io.save_table(align, "06_embedding_alignment", index=False)
display(align)
dec.record(Decision(id="cross-lingual-parallelism", section="Cross-language Analysis",
  observation="NO parallelism: 0% shared IDs; non-Eng->Eng cosine ~0.85-0.87 vs 0.94-0.98 Eng-Eng.",
  evidence="06_shared_id_analysis, 06_embedding_alignment",
  impact="No aligned/parallel data; English answers are not guaranteed translations.",
  recommendation="No pivot-through-English assumption; retrieve within-language; treat cross-lingual reuse as topical.",
  priority="High", tags=["architecture","retrieval","data"]))
""")
implications("""
- **Not translations**: IDs are unique per subset and non-English questions are only ~0.86 cosine to English (vs 0.95+ English–English) → **do not build a pivot-through-English pipeline** assuming parallel data.
- Cross-lingual answer reuse is **topical, not exact** → prefer **within-language retrieval**, use cross-lingual only as a fallback.
""")

# --------------------------------------------------------------------------- #
md("""
## 7. Vocabulary Analysis · 12. Answer Style · 13. Language Complexity
**Why.** Tokenizer choice, OOV risk, and whether a generator should imitate style. **What.** Per-language
vocab/OOV/medical density; sentence counts, disclaimers, readability; words-per-sentence, TTR, Unicode.
**How.** High OOV → subword models & dense retrieval; rare disclaimers → don't add boilerplate; per-subset
length/style → per-subset generation targets.
""")
code("""
display(V.per_language_vocab(train, C.INPUT_COL))
display(V.oov_between_splits(train, dfs["val"], C.INPUT_COL))
display(ST.style_by_subset(train)); display(ST.disclaimer_frequency(train))
display(CX.complexity_by_subset(train, C.OUTPUT_COL))
display(CX.unicode_blocks(train, C.OUTPUT_COL))
dec.record(Decision(id="answer-style-and-disclaimers", section="Answer Style",
  observation="Disclaimers <1%; per-subset length/complexity varies widely; bullets <2%.",
  evidence="12_style_by_subset.md, 12_disclaimers.md",
  impact="Appending disclaimers/bullets diverges from gold and loses ROUGE.",
  recommendation="Match per-subset length & plain prose; suppress unsolicited disclaimers.",
  priority="Medium", tags=["prompting","fine-tuning","post-processing"]))
dec.record(Decision(id="oov-subword-tokenization", section="Vocabulary Analysis",
  observation="High train->val OOV: Amharic 37%/16% type/token, Lug 26%, Swa 25%, Aka 20%.",
  evidence="07_oov_train_val_questions.md",
  impact="Word-level features miss surface forms in morphologically rich languages; hurts BM25 & word heads.",
  recommendation="Use subword/byte tokenizers and dense retrieval for these languages.",
  priority="Medium", tags=["tokenization","retrieval","fine-tuning"]))
""")
implications("""
- **Severe OOV** for Amharic/Luganda/Swahili/Akan → **subword/byte-level** models & **dense** retrieval, not lexical.
- **Disclaimers are rare & bullets minimal** → a generator must **not** add safety boilerplate → protect ROUGE.
- **Per-subset length/complexity** varies 5× → set **per-subset max-length & style targets**.
""")

# --------------------------------------------------------------------------- #
md("""
## 14–15. Validation & Test Distribution
**Why.** Can we trust Val as a proxy for Test, and will retrieval generalise? **What.** Language mix,
question-length KS tests, and each Val/Test question's nearest-Train cosine. **How.** No shift ⇒ tune
thresholds on Val and trust them on Test; a low-similarity tail marks questions needing generation.
""")
code("""
display(DIST.language_mix(train, dfs["val"]))
display(DIST.length_ks(train, dfs["val"], C.INPUT_COL))
val_sim = DIST.nearest_train_similarity(va_emb, tr_emb, dfs["val"][C.SUBSET_COL].to_numpy())
test_sim = DIST.nearest_train_similarity(te_emb, tr_emb, dfs["test"][C.SUBSET_COL].to_numpy())
io.save_table(val_sim, "14_val_nearest_train_sim", index=False); io.save_table(test_sim, "15_test_nearest_train_sim", index=False)
print("Val nearest-train:"); display(val_sim); print("Test nearest-train:"); display(test_sim)
dec.record(Decision(id="no-distribution-shift", section="Validation / Test Distribution",
  observation="Val & Test ~= Train (nearest-Train cosine ~0.96, p10 0.93); language mix +/-5pp.",
  evidence="14_val_nearest_train_sim.md, 15_test_nearest_train_sim.md",
  impact="Low shift risk; Val is a reliable Test proxy; retrieval generalises; Eng_Gha most novel.",
  recommendation="Tune routing/rerank thresholds on Val and trust for Test; extra attention to Eng_Gha.",
  priority="High", tags=["evaluation","architecture","retrieval"]))
dec.record(Decision(id="rouge-morphology-gap", section="Retrieval Feasibility / Complexity",
  observation="Amharic/Akan: high topic-match (0.97-0.99) but low ROUGE (0.17-0.20); rich morphology, 16% Amharic token-OOV.",
  evidence="10_retrieval_hybrid_summary.md, 13_complexity_answers.md",
  impact="Word-ROUGE under-scores correct morphologically-rich answers; generation unlikely to beat copy on surface ROUGE.",
  recommendation="Report per-subset ROUGE; char/stemmed ROUGE diagnostics; optimise these subsets toward the LLM judge.",
  priority="High", tags=["evaluation","tokenization","architecture"]))
""")
implications("""
- **No distribution shift** → **tune all thresholds on Val**; results transfer to Test. **Eng_Gha** is the most novel/hardest.
- The **ROUGE/morphology gap** (right answer, low word-overlap) means Amharic/Akan should be **optimised toward the LLM judge**, not surface ROUGE.
""")

# --------------------------------------------------------------------------- #
md("""
## 16. Modelling Recommendations & 17. Final Report
Synthesise all evidence into `DATASET_ANALYSIS_REPORT.md` and print the prioritised
`MODELLING_DECISIONS.md`. See the report for the full recommended architecture and rationale.
""")
code("""
text = report.build_report()
print(f"Wrote {C.REPORT_MD} ({len(text):,} chars) and {C.DECISIONS_MD}")
print("\\n=== MODELLING_DECISIONS.md (head) ===\\n")
print(C.DECISIONS_MD.read_text()[:1500])
""")
md("""
### Recommended pipeline (evidence → decision)
1. **Dense e5 retrieval** over the Train corpus — *strongest zero-training baseline, canonical answers* (§4, §10).
2. **Rerank** top-k (cross-encoder / answer-side) — *Q→A coupling is real* (§11).
3. **Confidence routing** (nearest-Train cosine, tuned on Val≈Test): high→copy/edit; low→**generate (RAG)** — *per-subset variance & failure tail* (§10, §14).
4. **Multilingual instruction-tuned decoder LLM** for generation; **LoRA** for style/low-resource adaptation with **subset-balanced** data (§2).
5. **Per-subset post-processing**: length/style match, Eng_Eth prefix, no added disclaimers (§4, §12).
6. **Evaluate per subset** on ROUGE **and** an LLM-judge proxy; optimise Amharic/Akan toward the judge (§13).

**Bottom line:** a **hybrid retrieval + multilingual generation** system, routed by confidence and tuned per subset,
should outperform both pure retrieval (fails low-resource) and a pure generator (ignores the canonical bank the metric rewards).
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3 (health_eda)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
out = ROOT / "notebooks" / "health_qa_eda.ipynb"
nbf.write(nb, str(out))
print("wrote", out, "cells:", len(cells))
