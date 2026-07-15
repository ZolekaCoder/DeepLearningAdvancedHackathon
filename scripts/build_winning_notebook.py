"""Generate notebooks/winning_solution.ipynb — the definitive, reproducible
winning pipeline (public 0.6567): a 4-way ensemble of q->q + hard-negative
fine-tuned dense retrievers, retrieve-and-copy, within-subset, train+val corpus.

Self-contained Colab notebook (A100 recommended). Run locally to (re)generate:
    python scripts/build_winning_notebook.py
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

cells=[]
def md(t): cells.append(new_markdown_cell(t.strip("\n")))
def code(t): cells.append(new_code_cell(t.strip("\n")))

md(r"""
# Winning Solution — IndabaX Multilingual Health QA (public 0.6567)

**Approach:** retrieve-and-copy (no generation). For each test question, retrieve
the most similar train/val question with a fine-tuned multilingual dense retriever
and copy its answer verbatim. Winner = **4-way ensemble** of two bge-m3 and two
e5-large models, each fine-tuned on **question→question same-answer pairs + hard
negatives**, with **within-subset (language-safe)** retrieval over the **train+val**
answer pool.

**Why this wins:** ~39% of answers are canonical duplicates → gold answers live in
the pool → copying the right one maximises ROUGE (74% of the metric). See
`SOLUTION.md` and `DATASET_ANALYSIS_REPORT.md`.

**Run:** GPU (A100) runtime → Run All → upload the 4 CSVs when prompted. End-to-end
~50-70 min (4 fine-tunes + encoding). Writes `submission_winning.csv`.
""")

# ---- Part 0 -------------------------------------------------------------- #
md("## Part 0 — Setup, data, official ROUGE, helpers")
code(r"""
!pip -q install "sentence-transformers>=3.0" "rouge-score>=0.1.2" "accelerate>=1.1.0" datasets
import torch; print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
""")
code(r"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
need=["Train.csv","Val.csv","Test.csv","SampleSubmission.csv"]
if [f for f in need if not os.path.exists(f)]:
    from google.colab import files; files.upload()
print("present:", [f for f in need if os.path.exists(f)])
""")
code(r"""
import numpy as np, pandas as pd, random, torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
ID,INP,OUT,SUB="ID","input","output","subset"
def load(f): return pd.read_csv(f,dtype=str,keep_default_na=False).replace({"":pd.NA})
train=load("Train.csv").reset_index(drop=True); val=load("Val.csv").reset_index(drop=True); test=load("Test.csv").reset_index(drop=True)
print("shapes:", train.shape, val.shape, test.shape)

from rouge_score import rouge_scorer
_sc=rouge_scorer.RougeScorer(["rouge1","rougeL"],use_stemmer=False); W1,WL=0.37,0.37
def score_frame(preds,golds,subs):
    rows=[{SUB:s,"r1":(o:=_sc.score(g,p))["rouge1"].fmeasure,"rL":o["rougeL"].fmeasure} for p,g,s in zip(preds,golds,subs)]
    df=pd.DataFrame(rows); out=pd.concat([df[["r1","rL"]].mean().to_frame("overall").T, df.groupby(SUB)[["r1","rL"]].mean()])
    out["rouge_wtd"]=W1*out.r1+WL*out.rL; return out.round(4)
print("setup ok")
""")
code(r"""
MODELS={}; _EMB={}
def enc(name, texts, kind, bs=128):
    key=(name,kind,len(texts),str(texts[0])[:40] if texts else "",str(texts[-1])[:40] if texts else "")
    if key in _EMB: return _EMB[key]
    m,is_e5=MODELS[name]; pre=("query: " if kind=="query" else "passage: ") if is_e5 else ""
    e=m.encode([pre+str(t) for t in texts], batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    _EMB[key]=e; return e
def ensemble_top1(specs,q_df,c_df):
    c_sub=c_df[SUB].to_numpy(); q_sub=q_df[SUB].to_numpy()
    c_e={n:enc(n,c_df[INP].tolist(),"query") for n,sp,w in specs}; q_e={n:enc(n,q_df[INP].tolist(),"query") for n,sp,w in specs}
    n_=len(q_df); top=np.empty(n_,dtype=np.int64)
    for s in range(0,n_,128):
        acc=None
        for n,sp,w in specs: sc=q_e[n][s:s+128]@c_e[n].T; acc=w*sc if acc is None else acc+w*sc
        top[s:s+128]=np.where(c_sub[None,:]==q_sub[s:s+128][:,None],acc,-1e9).argmax(1)
    return top
def eval_specs(specs):
    top=ensemble_top1(specs,val,train); return score_frame([train[OUT].tolist()[j] for j in top],val[OUT].tolist(),val[SUB].tolist())

def qq_pairs(prefix=""):
    from itertools import combinations
    by=defaultdict(list)
    for q,a in zip(train[INP],train[OUT]):
        if pd.isna(q) or pd.isna(a): continue
        by[a].append(prefix+str(q))
    ex=[]
    for a,qs in by.items():
        qs=list(dict.fromkeys(qs))
        if len(qs)<2: continue
        cb=list(combinations(qs,2)); random.shuffle(cb)
        for c in cb[:4]: ex.append(InputExample(texts=[c[0],c[1]]))
    return ex

def mine_triplets(miner, pos_per=1, neg_per=1, k=25, neardup=0.9, prefix=""):
    # (anchor, same-answer positive, similar-but-wrong-answer hard negative).
    # Hard negatives filtered to exclude near-duplicate answers (avoid false negs).
    qtr=train[INP].tolist(); atr=train[OUT].tolist(); subtr=train[SUB].to_numpy()
    qE=enc(miner, qtr, "query"); aE=enc(miner, atr, "passage")
    by=defaultdict(list)
    for i,(q,a) in enumerate(zip(qtr,atr)):
        if pd.isna(q) or pd.isna(a): continue
        by[a].append(i)
    trip=[]; N=len(qtr)
    for s in range(0,N,512):
        idx=np.arange(s,min(s+512,N)); sims=np.where(subtr[None,:]==subtr[idx][:,None], qE[idx]@qE.T, -1e9)
        for r,i in enumerate(idx):
            grp=[j for j in by.get(atr[i],[]) if j!=i]
            if not grp: continue
            order=np.argpartition(-sims[r],k)[:k]; order=order[np.argsort(-sims[r][order])]
            negs=[j for j in order if atr[j]!=atr[i] and float(aE[i]@aE[j])<neardup][:neg_per]
            for pos in random.sample(grp, min(pos_per,len(grp))):
                for neg in negs:
                    trip.append(InputExample(texts=[prefix+qtr[i], prefix+qtr[pos], prefix+qtr[neg]]))
    return trip

def train_model(base_id, examples, epochs, is_e5, save_name, lr=1e-5, batch=32, maxlen=128):
    m=SentenceTransformer(base_id, device="cuda"); m.max_seq_length=maxlen
    m.fit(train_objectives=[(DataLoader(examples,shuffle=True,batch_size=batch,drop_last=True),
          losses.MultipleNegativesRankingLoss(m))],
          epochs=epochs, warmup_steps=max(10,int(0.1*len(examples)/batch)),
          optimizer_params={"lr":lr}, use_amp=True, show_progress_bar=True)
    m.save(save_name); return m
def make_submission(top_idx, corpus_ans, fname):
    import re
    def clean(x):
        x="" if x is None or isinstance(x,float) else str(x); return re.sub(r"\s+"," ",x.replace("\r"," ").replace("\n"," ")).strip() or "N/A"
    ans=[clean(corpus_ans[j]) for j in top_idx]; sub=pd.DataFrame({ID:test[ID]})
    for c in ["TargetRLF1","TargetR1F1","TargetLLM"]: sub[c]=ans
    sub.to_csv(fname,index=False,encoding="utf-8"); print("wrote",fname,len(sub))
    from google.colab import files; files.download(fname)
print("helpers ok")
""")

# ---- Part 1 -------------------------------------------------------------- #
md("""## Part 1 — Fine-tune the four retrievers
`fthn`/`fthn2` = bge-m3 (round-1 / round-2 hard-neg); `fte`/`fte2` = e5-large
(q→q / round-2 hard-neg). Round-2 mines hard negatives with the round-1 model.""")
code(r"""
BGE="BAAI/bge-m3"; E5="intfloat/multilingual-e5-large"

# fthn: bge-m3, hard negatives mined with OFF-THE-SHELF bge, 1 epoch
MODELS["bge_base"]=(SentenceTransformer(BGE, device="cuda"), False)
fthn=train_model(BGE, mine_triplets("bge_base", pos_per=1, neg_per=1), epochs=1, is_e5=False, save_name="fthn_bgem3")
MODELS["fthn"]=(fthn, False); MODELS.pop("bge_base", None); _EMB.clear()
print("fthn done")
""")
code(r"""
# fthn2: bge-m3, round-2 hard negatives mined with fthn, 2 pos/2 neg, 2 epochs
fthn2=train_model(BGE, mine_triplets("fthn", pos_per=2, neg_per=2), epochs=2, is_e5=False, save_name="fthn2_bgem3")
MODELS["fthn2"]=(fthn2, False); _EMB.clear(); print("fthn2 done")
""")
code(r"""
# fte: e5-large, q->q same-answer pairs, 1 epoch (e5 'query:' prefix baked in)
fte=train_model(E5, qq_pairs(prefix="query: "), epochs=1, is_e5=True, save_name="fte_e5large_qq")
MODELS["fte"]=(fte, True); _EMB.clear(); print("fte done")
""")
code(r"""
# fte2: e5-large, round-2 hard negatives mined with fte, 2 pos/2 neg, 2 epochs
fte2=train_model(E5, mine_triplets("fte", pos_per=2, neg_per=2, prefix="query: "), epochs=2, is_e5=True, save_name="fte2_e5large")
MODELS["fte2"]=(fte2, True); _EMB.clear(); print("fte2 done")
""")

# ---- Part 2 -------------------------------------------------------------- #
md("## Part 2 — 4-way ensemble: Val check + Test submission")
code(r"""
WIN=[("fthn2","question",.3),("fthn","question",.2),("fte","question",.2),("fte2","question",.3)]
sf=eval_specs(WIN)
print(sf.to_string()); print("\nWINNING 4-way Val rouge_wtd =", sf.loc['overall','rouge_wtd'], "(public 0.6567)")
""")
code(r"""
# Test submission: retrieve over the COMBINED train+val corpus (more canonical candidates)
corpus=pd.concat([train,val],ignore_index=True); corpus_ans=corpus[OUT].tolist()
top=ensemble_top1(WIN, test, corpus)
make_submission(top, corpus_ans, "submission_winning.csv")
print("val-sourced test answers:", int((top>=len(train)).sum()), "/", len(test))
""")

nb=new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                                       "language_info":{"name":"python"},"accelerator":"GPU"})
out=Path(__file__).resolve().parents[1]/"notebooks"/"winning_solution.ipynb"
nbf.write(nb,str(out)); print("wrote", out, "cells:", len(cells))
