"""Generate notebooks/colab_advanced_reranker_qwen3.ipynb — a fresh, standalone
Colab notebook for the two chase-1st experiments:
  Part 2: fine-tuned cross-encoder reranker (bge-reranker-v2-m3) over top-10
  Part 3: Qwen3-Embedding-0.6B backbone fine-tune + ensemble

It LOADS the fine-tuned dense models (fthn, fte) from Google Drive (saved by the
main notebook). Run the main notebook first (or ensure the Drive folder exists).

Run locally: python scripts/build_colab_advanced.py
Then open the .ipynb in Colab (A100), upload the 4 CSVs, mount Drive, Run All.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

cells=[]
def md(t): cells.append(new_markdown_cell(t.strip("\n")))
def code(t): cells.append(new_code_cell(t.strip("\n")))

md(r"""
# Health QA — Advanced: Fine-tuned Reranker + Qwen3 Backbone (Colab, A100)

Chase back to 1st. Two experiments, both Val-validated before any submission:
- **Part 2 — fine-tuned cross-encoder reranker** over the top-10 of your dense
  ensemble (highest EV; likely the leader's move).
- **Part 3 — Qwen3-Embedding-0.6B backbone** fine-tuned with the q→q recipe.

**Prereq:** the main notebook already saved `fthn_bgem3/` and `fte_e5large_qq/`
to `MyDrive/health_qa_models/`. This notebook loads them from there.

**Run:** A100 runtime → Run Part 0 (upload the 4 CSVs + mount Drive) → Parts 1–3
(read each `rouge_wtd`) → Part 4 (set `BEST`, generate submission).
**Beat 0.4061 on Val** (current 1st-equiv). Est LB ≈ Val + ~0.23 (so 0.417→~0.65).
""")

# ---- Part 0 -------------------------------------------------------------- #
md("## Part 0 — Setup (install, CSVs, Drive, data, helpers)")
code(r"""
!pip -q install "sentence-transformers>=5.0" "rouge-score>=0.1.2" "accelerate>=1.1.0" datasets
import torch; print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
""")
code(r"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
need=["Train.csv","Val.csv","Test.csv","SampleSubmission.csv"]
if [f for f in need if not os.path.exists(f)]:
    from google.colab import files; files.upload()
from google.colab import drive; drive.mount('/content/drive')
DRV="/content/drive/MyDrive/health_qa_models"; assert os.path.isdir(DRV), f"missing {DRV} - run the main notebook first"
print("Drive models:", os.listdir(DRV))
""")
code(r"""
import numpy as np, pandas as pd, random, torch
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
ID,INP,OUT,SUB="ID","input","output","subset"
def load(f): return pd.read_csv(f,dtype=str,keep_default_na=False).replace({"":pd.NA})
train=load("Train.csv").reset_index(drop=True); val=load("Val.csv").reset_index(drop=True); test=load("Test.csv").reset_index(drop=True)
print("shapes:", train.shape, val.shape, test.shape)

from rouge_score import rouge_scorer
_sc=rouge_scorer.RougeScorer(["rouge1","rougeL"],use_stemmer=False); W1,WL,WLLM=0.37,0.37,0.26
def score_frame(preds,golds,subs):
    rows=[{SUB:s,"r1":(o:=_sc.score(g,p))["rouge1"].fmeasure,"rL":o["rougeL"].fmeasure} for p,g,s in zip(preds,golds,subs)]
    df=pd.DataFrame(rows); out=pd.concat([df[["r1","rL"]].mean().to_frame("overall").T, df.groupby(SUB)[["r1","rL"]].mean()])
    out["rouge_wtd"]=W1*out.r1+WL*out.rL; return out.round(4)
def est_lb(rw, llm=0.77): return round(rw+WLLM*llm,4)
print("setup ok")
""")
code(r"""
from sentence_transformers import SentenceTransformer
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
def ensemble_topk(specs,q_df,c_df,k=10):
    c_sub=c_df[SUB].to_numpy(); q_sub=q_df[SUB].to_numpy()
    c_e={n:enc(n,c_df[INP].tolist(),"query") for n,sp,w in specs}; q_e={n:enc(n,q_df[INP].tolist(),"query") for n,sp,w in specs}
    n_=len(q_df); cand=np.empty((n_,k),dtype=np.int64)
    for s in range(0,n_,128):
        acc=None
        for n,sp,w in specs: sc=q_e[n][s:s+128]@c_e[n].T; acc=w*sc if acc is None else acc+w*sc
        acc=np.where(c_sub[None,:]==q_sub[s:s+128][:,None],acc,-1e9)
        for r in range(acc.shape[0]):
            t=np.argpartition(-acc[r],k)[:k]; cand[s+r]=t[np.argsort(-acc[r][t])]
    return cand
def eval_specs(specs):
    top=ensemble_top1(specs,val,train); return score_frame([train[OUT].tolist()[j] for j in top],val[OUT].tolist(),val[SUB].tolist())
def qq_pairs(prefix=""):
    from itertools import combinations; from collections import defaultdict
    by=defaultdict(list)
    for q,a in zip(train[INP],train[OUT]):
        if pd.isna(q) or pd.isna(a): continue
        by[a].append(prefix+str(q))
    from sentence_transformers import InputExample
    ex=[]
    for a,qs in by.items():
        qs=list(dict.fromkeys(qs))
        if len(qs)<2: continue
        cb=list(combinations(qs,2)); random.shuffle(cb)
        for c in cb[:4]: ex.append(InputExample(texts=[c[0],c[1]]))
    return ex
def make_submission(top_idx, corpus_ans, fname):
    import re
    def clean(x):
        x="" if x is None or isinstance(x,float) else str(x); x=re.sub(r"\s+"," ",x.replace("\r"," ").replace("\n"," ")).strip(); return x or "N/A"
    ans=[clean(corpus_ans[j]) for j in top_idx]; sub=pd.DataFrame({ID:test[ID]})
    for c in ["TargetRLF1","TargetR1F1","TargetLLM"]: sub[c]=ans
    samp=load("SampleSubmission.csv"); assert sub[ID].tolist()==samp[ID].tolist() and (sub["TargetRLF1"].str.len()>0).all()
    sub.to_csv(fname,index=False,encoding="utf-8"); print("wrote",fname,len(sub))
    from google.colab import files; files.download(fname)
print("helpers ok")
""")

# ---- Part 1 -------------------------------------------------------------- #
md("## Part 1 — Load fine-tuned dense models from Drive (sanity check)")
code(r"""
MODELS["fthn"]=(SentenceTransformer(f"{DRV}/fthn_bgem3",device="cuda"), False)   # bge, raw
MODELS["fte"] =(SentenceTransformer(f"{DRV}/fte_e5large_qq",device="cuda"), True) # e5, 'query:' prefix
sf=eval_specs([("fthn","question",.5),("fte","question",.5)])
print("fthn+fte Val rouge_wtd=", sf.loc['overall','rouge_wtd'], "(expect ~0.4061)")
""")

# ---- Part 2 -------------------------------------------------------------- #
md("## Part 2 — Fine-tuned cross-encoder reranker (train → rerank top-10 → eval)")
code(r"""
import gc
from collections import defaultdict
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
random.seed(42); gc.collect(); torch.cuda.empty_cache()

qtr=train[INP].tolist(); atr=train[OUT].tolist(); subtr=train[SUB].to_numpy()
qE=enc("fthn", qtr, "query")                       # mine with the fine-tuned model
by=defaultdict(list)
for i,(q,a) in enumerate(zip(qtr,atr)):
    if pd.isna(q) or pd.isna(a): continue
    by[a].append(i)
ce_pairs=[]; N=len(qtr)
for s in range(0,N,512):
    idx=np.arange(s,min(s+512,N)); sims=np.where(subtr[None,:]==subtr[idx][:,None], qE[idx]@qE.T, -1e9)
    for r,i in enumerate(idx):
        grp=by.get(atr[i],[])
        if len(grp)<2: continue
        pos=random.choice([j for j in grp if j!=i]); ce_pairs.append(InputExample(texts=[qtr[i],qtr[pos]],label=1.0))
        for j in np.argpartition(-sims[r],10)[:10]:
            if j!=i and atr[j]!=atr[i]: ce_pairs.append(InputExample(texts=[qtr[i],qtr[j]],label=0.0)); break
print("reranker training pairs:", len(ce_pairs))
ce=CrossEncoder("BAAI/bge-reranker-v2-m3", num_labels=1, device="cuda", max_length=192)
ce.fit(train_dataloader=DataLoader(ce_pairs,shuffle=True,batch_size=32), epochs=1,
       warmup_steps=int(0.1*len(ce_pairs)/32), use_amp=True, show_progress_bar=True)
ce.save(f"{DRV}/ce_reranker"); print("reranker trained")
""")
code(r"""
specs=[("fthn","question",.5),("fte","question",.5)]
cand=ensemble_topk(specs, val, train, 10)
cq=train[INP].tolist(); ca=train[OUT].tolist(); vq=val[INP].tolist()
pairs=[[vq[i],cq[cand[i,r]]] for i in range(len(val)) for r in range(10)]
sc=np.array(ce.predict(pairs,batch_size=128,show_progress_bar=True)).reshape(len(val),10)
best=cand[np.arange(len(val)), sc.argmax(1)]
sf=score_frame([ca[j] for j in best], val[OUT].tolist(), val[SUB].tolist())
print(sf.to_string()); print("RERANK(FT) rouge_wtd=", round(sf.loc['overall','rouge_wtd'],4),
      "| est LB=", est_lb(sf.loc['overall','rouge_wtd']), "| beat 0.4061?")
""")

# ---- Part 3 -------------------------------------------------------------- #
md("## Part 3 — Qwen3-Embedding-0.6B backbone (fine-tune q→q + ensemble)")
code(r"""
from sentence_transformers import losses
from torch.utils.data import DataLoader
gc.collect(); torch.cuda.empty_cache()
qw=SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda", model_kwargs={"torch_dtype":torch.bfloat16})
qw.max_seq_length=128
ex=qq_pairs()  # raw q->q same-answer pairs
qw.fit(train_objectives=[(DataLoader(ex,shuffle=True,batch_size=32,drop_last=True), losses.MultipleNegativesRankingLoss(qw))],
       epochs=1, warmup_steps=int(0.1*len(ex)/32), optimizer_params={"lr":1e-5}, use_amp=False, show_progress_bar=True)  # bf16 -> no GradScaler
MODELS["qw"]=(qw, False); _EMB.clear(); qw.save(f"{DRV}/qw_qwen3_qq")
for tag,specs in [("qw", [("qw","question",1.0)]),
                  ("fthn+qw", [("fthn","question",.5),("qw","question",.5)]),
                  ("fthn+fte+qw", [("fthn","question",1/3),("fte","question",1/3),("qw","question",1/3)])]:
    print(tag, "->", round(eval_specs(specs).loc['overall','rouge_wtd'],4))
""")

# ---- Part 4 -------------------------------------------------------------- #
md(r"""## Part 4 — Generate the submission from the BEST Val config
Set `BEST`. Options: `"rerank"`, or any dense spec like
`[("fthn","question",.5),("fte","question",.5)]`. Uses train+val corpus for Test.""")
code(r"""
BEST = "rerank"    # or a dense spec list, e.g. [("fthn","question",1/3),("fte","question",1/3),("qw","question",1/3)]
corpus=pd.concat([train,val],ignore_index=True); corpus_ans=corpus[OUT].tolist()

if BEST=="rerank":
    cand=ensemble_topk([("fthn","question",.5),("fte","question",.5)], test, corpus, 10)
    cq=corpus[INP].tolist(); tq=test[INP].tolist()
    pairs=[[tq[i],cq[cand[i,r]]] for i in range(len(test)) for r in range(10)]
    sc=np.array(ce.predict(pairs,batch_size=128,show_progress_bar=True)).reshape(len(test),10)
    top=cand[np.arange(len(test)), sc.argmax(1)]
else:
    top=ensemble_top1(BEST, test, corpus)
make_submission(top, corpus_ans, "submission_advanced.csv")
print("val-sourced test answers:", int((top>=len(train)).sum()), "/", len(test))
""")

# ---- Part 5: Stage-1 gate (off-the-shelf Qwen3-4B) ----------------------- #
md(r"""## Part 5 — Stage 1 GATE: does a bigger backbone help *at all*?
Load **off-the-shelf** Qwen3-Embedding-4B (no training) and test whether adding it
to the ensemble beats `fthn+fte` on a Val sample. **If it doesn't help off-the-shelf,
stop — LoRA won't save it** and moselim's edge is elsewhere. ~6–8 min.""")
code(r"""
import numpy as np
qw4=SentenceTransformer("Qwen/Qwen3-Embedding-4B", device="cuda",
     model_kwargs={"torch_dtype":torch.bfloat16, "attn_implementation":"sdpa"})
qw4.max_seq_length=128; MODELS["qw4"]=(qw4, False)

rng=np.random.default_rng(42); idx=[]
for s,g in val.groupby(SUB): idx+=list(rng.choice(g.index.to_numpy(), min(250,len(g)), replace=False))
vs=val.loc[idx].reset_index(drop=True)
def eval_on(specs, q_df):
    top=ensemble_top1(specs, q_df, train)
    return round(score_frame([train[OUT].tolist()[j] for j in top], q_df[OUT].tolist(), q_df[SUB].tolist()).loc["overall","rouge_wtd"],4)

base=eval_on([("fthn","question",.5),("fte","question",.5)], vs)
print("baseline fthn+fte (sample):", base)
for tag,specs in [("fthn+fte+qw4 (1/3)", [("fthn","question",1/3),("fte","question",1/3),("qw4","question",1/3)]),
                  ("fthn+fte+qw4 (.4/.4/.2)", [("fthn","question",.4),("fte","question",.4),("qw4","question",.2)])]:
    print(tag, "->", eval_on(specs, vs))
print("\nGATE: if the +qw4 rows don't clearly beat baseline, SKIP Part 6 (LoRA won't help).")
""")

# ---- Part 6: Stage-2 LoRA fine-tune Qwen3-4B ----------------------------- #
md(r"""## Part 6 — Stage 2: LoRA fine-tune Qwen3-4B *(only if Stage 1 was promising)*
Quick LoRA (r=16) on q→q pairs, bf16, short seq. Then full-Val eval vs 0.4061.
~15–20 min.""")
code(r"""
!pip -q install peft
import gc
from peft import LoraConfig
from sentence_transformers import losses
from torch.utils.data import DataLoader
MODELS.pop("qw4", None); gc.collect(); torch.cuda.empty_cache()   # free the off-the-shelf 4B

qw4l=SentenceTransformer("Qwen/Qwen3-Embedding-4B", device="cuda",
      model_kwargs={"torch_dtype":torch.bfloat16, "attn_implementation":"sdpa"})
qw4l.add_adapter(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
      target_modules=["q_proj","k_proj","v_proj","o_proj"]))
qw4l.max_seq_length=128
ex=qq_pairs()
qw4l.fit(train_objectives=[(DataLoader(ex,shuffle=True,batch_size=32,drop_last=True),
          losses.MultipleNegativesRankingLoss(qw4l))],
         epochs=1, warmup_steps=20, optimizer_params={"lr":2e-4}, use_amp=False, show_progress_bar=True)
MODELS["qw4l"]=(qw4l, False); _EMB.clear(); qw4l.save(f"{DRV}/qw4l_lora")
for tag,specs in [("qw4l", [("qw4l","question",1.0)]),
                  ("fthn+qw4l", [("fthn","question",.5),("qw4l","question",.5)]),
                  ("fthn+fte+qw4l", [("fthn","question",1/3),("fte","question",1/3),("qw4l","question",1/3)])]:
    sf=eval_specs(specs); print(tag, "->", round(sf.loc['overall','rouge_wtd'],4),
        "| est LB=", est_lb(sf.loc['overall','rouge_wtd']), "(beat 0.4061?)")
""")
md(r"""### Submission from a Qwen3-4B config
If a `qw4l` config beat 0.4061, set `BEST` in the **Part 4** cell to that spec, e.g.
`[("fthn","question",1/3),("fte","question",1/3),("qw4l","question",1/3)]`, and re-run Part 4.""")

nb=new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                                       "language_info":{"name":"python"},"accelerator":"GPU"})
out=Path(__file__).resolve().parents[1]/"notebooks"/"colab_advanced_reranker_qwen3.ipynb"
nbf.write(nb,str(out)); print("wrote", out, "cells:", len(cells))
