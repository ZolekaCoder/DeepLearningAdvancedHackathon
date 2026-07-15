"""Generate notebooks/colab_finetune_retriever.ipynb — self-contained Colab
notebook (no dependency on src/health_eda) for the retrieval system on GPU.

Parts:
  0 setup (install, upload CSVs, data, official ROUGE, cached helpers)
  1 off-the-shelf bge-m3 + e5-large + ensemble  (baseline, Val ~0.3705)
  2 strong cross-encoder reranker               (optional; was flat)
  3 fine-tune bge-m3 q->q (same-answer)          (Val ~0.3918)  <- current 1st place
  4 hard-negative fine-tune bge-m3               (mine similar-but-wrong, filtered)
  5 fine-tune e5-large q->q + weight-swept ensembles
  6 RRF vs score-averaging sanity check
  7 generate the Test submission from the BEST Val config

Run locally: python scripts/build_colab_notebook.py
Recommended Colab GPU: A100 (holds several models at once).
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

cells = []
def md(t): cells.append(new_markdown_cell(t.strip("\n")))
def code(t): cells.append(new_code_cell(t.strip("\n")))

md(r"""
# Multilingual Health QA — Retrieval System (Colab, GPU)

Retrieval-based solution; writes a submission CSV. Open-source & reproducible
(GPU only speeds things up). **Use an A100** (several models are held at once).

**Run order:** Part 0 (upload the 4 CSVs) → Parts 1,3,4,5 (read each `rouge_wtd`) →
Part 7 (set `BEST` to the winner, run → downloads `submission.csv`). Part 2
(reranker) and Part 6 (RRF) are optional sanity checks.

**Metric:** 0.37·ROUGE-1 + 0.37·ROUGE-L + 0.26·LLM-judge (rouge-score lib strips
non-ASCII → Amharic ROUGE≈0, expected). **Calibration:** actual LB ≈ Val
`rouge_wtd` (train-only) + ~0.217. Current 1st = q->q fine-tune, Val 0.3932 → LB 0.628.
""")

# ---- Part 0 -------------------------------------------------------------- #
md("## Part 0 — Setup")
code(r"""
!pip -q install "sentence-transformers>=3.0" "rouge-score>=0.1.2" "accelerate>=1.1.0" datasets
import torch; print("CUDA:", torch.cuda.is_available(),
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
""")
code(r"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
need=["Train.csv","Val.csv","Test.csv","SampleSubmission.csv"]
missing=[f for f in need if not os.path.exists(f)]
if missing:
    from google.colab import files; print("Upload:", missing); files.upload()
print("present:", [f for f in need if os.path.exists(f)])
""")
code(r"""
import numpy as np, pandas as pd, random, torch
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
ID,INP,OUT,SUB="ID","input","output","subset"
def load(f): return pd.read_csv(f,dtype=str,keep_default_na=False).replace({"":pd.NA})
train=load("Train.csv").reset_index(drop=True)
val  =load("Val.csv").reset_index(drop=True)
test =load("Test.csv").reset_index(drop=True)
print("shapes:", train.shape, val.shape, test.shape)

from rouge_score import rouge_scorer
_sc=rouge_scorer.RougeScorer(["rouge1","rougeL"],use_stemmer=False)
W1,WL,WLLM=0.37,0.37,0.26
def score_frame(preds,golds,subs):
    rows=[{SUB:s,"r1":(o:=_sc.score(g,p))["rouge1"].fmeasure,"rL":o["rougeL"].fmeasure}
          for p,g,s in zip(preds,golds,subs)]
    df=pd.DataFrame(rows)
    out=pd.concat([df[["r1","rL"]].mean().to_frame("overall").T, df.groupby(SUB)[["r1","rL"]].mean()])
    out["rouge_wtd"]=W1*out.r1+WL*out.rL
    return out.round(4)
def est_lb(rw, llm=0.72): return round(rw+WLLM*llm,4)
print("setup ok")
""")

md("### Helpers (model cache, cached encoding, retrieval, submission)")
code(r"""
from sentence_transformers import SentenceTransformer
MODELS={}      # name -> (model, is_e5)
_EMB={}        # (name,kind,len,first,last) -> embeddings  (avoid recompute in sweeps)
def get_model(name, mid, is_e5):
    if name not in MODELS:
        print("loading", mid); MODELS[name]=(SentenceTransformer(mid, device="cuda"), is_e5)
    return MODELS[name]
def enc(name, texts, kind, bs=256):
    key=(name, kind, len(texts), str(texts[0])[:40] if texts else "", str(texts[-1])[:40] if texts else "")
    if key in _EMB: return _EMB[key]
    m,is_e5=MODELS[name]
    pre=("query: " if kind=="query" else "passage: ") if is_e5 else ""
    e=m.encode([pre+str(t) for t in texts], batch_size=bs, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=True)
    _EMB[key]=e; return e

def ensemble_top1(specs, q_df, c_df):
    # specs: list of (name, space, weight); space in {"question","answer"}
    c_sub=c_df[SUB].to_numpy(); q_sub=q_df[SUB].to_numpy()
    c_embs={(n,sp):enc(n,(c_df[OUT] if sp=="answer" else c_df[INP]).tolist(),
                       "passage" if sp=="answer" else "query") for n,sp,w in specs}
    q_embs={n:enc(n,q_df[INP].tolist(),"query") for n,sp,w in specs}
    n_=len(q_df); top=np.empty(n_,dtype=np.int64)
    for s in range(0,n_,128):
        acc=None
        for n,sp,w in specs:
            sc=q_embs[n][s:s+128]@c_embs[(n,sp)].T
            acc=w*sc if acc is None else acc+w*sc
        m=(c_sub[None,:]==q_sub[s:s+128][:,None])
        top[s:s+128]=np.where(m,acc,-1e9).argmax(1)
    return top
def eval_specs(specs):
    top=ensemble_top1(specs, val, train)
    return score_frame([train[OUT].tolist()[j] for j in top], val[OUT].tolist(), val[SUB].tolist())
def rw_of(specs): return eval_specs(specs).loc["overall","rouge_wtd"]

def make_submission(top_idx, corpus_ans, fname="submission.csv"):
    def clean(x):
        import re
        x="" if x is None or (isinstance(x,float)) else str(x)
        x=re.sub(r"\s+"," ", x.replace("\r"," ").replace("\n"," ")).strip()
        return x if x else "N/A"
    ans=[clean(corpus_ans[j]) for j in top_idx]
    sub=pd.DataFrame({ID:test[ID]})
    for c in ["TargetRLF1","TargetR1F1","TargetLLM"]: sub[c]=ans
    samp=load("SampleSubmission.csv")
    assert list(sub.columns)==list(samp.columns) and sub[ID].tolist()==samp[ID].tolist()
    assert (sub["TargetRLF1"]==sub["TargetLLM"]).all() and (sub["TargetRLF1"].str.len()>0).all()
    sub.to_csv(fname,index=False,encoding="utf-8"); print("wrote",fname,"rows",len(sub))
    from google.colab import files; files.download(fname)
print("helpers ok")
""")

md("""### (Recommended) Persist models + submissions to Google Drive
Survives Colab disconnects and gives you the reproducible model artifacts for the
top-10 code review. Set `USE_DRIVE=False` to skip. Call `backup_to_drive()` after
training checkpoints (already wired into Parts 5 & 9).""")
code(r"""
USE_DRIVE=True
def backup_to_drive(): print("(drive backup disabled)")
if USE_DRIVE:
    try:
        from google.colab import drive; drive.mount('/content/drive')
        import os, shutil
        SAVE_DIR="/content/drive/MyDrive/health_qa_models"; os.makedirs(SAVE_DIR, exist_ok=True)
        def backup_to_drive():
            import os, shutil
            for p in ["ftq_bgem3_qq","fthn_bgem3","fte_e5large_qq","fthn_tv_bgem3",
                      "fte_tv_e5large","submission.csv","submission_trainval.csv"]:
                if os.path.exists(p):
                    dst=os.path.join(SAVE_DIR, os.path.basename(p))
                    if os.path.isdir(p): shutil.copytree(p, dst, dirs_exist_ok=True)
                    else: shutil.copy(p, dst)
                    print("backed up", p)
        print("Drive mounted ->", SAVE_DIR)
    except Exception as e:
        print("Drive mount skipped:", e)
""")

# ---- Part 1 -------------------------------------------------------------- #
md("## Part 1 — Off-the-shelf bge-m3 + e5-large + ensemble (baseline)")
code(r"""
get_model("bge","BAAI/bge-m3",False)
get_model("e5l","intfloat/multilingual-e5-large",True)
for tag,specs in [("bge-m3",[("bge","question",1.0)]),
                  ("e5-large",[("e5l","question",1.0)]),
                  ("ENSEMBLE bge+e5l",[("bge","question",0.5),("e5l","question",0.5)])]:
    sf=eval_specs(specs); print(f"\n--- {tag} ---")
    print("rouge_wtd=", sf.loc['overall','rouge_wtd'], "| est LB=", est_lb(sf.loc['overall','rouge_wtd']))
""")

# ---- Part 2 (optional) --------------------------------------------------- #
md("## Part 2 — Cross-encoder reranker *(optional; was flat — skip to save time)*")
code(r"""
RUN_RERANK=False
if RUN_RERANK:
    from sentence_transformers import CrossEncoder
    def rerank_top1(base,q_df,c_df,k=10):
        c_sub=c_df[SUB].to_numpy(); q_sub=q_df[SUB].to_numpy()
        c_emb=enc(base,c_df[INP].tolist(),"query"); q_emb=enc(base,q_df[INP].tolist(),"query")
        n=len(q_df); cand=np.empty((n,k),dtype=np.int64)
        for s in range(0,n,256):
            sc=np.where(c_sub[None,:]==q_sub[s:s+256][:,None], q_emb[s:s+256]@c_emb.T, -1e9)
            idx=np.argpartition(-sc,k,axis=1)[:,:k]
            for r in range(len(idx)): cand[s+r]=idx[r][np.argsort(-sc[r,idx[r]])]
        if "rr" not in MODELS: MODELS["rr"]=CrossEncoder("BAAI/bge-reranker-v2-m3",device="cuda",max_length=512)
        cq=c_df[INP].tolist(); qq=q_df[INP].tolist()
        pairs=[[qq[i],cq[cand[i,r]]] for i in range(n) for r in range(k)]
        sc=np.array(MODELS["rr"].predict(pairs,batch_size=128,show_progress_bar=True)).reshape(n,k)
        return cand[np.arange(n),sc.argmax(1)]
    top=rerank_top1("bge",val,train,10)
    sf=score_frame([train[OUT].tolist()[j] for j in top],val[OUT].tolist(),val[SUB].tolist())
    print("RERANK rouge_wtd=", sf.loc['overall','rouge_wtd'])
else:
    print("skipped reranker")
""")

# ---- Part 3: q->q fine-tune (current best) ------------------------------- #
md("## Part 3 — Fine-tune bge-m3 on question→question (same-answer positives)")
code(r"""
import gc
from itertools import combinations
from collections import defaultdict
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader
gc.collect(); torch.cuda.empty_cache(); random.seed(42)

def qq_pairs(prefix=""):
    by=defaultdict(list)
    for q,a in zip(train[INP],train[OUT]):
        if pd.isna(q) or pd.isna(a): continue
        by[a].append(prefix+str(q))
    ex=[]; MAXP=4
    for a,qs in by.items():
        qs=list(dict.fromkeys(qs))
        if len(qs)<2: continue
        cb=list(combinations(qs,2)); random.shuffle(cb)
        for c in cb[:MAXP]: ex.append(InputExample(texts=[c[0],c[1]]))
    return ex

ex=qq_pairs(); print("q->q pairs:", len(ex))
ftq=SentenceTransformer("BAAI/bge-m3",device="cuda"); ftq.max_seq_length=128
loader=DataLoader(ex,shuffle=True,batch_size=64,drop_last=True)
ftq.fit(train_objectives=[(loader,losses.MultipleNegativesRankingLoss(ftq))],
        epochs=1, warmup_steps=int(0.1*len(loader)), optimizer_params={"lr":1e-5},
        use_amp=True, show_progress_bar=True)
MODELS["ftq"]=(ftq,False); _EMB.clear(); ftq.save("ftq_bgem3_qq")
sf=eval_specs([("ftq","question",1.0)]); print("ftq rouge_wtd=", sf.loc['overall','rouge_wtd'],
      "| est LB=", est_lb(sf.loc['overall','rouge_wtd']))
""")

# ---- Part 4: hard-negative fine-tune ------------------------------------- #
md(r"""## Part 4 — Hard-negative fine-tune of bge-m3
Mine, per anchor question, a **similar-but-wrong-answer** question as an explicit
hard negative. **Filter out** candidates whose answer equals or is near-duplicate
(cos ≥ 0.9) of the anchor's answer — otherwise we'd mislabel a valid paraphrase as
negative (≈40% of answers are duplicated). Uses off-the-shelf bge for mining.""")
code(r"""
gc.collect(); torch.cuda.empty_cache(); random.seed(42)
qtr=train[INP].tolist(); atr=train[OUT].tolist(); subtr=train[SUB].to_numpy()
qE=enc("bge",qtr,"query"); aE=enc("bge",atr,"passage")   # off-the-shelf embeddings for mining
by=defaultdict(list)
for i,(q,a) in enumerate(zip(qtr,atr)):
    if pd.isna(q) or pd.isna(a): continue
    by[a].append(i)

N=len(qtr); K=20; NEARDUP=0.9; triplets=[]
for s in range(0,N,512):
    idx=np.arange(s,min(s+512,N))
    sims=qE[idx]@qE.T
    m=(subtr[None,:]==subtr[idx][:,None])       # same-subset candidates only
    sims=np.where(m,sims,-1e9)
    for r,i in enumerate(idx):
        a_i=atr[i]; grp=by.get(a_i,[])
        if len(grp)<2: continue                  # need a same-answer positive
        pos=random.choice([j for j in grp if j!=i])
        order=np.argpartition(-sims[r],K)[:K]; order=order[np.argsort(-sims[r][order])]
        neg=None
        for j in order:
            if j==i or atr[j]==a_i: continue
            if float(aE[i]@aE[j])>NEARDUP: continue   # skip near-duplicate answers
            neg=j; break
        if neg is not None:
            triplets.append(InputExample(texts=[qtr[i],qtr[pos],qtr[neg]]))
print("hard-negative triplets:", len(triplets))

fthn=SentenceTransformer("BAAI/bge-m3",device="cuda"); fthn.max_seq_length=128
loader=DataLoader(triplets,shuffle=True,batch_size=32,drop_last=True)
fthn.fit(train_objectives=[(loader,losses.MultipleNegativesRankingLoss(fthn))],
         epochs=1, warmup_steps=int(0.1*len(loader)), optimizer_params={"lr":1e-5},
         use_amp=True, show_progress_bar=True)
MODELS["fthn"]=(fthn,False); _EMB.clear(); fthn.save("fthn_bgem3")
sf=eval_specs([("fthn","question",1.0)]); print("fthn rouge_wtd=", sf.loc['overall','rouge_wtd'],
      "| est LB=", est_lb(sf.loc['overall','rouge_wtd']), " (beat ftq 0.3918?)")
""")

# ---- Part 5: e5 fine-tune + weight sweep --------------------------------- #
md("## Part 5 — Fine-tune e5-large (q→q) + weight-swept ensembles")
code(r"""
gc.collect(); torch.cuda.empty_cache(); random.seed(42)
ex=qq_pairs(prefix="query: ")               # e5 wants the query prefix; keep it consistent
fte=SentenceTransformer("intfloat/multilingual-e5-large",device="cuda"); fte.max_seq_length=128
loader=DataLoader(ex,shuffle=True,batch_size=64,drop_last=True)
fte.fit(train_objectives=[(loader,losses.MultipleNegativesRankingLoss(fte))],
        epochs=1, warmup_steps=int(0.1*len(loader)), optimizer_params={"lr":1e-5},
        use_amp=True, show_progress_bar=True)
MODELS["fte"]=(fte,True); _EMB.clear(); fte.save("fte_e5large_qq")
print("fte trained")
""")
code(r"""
# Compare fine-tuned singles + ensembles + a weight sweep (all on Val)
cands = {
 "ftq":          [("ftq","question",1.0)],
 "fthn":         [("fthn","question",1.0)],
 "fte":          [("fte","question",1.0)],
 "ftq+e5l(.5)":  [("ftq","question",.5),("e5l","question",.5)],
 "fthn+fte(.5)": [("fthn","question",.5),("fte","question",.5)],
 "ftq+fte(.5)":  [("ftq","question",.5),("fte","question",.5)],
 "fthn+ftq+fte": [("fthn","question",1/3),("ftq","question",1/3),("fte","question",1/3)],
}
res={k: rw_of(v) for k,v in cands.items()}
# weight sweep on the strongest pair (bge-side FT + fte)
best_pair = "fthn" if res["fthn"]>=res["ftq"] else "ftq"
for w in [0.3,0.4,0.5,0.6,0.7]:
    res[f"{best_pair}+fte({w})"]=rw_of([(best_pair,"question",w),("fte","question",1-w)])
import pandas as pd
tab=pd.Series(res).sort_values(ascending=False)
print(tab.to_string()); print("\nBEST:", tab.index[0], "=", round(tab.iloc[0],4),
      "| est LB=", est_lb(tab.iloc[0]), "| current 1st Val=0.3932")
backup_to_drive()   # persist ftq/fthn/fte to Drive
""")

# ---- Part 6: RRF vs averaging ------------------------------------------- #
md("## Part 6 — RRF vs score-averaging *(sanity check for the best pair)*")
code(r"""
def rrf_top1(specs, q_df, c_df, k=50, C=60):
    c_sub=c_df[SUB].to_numpy(); q_sub=q_df[SUB].to_numpy()
    q_embs={n:enc(n,q_df[INP].tolist(),"query") for n,sp,w in specs}
    c_embs={n:enc(n,(c_df[OUT] if sp=="answer" else c_df[INP]).tolist(),
                  "passage" if sp=="answer" else "query") for n,sp,w in specs}
    n_=len(q_df); top=np.empty(n_,dtype=np.int64)
    for s in range(0,n_,128):
        rrf=None
        for n,sp,w in specs:
            sc=np.where(c_sub[None,:]==q_sub[s:s+128][:,None], q_embs[n][s:s+128]@c_embs[n].T, -1e9)
            ranks=(-sc).argsort(1).argsort(1)          # 0=best
            rr=w/(C+ranks)
            rrf=rr if rrf is None else rrf+rr
        top[s:s+128]=rrf.argmax(1)
    return top
pair=[( "fthn" if rw_of([("fthn","question",1.0)])>=rw_of([("ftq","question",1.0)]) else "ftq"),"question",0.5]
specs=[tuple(pair),("fte","question",0.5)]
avg=eval_specs(specs).loc["overall","rouge_wtd"]
top=rrf_top1(specs,val,train); rrf=score_frame([train[OUT].tolist()[j] for j in top],val[OUT].tolist(),val[SUB].tolist()).loc["overall","rouge_wtd"]
print("averaging=",avg," RRF=",rrf," -> use", "RRF" if rrf>avg else "averaging")
""")

# ---- Part 7: submission -------------------------------------------------- #
md(r"""## Part 7 — Generate the Test submission from the BEST config
Set `BEST_SPECS` to the winning specs from Part 5 (copy its name's spec), then run.
Uses the **train+val** corpus for Test. Only submit if Part 5 beat **0.3932**.""")
code(r"""
# EDIT this to the winner from Part 5 (examples shown):
BEST_SPECS = [("fthn","question",0.5), ("fte","question",0.5)]   # <- set to your best
# e.g. [("ftq","question",0.5),("e5l","question",0.5)]  (the current 0.628 config)

corpus=pd.concat([train,val],ignore_index=True); corpus_ans=corpus[OUT].tolist()
top=ensemble_top1(BEST_SPECS, test, corpus)
make_submission(top, corpus_ans, "submission.csv")
print("val-sourced test answers:", int((top>=len(train)).sum()), "/", len(test))
""")

# ---- Part 8: bootstrap CIs (decision hygiene) ---------------------------- #
md(r"""## Part 8 — Bootstrap confidence intervals (are the gains real or noise?)
Resamples the 6,686 Val rows 1,000× to put a 95% CI on each config's `rouge_wtd`
and on the *difference* between configs. Guards against selecting a noise-level
winner (the private-LB overfitting risk). Uses the train-only models above.""")
code(r"""
def per_row_wtd(specs):
    top=ensemble_top1(specs, val, train)
    preds=[train[OUT].tolist()[j] for j in top]; golds=val[OUT].tolist()
    out=np.empty(len(preds))
    for i,(p,g) in enumerate(zip(preds,golds)):
        o=_sc.score(g,p); out[i]=W1*o["rouge1"].fmeasure+WL*o["rougeL"].fmeasure
    return out

configs={
 "ftq+e5l(.5)":  [("ftq","question",.5),("e5l","question",.5)],   # old 0.628 config
 "fthn":         [("fthn","question",1.0)],
 "fthn+fte(.5)": [("fthn","question",.5),("fte","question",.5)],   # current best (0.6375 LB)
}
rows={k:per_row_wtd(v) for k,v in configs.items()}
rng=np.random.default_rng(42); N=len(val); B=1000
idxs=rng.integers(0,N,size=(B,N))
boot={k: rows[k][idxs].mean(1) for k in configs}
print("Config bootstrap  mean [95% CI]  sd:")
for k in configs:
    lo,hi=np.percentile(boot[k],[2.5,97.5])
    print(f"  {k:14s} {boot[k].mean():.4f}  [{lo:.4f}, {hi:.4f}]  sd={boot[k].std():.4f}")
d=boot["fthn+fte(.5)"]-boot["ftq+e5l(.5)"]
lo,hi=np.percentile(d,[2.5,97.5])
print(f"\n(fthn+fte) - (ftq+e5l): {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  P(better)={np.mean(d>0):.3f}")
print("Rule: only treat a config as better if its CI clears the other's; else keep the simpler one.")
""")

# ---- Part 9: retrain selected config on TRAIN+VAL ------------------------ #
md(r"""## Part 9 — Retrain the selected config on **Train + Val** (final model)
More labelled data & duplicate-answer groups → a more robust retriever for the
private board. **No hyperparameter changes.** ⚠️ Trained on Val, so it **cannot be
validated locally** — submit it, and only promote it if public ≥ 0.6375; otherwise
keep the train-only 0.6375 as your selected submission. Backup is never at risk.""")
code(r"""
import gc
for n in ["ftq","fthn","fte","e5l","rr"]:      # free round-1 models; keep base bge for mining
    MODELS.pop(n, None)
_EMB.clear(); gc.collect(); torch.cuda.empty_cache()

TV=pd.concat([train,val],ignore_index=True)
qtv=TV[INP].tolist(); atv=TV[OUT].tolist(); subtv=TV[SUB].to_numpy()

def qq_pairs_df(df, prefix=""):
    by=defaultdict(list)
    for q,a in zip(df[INP],df[OUT]):
        if pd.isna(q) or pd.isna(a): continue
        by[a].append(prefix+str(q))
    ex=[]
    for a,qs in by.items():
        qs=list(dict.fromkeys(qs))
        if len(qs)<2: continue
        cb=list(combinations(qs,2)); random.shuffle(cb)
        for c in cb[:4]: ex.append(InputExample(texts=[c[0],c[1]]))
    return ex

# hard-negative triplets on train+val (mine with off-the-shelf bge, filtered)
get_model("bge","BAAI/bge-m3",False)
qE=enc("bge",qtv,"query"); aE=enc("bge",atv,"passage")
by=defaultdict(list)
for i,(q,a) in enumerate(zip(qtv,atv)):
    if pd.isna(q) or pd.isna(a): continue
    by[a].append(i)
Nn=len(qtv); trip=[]
for s in range(0,Nn,512):
    idx=np.arange(s,min(s+512,Nn)); sims=np.where(subtv[None,:]==subtv[idx][:,None], qE[idx]@qE.T, -1e9)
    for r,i in enumerate(idx):
        a_i=atv[i]; grp=by.get(a_i,[])
        if len(grp)<2: continue
        pos=random.choice([j for j in grp if j!=i])
        order=np.argpartition(-sims[r],20)[:20]; order=order[np.argsort(-sims[r][order])]
        neg=None
        for j in order:
            if j==i or atv[j]==a_i: continue
            if float(aE[i]@aE[j])>0.9: continue
            neg=j; break
        if neg is not None: trip.append(InputExample(texts=[qtv[i],qtv[pos],qtv[neg]]))
print("train+val hard-negative triplets:", len(trip))
MODELS.pop("bge", None); _EMB.clear(); gc.collect(); torch.cuda.empty_cache()

fthn_tv=SentenceTransformer("BAAI/bge-m3",device="cuda"); fthn_tv.max_seq_length=128
ld=DataLoader(trip,shuffle=True,batch_size=32,drop_last=True)
fthn_tv.fit(train_objectives=[(ld,losses.MultipleNegativesRankingLoss(fthn_tv))], epochs=1,
            warmup_steps=int(0.1*len(ld)), optimizer_params={"lr":1e-5}, use_amp=True, show_progress_bar=True)
MODELS["fthn_tv"]=(fthn_tv,False); fthn_tv.save("fthn_tv_bgem3"); gc.collect(); torch.cuda.empty_cache()

ex=qq_pairs_df(TV, prefix="query: ")
fte_tv=SentenceTransformer("intfloat/multilingual-e5-large",device="cuda"); fte_tv.max_seq_length=128
ld=DataLoader(ex,shuffle=True,batch_size=64,drop_last=True)
fte_tv.fit(train_objectives=[(ld,losses.MultipleNegativesRankingLoss(fte_tv))], epochs=1,
           warmup_steps=int(0.1*len(ld)), optimizer_params={"lr":1e-5}, use_amp=True, show_progress_bar=True)
MODELS["fte_tv"]=(fte_tv,True); fte_tv.save("fte_tv_e5large"); _EMB.clear()
print("retrained fthn_tv + fte_tv on train+val")
""")
code(r"""
# Final submission from the train+val-retrained ensemble (corpus = train+val)
corpus=pd.concat([train,val],ignore_index=True); corpus_ans=corpus[OUT].tolist()
top=ensemble_top1([("fthn_tv","question",.5),("fte_tv","question",.5)], test, corpus)
make_submission(top, corpus_ans, "submission_trainval.csv")
print("val-sourced test answers:", int((top>=len(train)).sum()), "/", len(test))
print("NOTE: cannot validate locally (trained on val). Submit; keep 0.6375 if this regresses.")
backup_to_drive()   # persist fthn_tv/fte_tv + submission to Drive
""")

nb=new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                                       "language_info":{"name":"python"},"accelerator":"GPU"})
out=Path(__file__).resolve().parents[1]/"notebooks"/"colab_finetune_retriever.ipynb"
nbf.write(nb, str(out)); print("wrote", out, "cells:", len(cells))
