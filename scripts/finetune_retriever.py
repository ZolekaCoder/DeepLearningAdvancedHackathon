"""Stage 3 — fine-tune the retriever on train question->answer pairs.

Contrastive fine-tuning (MultipleNegativesRankingLoss / in-batch negatives) so a
question embeds close to its correct canonical ANSWER. At inference we retrieve
in ANSWER-space (query -> nearest train/val answer) and copy it.

Base: multilingual-e5-base (tractable on M4 MPS). e5 prefixes: 'query:' for
questions, 'passage:' for answers. Seeded for reproducibility.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

from health_eda import config as C, io_utils as io, eval_official as EO

C.set_seed()
torch.manual_seed(C.SEED)

BASE = "intfloat/multilingual-e5-base"
OUT_DIR = C.ART_DIR / "ft_e5base_qa"
EPOCHS = 2
BATCH = 8           # tight MPS memory budget on 17GB M4
MAX_SEQ = 128       # truncate long answers; questions are short


def build_pairs(train):
    """(query: question, passage: answer) InputExamples.

    Dedup by ANSWER (keep one question per unique answer). This removes MNRL
    in-batch false negatives (the same canonical answer appearing as another
    row's positive), which matters here given ~40% duplicate answers.
    """
    seen_ans = set(); ex = []
    for q, a in zip(train[C.INPUT_COL], train[C.OUTPUT_COL]):
        if a in seen_ans:
            continue
        seen_ans.add(a)
        ex.append(InputExample(texts=[f"query: {q}", f"passage: {a}"]))
    return ex


def answer_space_eval(model, train, val):
    """Retrieve nearest train ANSWER (subset-pooled) for each val question."""
    tr_ans = train[C.OUTPUT_COL].tolist()
    a_emb = model.encode([f"passage: {a}" for a in tr_ans], batch_size=64,
                         normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=True)
    q_emb = model.encode([f"query: {q}" for q in val[C.INPUT_COL]], batch_size=64,
                         normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=True)
    trsub = train[C.SUBSET_COL].to_numpy(); vasub = val[C.SUBSET_COL].to_numpy()
    n = len(val); top = np.empty(n, dtype=np.int64)
    for s in range(0, n, 128):
        with np.errstate(all="ignore"):
            sc = q_emb[s:s+128] @ a_emb.T
        m = (trsub[None, :] == vasub[s:s+128][:, None])
        top[s:s+128] = np.where(m, sc, -1e9).argmax(axis=1)
    preds = [tr_ans[j] for j in top]
    return EO.score_frame(preds, val[C.OUTPUT_COL].tolist(), vasub.tolist())


def main():
    train = io.load_split("train").reset_index(drop=True)
    val = io.load_split("val").reset_index(drop=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(BASE, device=dev)
    model.max_seq_length = MAX_SEQ

    ex = build_pairs(train)
    print(f"training pairs: {len(ex):,} | device {dev} | epochs {EPOCHS} batch {BATCH}")
    loader = DataLoader(ex, shuffle=True, batch_size=BATCH, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    with io.timer("fine-tune"):
        model.fit(train_objectives=[(loader, loss)], epochs=EPOCHS,
                  warmup_steps=int(0.1*len(loader)), show_progress_bar=True,
                  optimizer_params={"lr": 2e-5})
    model.save(str(OUT_DIR))
    print("saved ->", OUT_DIR)

    print("\n=== Val answer-space eval (fine-tuned e5-base) ===")
    sf = answer_space_eval(model, train, val)
    io.save_table(sf, "stage3_ft_val_answerspace")
    print(sf.to_string())
    print(f"\nROUGE-weighted={sf.loc['overall','rouge_weighted']:.4f} "
          f"(ensemble baseline 0.3705, oracle 0.4573)")


if __name__ == "__main__":
    main()
