"""Multilingual embeddings via intfloat/multilingual-e5-base (chosen with user).

e5 expects instruction prefixes: 'query: ' for queries and 'passage: ' for
documents. We honour that. Embeddings are L2-normalized so dot product == cosine
similarity, and cached to outputs/artifacts so nothing is recomputed on re-run.

CPU/MPS only — we auto-select MPS on Apple silicon, else CPU, and show a
progress bar + timing.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import config as C
from . import io_utils as io

_MODEL = None


def _device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


_MODELS: dict = {}


def get_model(model_name: str = C.EMBED_MODEL):
    """Lazily load a SentenceTransformer (cached per model name)."""
    if model_name not in _MODELS:
        from sentence_transformers import SentenceTransformer

        dev = _device()
        print(f"[embeddings] loading {model_name} on {dev} ...", flush=True)
        _MODELS[model_name] = SentenceTransformer(model_name, device=dev)
    return _MODELS[model_name]


def _cache_key(texts: list[str], prefix: str) -> str:
    h = hashlib.sha1()
    h.update(prefix.encode())
    h.update(str(len(texts)).encode())
    # sample a few texts to key the cache without hashing megabytes
    for i in (0, len(texts) // 2, len(texts) - 1):
        if 0 <= i < len(texts):
            h.update(str(texts[i]).encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def encode(texts, kind: str = "query", cache_name: str | None = None,
           batch_size: int = C.EMBED_BATCH, model_name: str = C.EMBED_MODEL) -> np.ndarray:
    """Encode texts to L2-normalized embeddings, with on-disk caching.

    kind: 'query' or 'passage' -> selects the e5 instruction prefix.
    e5 models require the prefix; non-e5 models (e.g. bge-m3) do not.
    cache_name: if given, embeddings are cached under this name (+model tag).
    """
    texts = [("" if t is None else str(t)) for t in texts]
    prefix = ("query: " if kind == "query" else "passage: ") if "e5" in model_name else ""

    # short model tag so different models don't collide in the cache
    tag = model_name.split("/")[-1].replace("multilingual-", "")

    cache_path = None
    if cache_name:
        key = _cache_key(texts, prefix)
        cache_path = C.ART_DIR / f"emb_{cache_name}_{tag}_{key}.npy"
        # backward-compat: original e5-base cache had no model tag
        legacy = C.ART_DIR / f"emb_{cache_name}_{key}.npy"
        if cache_path.exists():
            print(f"[embeddings] cache hit: {cache_path.name}", flush=True)
            return np.load(cache_path)
        if model_name == C.EMBED_MODEL and legacy.exists():
            print(f"[embeddings] cache hit (legacy): {legacy.name}", flush=True)
            return np.load(legacy)

    model = get_model(model_name)
    with io.timer(f"encode {len(texts)} texts ({kind})"):
        emb = model.encode(
            [prefix + t for t in texts],
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    if cache_path is not None:
        np.save(cache_path, emb)
        print(f"[embeddings] cached -> {cache_path.name}", flush=True)
    return emb.astype(np.float32)
