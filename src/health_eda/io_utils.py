"""Loading data and persisting artefacts (tables, figures, dataframes).

Keeping all I/O in one place means the notebook never hard-codes a path and
every table/figure lands in the right `outputs/` sub-directory automatically.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_split(split: str) -> pd.DataFrame:
    """Load one of 'train' | 'val' | 'test' | 'sample_sub' as a DataFrame."""
    paths = {
        "train": C.TRAIN_CSV,
        "val": C.VAL_CSV,
        "test": C.TEST_CSV,
        "sample_sub": C.SAMPLE_SUB_CSV,
    }
    if split not in paths:
        raise ValueError(f"Unknown split {split!r}; choose from {list(paths)}")
    df = pd.read_csv(paths[split], dtype=str, keep_default_na=False)
    # Preserve genuine emptiness: keep_default_na=False turns NaN into "" so we
    # can distinguish "missing" from the literal string 'nan'. We re-mark
    # truly empty cells as NA for missing-value accounting.
    df = df.replace({"": pd.NA})
    df.attrs["split"] = split
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Load train/val/test/sample_sub into a dict."""
    return {s: load_split(s) for s in ("train", "val", "test", "sample_sub")}


# --------------------------------------------------------------------------- #
# Saving tables and figures
# --------------------------------------------------------------------------- #
def save_table(df: pd.DataFrame, name: str, index: bool = True) -> Path:
    """Save a table as both CSV (machine) and Markdown (report) under tables/."""
    csv_path = C.TAB_DIR / f"{name}.csv"
    md_path = C.TAB_DIR / f"{name}.md"
    df.to_csv(csv_path, index=index)
    try:
        md_path.write_text(df.to_markdown(index=index))
    except Exception:
        # to_markdown needs `tabulate`; fall back silently to CSV only.
        pass
    return csv_path


def save_fig(fig, name: str, dpi: int = 130) -> Path:
    """Save a matplotlib figure to figures/ as PNG and close it."""
    import matplotlib.pyplot as plt

    path = C.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_artifact(obj, name: str) -> Path:
    """Pickle an arbitrary intermediate artefact (embeddings, indices, ...)."""
    import pickle

    path = C.ART_DIR / f"{name}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    return path


def load_artifact(name: str):
    """Load a previously pickled artefact, or return None if absent."""
    import pickle

    path = C.ART_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------------- #
# Timing helper (for the "include timing information" requirement)
# --------------------------------------------------------------------------- #
@contextmanager
def timer(label: str):
    """Context manager that prints wall-clock time for an expensive block."""
    t0 = time.perf_counter()
    print(f"[timer] {label} ...", flush=True)
    yield
    dt = time.perf_counter() - t0
    print(f"[timer] {label} done in {dt:,.1f}s", flush=True)
