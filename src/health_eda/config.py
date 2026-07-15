"""Central configuration for the Multilingual Health QA EDA.

Everything that another engineer might want to tweak (paths, seeds, the
embedding model, subset metadata) lives here so the rest of the code stays
free of magic constants.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Resolve the project root as two levels up from this file (src/health_eda/).
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT
TRAIN_CSV = DATA_DIR / "Train.csv"
VAL_CSV = DATA_DIR / "Val.csv"
TEST_CSV = DATA_DIR / "Test.csv"
SAMPLE_SUB_CSV = DATA_DIR / "SampleSubmission.csv"

OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
TAB_DIR = OUTPUTS / "tables"
ART_DIR = OUTPUTS / "artifacts"

REPORT_MD = ROOT / "DATASET_ANALYSIS_REPORT.md"
DECISIONS_MD = ROOT / "MODELLING_DECISIONS.md"

for _d in (FIG_DIR, TAB_DIR, ART_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG we might touch, for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #
ID_COL = "ID"
INPUT_COL = "input"      # the question
OUTPUT_COL = "output"    # the answer (absent in Test)
SUBSET_COL = "subset"    # language_country tag

# --------------------------------------------------------------------------- #
# Subset metadata: language / country / script for the 8 language-country tags.
# `african` flags the non-English languages (used in cross-lingual analysis).
# --------------------------------------------------------------------------- #
SUBSET_META = {
    "Aka_Gha": {"language": "Akan",     "country": "Ghana",    "script": "Latin",  "african": True},
    "Amh_Eth": {"language": "Amharic",  "country": "Ethiopia", "script": "Ge'ez",  "african": True},
    "Eng_Eth": {"language": "English",  "country": "Ethiopia", "script": "Latin",  "african": False},
    "Eng_Gha": {"language": "English",  "country": "Ghana",    "script": "Latin",  "african": False},
    "Eng_Ken": {"language": "English",  "country": "Kenya",    "script": "Latin",  "african": False},
    "Eng_Uga": {"language": "English",  "country": "Uganda",   "script": "Latin",  "african": False},
    "Lug_Uga": {"language": "Luganda",  "country": "Uganda",   "script": "Latin",  "african": True},
    "Swa_Ken": {"language": "Swahili",  "country": "Kenya",    "script": "Latin",  "african": True},
}

# Stable colour per subset for consistent plots across the whole notebook.
SUBSET_PALETTE = {
    "Eng_Uga": "#4C72B0",
    "Aka_Gha": "#DD8452",
    "Eng_Gha": "#55A868",
    "Eng_Eth": "#C44E52",
    "Lug_Uga": "#8172B3",
    "Eng_Ken": "#937860",
    "Swa_Ken": "#DA8BC3",
    "Amh_Eth": "#8C8C8C",
}

# --------------------------------------------------------------------------- #
# Models (chosen with the user: multilingual-e5-base only, CPU/MPS).
# e5 requires "query: " / "passage: " prefixes — encoded in retrieval.py.
# --------------------------------------------------------------------------- #
EMBED_MODEL = "intfloat/multilingual-e5-base"
EMBED_BATCH = 32
