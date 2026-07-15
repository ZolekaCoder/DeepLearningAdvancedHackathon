"""Thin plotting helpers so the notebook stays declarative.

Every helper returns a matplotlib Figure; the notebook passes it to
`io_utils.save_fig` so all figures land in outputs/figures with consistent
styling. Fonts are set to render Latin + Ethiopic where the installed fonts
allow; missing glyphs degrade gracefully (boxes) without crashing.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

matplotlib.rcParams.update({
    "figure.autolayout": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def _colors(subsets) -> list[str]:
    return [C.SUBSET_PALETTE.get(s, "#333333") for s in subsets]


def bar(counts: pd.Series, title: str, xlabel: str = "", ylabel: str = "count",
        color_by_subset: bool = False, rotate: int = 45, figsize=(8, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = _colors(counts.index) if color_by_subset else "#4C72B0"
    ax.bar(counts.index.astype(str), counts.values, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    return fig


def hist_by_group(df: pd.DataFrame, value_col: str, group_col: str,
                  title: str, xlabel: str, bins: int = 50, clip_q: float = 0.99,
                  figsize=(9, 5)):
    """Overlaid histograms of `value_col` per group, clipped at a high quantile
    so a few extreme outliers don't flatten the plot."""
    fig, ax = plt.subplots(figsize=figsize)
    hi = df[value_col].quantile(clip_q)
    for g, sub in df.groupby(group_col):
        vals = sub[value_col].clip(upper=hi)
        ax.hist(vals, bins=bins, histtype="step", linewidth=1.6,
                label=str(g), color=C.SUBSET_PALETTE.get(g))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frequency")
    ax.legend(fontsize=8, ncol=2)
    return fig


def box_by_group(df: pd.DataFrame, value_col: str, group_col: str,
                 title: str, ylabel: str, clip_q: float = 0.99, figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    hi = df[value_col].quantile(clip_q)
    groups = sorted(df[group_col].unique())
    data = [df.loc[df[group_col] == g, value_col].clip(upper=hi) for g in groups]
    bp = ax.boxplot(data, tick_labels=groups, patch_artist=True, showfliers=False)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(C.SUBSET_PALETTE.get(g, "#999999"))
        patch.set_alpha(0.7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig


def scatter_2d(xy: np.ndarray, labels, title: str, figsize=(9, 8),
               palette: dict | None = None, alpha: float = 0.5, s: float = 6):
    """2-D scatter (for UMAP/t-SNE projections) coloured by a categorical label."""
    fig, ax = plt.subplots(figsize=figsize)
    labels = np.asarray(labels)
    uniq = sorted(pd.unique(labels), key=lambda v: str(v))
    for lab in uniq:
        m = labels == lab
        col = (palette or C.SUBSET_PALETTE).get(lab) if palette or True else None
        ax.scatter(xy[m, 0], xy[m, 1], s=s, alpha=alpha, label=str(lab),
                   color=(palette.get(lab) if palette else None))
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8, ncol=2, loc="best")
    return fig


def heatmap(matrix: np.ndarray, labels, title: str, figsize=(7, 6), fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] < matrix.max() * 0.6 else "black",
                    fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig
