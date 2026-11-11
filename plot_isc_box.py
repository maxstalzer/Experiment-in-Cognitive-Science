"""
Box-and-whisker plot of the pairwise ISC per condition, with every pair as a dot.

Usage:  python plot_isc_box.py [--out output]

Reads output/isc_pairwise.csv (written by sync_hr_isc.py) and saves
output/isc_box.png. Boxes show median and quartiles, whiskers 1.5 x IQR;
outliers are not drawn separately because every pair is already a dot.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DOT = "#2a78d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ORDER = ["music_90", "music_120", "music_150",
         "rest_after_90", "rest_after_120", "rest_after_150", "baseline"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output")
    a = ap.parse_args()
    pairs = pd.read_csv(a.out / "isc_pairwise.csv")

    labels = [l for l in ORDER if l in pairs["label"].unique()]
    data = [pairs.loc[pairs["label"] == l, "r"].to_numpy() for l in labels]
    pos = np.arange(1, len(labels) + 1)

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=INK2, lw=1, zorder=1)
    ax.boxplot(data, positions=pos, widths=0.55, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="none", edgecolor=INK2, lw=1.2),
               medianprops=dict(color=INK, lw=2),
               whiskerprops=dict(color=INK2, lw=1.2),
               capprops=dict(color=INK2, lw=1.2), zorder=2)

    rng = np.random.default_rng(0)   # fixed jitter so the figure is reproducible
    for x, r in zip(pos, data):
        jitter = rng.uniform(-0.17, 0.17, len(r))
        ax.scatter(x + jitter, r, s=30, color=DOT, alpha=0.75,
                   edgecolors=SURFACE, linewidths=0.8, zorder=3)

    ax.set_xticks(pos)
    ax.set_xticklabels([f"{l.replace('_', ' ')}\n(n = {len(r)} pairs)" for l, r in zip(labels, data)],
                       fontsize=9, color=INK2)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("pairwise Pearson r (HR)", color=INK2, fontsize=10)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(axis="y", colors=INK2, labelsize=9)
    ax.set_title("Inter-subject correlation of heart rate per condition (one dot per participant pair)",
                 loc="left", color=INK, fontsize=12)

    fig.tight_layout()
    out = a.out / "isc_box.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
