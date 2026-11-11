"""
Overlay the resampled HR series that enter the ISC correlations.

Usage:  python plot_isc_overlay.py [--out output]

Reads output/synced_hr_long.csv (written by sync_hr_isc.py) and saves
output/isc_overlay.png: one row per segment (music blocks per tempo, then
rests and baseline), HR in bpm as resampled. Every participant is drawn as a
thin line in one colour, so the spread stays readable however many there are.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PARTICIPANT = "#2a78d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ORDER = ["music_90", "music_120", "music_150",
         "rest_after_90", "rest_after_120", "rest_after_150", "baseline"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output")
    a = ap.parse_args()
    hr = pd.read_csv(a.out / "synced_hr_long.csv").dropna(subset=["hr_bpm"])

    labels = [l for l in ORDER if l in hr["label"].unique()]
    fig, axes = plt.subplots(len(labels), 1, figsize=(9, 2.4 * len(labels)),
                             squeeze=False, facecolor=SURFACE, sharex=True, sharey=True)
    for row, label in enumerate(labels):
        ax = axes[row, 0]
        d = hr[hr["label"] == label]
        ax.set_facecolor(SURFACE)
        for _, g in d.groupby("participant"):
            ax.plot(g["t_s"], g["hr_bpm"], lw=1.1, color=PARTICIPANT, alpha=0.6)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8, labelbottom=True)
        ax.set_ylabel("HR (bpm)", color=INK2, fontsize=9)
        ax.set_title(f"{label.replace('_', ' ')}   (n = {d['participant'].nunique()})",
                     loc="left", color=INK, fontsize=11, fontweight="bold")
    axes[-1, 0].set_xlabel("time since segment onset (s)", color=INK2, fontsize=9)

    fig.suptitle("Resampled heart rate per segment (one line per participant)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = a.out / "isc_overlay.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
