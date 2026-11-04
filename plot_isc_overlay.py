"""
Overlay the resampled HR series that enter the ISC correlations.

Usage:  python plot_isc_overlay.py [--out output]

Reads output/synced_hr_long.csv and output/isc_pairwise.csv (written by
sync_hr_isc.py) and saves output/isc_overlay.png:
  rows    = segments (music blocks per tempo, then rests and baseline)
  left    = HR in bpm as resampled
  right   = same series z-scored per participant (Pearson r ignores offset and
            scale, so this is the shape the correlation actually compares)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# fixed categorical order (colour follows the participant, never the rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ORDER = ["music_90", "music_120", "music_150",
         "rest_after_90", "rest_after_120", "rest_after_150", "baseline"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output")
    a = ap.parse_args()
    hr = pd.read_csv(a.out / "synced_hr_long.csv").dropna(subset=["hr_bpm"])
    pairs = pd.read_csv(a.out / "isc_pairwise.csv")

    labels = [l for l in ORDER if l in hr["label"].unique()]
    pids = sorted(hr["participant"].unique())
    color = {p: SERIES[i % len(SERIES)] for i, p in enumerate(pids)}

    fig, axes = plt.subplots(len(labels), 2, figsize=(13, 2.6 * len(labels)),
                             squeeze=False, facecolor=SURFACE)
    for row, label in enumerate(labels):
        d = hr[hr["label"] == label]
        pr = pairs[pairs["label"] == label]
        rtxt = "   ".join(f"P{int(r.p1):02d}–P{int(r.p2):02d} {r.r:+.2f}"
                          for r in pr.itertuples())
        for col, ax in enumerate(axes[row]):
            ax.set_facecolor(SURFACE)
            for p, g in d.groupby("participant"):
                y = g["hr_bpm"]
                if col == 1:
                    y = (y - y.mean()) / y.std()
                ax.plot(g["t_s"], y, lw=2, color=color[p], label=f"P{p:02d}")
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8)
            ax.set_ylabel("HR (bpm)" if col == 0 else "HR (z)", color=INK2, fontsize=9)
            if row == len(labels) - 1:
                ax.set_xlabel("time since segment onset (s)", color=INK2, fontsize=9)
        axes[row, 0].set_title(label.replace("_", " "), loc="left", color=INK,
                               fontsize=11, fontweight="bold")
        axes[row, 1].set_title(rtxt, loc="left", color=INK2, fontsize=8)

    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", ncol=len(pids), frameon=False,
               fontsize=9, labelcolor=INK)
    fig.suptitle("Resampled heart rate per segment (left: bpm, right: z-scored, "
                 "what Pearson r compares)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = a.out / "isc_overlay.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
