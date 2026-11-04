"""
Sync Firstbeat Bodyguard IBI recordings to the experiment logs and compute
simple inter-subject correlations (ISC) of heart rate per music tempo.

Usage:
    python sync_hr_isc.py                       # uses ./data, writes ./output
    python sync_hr_isc.py --data data --out output --fs 4 --detrend

Sync logic
----------
* The Firstbeat header "Start time: dd.mm.yyyy HH:MM:SS" is device-local time
  with no timezone. Beat times = start + cumsum(artifact-corrected RR).
* blocks.csv is split into "runs" at every `device_attached` row (restarts).
  A run is complete if it contains as many completed blocks as the run's
  highest block index (>= MIN_BLOCKS).
* For every whole-hour offset in [-MAX_H, +MAX_H] we measure how much of each
  complete run is covered by the shifted HR recording. The (run, offset) pair
  with the highest coverage wins; ties go to the smaller |offset| and then the
  later run. Use MANUAL_OFFSETS_H to override if needed.
* Session timestamps are taken from date + start_local (PC local time), so
  both clocks are compared as naive local wall-clock times.

ISC
---
For each segment (baseline, block per BPM, rest after each block) the HR
(60000 / RR) is linearly interpolated to a common grid (default 4 Hz) from
segment onset to the shortest common segment duration. Interpolation is
needed because beats occur at different times/rates per participant, and
Pearson r requires time-matched samples. For each condition we report
(no averaging / Fisher z):
  * Pearson r for every participant pair (isc_summary.csv, isc_pairwise.csv)
  * leave-one-out r (each participant vs. mean of the others)
"""
from __future__ import annotations

import argparse
import itertools
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend as sp_detrend

MAX_H = 3                    # search offsets -MAX_H..+MAX_H hours
MIN_BLOCKS = 3               # a complete run must have at least this many blocks
MIN_COVERAGE = 0.95          # required HR coverage of a run to accept the sync
MAX_GAP_S = 3.0              # segments with a beat gap longer than this get flagged
MANUAL_OFFSETS_H: dict[int, float] = {}   # e.g. {2: 2.0} to force participant 2


# --------------------------------------------------------------------------- IO
@dataclass
class HRRecording:
    pid: int
    path: Path
    start: datetime          # naive, device-local
    beat_t: np.ndarray       # seconds since start, time of each beat
    rr_ms: np.ndarray        # artifact-corrected RR
    artifact: np.ndarray     # raw artifact flag (bool)

    @property
    def duration_s(self) -> float:
        return float(self.beat_t[-1]) if len(self.beat_t) else 0.0


def read_hr(path: Path) -> HRRecording:
    m = re.search(r"participant_(\d+)", path.stem)
    if not m:
        raise ValueError(f"cannot find participant id in {path.name}")
    pid = int(m.group(1))
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start_line = next(l for l in lines if l.lower().startswith("start time"))
    start = datetime.strptime(start_line.split(":", 1)[1].strip(), "%d.%m.%Y %H:%M:%S")
    hdr_idx = next(i for i, l in enumerate(lines) if l.startswith("RR;"))
    # The raw and corrected columns are separate streams of different length
    # (correction inserts/merges beats), so parse each column on its own.
    rows = [(l.split(";") + ["", "", ""])[:3] for l in lines[hdr_idx + 1:] if l.strip()]
    col = lambda i: np.array([float(r[i]) for r in rows if r[i].strip() != ""])
    rr_raw, rr_corr, art = col(0), col(1), col(2)
    beat_t = np.cumsum(rr_corr) / 1000.0
    d_raw, d_corr = rr_raw.sum() / 1000, rr_corr.sum() / 1000
    if abs(d_raw - d_corr) > 2.0:
        print(f"[warn] {path.name}: raw {d_raw:.1f}s vs corrected {d_corr:.1f}s duration")
    return HRRecording(pid, path, start, beat_t, rr_corr, art != 0)


def read_blocks(pdir: Path) -> pd.DataFrame:
    df = pd.read_csv(pdir / "blocks.csv")
    df["start_dt"] = pd.to_datetime(df["date"] + " " + df["start_local"])
    df["stop_dt"] = pd.to_datetime(df["date"] + " " + df["stop_local"])
    df = df.sort_values("start_unix").reset_index(drop=True)
    df["run"] = (df["segment"] == "device_attached").cumsum()
    return df


def complete_runs(blocks: pd.DataFrame) -> list[pd.DataFrame]:
    runs = []
    for _, r in blocks.groupby("run"):
        b = r[(r["segment"] == "block") & (r["completed"] == "yes")]
        if len(b) >= MIN_BLOCKS and b["block_index"].nunique() == len(b):
            runs.append(r[r["segment"] != "device_attached"].copy())
    return runs


# ------------------------------------------------------------------------ sync
def coverage(rec: HRRecording, run: pd.DataFrame, offset_h: float) -> float:
    hr0 = rec.start + timedelta(hours=offset_h)
    hr1 = hr0 + timedelta(seconds=rec.duration_s)
    r0, r1 = run["start_dt"].min(), run["stop_dt"].max()
    ov = (min(hr1, r1) - max(hr0, r0)).total_seconds()
    return max(ov, 0.0) / (r1 - r0).total_seconds()


def sync(rec: HRRecording, blocks: pd.DataFrame) -> tuple[pd.DataFrame, float, float, list]:
    runs = complete_runs(blocks)
    if not runs:
        raise RuntimeError("no complete run in blocks.csv")
    offsets = ([MANUAL_OFFSETS_H[rec.pid]] if rec.pid in MANUAL_OFFSETS_H
               else list(range(-MAX_H, MAX_H + 1)))
    cands = []
    for ri, run in enumerate(runs):
        for h in offsets:
            cands.append((coverage(rec, run, h), -abs(h), ri, h))
    cands.sort(reverse=True)
    cov, _, ri, h = cands[0]
    return runs[ri], float(h), cov, cands


# ------------------------------------------------------------------ resampling
def segment_series(rec: HRRecording, offset_h: float, t0: datetime, dur: float, fs: float):
    """HR (bpm) on a regular grid from t0 for dur seconds. Returns (grid, hr, info)."""
    hr_start = rec.start + timedelta(hours=offset_h)
    seg0 = (t0 - hr_start).total_seconds()
    grid = np.arange(0, dur, 1.0 / fs)
    t = rec.beat_t - seg0
    hr = 60000.0 / rec.rr_ms
    inwin = (t >= -2) & (t <= dur + 2)            # small margin for interpolation
    tw, hw = t[inwin], hr[inwin]
    info = {"n_beats": int(((t >= 0) & (t <= dur)).sum()),
            "artifact_pct": float(100 * rec.artifact[(t >= 0) & (t <= dur)].mean()) if inwin.any() else np.nan}
    if len(tw) < 2 or tw[0] > 0 or tw[-1] < dur - 1.0 / fs:
        info["ok"] = False
        info["max_gap_s"] = np.nan
        return grid, np.full_like(grid, np.nan), info
    info["max_gap_s"] = float(np.diff(tw).max())
    info["ok"] = info["max_gap_s"] <= MAX_GAP_S
    return grid, np.interp(grid, tw, hw), info


# ------------------------------------------------------------------------- ISC
def isc(mat: dict[int, np.ndarray]):
    pids = sorted(mat)
    pair = [(a, b, float(np.corrcoef(mat[a], mat[b])[0, 1]))
            for a, b in itertools.combinations(pids, 2)]
    loo = []
    for p in pids:
        others = np.mean([mat[q] for q in pids if q != p], axis=0)
        loo.append((p, float(np.corrcoef(mat[p], others)[0, 1])))
    return pair, loo


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=here / "output")
    ap.add_argument("--fs", type=float, default=4.0, help="resampling rate (Hz)")
    ap.add_argument("--detrend", action="store_true",
                    help="remove linear trend per segment before ISC")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sync_rows, seg_rows, long_rows = [], [], []
    series = {}   # (condition_label) -> {pid: (grid, hr)}

    for f in sorted((args.data / "raw_hr").glob("*.csv")):
        rec = read_hr(f)
        pdir = args.data / f"participant-{rec.pid:02d}"
        if not pdir.exists():
            print(f"[skip] {f.name}: no {pdir.name}")
            continue
        run, h, cov, cands = sync(rec, read_blocks(pdir))
        alt = [c for c in cands if c[0] > 0 and (c[2], c[3]) != (cands[0][2], cands[0][3])]
        status = "ok" if cov >= MIN_COVERAGE else "LOW_COVERAGE"
        print(f"P{rec.pid:02d}: HR start {rec.start}  dur {rec.duration_s:6.1f}s  "
              f"offset {h:+.0f} h  run {run['start_dt'].min():%Y-%m-%d %H:%M:%S}"
              f"-{run['stop_dt'].max():%H:%M:%S}  coverage {cov:.0%}  [{status}]")
        sync_rows.append(dict(
            participant=rec.pid, hr_file=f.name, hr_start_device=rec.start,
            hr_duration_s=rec.duration_s, offset_h=h,
            hr_start_aligned=rec.start + timedelta(hours=h),
            run_start=run["start_dt"].min(), run_stop=run["stop_dt"].max(),
            run_coverage=cov, status=status,
            runner_up=(f"cov={alt[0][0]:.2f} offset={alt[0][3]:+d}h" if alt else ""),
            artifact_pct_total=100 * rec.artifact.mean()))
        if status != "ok":
            continue

        for _, s in run.iterrows():
            seg = s["segment"]
            bi = int(s["block_index"]) if pd.notna(s["block_index"]) else 0
            if seg == "block":
                bpm = int(s["condition_bpm"])
                label = f"music_{bpm}"
            elif seg == "rest":
                prev = run[(run["segment"] == "block") & (run["block_index"] == bi)]
                bpm = int(prev["condition_bpm"].iloc[0])
                label = f"rest_after_{bpm}"
            else:
                bpm, label = np.nan, seg
            seg_rows.append(dict(participant=rec.pid, segment=seg, label=label,
                                 block_index=bi or np.nan, condition_bpm=bpm,
                                 start=s["start_dt"], stop=s["stop_dt"],
                                 duration_s=s["duration_s"]))
            series.setdefault(label, {})[rec.pid] = (s["start_dt"], s["duration_s"], rec, h, bi)

    pd.DataFrame(sync_rows).to_csv(args.out / "sync_report.csv", index=False)

    # resample and compute ISC per condition on a common duration
    pair_rows, loo_rows, sum_rows, qc_rows = [], [], [], []
    for label, d in sorted(series.items()):
        dur = min(v[1] for v in d.values())
        mat = {}
        for pid, (t0, _, rec, h, bi) in d.items():
            grid, hr, info = segment_series(rec, h, t0, dur, args.fs)
            qc_rows.append(dict(participant=pid, label=label, block_index=bi or np.nan,
                                duration_used_s=dur, **info))
            for tt, v in zip(grid, hr):
                long_rows.append(dict(participant=pid, label=label, block_index=bi or np.nan,
                                      t_s=round(tt, 4), hr_bpm=v))
            if info["ok"]:
                mat[pid] = sp_detrend(hr) if args.detrend else hr
        if len(mat) < 2:
            continue
        pair, loo = isc(mat)
        pair_rows += [dict(label=label, p1=a, p2=b, r=r) for a, b, r in pair]
        loo_rows += [dict(label=label, participant=p, r_loo=r) for p, r in loo]
        row = dict(label=label, n=len(mat), duration_s=dur)
        row.update({f"r_P{a:02d}_P{b:02d}": r for a, b, r in pair})
        sum_rows.append(row)

    pd.DataFrame(seg_rows).to_csv(args.out / "segments.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(args.out / "segment_qc.csv", index=False)
    pd.DataFrame(long_rows).to_csv(args.out / "synced_hr_long.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(args.out / "isc_pairwise.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(args.out / "isc_leave_one_out.csv", index=False)
    summary = pd.DataFrame(sum_rows)
    summary.to_csv(args.out / "isc_summary.csv", index=False)
    print("\nISC summary" + (" (detrended)" if args.detrend else "") + ":")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nOutputs written to {args.out}")


if __name__ == "__main__":
    main()
