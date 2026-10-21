#!/usr/bin/env python
"""Prepare the three tempo stimuli for playback by run_experiment.py.

The source MP3s in stimuli/ are never modified. For each one this writes a
48 kHz / stereo / 24-bit WAV into stimuli_prepared/ holding exactly 210.000 s
of audio starting at the musical onset, loudness-matched across conditions and
with the clipping present in the sources removed.

Why each step is needed (measured from the actual source files):

  file            duration   rate      loudness    true peak
  90_BPM.mpeg     372.41 s   48 kHz    -9.6 LUFS   +1.4 dBTP
  120_BPM.mpeg    280.80 s   48 kHz    -9.6 LUFS   +1.0 dBTP
  150_BPM.mpeg    224.05 s   44.1 kHz  -9.9 LUFS   +0.5 dBTP

  * sample rates disagree, so one file would be resampled at run time
  * all three exceed 0 dBTP, i.e. they clip on playback
  * loudness differs by 0.3 LU, and playback level is a control variable
  * MP3 decoding adds a variable leading delay, which would smear stimulus onset

Normalisation is a single constant gain per file, computed from the loudness of
the 210 s excerpt actually used (truncation changes integrated loudness, so
measuring the full file would give the wrong number). No compression or
limiting is applied: the required gain is an attenuation, so nothing clips and
the music's dynamics are left untouched.

Run once:

    py -3.11 prepare_stimuli.py

Requires numpy and ffmpeg/ffprobe. On this machine ffmpeg ships with
mne-python; the script finds it automatically or takes --ffmpeg.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "stimuli"
OUT_DIR = ROOT / "stimuli_prepared"

CONDITIONS = {90: "90_BPM.mpeg", 120: "120_BPM.mpeg", 150: "150_BPM.mpeg"}

SAMPLE_RATE = 48_000
CHANNELS = 2
BLOCK_SECONDS = 210.0
BLOCK_FRAMES = int(round(BLOCK_SECONDS * SAMPLE_RATE))  # 10_080_000

TARGET_LUFS = -16.0
MAX_TRUE_PEAK_DBTP = -1.5
LUFS_TOLERANCE = 0.3

FADE_IN_S = 0.100
FADE_OUT_S = 0.500

# Musical-onset detection. Anything quieter than this counts as lead-in silence.
ONSET_THRESHOLD_DBFS = -50.0
ONSET_WINDOW_S = 0.010
ONSET_BACKOFF_S = 0.010
# If the detected onset is later than this, the file is not what we think it is.
ONSET_SANITY_LIMIT_S = 5.0

# Fallback search locations for ffmpeg, beyond PATH.
FFMPEG_HINTS = [Path(r"C:\Users\Bruger\mne-python\1.8.0_0\Library\bin")]


class PrepError(RuntimeError):
    """A QC gate failed. The stimulus set must not be used."""


# --------------------------------------------------------------------------
# ffmpeg plumbing
# --------------------------------------------------------------------------


def locate_tool(name: str, override: Path | None) -> Path:
    if override is not None:
        candidate = override if override.is_file() else override / f"{name}.exe"
        if candidate.is_file():
            return candidate
        raise PrepError(f"--ffmpeg given but {name} not found at {candidate}")

    found = shutil.which(name)
    if found:
        return Path(found)

    for hint in FFMPEG_HINTS:
        for suffix in (".exe", ""):
            candidate = hint / f"{name}{suffix}"
            if candidate.is_file():
                return candidate

    raise PrepError(
        f"{name} not found on PATH. Pass --ffmpeg <dir containing ffmpeg and ffprobe>."
    )


def run_capture(cmd: list[str], stdin_data: bytes | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        input=stdin_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-15:]
        raise PrepError(
            "ffmpeg failed (exit %d):\n  %s\n%s"
            % (proc.returncode, " ".join(cmd[:2]), "\n".join("    " + t for t in tail))
        )
    return proc


_RESAMPLE_FILTER: str | None = None


def resample_filter(ffmpeg: Path) -> str:
    """Pick the best resampler this ffmpeg build actually has.

    Only the 150 BPM source needs resampling (44.1 -> 48 kHz). soxr is preferred,
    but not every build ships it -- the mne-python build on this machine does not --
    and swresample's default is perfectly adequate for a music stimulus.
    """
    global _RESAMPLE_FILTER
    if _RESAMPLE_FILTER is not None:
        return _RESAMPLE_FILTER

    probe = subprocess.run(
        [str(ffmpeg), "-nostdin", "-v", "error", "-f", "lavfi",
         "-i", "anullsrc=r=44100:cl=stereo", "-t", "0.01",
         "-af", "aresample=48000:resampler=soxr:precision=28",
         "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if probe.returncode == 0:
        _RESAMPLE_FILTER = f"aresample={SAMPLE_RATE}:resampler=soxr:precision=28"
    else:
        _RESAMPLE_FILTER = f"aresample={SAMPLE_RATE}"
        print("  note: this ffmpeg has no soxr; using the built-in resampler")
    return _RESAMPLE_FILTER


def decode_to_array(ffmpeg: Path, path: Path) -> np.ndarray:
    """Decode any input to float32 stereo at SAMPLE_RATE, shape (frames, 2)."""
    cmd = [
        str(ffmpeg), "-nostdin", "-v", "error",
        "-i", str(path),
        "-map", "0:a:0", "-vn",
        "-af", resample_filter(ffmpeg),
        "-ac", str(CHANNELS),
        "-f", "f32le", "-",
    ]
    raw = run_capture(cmd).stdout
    audio = np.frombuffer(raw, dtype="<f4")
    if audio.size % CHANNELS:
        raise PrepError(f"{path.name}: decoded sample count is not a whole number of frames")
    return audio.reshape(-1, CHANNELS)


def write_wav(ffmpeg: Path, audio: np.ndarray, path: Path) -> None:
    """Write float32 (frames, 2) to a 24-bit PCM WAV."""
    cmd = [
        str(ffmpeg), "-nostdin", "-v", "error", "-y",
        "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-i", "-",
        "-c:a", "pcm_s24le",
        str(path),
    ]
    run_capture(cmd, stdin_data=audio.astype("<f4", copy=False).tobytes())


def measure_loudness(ffmpeg: Path, path: Path) -> dict:
    """Return EBU R128 measurements for a file via ffmpeg's loudnorm analysis pass.

    Used only as a measurement tool -- the gain is applied separately in numpy so
    that the output length is exact and no dynamics processing is involved.
    """
    cmd = [
        str(ffmpeg), "-nostdin", "-v", "info", "-nostats",
        "-i", str(path),
        "-af", f"loudnorm=I={TARGET_LUFS}:TP={MAX_TRUE_PEAK_DBTP}:LRA=11:print_format=json",
        "-f", "null", "-",
    ]
    stderr = run_capture(cmd).stderr.decode("utf-8", "replace")

    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1:
        raise PrepError(f"could not parse loudnorm output for {path.name}")
    payload = json.loads(stderr[start : end + 1])

    return {
        "integrated_lufs": float(payload["input_i"]),
        "true_peak_dbtp": float(payload["input_tp"]),
        "lra_lu": float(payload["input_lra"]),
    }


# --------------------------------------------------------------------------
# audio processing
# --------------------------------------------------------------------------


def find_musical_onset(audio: np.ndarray) -> int:
    """First frame at which the music starts, in frames.

    The three files are one piece time-stretched, so their lead-in silences differ
    in proportion to tempo. Trimming each to its own onset is what makes the
    musical start -- not the file start -- line up with the logged MUSIC_ON event.
    """
    window = max(1, int(round(ONSET_WINDOW_S * SAMPLE_RATE)))
    mono = audio.mean(axis=1)

    usable = (mono.size // window) * window
    frames = mono[:usable].reshape(-1, window)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))

    threshold = 10.0 ** (ONSET_THRESHOLD_DBFS / 20.0)
    loud = np.flatnonzero(rms > threshold)
    if loud.size == 0:
        raise PrepError("no audio above the silence threshold -- is the file empty?")

    onset = loud[0] * window - int(round(ONSET_BACKOFF_S * SAMPLE_RATE))
    return max(0, onset)


def apply_fades(audio: np.ndarray) -> np.ndarray:
    """Raised-cosine fades. Without these the 210 s cut starts and ends on a step,
    which is an audible click -- itself a startle stimulus at the exact moment we
    care most about the cardiac response."""
    out = audio.copy()

    n_in = int(round(FADE_IN_S * SAMPLE_RATE))
    n_out = int(round(FADE_OUT_S * SAMPLE_RATE))

    ramp_in = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_in, dtype=np.float64)))
    out[:n_in] *= ramp_in[:, None].astype(np.float32)

    ramp_out = 0.5 * (1.0 - np.cos(np.linspace(np.pi, 0.0, n_out, dtype=np.float64)))
    out[-n_out:] *= ramp_out[:, None].astype(np.float32)

    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# per-file pipeline
# --------------------------------------------------------------------------


def prepare_one(ffmpeg: Path, bpm: int, src: Path, tmp_dir: Path) -> dict:
    print(f"\n[{bpm} BPM] {src.name}")

    audio = decode_to_array(ffmpeg, src)
    src_seconds = audio.shape[0] / SAMPLE_RATE
    print(f"  decoded          {src_seconds:8.3f} s at {SAMPLE_RATE} Hz, {CHANNELS} ch")

    onset = find_musical_onset(audio)
    onset_s = onset / SAMPLE_RATE
    if onset_s > ONSET_SANITY_LIMIT_S:
        raise PrepError(
            f"{src.name}: musical onset detected at {onset_s:.3f} s, which is beyond the "
            f"{ONSET_SANITY_LIMIT_S:.1f} s sanity limit. Inspect the file before trusting this."
        )
    print(f"  musical onset    {onset_s:8.3f} s")

    available = audio.shape[0] - onset
    if available < BLOCK_FRAMES:
        raise PrepError(
            f"{src.name}: only {available / SAMPLE_RATE:.3f} s of audio after the musical "
            f"onset, but a block needs {BLOCK_SECONDS:.3f} s."
        )

    excerpt = apply_fades(audio[onset : onset + BLOCK_FRAMES])
    if excerpt.shape[0] != BLOCK_FRAMES:
        raise PrepError(f"{src.name}: excerpt is {excerpt.shape[0]} frames, expected {BLOCK_FRAMES}")

    # Measure the excerpt, not the source file: truncation changes integrated loudness.
    probe_path = tmp_dir / f"{bpm}_probe.wav"
    write_wav(ffmpeg, excerpt, probe_path)
    before = measure_loudness(ffmpeg, probe_path)
    print(
        f"  excerpt loudness {before['integrated_lufs']:8.2f} LUFS, "
        f"true peak {before['true_peak_dbtp']:+.2f} dBTP"
    )

    gain_db = TARGET_LUFS - before["integrated_lufs"]
    projected_peak = before["true_peak_dbtp"] + gain_db
    if projected_peak > MAX_TRUE_PEAK_DBTP:
        raise PrepError(
            f"{src.name}: reaching {TARGET_LUFS} LUFS needs {gain_db:+.2f} dB, which would put "
            f"true peak at {projected_peak:+.2f} dBTP, above the {MAX_TRUE_PEAK_DBTP} dBTP "
            f"ceiling. Lower TARGET_LUFS rather than adding a limiter."
        )

    normalised = excerpt * np.float32(10.0 ** (gain_db / 20.0))
    print(f"  applied gain     {gain_db:+8.2f} dB")

    out_path = OUT_DIR / f"{bpm}.wav"
    write_wav(ffmpeg, normalised, out_path)

    # ---- QC gates. Verify the file on disk, not the array we think we wrote. ----
    after = measure_loudness(ffmpeg, out_path)

    written = decode_to_array(ffmpeg, out_path)
    if written.shape[0] != BLOCK_FRAMES:
        raise PrepError(
            f"{out_path.name}: {written.shape[0]} frames on disk, expected exactly {BLOCK_FRAMES}"
        )

    lufs_error = abs(after["integrated_lufs"] - TARGET_LUFS)
    if lufs_error > LUFS_TOLERANCE:
        raise PrepError(
            f"{out_path.name}: {after['integrated_lufs']:.2f} LUFS, off target by "
            f"{lufs_error:.2f} LU (tolerance {LUFS_TOLERANCE} LU)"
        )

    if after["true_peak_dbtp"] > MAX_TRUE_PEAK_DBTP:
        raise PrepError(
            f"{out_path.name}: true peak {after['true_peak_dbtp']:+.2f} dBTP exceeds the "
            f"{MAX_TRUE_PEAK_DBTP} dBTP ceiling"
        )

    print(
        f"  WROTE {out_path.name}  {written.shape[0]} frames "
        f"({written.shape[0] / SAMPLE_RATE:.3f} s), "
        f"{after['integrated_lufs']:.2f} LUFS, {after['true_peak_dbtp']:+.2f} dBTP  [OK]"
    )

    probe_path.unlink(missing_ok=True)

    return {
        "bpm": bpm,
        "filename": out_path.name,
        "sha256": sha256(out_path),
        "frames": int(written.shape[0]),
        "duration_s": round(written.shape[0] / SAMPLE_RATE, 6),
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "integrated_lufs": round(after["integrated_lufs"], 2),
        "true_peak_dbtp": round(after["true_peak_dbtp"], 2),
        "lra_lu": round(after["lra_lu"], 2),
        "gain_applied_db": round(gain_db, 3),
        "source_file": src.name,
        "source_sha256": sha256(src),
        "source_duration_s": round(src_seconds, 3),
        "musical_onset_s": round(onset_s, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=None,
        help="path to ffmpeg, or to the directory containing ffmpeg and ffprobe",
    )
    args = parser.parse_args()

    try:
        ffmpeg = locate_tool("ffmpeg", args.ffmpeg)
    except PrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"ffmpeg: {ffmpeg}")
    print(f"target: {BLOCK_SECONDS:.3f} s ({BLOCK_FRAMES} frames) @ {TARGET_LUFS} LUFS")

    OUT_DIR.mkdir(exist_ok=True)
    tmp_dir = OUT_DIR / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    entries = []
    try:
        for bpm, filename in sorted(CONDITIONS.items()):
            src = SRC_DIR / filename
            if not src.is_file():
                raise PrepError(f"missing source stimulus: {src}")
            entries.append(prepare_one(ffmpeg, bpm, src, tmp_dir))
    except PrepError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        print("Stimulus set is NOT usable. Nothing was written to manifest.json.", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": os.path.basename(__file__),
        "block_seconds": BLOCK_SECONDS,
        "block_frames": BLOCK_FRAMES,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "target_lufs": TARGET_LUFS,
        "max_true_peak_dbtp": MAX_TRUE_PEAK_DBTP,
        "fade_in_s": FADE_IN_S,
        "fade_out_s": FADE_OUT_S,
        "resample_filter": resample_filter(ffmpeg),
        "stimuli": {str(entry["bpm"]): entry for entry in entries},
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    spread = max(e["integrated_lufs"] for e in entries) - min(e["integrated_lufs"] for e in entries)
    print(f"\nAll QC gates passed. Wrote {manifest_path}")
    print(f"Loudness spread across conditions: {spread:.2f} LU")
    print(
        "\nRemaining manual check before the pilot: play each prepared WAV against a\n"
        "metronome at its nominal BPM and confirm they match. Tempo is the independent\n"
        "variable, so it should be verified by ear once rather than assumed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
