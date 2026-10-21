#!/usr/bin/env python
"""Music tempo and heart rate experiment -- single-participant session script.

One participant wears a Firstbeat device that records on its own. This script
plays the three tempo blocks with a 2 min rest between them and records when
every block starts and stops. The Firstbeat data is lined up with these times
by hand afterwards, to the nearest second.

Session structure:

    baseline (5 min, silent)
    block 1  (210 s music)  ->  rest (2 min)
    block 2  (210 s music)  ->  rest (2 min)
    block 3  (210 s music)

Only the baseline needs SPACE to start; everything after runs on its own. The
last 10 s of the baseline and of each rest show "Block N, music will start
shortly", and the music starts as soon as that period ends.

Each session writes to data/participant-XX/:

    blocks.csv    device attach time, then one row per baseline/block/rest:
                  start/stop as local clock time (HH:MM:SS) and Unix time.
                  This is the file to sync against.
    events.csv    every event in order, including 30 s heartbeats.
    session_*.json  settings, block order, stimuli manifest, PC clock offset note.

The Firstbeat device starts recording when the electrodes are clipped on. The
first screen asks the operator to press SPACE at that moment; the PC time is
logged as DEVICE_ATTACHED (first row of blocks.csv). Compare it with the
recording start in the Firstbeat file to get the clock offset.

Usage:

    .venv\\Scripts\\python run_experiment.py                  # dialog for participant ID
    .venv\\Scripts\\python run_experiment.py --participant 3
    .venv\\Scripts\\python run_experiment.py --participant 99 --time-scale 0.05 --windowed --auto-confirm
    .venv\\Scripts\\python run_experiment.py --list-audio     # print output devices

Abort at any time by HOLDING Escape for 1 second and releasing. A momentary
press is ignored, so a brush against the key cannot end a session.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime
from itertools import permutations
from pathlib import Path

# Audio backend must be chosen before psychopy.sound is imported anywhere.
from psychopy import prefs  # noqa: E402

# Lowercase "ptb" -- PsychoPy 2026.2 validates this against a lowercase list and
# rejects the older "PTB" spelling outright.
prefs.hardware["audioLib"] = ["ptb"]
prefs.hardware["audioLatencyMode"] = 3  # aggressive / exclusive low-latency mode

import psychopy  # noqa: E402
from psychopy import core, gui, sound, visual  # noqa: E402
from psychopy.hardware import keyboard  # noqa: E402

ROOT = Path(__file__).resolve().parent
STIM_DIR = ROOT / "stimuli_prepared"
MANIFEST_PATH = STIM_DIR / "manifest.json"
DATA_DIR = ROOT / "data"

# --- timing, in seconds. All of these scale with --time-scale. ---
BASELINE_S = 15.0 # 300.0 
BLOCK_S = 60.0
ANNOUNCE_S = 10.0  # "music will start shortly" shown for the last 10 s of baseline/rest
REST_S = 30.0
HEARTBEAT_S = 30.0

ABORT_HOLD_S = 1.0
FADE_OUT_S = 0.5

CONDITIONS = (90, 120, 150)
ORDERS = list(permutations(CONDITIONS))  # 6 counterbalanced orders

WIN_COLOR = [-0.7, -0.7, -0.7]
TEXT_COLOR = [0.55, 0.55, 0.55]
FIXATION_COLOR = [0.35, 0.35, 0.35]

PREFLIGHT_ITEMS = [
    "Firstbeat device is fitted and RECORDING.",
    "Playback volume at the pinned level.",
    "Participant briefed: no talking, stay still, look at the cross.",
]


class Aborted(Exception):
    """Operator ended the session early."""


class PreflightFailed(Exception):
    """A pre-flight check failed. Do not run."""


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------


def stamp(t_unix: float) -> tuple[str, str]:
    """Local date and local HH:MM:SS.mmm for a Unix time."""
    dt = datetime.fromtimestamp(t_unix)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


class CsvFile:
    """Append-only CSV, fsynced after every row so a crash loses nothing."""

    def __init__(self, path: Path, columns: list[str]):
        is_new = not path.exists() or path.stat().st_size == 0
        self._fh = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        if is_new:
            self.write(columns)

    def write(self, row):
        self._writer.writerow(row)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


class EventLog:
    """events.csv (everything) plus blocks.csv (one row per block/rest, for syncing)."""

    EVENT_COLUMNS = ["event", "block_index", "condition_bpm", "date", "local_time", "t_unix", "t_session_s", "note"]
    BLOCK_COLUMNS = [
        "segment", "block_index", "condition_bpm", "date",
        "start_local", "stop_local", "start_unix", "stop_unix", "duration_s", "completed",
    ]

    def __init__(self, session_dir: Path):
        self.events = CsvFile(session_dir / "events.csv", self.EVENT_COLUMNS)
        self.blocks = CsvFile(session_dir / "blocks.csv", self.BLOCK_COLUMNS)
        self.t0 = time.time()
        self._open = None  # (segment, block_index, bpm, start_unix) of the running segment

    def log(self, event_name, block_index="", bpm="", note="", t_unix=None):
        t = time.time() if t_unix is None else t_unix
        date, local = stamp(t)
        self.events.write([event_name, block_index, bpm, date, local, f"{t:.3f}", f"{t - self.t0:.3f}", note])
        print(f"  [{local}] {event_name} {block_index} {bpm} {note}".rstrip())
        return t

    def start_segment(self, segment, block_index="", bpm="", t_unix=None):
        t = self.log(f"{segment.upper()}_START", block_index, bpm, t_unix=t_unix)
        self._open = (segment, block_index, bpm, t)

    def mark(self, segment, note=""):
        """A single point in time, logged to both files (start = stop)."""
        t = self.log(segment.upper(), note=note)
        date, local = stamp(t)
        self.blocks.write([segment, "", "", date, local, local, f"{t:.3f}", f"{t:.3f}", "0.000", "yes"])
        return t

    def stop_segment(self, completed=True, note=""):
        if self._open is None:
            return
        segment, block_index, bpm, start = self._open
        self._open = None
        stop = self.log(f"{segment.upper()}_STOP", block_index, bpm, note=note)
        date, start_local = stamp(start)
        _, stop_local = stamp(stop)
        self.blocks.write([
            segment, block_index, bpm, date, start_local, stop_local,
            f"{start:.3f}", f"{stop:.3f}", f"{stop - start:.3f}", "yes" if completed else "no",
        ])

    def close(self):
        self.events.close()
        self.blocks.close()


# --------------------------------------------------------------------------
# setup helpers
# --------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_verify_manifest() -> dict:
    """Refuse to run against stimuli that are missing, stale or modified."""
    if not MANIFEST_PATH.is_file():
        raise PreflightFailed(f"{MANIFEST_PATH} not found. Run prepare_stimuli.py first.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    for bpm in CONDITIONS:
        entry = manifest["stimuli"].get(str(bpm))
        if entry is None:
            raise PreflightFailed(f"manifest has no entry for {bpm} BPM")
        path = STIM_DIR / entry["filename"]
        if not path.is_file():
            raise PreflightFailed(f"missing prepared stimulus: {path}")
        if sha256(path) != entry["sha256"]:
            raise PreflightFailed(f"{path.name} does not match the manifest. Re-run prepare_stimuli.py.")
    return manifest


def resolve_order(participant_id: int) -> tuple[tuple[int, int, int], int]:
    """Block order for a participant: permutation (participant_id - 1) % 6."""
    perm_index = (participant_id - 1) % len(ORDERS)
    return ORDERS[perm_index], perm_index


def safe(text: str) -> str:
    """Make a device name printable on a Windows console with a legacy code page."""
    encoding = sys.stdout.encoding or "ascii"
    return str(text).encode(encoding, "replace").decode(encoding, "replace")


def available_speakers() -> list[dict]:
    from psychopy.hardware.speaker import SpeakerDevice

    return SpeakerDevice.getAvailableDevices()


def pin_speaker(name: str | None) -> str:
    """Pick the output device. Pin it by name with --speaker so it can't change between sessions."""
    devices = available_speakers()
    if not devices:
        raise PreflightFailed("no audio output devices found")

    if name:
        names = [d["deviceName"] for d in devices]
        if name not in names:
            listing = "\n".join(f"  - {safe(n)}" for n in names)
            raise PreflightFailed(f"--speaker {name!r} does not match any output device. Available:\n{listing}")
        prefs.hardware["audioDevice"] = name
        return name

    configured = prefs.hardware["audioDevice"]
    configured = configured[0] if isinstance(configured, (list, tuple)) else configured
    if configured not in ("default", "None", None, ""):
        return str(configured)
    return devices[0]["deviceName"]


def list_audio_devices() -> int:
    try:
        devices = available_speakers()
    except Exception as exc:
        print(f"Could not query audio devices: {exc}")
        return 1
    print("Output devices (pass the exact name to --speaker):")
    for device in devices:
        print(f"  - {safe(device.get('deviceName'))}")
    return 0


def ask_session_details(args) -> dict:
    if args.participant is not None:
        return {
            "participant_id": args.participant,
            "participant_code": args.code or f"P{args.participant:02d}",
            "operator": args.operator or "",
        }

    dlg = gui.Dlg(title="Music tempo experiment -- session setup")
    dlg.addField("Participant ID (integer):", 1)
    dlg.addField("Participant code:", "")
    dlg.addField("Operator initials:", "")
    values = dlg.show()
    if not dlg.OK:
        print("Cancelled at the setup dialog.")
        core.quit()

    values = list(values.values()) if isinstance(values, dict) else list(values)
    participant_id = int(values[0])
    return {
        "participant_id": participant_id,
        "participant_code": str(values[1]) or f"P{participant_id:02d}",
        "operator": str(values[2]),
    }


# --------------------------------------------------------------------------
# screen primitives
# --------------------------------------------------------------------------


class Session:
    def __init__(self, win, kb, log, stims, time_scale, dry_run_banner, auto_confirm=False):
        self.win = win
        self.kb = kb
        self.log = log
        self.stims = stims
        self.time_scale = time_scale
        self.banner = dry_run_banner
        self.auto_confirm = auto_confirm
        self.message = visual.TextStim(
            win, text="", color=TEXT_COLOR, height=0.045, wrapWidth=1.5, alignText="center"
        )
        self.fixation = visual.TextStim(win, text="+", color=FIXATION_COLOR, height=0.10)
        self.announcement = visual.TextStim(
            win, text="", color=TEXT_COLOR, height=0.045, wrapWidth=1.5, alignText="center"
        )

    def _decorate(self):
        if self.banner is not None:
            self.banner.draw()

    def show_text(self, text, wait_keys=("space",)):
        """Display text until one of wait_keys is pressed. Escape aborts immediately here."""
        self.message.text = text
        allowed = list(wait_keys) + ["escape"]
        self.kb.clearEvents()

        if self.auto_confirm:
            # Dry runs only -- main() refuses this flag at time_scale 1.0.
            self.message.draw()
            self._decorate()
            self.win.flip()
            core.wait(0.25)
            return allowed[0]

        while True:
            self.message.draw()
            self._decorate()
            self.win.flip()
            for press in self.kb.getKeys(allowed, waitRelease=False):
                if press.name == "escape":
                    raise Aborted("escape at an instruction screen")
                return press.name

    def _check_abort(self):
        for press in self.kb.getKeys(["escape"], waitRelease=True):
            if press.duration is not None and press.duration >= ABORT_HOLD_S:
                raise Aborted(f"escape held {press.duration:.2f}s")

    def wait(self, duration, block_index="", bpm="", label="", stim=None, announce=None):
        """Hold a fixation cross (or `stim`) for `duration`, logging heartbeats and watching for abort.

        `announce` = (next_block_index, next_bpm): for the final ANNOUNCE_S of the wait,
        show the "music will start shortly" text for that block instead.
        """
        stim = stim or self.fixation
        clock = core.Clock()
        heartbeat_every = HEARTBEAT_S * self.time_scale
        next_heartbeat = heartbeat_every
        announce_at = duration - ANNOUNCE_S * self.time_scale if announce else None

        self.kb.clearEvents()
        while clock.getTime() < duration:
            if announce_at is not None and clock.getTime() >= announce_at:
                next_block, next_bpm = announce
                self.announcement.text = (
                    f"Block {next_block} of {len(CONDITIONS)}.\n\n"
                    "Music will start shortly. Please look at the cross,\n"
                    "stay still, and simply listen."
                )
                self.log.log("ANNOUNCE", next_block, next_bpm, note=f"{label} ends in {duration - clock.getTime():.1f}s")
                stim = self.announcement
                announce_at = None

            stim.draw()
            self._decorate()
            self.win.flip()
            self._check_abort()

            elapsed = clock.getTime()
            if elapsed >= next_heartbeat:
                self.log.log("HEARTBEAT", block_index, bpm, note=f"{label} t+{elapsed:.1f}s")
                while next_heartbeat <= elapsed:
                    next_heartbeat += heartbeat_every

    # -- session components --------------------------------------------------

    def device_attached(self) -> float:
        """Log the PC time at which the Firstbeat electrodes were clipped on (= its recording start)."""
        self.show_text(
            "ATTACH THE FIRSTBEAT DEVICE\n\n\n"
            "Clip the device onto the electrodes and press SPACE\n"
            "at the same moment.\n\n"
            "The PC time is logged and later compared with the\n"
            "recording start time in the Firstbeat file."
        )
        return self.log.mark("device_attached", note="SPACE pressed as electrodes clipped on")

    def preflight(self, details, audio_device) -> list[str]:
        confirmed = []
        total = len(PREFLIGHT_ITEMS)
        for index, item in enumerate(PREFLIGHT_ITEMS, start=1):
            key = self.show_text(
                f"PRE-FLIGHT  {index} / {total}\n\n\n{item}\n\n\ny = confirmed        n = stop",
                wait_keys=("y", "n"),
            )
            if key == "n":
                raise Aborted(f"pre-flight item {index} not confirmed: {item.splitlines()[0]}")
            confirmed.append(item.replace("\n", " "))

        order_text = "  ".join(f"{i}. {bpm} BPM" for i, bpm in enumerate(details["order"], 1))
        self.show_text(
            f"Participant {details['participant_id']}   ({details['participant_code']})\n\n"
            f"Block order:  {order_text}\n\n"
            f"Audio out:  {audio_device}\n\n\n"
            f"Check the order against the run sheet, then press SPACE to begin."
        )
        return confirmed

    def baseline(self, first_bpm):
        self.show_text(
            "Baseline.\n\n"
            "Please sit comfortably, look at the cross, and stay still.\n"
            "There will be no sound during this period.\n\n\n"
            "Press SPACE to start."
        )
        self.log.start_segment("baseline")
        self.wait(BASELINE_S * self.time_scale, label="baseline", announce=(1, first_bpm))
        self.log.stop_segment()

    def play_block(self, block_index, bpm):
        """Starts immediately -- the preceding baseline/rest has already announced it."""
        stim = self.stims[bpm]
        try:
            stim.stop()
        except Exception:
            pass
        started = {}

        def fire(store=started):
            stim.play()
            store["t"] = time.time()

        self.kb.clearEvents()
        self.win.callOnFlip(fire)
        self.fixation.draw()
        self._decorate()
        self.win.flip()
        self.log.start_segment("block", block_index, bpm, t_unix=started.get("t"))

        try:
            self.wait(BLOCK_S * self.time_scale, block_index, bpm, "music")
        finally:
            stim.stop()
        self.log.stop_segment()

    def rest(self, after_block, next_bpm):
        self.message.text = "Rest.\n\nPlease sit quietly until the next block begins."
        self.log.start_segment("rest", after_block)
        self.wait(
            REST_S * self.time_scale, after_block, label="rest", stim=self.message,
            announce=(after_block + 1, next_bpm),
        )
        self.log.stop_segment()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--participant", type=int, default=None, help="participant ID; omit to get a dialog")
    parser.add_argument("--code", default=None, help="participant code (default P<ID>)")
    parser.add_argument("--operator", default=None)
    parser.add_argument("--time-scale", type=float, default=1.0, help="multiply all durations, for dry runs (e.g. 0.05)")
    parser.add_argument("--windowed", action="store_true", help="run in a window, for development")
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="advance every screen automatically. Dry runs only -- rejected at --time-scale 1.0.",
    )
    parser.add_argument("--list-audio", action="store_true", help="print output devices and exit")
    parser.add_argument("--speaker", default=None, help="exact output device name to pin playback to (see --list-audio)")
    args = parser.parse_args()

    if args.list_audio:
        return list_audio_devices()

    if args.time_scale <= 0:
        print("ERROR: --time-scale must be positive", file=sys.stderr)
        return 2

    if args.auto_confirm and args.time_scale == 1.0:
        print("ERROR: --auto-confirm is only accepted together with --time-scale != 1.0.", file=sys.stderr)
        return 2

    try:
        manifest = load_and_verify_manifest()
        audio_device = pin_speaker(args.speaker)
    except PreflightFailed as exc:
        print(f"\nPRE-FLIGHT FAILED:\n{exc}\n", file=sys.stderr)
        return 1
    print(f"Audio output: {safe(audio_device)}{'' if args.speaker else '  (not pinned)'}")

    details = ask_session_details(args)
    order, perm_index = resolve_order(details["participant_id"])
    details["order"] = list(order)

    session_dir = DATA_DIR / f"participant-{details['participant_id']:02d}"
    session_dir.mkdir(parents=True, exist_ok=True)
    log = EventLog(session_dir)

    started_local = datetime.now().strftime("%Y%m%dT%H%M%S")
    record = {
        **details,
        "perm_index": perm_index,
        "time_scale": args.time_scale,
        "is_dry_run": args.time_scale != 1.0,
        "auto_confirmed": args.auto_confirm,
        "started_local": datetime.now().isoformat(timespec="seconds"),
        "started_unix": time.time(),
        "pc_utc_offset": datetime.now().astimezone().strftime("%z"),
        "durations_s": {
            "baseline": BASELINE_S * args.time_scale,
            "announce_before_block": ANNOUNCE_S * args.time_scale,
            "block": BLOCK_S * args.time_scale,
            "rest": REST_S * args.time_scale,
        },
        "manifest": manifest,
        "audio_device": audio_device,
        "audio_device_pinned": bool(args.speaker),
        "psychopy_version": psychopy.__version__,
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }

    win = visual.Window(
        fullscr=not args.windowed,
        size=(1280, 720),
        color=WIN_COLOR,
        units="height",
        allowGUI=args.windowed,
        winType="pyglet",
    )
    win.mouseVisible = False

    banner = None
    if args.time_scale != 1.0:
        banner = visual.TextStim(
            win,
            text=f"DRY RUN  x{args.time_scale:g}  -- NOT VALID DATA",
            color=[1.0, -0.6, -0.6],
            height=0.035,
            pos=(0, 0.45),
        )

    kb = keyboard.Keyboard()

    stims = {
        bpm: sound.Sound(
            str(STIM_DIR / manifest["stimuli"][str(bpm)]["filename"]),
            sampleRate=manifest["sample_rate"],
            stereo=True,
        )
        for bpm in CONDITIONS
    }

    session = Session(win, kb, log, stims, args.time_scale, banner, args.auto_confirm)

    status = 0
    try:
        record["device_attached_unix"] = session.device_attached()
        record["device_attached_local"] = datetime.fromtimestamp(record["device_attached_unix"]).isoformat(
            timespec="milliseconds"
        )
        audio_label = audio_device if args.speaker else f"{audio_device}   [NOT PINNED]"
        record["preflight_confirmed"] = session.preflight(details, audio_label)

        log.log(
            "SESSION_START",
            note=f"participant={details['participant_id']} order={'-'.join(map(str, order))} "
            f"time_scale={args.time_scale:g}",
        )

        session.baseline(order[0])
        for block_index, bpm in enumerate(order, start=1):
            session.play_block(block_index, bpm)
            if block_index < len(order):
                session.rest(block_index, order[block_index])

        log.log("SESSION_END")
        session.show_text(
            "Session complete.\n\n\n"
            "OPERATOR: stop the Firstbeat recording and note the time.\n\n\n"
            "Press SPACE to exit."
        )

    except Aborted as exc:
        for stim in stims.values():
            fade_out(stim)
        log.stop_segment(completed=False, note="aborted")
        log.log("ABORT", note=str(exc))
        record["aborted"] = str(exc)
        status = 1
        try:
            session.show_text(
                f"SESSION ABORTED\n\n{exc}\n\n\n"
                "Stop the Firstbeat recording and note the time.\n\n"
                "Press SPACE to exit."
            )
        except Aborted:
            pass

    finally:
        for stim in stims.values():
            try:
                stim.stop()
            except Exception:
                pass
        record["ended_unix"] = time.time()
        record["ended_local"] = datetime.now().isoformat(timespec="seconds")
        (session_dir / f"session_{started_local}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        log.close()
        win.close()

    print(f"\nBlock times: {session_dir / 'blocks.csv'}")
    print(f"Event log:   {session_dir / 'events.csv'}")
    return status


def fade_out(stim, seconds=FADE_OUT_S, steps=20):
    """Ramp a stimulus down rather than cutting it -- an abrupt stop is a startle."""
    try:
        start_volume = stim.volume
        for step in range(steps, -1, -1):
            stim.volume = start_volume * (step / steps)
            core.wait(seconds / steps)
        stim.stop()
        stim.volume = start_volume
    except Exception:
        try:
            stim.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
