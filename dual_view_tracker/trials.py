"""Flash trials: sample both cameras around a light event and compare onsets.

`FlashTrial` is one manual trial (the user fires a flash). `TorchBatch` runs
many trials unattended by switching the phone's LED torch through its remote
API. Both are pure step machines driven by the viewer loop with an explicit
clock, so they can be tested without cameras or a phone.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from dual_view_tracker.capture import CameraStream
from dual_view_tracker.flash import (
    BrightnessSample,
    find_flash_onset,
    largest_rise,
    mean_brightness,
)
from dual_view_tracker.remote import DroidCamRemote

# Brightness is 0..255; a camera flash on a normal room easily exceeds this.
FLASH_MIN_JUMP = 20.0

# Torch batch timing, validated by hand. The phone's auto-exposure needs
# ~1.5 s to adapt to the lit scene; locking exposure while lit keeps it at
# 30 fps in a dim room, where an unlocked phone would drop its frame rate.
LIGHT_SECONDS = 1.5
SETTLE_SECONDS = 1.0
BASELINE_SECONDS = 0.5
LIT_SECONDS = 1.5
GAP_SECONDS = 1.0
# Below this a camera has lost frames or slowed down, so the onset is coarse.
SLOW_FPS = 27.0


class Phase(Enum):
    LIGHTING = "lighting"
    SETTLING = "settling"
    BASELINE = "baseline"
    LIT = "lit"
    OFF = "off"
    DONE = "done"


class FlashTrial:
    """Collects one brightness sample per new frame on each stream while armed."""

    def __init__(
        self,
        streams: list[CameraStream],
        duration: float,
        started_at: float,
        min_jump: float = FLASH_MIN_JUMP,
    ):
        self.streams = streams
        self.min_jump = min_jump
        self.ends_at = started_at + duration
        self.samples: list[list[BrightnessSample]] = [[] for _ in streams]
        self.last_index = [-1] * len(streams)
        # Frames evicted from the ring buffer before we sampled them (the main
        # loop stalls when the window is dragged, for example). A missed frame
        # around the flash makes the onset timestamp unreliable.
        self.missed = [0] * len(streams)

    def collect(self) -> None:
        for i, stream in enumerate(self.streams):
            for frame in stream.frames():
                if frame.index <= self.last_index[i]:
                    continue
                if self.last_index[i] >= 0:
                    self.missed[i] += frame.index - self.last_index[i] - 1
                self.last_index[i] = frame.index
                self.samples[i].append(
                    BrightnessSample(frame.timestamp, mean_brightness(frame.image))
                )

    def finished(self, now: float) -> bool:
        return now >= self.ends_at

    def offset(self) -> float | None:
        """Camera 1 onset minus camera 0 onset; positive means camera 1 saw the flash later."""
        onsets = [find_flash_onset(s, self.min_jump) for s in self.samples]
        if any(onset is None for onset in onsets):
            return None
        return onsets[1] - onsets[0]

    def rises(self) -> list[float]:
        """Largest brightness rise seen per camera."""
        return [largest_rise(samples)[0] for samples in self.samples]

    def fps(self) -> list[float]:
        """Frame rate per camera over the sampled frames; 0.0 with fewer than 2 samples."""
        rates = []
        for samples in self.samples:
            span = samples[-1].timestamp - samples[0].timestamp if len(samples) > 1 else 0.0
            rates.append((len(samples) - 1) / span if span > 0 else 0.0)
        return rates

    def rise_summary(self) -> str:
        """Largest brightness rise seen per camera, so a missed flash can be diagnosed."""
        parts = [
            f"cam{i} rise {jump:.1f} in {len(samples)} frames"
            for i, (jump, samples) in enumerate(zip(self.rises(), self.samples))
        ]
        return f"largest rise (need {self.min_jump:g}): " + ", ".join(parts)

    def missed_summary(self) -> str:
        total = sum(self.missed)
        per_camera = ", ".join(f"cam{i} {n}" for i, n in enumerate(self.missed))
        return f"{total} frames missed ({per_camera})"


@dataclass(frozen=True)
class TrialResult:
    offset: float | None  # camera 1 onset minus camera 0 onset, seconds; None if not detected
    fps: tuple[float, float]
    rise: tuple[float, float]
    missed: tuple[int, int]
    slow: bool


class TorchBatch:
    """Runs `trials` torch flashes unattended, one FlashTrial per flash.

    Phases: LIGHTING (torch on) -> SETTLING (lock exposure, torch off) ->
    per trial BASELINE (sampling, torch off) -> LIT (torch on) -> OFF (torch
    off, trial evaluated) -> ... -> DONE (exposure unlocked). Each remote
    command is sent exactly once, at a phase transition. The torch and lock
    state are tracked here because the phone's API only offers toggles.
    """

    def __init__(
        self,
        streams: list[CameraStream],
        remote: DroidCamRemote,
        trials: int,
        min_jump: float,
        now: float,
    ):
        self.streams = streams
        self.remote = remote
        self.trials = trials
        self.min_jump = min_jump
        self.results: list[TrialResult] = []
        self.trial: FlashTrial | None = None
        self.torch_on = False
        self.locked = False
        self._enter(Phase.LIGHTING, now + LIGHT_SECONDS)
        self._set_torch(True)

    @property
    def finished(self) -> bool:
        return self.phase == Phase.DONE

    @property
    def trial_number(self) -> int:
        """The trial in progress, or the last one completed (0 before the first)."""
        return len(self.results) + (1 if self.trial is not None else 0)

    def offsets(self) -> list[float]:
        return [r.offset for r in self.results if r.offset is not None]

    def update(self, now: float) -> None:
        if self.trial is not None:
            self.trial.collect()
        if self.phase == Phase.DONE or now < self.phase_ends_at:
            return
        if self.phase == Phase.LIGHTING:
            self._set_lock(True)
            self._set_torch(False)
            self._enter(Phase.SETTLING, now + SETTLE_SECONDS)
        elif self.phase in (Phase.SETTLING, Phase.OFF):
            if len(self.results) == self.trials:
                self._set_lock(False)
                self._enter(Phase.DONE, now)
            else:
                self.trial = FlashTrial(
                    self.streams, BASELINE_SECONDS + LIT_SECONDS, now, self.min_jump
                )
                self._enter(Phase.BASELINE, now + BASELINE_SECONDS)
        elif self.phase == Phase.BASELINE:
            self._set_torch(True)
            # Timed from the command, not from the trial start: if the loop
            # stalled during the baseline the pulse must still last the full
            # length, or the cameras get only a blink.
            self._enter(Phase.LIT, now + LIT_SECONDS)
        elif self.phase == Phase.LIT:
            self._set_torch(False)
            self.results.append(self._evaluate(self.trial))
            self.trial = None
            self._enter(Phase.OFF, now + GAP_SECONDS)

    def abort(self) -> None:
        """Leave the phone as it was found: torch off, exposure unlocked.

        Both resets are attempted even if the first fails; the error raised
        afterwards says what may still be on, so the operator can fix it by hand.
        """
        problems = []
        try:
            self._set_torch(False)
        except RuntimeError as error:
            problems.append(f"{error}: torch may still be on")
        try:
            self._set_lock(False)
        except RuntimeError as error:
            problems.append(f"{error}: exposure may still be locked")
        self.trial = None
        self._enter(Phase.DONE, 0.0)
        if problems:
            raise RuntimeError("; ".join(problems))

    def _evaluate(self, trial: FlashTrial) -> TrialResult:
        fps = trial.fps()
        rise = trial.rises()
        return TrialResult(
            offset=trial.offset(),
            fps=(fps[0], fps[1]),
            rise=(rise[0], rise[1]),
            missed=(trial.missed[0], trial.missed[1]),
            slow=any(rate < SLOW_FPS for rate in fps),
        )

    def _enter(self, phase: Phase, ends_at: float) -> None:
        self.phase = phase
        self.phase_ends_at = ends_at

    def _set_torch(self, on: bool) -> None:
        if self.torch_on != on:
            self.remote.toggle_torch()
            self.torch_on = on

    def _set_lock(self, on: bool) -> None:
        if self.locked != on:
            self.remote.toggle_exposure_lock()
            self.locked = on


def format_trial_result(number: int, result: TrialResult, min_jump: float) -> str:
    if result.offset is None:
        head = f"torch trial {number}: not detected (need rise {min_jump:g})"
    else:
        head = f"torch trial {number}: offset {result.offset * 1000:+.1f} ms (cam1 minus cam0)"
    cameras = "; ".join(
        f"cam{i} fps {fps:.1f} rise {rise:.1f}"
        for i, (fps, rise) in enumerate(zip(result.fps, result.rise))
    )
    missed = ", ".join(f"cam{i} {n}" for i, n in enumerate(result.missed))
    line = f"{head}; {cameras}; missed {missed}"
    return line + " [slow]" if result.slow else line


def write_torch_csv(path: Path, results: list[TrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "trial",
                "offset_cam1_minus_cam0_seconds",
                "fps_cam0",
                "fps_cam1",
                "rise_cam0",
                "rise_cam1",
                "missed_cam0",
                "missed_cam1",
                "slow",
            ]
        )
        for i, r in enumerate(results, start=1):
            offset = "" if r.offset is None else f"{r.offset:.6f}"
            writer.writerow(
                [
                    i,
                    offset,
                    f"{r.fps[0]:.2f}",
                    f"{r.fps[1]:.2f}",
                    f"{r.rise[0]:.1f}",
                    f"{r.rise[1]:.1f}",
                    r.missed[0],
                    r.missed[1],
                    int(r.slow),
                ]
            )
