"""Live side-by-side viewer for two cameras with a flash sync test.

Run: python -m dual_view_tracker.view
Keys: f = arm a 3 s flash trial, q or Esc = quit.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from dual_view_tracker.capture import CameraConfig, CameraStream
from dual_view_tracker.flash import (
    BrightnessSample,
    find_flash_onset,
    mean_brightness,
    summarize_offsets,
)
from dual_view_tracker.sync import FramePair, FramePairer

BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}
FLASH_TRIAL_SECONDS = 3.0
# Brightness is 0..255; a camera flash on a normal room easily exceeds this.
FLASH_MIN_JUMP = 20.0
STARTUP_CHECK_SECONDS = 1.0
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
# Overlay rows, top to bottom: per-camera label, pair gap, flash prompt.
LABEL_Y, GAP_Y, PROMPT_Y = 25, 50, 75


def parse_resolution(text: str) -> tuple[int, int]:
    width, _, height = text.lower().partition("x")
    try:
        return int(width), int(height)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WxH, got {text!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cam0", type=int, default=0, help="reference camera index")
    parser.add_argument("--cam1", type=int, default=1, help="other camera index")
    parser.add_argument("--res0", type=parse_resolution, default="1280x720", metavar="WxH")
    parser.add_argument("--res1", type=parse_resolution, default="1920x1080", metavar="WxH")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="msmf")
    parser.add_argument("--max-wait", type=float, default=0.1, help="seconds")
    parser.add_argument("--display-height", type=int, default=480)
    return parser.parse_args(argv)


class FlashTrial:
    """Collects one brightness sample per new frame on each stream while armed."""

    def __init__(self, streams: list[CameraStream], duration: float, started_at: float):
        self.streams = streams
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
        onsets = [find_flash_onset(s, FLASH_MIN_JUMP) for s in self.samples]
        if any(onset is None for onset in onsets):
            return None
        return onsets[1] - onsets[0]

    def missed_summary(self) -> str:
        total = sum(self.missed)
        per_camera = ", ".join(f"cam{i} {n}" for i, n in enumerate(self.missed))
        return f"{total} frames missed ({per_camera})"


def scaled_to_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    return cv2.resize(image, (int(round(image.shape[1] * scale)), height))


def put_text(image: np.ndarray, text: str, x: int, y: int, color=(0, 255, 0)) -> None:
    # Dark outline first so the text stays readable on bright frames.
    cv2.putText(image, text, (x, y), FONT, FONT_SCALE, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), FONT, FONT_SCALE, color, 1, cv2.LINE_AA)


def put_text_centered(image: np.ndarray, text: str, center_x: int, y: int, color) -> None:
    (width, _), _ = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
    put_text(image, text, max(0, center_x - width // 2), y, color)


def compose(
    pair: FramePair,
    streams: list[CameraStream],
    display_height: int,
    offsets: list[float],
    trial_armed: bool,
) -> np.ndarray:
    panels = []
    for stream, frame in zip(streams, (pair.reference, pair.other)):
        panel = scaled_to_height(frame.image, display_height)
        settings = stream.actual_settings()
        put_text(
            panel,
            f"cam {stream.config.device}  {settings['width']}x{settings['height']}"
            f"  {stream.measured_fps():.1f} fps",
            10,
            LABEL_Y,
        )
        panels.append(panel)
    canvas = np.hstack(panels)

    status = "settled" if pair.settled else "UNSETTLED"
    color = (0, 255, 0) if pair.settled else (0, 0, 255)
    seam_x = panels[0].shape[1]
    put_text_centered(canvas, f"gap {pair.gap * 1000:+.1f} ms  {status}", seam_x, GAP_Y, color)

    stats = summarize_offsets(offsets)
    put_text(
        canvas,
        f"flash trials: {stats['count']}, mean {stats['mean'] * 1000:+.1f} ms,"
        f" std {stats['std'] * 1000:.1f} ms",
        10,
        display_height - 15,
    )
    if trial_armed:
        put_text(canvas, "FLASH TRIAL: fire the flash now", 10, PROMPT_Y, (0, 255, 255))
    return canvas


def print_settings(stream: CameraStream, requested: CameraConfig) -> None:
    actual = stream.actual_settings()
    print(
        f"cam {requested.device}: requested {requested.width}x{requested.height}"
        f" @ {requested.fps} {requested.fourcc}, actual {actual['width']}x{actual['height']}"
        f" @ {actual['fps']:g} {actual['fourcc']}"
    )


def print_silent_streams(streams: list[CameraStream], after_seconds: float) -> None:
    for stream in streams:
        if stream.latest() is None:
            print(
                f"cam {stream.config.device}: no frames after {after_seconds:g} s"
                f" ({stream.dropped_frames} dropped); the camera opened but is not delivering yet"
            )


def write_offsets_csv(path: Path, offsets: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "offset_cam1_minus_cam0_seconds"])
        for i, offset in enumerate(offsets, start=1):
            writer.writerow([i, f"{offset:.6f}"])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    backend = BACKENDS[args.backend]
    configs = [
        CameraConfig(args.cam0, *args.res0, fps=args.fps, backend=backend),
        CameraConfig(args.cam1, *args.res1, fps=args.fps, backend=backend),
    ]
    streams = [CameraStream(config) for config in configs]
    pairer = FramePairer(max_wait=args.max_wait)
    offsets: list[float] = []
    trial: FlashTrial | None = None
    window = "dual-view-tracker"

    try:
        for stream, config in zip(streams, configs):
            stream.start()
            print_settings(stream, config)
        started_at = time.perf_counter()
        startup_checked = False

        while True:
            now = time.perf_counter()
            if not startup_checked and now - started_at >= STARTUP_CHECK_SECONDS:
                print_silent_streams(streams, STARTUP_CHECK_SECONDS)
                startup_checked = True

            if trial is not None:
                trial.collect()
                if trial.finished(now):
                    offset = trial.offset()
                    if offset is None:
                        print(f"flash trial: flash not detected on both cameras; {trial.missed_summary()}")
                    else:
                        offsets.append(offset)
                        print(
                            f"flash trial {len(offsets)}: offset {offset * 1000:+.2f} ms"
                            f" (cam1 minus cam0); {trial.missed_summary()}"
                        )
                    trial = None

            pair = pairer.pair(streams[0].frames(), streams[1].frames(), now)
            if pair is not None:
                cv2.imshow(
                    window,
                    compose(pair, streams, args.display_height, offsets, trial is not None),
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f") and trial is None:
                trial = FlashTrial(streams, FLASH_TRIAL_SECONDS, now)
                print("flash trial armed: fire the flash now")
    finally:
        for stream in streams:
            stream.stop()
        cv2.destroyAllWindows()

    stats = summarize_offsets(offsets)
    print(
        f"flash offsets (cam1 minus cam0): count {stats['count']},"
        f" mean {stats['mean'] * 1000:+.2f} ms, std {stats['std'] * 1000:.2f} ms,"
        f" min {stats['min'] * 1000:+.2f} ms, max {stats['max'] * 1000:+.2f} ms"
    )
    for stream in streams:
        print(
            f"cam {stream.config.device}: measured {stream.measured_fps():.1f} fps,"
            f" {stream.dropped_frames} dropped frames"
        )
    if offsets:
        path = Path("output") / "flash_offsets.csv"
        write_offsets_csv(path, offsets)
        print(f"wrote {path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(f"error: {error}")
