"""Live side-by-side viewer for two cameras with a flash sync test.

Run: python -m dual_view_tracker.view
Keys: f = arm a 3 s flash trial, t = run an automated torch batch (needs
--remote), q or Esc = quit.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from dual_view_tracker.capture import CameraConfig, CameraStream, parse_source
from dual_view_tracker.flash import summarize_offsets
from dual_view_tracker.remote import DroidCamRemote
from dual_view_tracker.sync import FramePair, FramePairer
from dual_view_tracker.trials import (
    FLASH_MIN_JUMP,
    FlashTrial,
    TorchBatch,
    TrialResult,
    format_trial_result,
    write_torch_csv,
)

BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}
FLASH_TRIAL_SECONDS = 3.0
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


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}")
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a number of 1 or more, got {value}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cam0", type=parse_source, default=0, help="reference camera index or stream URL"
    )
    parser.add_argument(
        "--cam1", type=parse_source, default=1, help="other camera index or stream URL"
    )
    parser.add_argument("--res0", type=parse_resolution, default="1280x720", metavar="WxH")
    parser.add_argument("--res1", type=parse_resolution, default="1920x1080", metavar="WxH")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="msmf")
    parser.add_argument("--max-wait", type=float, default=0.1, help="seconds")
    parser.add_argument("--display-height", type=int, default=480)
    parser.add_argument(
        "--min-jump", type=float, default=FLASH_MIN_JUMP,
        help="smallest brightness rise (0-255) that counts as the flash",
    )
    parser.add_argument(
        "--remote", metavar="BASE_URL", default=None,
        help="DroidCam remote control base, e.g. http://192.168.0.20:4747;"
        " enables the torch batch (key t)",
    )
    parser.add_argument(
        "--trials", type=positive_int, default=20, help="flashes per torch batch"
    )
    return parser.parse_args(argv)


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
    prompt: str | None,
) -> np.ndarray:
    panels = []
    # Label panels by position, not by source: a stream URL would not fit.
    for i, (stream, frame) in enumerate(zip(streams, (pair.reference, pair.other))):
        panel = scaled_to_height(frame.image, display_height)
        settings = stream.actual_settings()
        put_text(
            panel,
            f"cam {i}  {settings['width']}x{settings['height']}"
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
    if prompt:
        put_text(canvas, prompt, 10, PROMPT_Y, (0, 255, 255))
    return canvas


def batch_prompt(batch: TorchBatch) -> str:
    mean = summarize_offsets(batch.offsets())["mean"]
    return f"TORCH BATCH trial {batch.trial_number}/{batch.trials}  mean {mean * 1000:+.1f} ms"


def print_settings(position: int, stream: CameraStream, requested: CameraConfig) -> None:
    actual = stream.actual_settings()
    if isinstance(requested.device, str):
        request = "requested stream"
    else:
        request = (
            f"requested {requested.width}x{requested.height}"
            f" @ {requested.fps} {requested.fourcc}"
        )
    print(
        f"cam {position} ({requested.device}): {request},"
        f" actual {actual['width']}x{actual['height']} @ {actual['fps']:g} {actual['fourcc']}"
    )


def print_silent_streams(streams: list[CameraStream], after_seconds: float) -> None:
    for i, stream in enumerate(streams):
        if stream.latest() is None:
            print(
                f"cam {i}: no frames after {after_seconds:g} s"
                f" ({stream.dropped_frames} dropped); the camera opened but is not delivering yet"
            )


def print_flash_trial(trial: FlashTrial, number: int, offset: float | None) -> None:
    if offset is None:
        print(
            "flash trial: flash not detected on both cameras;"
            f" {trial.rise_summary()}; {trial.missed_summary()}"
        )
    else:
        print(
            f"flash trial {number}: offset {offset * 1000:+.2f} ms"
            f" (cam1 minus cam0); {trial.rise_summary()}; {trial.missed_summary()}"
        )


def format_stats(offsets: list[float]) -> str:
    stats = summarize_offsets(offsets)
    return (
        f"mean {stats['mean'] * 1000:+.2f} ms, std {stats['std'] * 1000:.2f} ms,"
        f" min {stats['min'] * 1000:+.2f} ms, max {stats['max'] * 1000:+.2f} ms"
    )


def print_torch_summary(results: list[TrialResult]) -> None:
    # A slow camera makes the onset coarse, so slow trials stay out of the
    # statistics; they are still in the CSV with their flag.
    detected = [r for r in results if r.offset is not None]
    valid = [r.offset for r in detected if not r.slow]
    print(
        f"torch offsets (cam1 minus cam0): count {len(valid)} valid"
        f" ({len(detected) - len(valid)} slow excluded,"
        f" {len(results) - len(detected)} not detected), {format_stats(valid)}"
    )


def finish_batch(batch: TorchBatch, torch_results: list[TrialResult]) -> None:
    """Keep the finished trials and reset the phone; a failed reset is printed
    so the viewer keeps running and the operator can fix the phone by hand."""
    torch_results.extend(batch.results)
    try:
        batch.abort()
    except RuntimeError as error:
        print(f"torch batch: could not reset the phone: {error}")


def advance_batch(
    batch: TorchBatch, now: float, torch_results: list[TrialResult], min_jump: float
) -> TorchBatch | None:
    """One step of the batch; None once it is over and its results were
    moved to `torch_results`. Trials are numbered across batches to match
    the CSV rows."""
    reported = len(batch.results)
    try:
        batch.update(now)
    except RuntimeError as error:
        print(f"torch batch stopped: {error}")
        finish_batch(batch, torch_results)
        return None
    for number in range(reported + 1, len(batch.results) + 1):
        print(
            format_trial_result(
                len(torch_results) + number, batch.results[number - 1], min_jump
            )
        )
    if batch.finished:
        torch_results.extend(batch.results)
        print("torch batch done")
        return None
    return batch


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
    remote = DroidCamRemote(args.remote) if args.remote else None
    pairer = FramePairer(max_wait=args.max_wait)
    offsets: list[float] = []
    torch_results: list[TrialResult] = []
    trial: FlashTrial | None = None
    batch: TorchBatch | None = None
    window = "dual-view-tracker"

    try:
        for i, (stream, config) in enumerate(zip(streams, configs)):
            stream.start()
            print_settings(i, stream, config)
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
                    if offset is not None:
                        offsets.append(offset)
                    print_flash_trial(trial, len(offsets), offset)
                    trial = None

            if batch is not None:
                batch = advance_batch(batch, now, torch_results, args.min_jump)

            pair = pairer.pair(streams[0].frames(), streams[1].frames(), now)
            if pair is not None:
                if batch is not None:
                    prompt = batch_prompt(batch)
                elif trial is not None:
                    prompt = "FLASH TRIAL: fire the flash now"
                else:
                    prompt = None
                cv2.imshow(window, compose(pair, streams, args.display_height, offsets, prompt))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f") and trial is None and batch is None:
                trial = FlashTrial(streams, FLASH_TRIAL_SECONDS, now, args.min_jump)
                print("flash trial armed: fire the flash now")
            if key == ord("t") and trial is None and batch is None:
                if remote is None:
                    print("torch batch needs --remote BASE_URL (the DroidCam remote control)")
                else:
                    try:
                        batch = TorchBatch(streams, remote, args.trials, args.min_jump, now)
                        print(f"torch batch started: {args.trials} trials")
                    except RuntimeError as error:
                        print(f"torch batch could not start: {error}")
    finally:
        if batch is not None:
            finish_batch(batch, torch_results)
        for stream in streams:
            stream.stop()
        cv2.destroyAllWindows()

    print(f"flash offsets (cam1 minus cam0): count {len(offsets)}, {format_stats(offsets)}")
    for i, stream in enumerate(streams):
        print(
            f"cam {i}: measured {stream.measured_fps():.1f} fps,"
            f" {stream.dropped_frames} dropped frames"
        )
    if offsets:
        path = Path("output") / "flash_offsets.csv"
        write_offsets_csv(path, offsets)
        print(f"wrote {path}")
    if torch_results:
        print_torch_summary(torch_results)
        path = Path("output") / "torch_offsets.csv"
        write_torch_csv(path, torch_results)
        print(f"wrote {path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(f"error: {error}")
