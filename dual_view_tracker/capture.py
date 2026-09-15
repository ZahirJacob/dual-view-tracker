"""Threaded single-camera capture with a short timestamped history."""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp: float
    image: np.ndarray


@dataclass(frozen=True)
class CameraConfig:
    device: int | str  # local device index, or a video stream URL
    width: int
    height: int
    fps: int = 30
    backend: int = cv2.CAP_MSMF
    fourcc: str = "MJPG"


def parse_source(text: str) -> int | str:
    """Digits-only text is a device index; anything else is a stream URL."""
    return int(text) if text.isdigit() else text


def decode_fourcc(value: float) -> str:
    code = int(value)
    text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    # MSMF sometimes reports a media-subtype index (e.g. 22 = RGB32) instead of
    # a fourcc, and -1 means "unknown"; show the number rather than four
    # control or non-ASCII characters.
    if code < 0 or not (text.isascii() and text.isprintable()):
        return f"#{code}"
    return text


class CameraStream:
    """Reads one camera on a background thread into a bounded ring buffer.

    A ring buffer instead of a newest-only slot: the pairer needs a short
    history on each camera to pick the frame closest in time to the other
    camera's frame. The buffer is bounded so latency never grows; the viewer
    still always sees the newest frames.

    `capture` lets tests inject a fake VideoCapture-like object
    (grab, retrieve, get, set, release, isOpened) instead of a real device.
    """

    def __init__(self, config: CameraConfig, history_seconds: float = 0.5, capture=None):
        self.config = config
        self._capture = capture
        self._frames: collections.deque[Frame] = collections.deque(
            maxlen=max(2, round(config.fps * history_seconds))
        )
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._next_index = 0
        self.dropped_frames = 0  # failed grab() or retrieve() calls
        self.settings: dict = {}

    def start(self) -> None:
        source = self.config.device
        is_stream = isinstance(source, str)
        if self._capture is None:
            backend = cv2.CAP_FFMPEG if is_stream else self.config.backend
            self._capture = cv2.VideoCapture(source, backend)
        if not self._capture.isOpened():
            raise RuntimeError(f"cannot open camera source {source}")
        cap = self._capture
        if not is_stream:
            # fourcc first: MJPG is what unlocks the high resolutions at full
            # rate on most UVC webcams, and some drivers reject the size until
            # it is set. A stream's format is decided by the remote side, so
            # nothing is requested for it.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        # Smallest queue so a grab returns the freshest frame; FFmpeg queues
        # stream frames too. Not all backends honour this, and that is fine.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Read the negotiated settings once here; cap.get() from another
        # thread while the capture thread sits in grab() is not safe on MSMF.
        self.settings = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "fourcc": decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name=f"camera-{self.config.device}", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        cap = self._capture
        while self._running:
            if not cap.isOpened():
                break
            grabbed = cap.grab()
            # Stamp right after grab: retrieve() includes the decode time,
            # which would shift every timestamp by a variable amount.
            timestamp = time.perf_counter()
            if not grabbed:
                self.dropped_frames += 1
                time.sleep(0.001)
                continue
            ok, image = cap.retrieve()
            if not ok:
                self.dropped_frames += 1
                continue
            frame = Frame(self._next_index, timestamp, image)
            self._next_index += 1
            with self._lock:
                self._frames.append(frame)

    def frames(self) -> list[Frame]:
        with self._lock:
            return list(self._frames)

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def actual_settings(self) -> dict:
        """Width, height, fps and fourcc as the driver reported them at start()."""
        return dict(self.settings)

    def measured_fps(self) -> float:
        frames = self.frames()
        if len(frames) < 2:
            return 0.0
        span = frames[-1].timestamp - frames[0].timestamp
        return (len(frames) - 1) / span if span > 0 else 0.0

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            # A grab() blocked on an unplugged device may never return; do not
            # let that keep the whole program from exiting.
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                # Releasing under a live grab() can crash; leave it to process exit.
                print(f"cam {self.config.device}: capture thread did not stop, skipping release")
                return
            self._thread = None
        if self._capture is not None:
            self._capture.release()

    def __enter__(self) -> "CameraStream":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()
