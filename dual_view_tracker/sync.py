"""Pair frames from two cameras by timestamp. Pure logic: no cv2, no threads."""

from __future__ import annotations

from dataclasses import dataclass

from dual_view_tracker.capture import Frame


@dataclass(frozen=True)
class FramePair:
    reference: Frame
    other: Frame
    gap: float  # other.timestamp - reference.timestamp, seconds
    settled: bool


class FramePairer:
    """Picks, for each new reference frame, the other-camera frame closest in time.

    A reference frame is only paired once the other camera has a frame at or
    after it ("settled"). Before that, the best match might still be on its way,
    and pairing early would pick an older frame that is not the closest. Waiting
    costs at most one frame period of the other camera. If the other camera
    stalls for longer than `max_wait` (no new frame from it for that long), the
    newest reference frame is paired anyway and marked unsettled so the caller
    knows the match is unverified.
    """

    def __init__(self, max_wait: float):
        self.max_wait = max_wait
        self.last_reference_timestamp = float("-inf")

    def pair(
        self, reference_frames: list[Frame], other_frames: list[Frame], now: float
    ) -> FramePair | None:
        if not reference_frames or not other_frames:
            return None
        candidates = [
            f for f in reference_frames if f.timestamp > self.last_reference_timestamp
        ]
        if not candidates:
            return None

        newest_other = max(f.timestamp for f in other_frames)
        settled = [f for f in candidates if f.timestamp <= newest_other]
        if settled:
            reference = max(settled, key=lambda f: f.timestamp)
            is_settled = True
        else:
            # Nothing settled: the other camera is behind. Only give up waiting
            # when it has clearly stalled, measured on its own newest frame, so
            # a live reference camera cannot keep resetting the wait.
            if now - newest_other <= self.max_wait:
                return None
            reference = max(candidates, key=lambda f: f.timestamp)
            is_settled = False

        other = min(other_frames, key=lambda f: abs(f.timestamp - reference.timestamp))
        self.last_reference_timestamp = reference.timestamp
        return FramePair(
            reference=reference,
            other=other,
            gap=other.timestamp - reference.timestamp,
            settled=is_settled,
        )
