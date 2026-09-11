"""Brightness-based flash detection for measuring the offset between cameras."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np


def mean_brightness(image: np.ndarray) -> float:
    # Every 8th pixel is plenty for a whole-frame flash and keeps this cheap
    # enough for every frame. Averaging colour channels is a good enough grey.
    return float(image[::8, ::8].mean())


@dataclass(frozen=True)
class BrightnessSample:
    timestamp: float
    value: float


def find_flash_onset(samples: list[BrightnessSample], min_jump: float) -> float | None:
    """Timestamp of the largest brightness rise between consecutive samples,
    or None if that rise is below `min_jump` or there are fewer than 2 samples."""
    if len(samples) < 2:
        return None
    best_index = 1
    best_jump = samples[1].value - samples[0].value
    for i in range(2, len(samples)):
        jump = samples[i].value - samples[i - 1].value
        if jump > best_jump:
            best_jump = jump
            best_index = i
    if best_jump < min_jump:
        return None
    return samples[best_index].timestamp


def summarize_offsets(offsets: list[float]) -> dict:
    if not offsets:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(offsets),
        "mean": statistics.mean(offsets),
        "std": statistics.pstdev(offsets),
        "min": min(offsets),
        "max": max(offsets),
    }
