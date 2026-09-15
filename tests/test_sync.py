import pytest
import numpy as np

from dual_view_tracker.capture import Frame
from dual_view_tracker.sync import FramePairer

IMAGE = np.zeros((2, 2), dtype=np.uint8)


def frames(*timestamps: float) -> list[Frame]:
    return [Frame(i, t, IMAGE) for i, t in enumerate(timestamps)]


def test_picks_closest_other_frame_not_newest():
    pairer = FramePairer(max_wait=0.1)
    pair = pairer.pair(frames(1.0), frames(0.99, 1.02, 1.05), now=1.06)
    assert pair is not None
    assert pair.other.timestamp == 0.99
    assert pair.gap == 0.99 - 1.0
    assert pair.settled


def test_waits_while_other_camera_is_behind():
    pairer = FramePairer(max_wait=0.1)
    assert pairer.pair(frames(1.0), frames(0.98), now=1.05) is None


def test_pairs_unsettled_after_max_wait():
    pairer = FramePairer(max_wait=0.1)
    pair = pairer.pair(frames(1.0), frames(0.9), now=1.2)
    assert pair is not None
    assert pair.other.timestamp == 0.9
    assert not pair.settled


def test_each_reference_paired_once_and_in_time_order():
    pairer = FramePairer(max_wait=0.1)
    others = frames(1.0, 1.033, 1.066, 1.1)

    first = pairer.pair(frames(1.0, 1.033, 1.066), others, now=1.1)
    assert first is not None
    assert first.reference.timestamp == 1.066
    assert first.settled

    assert pairer.pair(frames(1.0, 1.033, 1.066), others, now=1.11) is None

    second = pairer.pair(frames(1.033, 1.066, 1.1), frames(1.066, 1.1, 1.133), now=1.15)
    assert second is not None
    assert second.reference.timestamp == 1.1
    assert second.reference.timestamp > first.reference.timestamp


def test_empty_inputs_return_none():
    pairer = FramePairer(max_wait=0.1)
    assert pairer.pair([], [], now=0.0) is None
    assert pairer.pair(frames(1.0), [], now=2.0) is None
    assert pairer.pair([], frames(1.0), now=2.0) is None


def test_best_match_is_oldest_other_frame():
    pairer = FramePairer(max_wait=0.1)
    pair = pairer.pair(frames(1.0), frames(1.01, 1.05, 1.1, 1.2), now=1.25)
    assert pair is not None
    assert pair.other.timestamp == 1.01
    assert pair.gap == pytest.approx(0.01)
    assert pair.settled


def test_exact_timestamp_match_counts_as_settled():
    pairer = FramePairer(max_wait=0.1)
    pair = pairer.pair(frames(1.0), frames(0.9, 1.0), now=1.0)
    assert pair is not None
    assert pair.other.timestamp == 1.0
    assert pair.gap == 0.0
    assert pair.settled


def test_duplicate_reference_timestamps_pair_only_once():
    pairer = FramePairer(max_wait=0.1)
    reference = frames(1.0, 1.0)
    others = frames(0.99, 1.02)
    first = pairer.pair(reference, others, now=1.05)
    assert first is not None
    assert first.reference.timestamp == 1.0
    assert pairer.pair(reference, others, now=1.06) is None


def test_duplicate_other_timestamps_still_give_closest_gap():
    pairer = FramePairer(max_wait=0.1)
    pair = pairer.pair(frames(1.0), frames(0.98, 0.98, 1.05), now=1.06)
    assert pair is not None
    assert pair.other.timestamp == 0.98
    assert pair.gap == pytest.approx(-0.02)


def test_other_camera_far_ahead_pairs_with_its_oldest_frame():
    pairer = FramePairer(max_wait=0.1)
    pair = pairer.pair(frames(1.0, 1.033), frames(3.0, 3.033, 3.066), now=3.1)
    assert pair is not None
    assert pair.reference.timestamp == 1.033
    assert pair.other.timestamp == 3.0
    assert pair.gap == pytest.approx(3.0 - 1.033)
    assert pair.settled


def test_reference_buffer_emptied_and_refilled_keeps_order():
    pairer = FramePairer(max_wait=0.1)
    others = frames(1.0, 1.033, 1.066, 1.1)
    first = pairer.pair(frames(1.0, 1.033), others, now=1.1)
    assert first is not None
    assert first.reference.timestamp == 1.033

    assert pairer.pair([], others, now=1.2) is None
    assert pairer.pair(frames(0.9, 1.033), others, now=1.2) is None

    second = pairer.pair(frames(1.066, 1.1), others, now=1.2)
    assert second is not None
    assert second.reference.timestamp == 1.1
    assert second.settled


def test_unsettled_reference_is_not_paired_again_when_other_catches_up():
    pairer = FramePairer(max_wait=0.1)
    stalled = pairer.pair(frames(1.0), frames(0.9), now=1.2)
    assert stalled is not None
    assert not stalled.settled

    assert pairer.pair(frames(1.0), frames(0.9, 1.01), now=1.21) is None

    resumed = pairer.pair(frames(1.0, 1.033), frames(0.9, 1.01, 1.04), now=1.25)
    assert resumed is not None
    assert resumed.reference.timestamp == 1.033
    assert resumed.other.timestamp == 1.04
    assert resumed.settled


def test_simulated_drifting_streams_pair_monotonically_within_half_a_period():
    pairer = FramePairer(max_wait=0.1)
    period = 1 / 30
    other_period = period * 1.02
    reference_all = [Frame(i, i * period, IMAGE) for i in range(60)]
    other_all = [Frame(i, 0.004 + i * other_period, IMAGE) for i in range(60)]
    delivery_lag = 0.01
    history = 8

    paired = []
    now = 0.0
    while now < 2.0:
        reference = [f for f in reference_all if f.timestamp <= now][-history:]
        other = [f for f in other_all if f.timestamp <= now - delivery_lag][-history:]
        pair = pairer.pair(reference, other, now=now)
        if pair is not None:
            paired.append(pair)
        now += 1 / 45

    stamps = [p.reference.timestamp for p in paired]
    assert len(paired) >= 50
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)
    assert all(p.settled for p in paired)
    assert all(abs(p.gap) <= other_period / 2 + 1e-9 for p in paired)


def test_keeps_pairing_unsettled_when_only_the_other_camera_stalls():
    pairer = FramePairer(max_wait=0.1)
    period = 1 / 30
    others = frames(0.5, 0.533, 0.566)
    history = 8

    pairs = []
    for i in range(30):
        now = 1.0 + i * period
        reference = [
            Frame(i + k, now - (history - 1 - k) * period, IMAGE) for k in range(history)
        ]
        pair = pairer.pair(reference, others, now=now)
        if pair is not None:
            pairs.append(pair)

    assert pairs
    assert not any(p.settled for p in pairs)
