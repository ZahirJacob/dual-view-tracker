import numpy as np
import pytest

from dual_view_tracker.capture import Frame
from dual_view_tracker.trials import (
    FlashTrial,
    TorchBatch,
    TrialResult,
    format_trial_result,
    write_torch_csv,
)

DARK = np.full((16, 16, 3), 40, dtype=np.uint8)
BRIGHT = np.full((16, 16, 3), 220, dtype=np.uint8)
FRAME_PERIOD = 1 / 30


class FakeStream:
    """Only `frames()` is needed by the trials, so no thread and no camera."""

    def __init__(self, frames=()):
        self.buffer = list(frames)
        self.next_index = 0

    def frames(self):
        return list(self.buffer)

    def push(self, timestamp, image):
        self.buffer.append(Frame(self.next_index, timestamp, image))
        self.next_index += 1
        # Same bounded history as the real ring buffer.
        del self.buffer[:-15]


def sequence(timestamps, bright_from):
    return [
        Frame(i, t, BRIGHT if i >= bright_from else DARK) for i, t in enumerate(timestamps)
    ]


class FakeRemote:
    """Records each toggle with the clock the test set on `now`."""

    def __init__(self):
        self.now = 0.0
        self.calls = []
        self.torch = False
        self.locked = False

    def toggle_torch(self):
        self.torch = not self.torch
        self.calls.append(("torch", self.now))

    def toggle_exposure_lock(self):
        self.locked = not self.locked
        self.calls.append(("lock", self.now))


def run_scene(
    batch, streams, remote, start, until, period=FRAME_PERIOD, lag=0.004, sees_torch=(True, True)
):
    """Advance the clock one frame period at a time. Each camera delivers one
    frame per step, bright while the remote's torch is on; camera 1 stamps its
    frames `lag` seconds later than camera 0."""
    now = start
    while now < until and not batch.finished:
        now += period
        for i, stream in enumerate(streams):
            lit = remote.torch and sees_torch[i]
            stream.push(now + i * lag, BRIGHT if lit else DARK)
        remote.now = now
        batch.update(now)
    return now


def new_batch(remote, trials, start=10.0):
    streams = [FakeStream(), FakeStream()]
    remote.now = start
    return TorchBatch(streams, remote, trials, min_jump=20.0, now=start), streams


# FlashTrial


def test_offset_is_other_onset_minus_reference_onset():
    cam0 = FakeStream(sequence([1.000, 1.033, 1.066, 1.100, 1.133], bright_from=3))
    cam1 = FakeStream(sequence([1.010, 1.043, 1.076, 1.109, 1.142, 1.175], bright_from=4))
    trial = FlashTrial([cam0, cam1], duration=3.0, started_at=1.0)
    trial.collect()
    assert trial.offset() == pytest.approx(1.142 - 1.100)


def test_offset_is_negative_when_other_camera_sees_flash_first():
    cam0 = FakeStream(sequence([1.000, 1.033, 1.066, 1.100], bright_from=3))
    cam1 = FakeStream(sequence([1.005, 1.038, 1.071, 1.104], bright_from=2))
    trial = FlashTrial([cam0, cam1], duration=3.0, started_at=1.0)
    trial.collect()
    assert trial.offset() == pytest.approx(1.071 - 1.100)


def test_overlapping_snapshots_sample_each_frame_once():
    all0 = sequence([1.0, 1.033, 1.066, 1.1], bright_from=3)
    all1 = sequence([1.01, 1.043, 1.076, 1.109], bright_from=3)
    cam0, cam1 = FakeStream(all0[:2]), FakeStream(all1[:2])
    trial = FlashTrial([cam0, cam1], duration=3.0, started_at=1.0)
    trial.collect()
    cam0.buffer, cam1.buffer = all0[1:], all1[1:]
    trial.collect()
    trial.collect()
    assert [s.timestamp for s in trial.samples[0]] == [f.timestamp for f in all0]
    assert [s.timestamp for s in trial.samples[1]] == [f.timestamp for f in all1]
    assert trial.offset() == pytest.approx(1.109 - 1.1)


def test_no_offset_when_one_camera_misses_the_flash():
    cam0 = FakeStream(sequence([1.0, 1.033, 1.066], bright_from=2))
    cam1 = FakeStream(sequence([1.0, 1.033, 1.066], bright_from=99))
    trial = FlashTrial([cam0, cam1], duration=3.0, started_at=1.0)
    trial.collect()
    assert trial.offset() is None


def test_trial_finishes_after_duration():
    trial = FlashTrial([FakeStream(), FakeStream()], duration=3.0, started_at=10.0)
    assert not trial.finished(12.999)
    assert trial.finished(13.0)


def test_missed_frames_are_counted_per_camera():
    all0 = sequence([1.0, 1.033, 1.066, 1.1, 1.133], bright_from=3)
    all1 = sequence([1.01, 1.043, 1.076, 1.109, 1.142], bright_from=3)
    cam0, cam1 = FakeStream(all0[:1]), FakeStream(all1[:1])
    trial = FlashTrial([cam0, cam1], duration=3.0, started_at=1.0)
    trial.collect()
    # Frames 1 and 2 of cam0 and frame 1 of cam1 were evicted before sampling.
    cam0.buffer, cam1.buffer = all0[3:], all1[2:]
    trial.collect()
    assert trial.missed == [2, 1]
    assert trial.missed_summary() == "3 frames missed (cam0 2, cam1 1)"


def test_frames_before_the_trial_are_not_counted_as_missed():
    cam0 = FakeStream(sequence([1.0, 1.033], bright_from=99)[1:])
    trial = FlashTrial([cam0], duration=3.0, started_at=1.0)
    trial.collect()
    assert trial.missed == [0]


def test_rise_summary_reports_largest_rise_per_camera():
    cam0 = FakeStream(sequence([1.0, 1.033, 1.066], bright_from=2))
    cam1 = FakeStream(sequence([1.0, 1.033, 1.066], bright_from=99))
    trial = FlashTrial([cam0, cam1], duration=3.0, started_at=1.0, min_jump=20.0)
    trial.collect()
    assert trial.rises() == [180.0, 0.0]
    summary = trial.rise_summary()
    assert summary.startswith("largest rise (need 20)")
    assert "cam0 rise 180.0 in 3 frames" in summary
    assert "cam1 rise 0.0 in 3 frames" in summary


def test_fps_from_sample_timestamps():
    cam0 = FakeStream(sequence([1.0, 1.05, 1.1, 1.15, 1.2], bright_from=99))
    cam1 = FakeStream(sequence([1.0], bright_from=99))
    trial = FlashTrial([cam0, cam1], duration=3.0, started_at=1.0)
    trial.collect()
    assert trial.fps() == pytest.approx([20.0, 0.0])


# TorchBatch


def test_commands_follow_the_validated_sequence():
    remote = FakeRemote()
    batch, _ = new_batch(remote, trials=1, start=10.0)
    assert remote.calls == [("torch", 10.0)]
    for now in (11.0, 11.5, 12.0, 12.5, 12.9, 13.0, 14.0, 14.5, 15.0, 15.5):
        remote.now = now
        batch.update(now)
    assert remote.calls == [
        ("torch", 10.0),  # light the scene
        ("lock", 11.5),  # exposure locked while lit
        ("torch", 11.5),
        ("torch", 13.0),  # trial 1: torch on after 0.5 s baseline
        ("torch", 14.5),  # torch off at trial end
        ("lock", 15.5),  # unlock after the 1 s gap
    ]
    assert batch.finished
    assert not remote.torch and not remote.locked


def test_every_trial_measures_the_camera_lag():
    remote = FakeRemote()
    batch, streams = new_batch(remote, trials=3)
    run_scene(batch, streams, remote, start=10.0, until=60.0, lag=0.004)
    assert batch.finished
    assert len(batch.results) == 3
    assert batch.offsets() == pytest.approx([0.004] * 3)
    for result in batch.results:
        assert result.fps == pytest.approx((30.0, 30.0))
        assert result.rise == pytest.approx((180.0, 180.0))
        assert result.missed == (0, 0)
        assert not result.slow


def test_slow_camera_is_flagged():
    remote = FakeRemote()
    batch, streams = new_batch(remote, trials=1)
    run_scene(batch, streams, remote, start=10.0, until=60.0, period=1 / 20)
    (result,) = batch.results
    assert result.fps == pytest.approx((20.0, 20.0))
    assert result.slow


def test_undetected_trial_has_no_offset_and_is_left_out_of_the_mean():
    remote = FakeRemote()
    batch, streams = new_batch(remote, trials=2)
    run_scene(batch, streams, remote, start=10.0, until=60.0, sees_torch=(True, False))
    assert [r.offset for r in batch.results] == [None, None]
    assert batch.results[0].rise[1] == 0.0
    assert batch.offsets() == []


def test_abort_mid_batch_turns_torch_off_and_unlocks():
    remote = FakeRemote()
    batch, streams = new_batch(remote, trials=5)
    # 4 s in: exposure locked, first trial lit.
    now = run_scene(batch, streams, remote, start=10.0, until=14.0)
    assert remote.torch and remote.locked
    batch.abort()
    assert not remote.torch and not remote.locked
    assert batch.finished
    sent = len(remote.calls)
    batch.update(now + 10.0)
    assert len(remote.calls) == sent


def test_trial_number_counts_the_trial_in_progress():
    remote = FakeRemote()
    batch, streams = new_batch(remote, trials=2)
    assert batch.trial_number == 0
    run_scene(batch, streams, remote, start=10.0, until=13.0)
    assert batch.trial_number == 1
    run_scene(batch, streams, remote, start=13.0, until=60.0)
    assert batch.trial_number == 2


def test_trial_line_marks_slow_and_undetected():
    detected = TrialResult(0.0042, (29.9, 30.1), (85.0, 60.2), (0, 1), False)
    assert format_trial_result(3, detected, 20.0) == (
        "torch trial 3: offset +4.2 ms (cam1 minus cam0);"
        " cam0 fps 29.9 rise 85.0; cam1 fps 30.1 rise 60.2; missed cam0 0, cam1 1"
    )
    undetected = TrialResult(None, (24.0, 30.0), (5.0, 60.0), (0, 0), True)
    assert format_trial_result(4, undetected, 20.0) == (
        "torch trial 4: not detected (need rise 20);"
        " cam0 fps 24.0 rise 5.0; cam1 fps 30.0 rise 60.0; missed cam0 0, cam1 0 [slow]"
    )


def test_torch_csv_columns(tmp_path):
    path = tmp_path / "out" / "torch_offsets.csv"
    results = [
        TrialResult(0.0042, (29.9, 30.1), (85.0, 60.2), (0, 1), False),
        TrialResult(None, (24.0, 30.0), (5.0, 60.0), (0, 0), True),
    ]
    write_torch_csv(path, results)
    assert path.read_text().splitlines() == [
        "trial,offset_cam1_minus_cam0_seconds,fps_cam0,fps_cam1,"
        "rise_cam0,rise_cam1,missed_cam0,missed_cam1,slow",
        "1,0.004200,29.90,30.10,85.0,60.2,0,1,0",
        "2,,24.00,30.00,5.0,60.0,0,0,1",
    ]


# FlashTrial edge cases


def test_onset_at_second_sample_and_at_last_sample():
    cam0 = FakeStream(sequence([1.000, 1.033, 1.066, 1.100], bright_from=1))
    cam1 = FakeStream(sequence([1.010, 1.043, 1.076, 1.109], bright_from=3))
    trial = FlashTrial([cam0, cam1], duration=3.0, started_at=1.0)
    trial.collect()
    assert trial.offset() == pytest.approx(1.109 - 1.033)


def test_camera_lit_from_its_first_sample_has_no_visible_onset():
    cam0 = FakeStream(sequence([1.000, 1.033, 1.066], bright_from=0))
    cam1 = FakeStream(sequence([1.000, 1.033, 1.066], bright_from=2))
    trial = FlashTrial([cam0, cam1], duration=3.0, started_at=1.0)
    trial.collect()
    assert trial.offset() is None
    assert trial.rises() == [0.0, 180.0]


def test_single_sample_per_camera_gives_no_offset_and_zero_fps():
    cam0 = FakeStream(sequence([1.0], bright_from=0))
    cam1 = FakeStream(sequence([1.0], bright_from=99))
    trial = FlashTrial([cam0, cam1], duration=3.0, started_at=1.0)
    trial.collect()
    assert trial.offset() is None
    assert trial.fps() == [0.0, 0.0]
    assert trial.rises() == [0.0, 0.0]


def test_fps_with_duplicate_timestamps_is_zero_not_an_error():
    cam0 = FakeStream(sequence([2.0, 2.0, 2.0], bright_from=99))
    cam1 = FakeStream(sequence([2.0, 2.5], bright_from=99))
    trial = FlashTrial([cam0, cam1], duration=3.0, started_at=1.0)
    trial.collect()
    assert trial.fps() == pytest.approx([0.0, 2.0])


# TorchBatch edge cases


def step(batch, remote, now):
    """One update; returns the commands it sent."""
    remote.now = now
    before = len(remote.calls)
    batch.update(now)
    return remote.calls[before:]


def test_stalled_loop_sends_one_command_per_transition_and_skips_none():
    remote = FakeRemote()
    batch, _ = new_batch(remote, trials=2, start=10.0)
    now = 10.0
    while not batch.finished:
        now += 3.0  # every step arrives long after the phase should have ended
        sent = step(batch, remote, now)
        assert len([c for c in sent if c[0] == "torch"]) <= 1
        assert len([c for c in sent if c[0] == "lock"]) <= 1
        assert remote.torch == batch.torch_on
        assert remote.locked == batch.locked
    assert [kind for kind, _ in remote.calls] == [
        "torch", "lock", "torch", "torch", "torch", "torch", "torch", "lock"
    ]
    assert len(batch.results) == 2
    assert not remote.torch and not remote.locked


def test_lit_phase_lasts_its_full_length_after_a_stall_in_baseline():
    remote = FakeRemote()
    batch, _ = new_batch(remote, trials=1, start=10.0)
    step(batch, remote, 11.5)  # lock, torch off
    step(batch, remote, 12.5)  # trial 1 baseline starts; the loop then stalls
    assert step(batch, remote, 16.0) == [("torch", 16.0)]
    # The pulse is timed from the torch-on command, so it still lasts 1.5 s.
    assert step(batch, remote, 16.0) == []
    assert step(batch, remote, 17.4) == []
    assert step(batch, remote, 17.5) == [("torch", 17.5)]
    assert batch.results[0].offset is None
    assert step(batch, remote, 18.5) == [("lock", 18.5)]
    assert batch.finished


def test_repeated_updates_within_a_phase_send_nothing():
    remote = FakeRemote()
    batch, _ = new_batch(remote, trials=1, start=10.0)
    for now in (10.0, 10.5, 11.0, 11.4, 11.499):
        assert step(batch, remote, now) == []
    assert step(batch, remote, 11.5) == [("lock", 11.5), ("torch", 11.5)]
    for now in (11.5, 11.5, 12.0, 12.499):
        assert step(batch, remote, now) == []


def test_abort_before_locking_only_turns_the_torch_off():
    remote = FakeRemote()
    batch, _ = new_batch(remote, trials=3, start=10.0)
    batch.abort()
    assert remote.calls == [("torch", 10.0), ("torch", 10.0)]
    assert not remote.torch and not remote.locked
    assert batch.finished
    assert batch.results == []


def test_abort_after_done_and_a_second_abort_send_nothing():
    remote = FakeRemote()
    batch, streams = new_batch(remote, trials=1)
    run_scene(batch, streams, remote, start=10.0, until=60.0)
    assert batch.finished
    sent = len(remote.calls)
    batch.abort()
    batch.abort()
    assert len(remote.calls) == sent

    remote = FakeRemote()
    batch, streams = new_batch(remote, trials=3)
    run_scene(batch, streams, remote, start=10.0, until=14.0)
    batch.abort()
    sent = len(remote.calls)
    batch.abort()
    assert len(remote.calls) == sent


class FailingRemote(FakeRemote):
    """Raises on the n-th command, as a phone that dropped off the network would."""

    def __init__(self, fail_at_call):
        super().__init__()
        self.fail_at_call = fail_at_call
        self.failed = False

    def _check(self):
        if not self.failed and len(self.calls) + 1 == self.fail_at_call:
            self.failed = True
            raise RuntimeError("PUT failed")

    def toggle_torch(self):
        self._check()
        super().toggle_torch()

    def toggle_exposure_lock(self):
        self._check()
        super().toggle_exposure_lock()


def test_abort_after_a_failed_torch_off_resets_torch_and_lock_exactly_once():
    remote = FailingRemote(fail_at_call=3)  # torch on, lock, [torch off fails]
    batch, _ = new_batch(remote, trials=3, start=10.0)
    with pytest.raises(RuntimeError):
        step(batch, remote, 11.5)
    assert remote.torch and remote.locked
    assert batch.torch_on and batch.locked
    batch.abort()
    assert remote.calls == [("torch", 10.0), ("lock", 11.5), ("torch", 11.5), ("lock", 11.5)]
    assert not remote.torch and not remote.locked
    assert batch.finished


def test_abort_tries_the_unlock_even_when_the_torch_off_fails():
    remote = FailingRemote(fail_at_call=5)  # torch on, lock, torch off, torch on, [abort torch off]
    batch, _ = new_batch(remote, trials=3, start=10.0)
    step(batch, remote, 11.5)
    step(batch, remote, 12.5)
    step(batch, remote, 13.0)
    assert remote.torch and remote.locked
    with pytest.raises(RuntimeError, match="torch may still be on") as info:
        batch.abort()
    assert "exposure may still be locked" not in str(info.value)
    assert remote.torch and not remote.locked
    assert batch.finished
    # The torch is still believed on, so a second abort retries only that.
    sent = len(remote.calls)
    batch.abort()
    assert remote.calls[sent:] == [("torch", 13.0)]
    assert not remote.torch and not remote.locked


def test_abort_after_a_failed_lock_does_not_send_an_unlock():
    remote = FailingRemote(fail_at_call=2)  # torch on, [lock fails]
    batch, _ = new_batch(remote, trials=3, start=10.0)
    with pytest.raises(RuntimeError):
        step(batch, remote, 11.5)
    batch.abort()
    assert remote.calls == [("torch", 10.0), ("torch", 11.5)]
    assert not remote.torch and not remote.locked


def test_each_trial_fps_comes_from_its_own_samples():
    remote = FakeRemote()
    batch, streams = new_batch(remote, trials=2, start=10.0)
    # Trial 1 runs at 20 fps (evaluated at 14.5 s); trial 2 at 30 fps.
    now = run_scene(batch, streams, remote, start=10.0, until=14.6, period=1 / 20)
    assert len(batch.results) == 1
    run_scene(batch, streams, remote, start=now, until=60.0, period=1 / 30)
    first, second = batch.results
    assert first.fps == pytest.approx((20.0, 20.0))
    assert first.slow
    assert second.fps == pytest.approx((30.0, 30.0))
    assert not second.slow


def test_one_slow_camera_flags_the_trial():
    remote = FakeRemote()
    batch, streams = new_batch(remote, trials=1, start=10.0)
    now, tick = 10.0, 0
    while not batch.finished:
        now += FRAME_PERIOD
        tick += 1
        image = BRIGHT if remote.torch else DARK
        streams[0].push(now, image)
        if tick % 2 == 0:  # camera 1 delivers every other frame
            streams[1].push(now, image)
        remote.now = now
        batch.update(now)
    (result,) = batch.results
    assert result.fps == pytest.approx((30.0, 15.0))
    assert result.slow
    assert result.offset is not None


def test_offsets_keeps_only_detected_trials_in_order():
    batch = TorchBatch([], FakeRemote(), trials=3, min_jump=20.0, now=0.0)
    batch.results.extend(
        [
            TrialResult(0.004, (30.0, 30.0), (100.0, 100.0), (0, 0), False),
            TrialResult(None, (30.0, 30.0), (0.0, 100.0), (0, 0), False),
            TrialResult(-0.002, (30.0, 30.0), (100.0, 100.0), (0, 0), True),
        ]
    )
    assert batch.offsets() == [0.004, -0.002]


def test_zero_trials_lights_locks_and_finishes_with_no_results():
    remote = FakeRemote()
    batch, _ = new_batch(remote, trials=0, start=10.0)
    assert step(batch, remote, 11.5) == [("lock", 11.5), ("torch", 11.5)]
    assert step(batch, remote, 12.5) == [("lock", 12.5)]
    assert batch.finished
    assert batch.results == []
    assert batch.trial_number == 0
    assert not remote.torch and not remote.locked


def test_torch_csv_with_no_results_has_only_the_header(tmp_path):
    path = tmp_path / "torch_offsets.csv"
    write_torch_csv(path, [])
    assert path.read_text().splitlines() == [
        "trial,offset_cam1_minus_cam0_seconds,fps_cam0,fps_cam1,"
        "rise_cam0,rise_cam1,missed_cam0,missed_cam1,slow"
    ]
