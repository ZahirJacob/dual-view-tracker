import numpy as np
import pytest

from dual_view_tracker.capture import Frame
from dual_view_tracker.view import FlashTrial, parse_args, write_offsets_csv

DARK = np.full((16, 16, 3), 40, dtype=np.uint8)
BRIGHT = np.full((16, 16, 3), 220, dtype=np.uint8)


class FakeStream:
    """Only `frames()` is needed by FlashTrial, so no thread and no camera."""

    def __init__(self, frames=()):
        self.buffer = list(frames)

    def frames(self):
        return list(self.buffer)


def sequence(timestamps, bright_from):
    return [
        Frame(i, t, BRIGHT if i >= bright_from else DARK) for i, t in enumerate(timestamps)
    ]


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


def test_parse_args_defaults_and_resolutions():
    args = parse_args([])
    assert args.res0 == (1280, 720)
    assert args.res1 == (1920, 1080)
    args = parse_args(["--res0", "640X480", "--cam1", "2"])
    assert args.res0 == (640, 480)
    assert args.cam1 == 2
    with pytest.raises(SystemExit):
        parse_args(["--res0", "wide"])


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


def test_offsets_csv_header_states_sign_convention(tmp_path):
    path = tmp_path / "out" / "flash_offsets.csv"
    write_offsets_csv(path, [0.004, -0.0061])
    assert path.read_text().splitlines() == [
        "trial,offset_cam1_minus_cam0_seconds",
        "1,0.004000",
        "2,-0.006100",
    ]
