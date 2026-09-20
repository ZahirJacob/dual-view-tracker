import numpy as np

from dual_view_tracker.flash import (
    BrightnessSample,
    find_flash_onset,
    largest_rise,
    mean_brightness,
    summarize_offsets,
)


def samples(*values: float) -> list[BrightnessSample]:
    return [BrightnessSample(float(i) / 30, v) for i, v in enumerate(values)]


def test_onset_is_sample_with_largest_jump():
    onset = find_flash_onset(samples(50, 52, 51, 200, 210, 60), min_jump=20)
    assert onset == 3 / 30


def test_no_onset_below_min_jump():
    assert find_flash_onset(samples(50, 52, 51, 55, 54), min_jump=20) is None


def test_no_onset_with_fewer_than_two_samples():
    assert find_flash_onset([], min_jump=20) is None
    assert find_flash_onset(samples(50), min_jump=20) is None


def test_summarize_empty():
    assert summarize_offsets([]) == {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}


def test_summarize_known_values():
    stats = summarize_offsets([0.010, 0.020, 0.030])
    assert stats["count"] == 3
    assert stats["mean"] == 0.020
    assert abs(stats["std"] - 0.008165) < 1e-6
    assert stats["min"] == 0.010
    assert stats["max"] == 0.030


def test_mean_brightness_of_flat_images():
    assert mean_brightness(np.full((48, 64, 3), 200, dtype=np.uint8)) == 200.0
    assert mean_brightness(np.zeros((48, 64), dtype=np.uint8)) == 0.0


def test_onset_at_second_sample_is_earliest_detectable():
    assert find_flash_onset(samples(50, 200, 200), min_jump=20) == 1 / 30


def test_flash_already_on_at_first_sample_is_not_detected():
    assert find_flash_onset(samples(200, 201, 200, 202), min_jump=20) is None


def test_onset_at_last_sample():
    assert find_flash_onset(samples(50, 51, 50, 200), min_jump=20) == 3 / 30


def test_slow_ramp_does_not_count_as_onset():
    assert find_flash_onset(samples(50, 60, 70, 80, 90, 100, 110), min_jump=20) is None


def test_sharp_jump_beats_larger_total_ramp():
    onset = find_flash_onset(samples(50, 60, 70, 80, 90, 100, 100, 125), min_jump=20)
    assert onset == 7 / 30


def test_brightness_drop_is_ignored():
    assert find_flash_onset(samples(200, 50, 52), min_jump=20) is None


def test_large_drop_does_not_hide_smaller_rise():
    assert find_flash_onset(samples(50, 80, 10, 12), min_jump=20) == 1 / 30


def test_first_of_equal_jumps_wins():
    assert find_flash_onset(samples(50, 100, 100, 150), min_jump=20) == 1 / 30


def test_min_jump_boundary_is_inclusive():
    assert find_flash_onset(samples(50, 70), min_jump=20) == 1 / 30


def test_onset_returns_sample_timestamp_not_position():
    uneven = [
        BrightnessSample(10.5, 50),
        BrightnessSample(10.7, 51),
        BrightnessSample(11.2, 200),
    ]
    assert find_flash_onset(uneven, min_jump=20) == 11.2


def test_mean_brightness_averages_colour_channels():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 1] = 90
    image[..., 2] = 210
    assert mean_brightness(image) == 100.0


def test_largest_rise_reports_size_and_timestamp():
    samples = [
        BrightnessSample(1.0, 10.0),
        BrightnessSample(1.1, 12.0),
        BrightnessSample(1.2, 40.0),
        BrightnessSample(1.3, 41.0),
    ]
    assert largest_rise(samples) == (28.0, 1.2)


def test_largest_rise_with_too_few_samples():
    assert largest_rise([]) == (0.0, None)
    assert largest_rise([BrightnessSample(1.0, 5.0)]) == (0.0, None)
