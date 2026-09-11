import time

import cv2
import numpy as np
import pytest

from dual_view_tracker.capture import CameraConfig, CameraStream, decode_fourcc


class FakeCapture:
    """Yields numbered frames as fast as asked, like a device with no wait."""

    def __init__(self):
        self.count = 0
        self.released = False
        self.props = {}
        self.grab_results = []
        self.retrieve_results = []

    def isOpened(self):
        return not self.released

    def grab(self):
        # A short pause keeps successive perf_counter() values distinct.
        time.sleep(0.0005)
        if self.grab_results:
            return self.grab_results.pop(0)
        return True

    def retrieve(self):
        if self.retrieve_results:
            return self.retrieve_results.pop(0)
        image = np.full((4, 4), self.count % 256, dtype=np.uint8)
        self.count += 1
        return True, image

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def release(self):
        self.released = True


def wait_until(condition, timeout=2.0):
    deadline = time.perf_counter() + timeout
    while not condition():
        assert time.perf_counter() < deadline, "timed out waiting for the capture thread"
        time.sleep(0.005)


def make_stream(fps=30, history_seconds=0.1):
    fake = FakeCapture()
    stream = CameraStream(CameraConfig(0, 4, 4, fps=fps), history_seconds, capture=fake)
    return stream, fake


def test_buffer_is_bounded_to_maxlen():
    stream, fake = make_stream(fps=30, history_seconds=0.1)  # maxlen 3
    with stream:
        wait_until(lambda: fake.count > 20)
        frames = stream.frames()
    assert 2 <= len(frames) <= 3


def test_buffer_never_smaller_than_two():
    stream, _ = make_stream(fps=30, history_seconds=0.0)
    assert stream._frames.maxlen == 2


def test_indices_and_timestamps_increase():
    stream, fake = make_stream(fps=30, history_seconds=1.0)
    with stream:
        wait_until(lambda: fake.count >= 10)
        frames = stream.frames()
    indices = [f.index for f in frames]
    timestamps = [f.timestamp for f in frames]
    assert indices == list(range(indices[0], indices[0] + len(indices)))
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)


def test_latest_and_frames():
    stream, fake = make_stream(fps=30, history_seconds=1.0)
    assert stream.latest() is None
    assert stream.frames() == []
    with stream:
        wait_until(lambda: fake.count >= 5)
        frames = stream.frames()
        latest = stream.latest()
    assert latest is not None
    assert latest.index >= frames[-1].index
    assert frames[0].image[0, 0] == frames[0].index % 256


def test_measured_fps_from_timestamps():
    stream, fake = make_stream(fps=30, history_seconds=1.0)
    assert stream.measured_fps() == 0.0
    with stream:
        wait_until(lambda: fake.count >= 10)
        assert stream.measured_fps() > 0.0


def test_stop_releases_capture():
    stream, fake = make_stream()
    stream.start()
    stream.stop()
    assert fake.released
    assert stream._thread is None


def test_start_applies_requested_settings():
    stream, fake = make_stream(fps=15)
    stream.start()
    stream.stop()
    assert fake.props[cv2.CAP_PROP_FRAME_WIDTH] == 4
    assert fake.props[cv2.CAP_PROP_FRAME_HEIGHT] == 4
    assert fake.props[cv2.CAP_PROP_FPS] == 15
    assert fake.props[cv2.CAP_PROP_BUFFERSIZE] == 1
    assert decode_fourcc(fake.props[cv2.CAP_PROP_FOURCC]) == "MJPG"


def test_start_fails_on_closed_capture():
    stream, fake = make_stream()
    fake.released = True
    with pytest.raises(RuntimeError, match="device 0"):
        stream.start()


def test_failed_grabs_are_counted_and_capture_continues():
    stream, fake = make_stream(history_seconds=1.0)
    fake.grab_results = [False, False, False]
    with stream:
        wait_until(lambda: fake.count >= 5)
    assert stream.dropped_frames == 3
    assert [f.index for f in stream.frames()][:3] == [0, 1, 2]


def test_failed_retrieve_is_counted_and_capture_continues():
    stream, fake = make_stream(history_seconds=1.0)
    fake.retrieve_results = [(False, None)]
    with stream:
        wait_until(lambda: fake.count >= 5)
    assert stream.dropped_frames == 1
    assert [f.index for f in stream.frames()][:3] == [0, 1, 2]


def test_actual_settings_are_read_once_at_start():
    stream, fake = make_stream(fps=15)
    fake.props[cv2.CAP_PROP_FRAME_WIDTH] = 4.0
    fake.props[cv2.CAP_PROP_FPS] = 15.0
    assert stream.actual_settings() == {}
    stream.start()
    fake.props[cv2.CAP_PROP_FPS] = 99.0
    stream.stop()
    assert stream.actual_settings() == {"width": 4, "height": 4, "fps": 15.0, "fourcc": "MJPG"}


def test_decode_fourcc_handles_odd_driver_values():
    assert decode_fourcc(22) == "#22"
    assert decode_fourcc(-1) == "#-1"
    assert decode_fourcc(0x47504A4D) == "MJPG"


def test_thread_stops_when_capture_closes():
    stream, fake = make_stream(history_seconds=1.0)
    stream.start()
    wait_until(lambda: fake.count >= 3)
    fake.released = True
    stream._thread.join(timeout=2.0)
    assert not stream._thread.is_alive()
    stream.stop()
