import pytest

from dual_view_tracker.remote import DroidCamRemote


class RecordingSend:
    def __init__(self, error: Exception | None = None):
        self.calls = []
        self.error = error

    def __call__(self, method, url, timeout):
        self.calls.append((method, url))
        if self.error is not None:
            raise self.error


def test_toggles_put_to_the_droidcam_endpoints():
    send = RecordingSend()
    remote = DroidCamRemote("http://172.26.104.168:4747/", send=send)
    remote.toggle_torch()
    remote.toggle_exposure_lock()
    assert send.calls == [
        ("PUT", "http://172.26.104.168:4747/v1/camera/torch_toggle"),
        ("PUT", "http://172.26.104.168:4747/v1/camera/el_toggle"),
    ]


def test_failure_raises_runtime_error_naming_the_url():
    send = RecordingSend(error=ConnectionRefusedError("refused"))
    remote = DroidCamRemote("http://phone:4747", send=send)
    with pytest.raises(RuntimeError, match="http://phone:4747/v1/camera/torch_toggle"):
        remote.toggle_torch()


def test_timeout_is_passed_to_every_request():
    seen = []
    remote = DroidCamRemote("http://phone:4747", timeout=0.25, send=lambda m, u, t: seen.append(t))
    remote.toggle_torch()
    remote.toggle_exposure_lock()
    assert seen == [0.25, 0.25]


def test_http_error_from_the_phone_is_reported_with_the_url():
    import urllib.error

    error = urllib.error.HTTPError("http://phone:4747/v1/camera/el_toggle", 404, "nf", {}, None)
    remote = DroidCamRemote("http://phone:4747", send=RecordingSend(error=error))
    with pytest.raises(RuntimeError, match="el_toggle"):
        remote.toggle_exposure_lock()
