"""Remote control of the phone camera app (DroidCam) over its HTTP API."""

from __future__ import annotations

import urllib.request
from typing import Callable


def send_request(method: str, url: str, timeout: float) -> None:
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


class DroidCamRemote:
    """Both endpoints are state-blind toggles: the app does not report whether
    the torch or the exposure lock is currently on, so the caller must track it.

    `send(method, url, timeout)` can be replaced in tests to avoid the network.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 3.0,
        send: Callable[[str, str, float], None] = send_request,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._send = send

    def toggle_torch(self) -> None:
        self._put("/v1/camera/torch_toggle")

    def toggle_exposure_lock(self) -> None:
        self._put("/v1/camera/el_toggle")

    def _put(self, path: str) -> None:
        url = self.base_url + path
        try:
            self._send("PUT", url, self.timeout)
        except OSError as error:
            # URLError and HTTPError are both OSError subclasses.
            raise RuntimeError(f"PUT {url} failed: {error}") from error
