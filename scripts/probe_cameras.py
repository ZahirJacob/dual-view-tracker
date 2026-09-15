"""Discover what the connected webcams really support. Prints everything, hides nothing.

Run: .venv/Scripts/python.exe scripts/probe_cameras.py
"""

import time

import cv2

from dual_view_tracker.capture import decode_fourcc

DEVICE_INDICES = range(6)
BACKENDS = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]
MODES = [(640, 480), (1280, 720), (1920, 1080)]
REQUESTED_FPS = 30
WARMUP_FRAMES = 30
TIMED_FRAMES = 60

CONTROL_PROPS = [
    ("AUTOFOCUS", cv2.CAP_PROP_AUTOFOCUS),
    ("FOCUS", cv2.CAP_PROP_FOCUS),
    ("AUTO_EXPOSURE", cv2.CAP_PROP_AUTO_EXPOSURE),
    ("EXPOSURE", cv2.CAP_PROP_EXPOSURE),
]


def probe_mode(cap, width, height):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)

    for _ in range(WARMUP_FRAMES):
        cap.read()
    read_ok = 0
    started = time.perf_counter()
    for _ in range(TIMED_FRAMES):
        ok, _ = cap.read()
        read_ok += ok
    elapsed = time.perf_counter() - started
    measured_fps = read_ok / elapsed if elapsed > 0 else 0.0

    print(
        f"    requested {width}x{height} @ {REQUESTED_FPS} MJPG -> actual"
        f" {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        f" @ {cap.get(cv2.CAP_PROP_FPS):g} {decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))},"
        f" measured {measured_fps:.1f} fps ({read_ok}/{TIMED_FRAMES} reads ok)"
    )


def probe_controls(cap):
    for name, prop in CONTROL_PROPS:
        print(f"    {name}: {cap.get(prop)!r}")
    original = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    accepted = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    print(f"    set(AUTOFOCUS, 0) returned {accepted}; now reads {cap.get(cv2.CAP_PROP_AUTOFOCUS)!r}")
    cap.set(cv2.CAP_PROP_AUTOFOCUS, original)


def open_device(index, backend):
    """Return an opened capture, or None after printing why it failed."""
    try:
        cap = cv2.VideoCapture(index, backend)
    except Exception as exc:  # cv2 may raise instead of returning a closed capture
        print(f"    open raised {type(exc).__name__}: {exc}")
        return None
    if cap.isOpened():
        return cap
    cap.release()
    return None


def probe_device(index, backend_name, backend):
    print(f"device {index} / {backend_name}:")
    cap = open_device(index, backend)
    if cap is None:
        print("    not opened")
        return
    cap.release()

    # Reopen for every mode: MSMF crashes inside OpenCV when the resolution
    # is changed on a capture that already delivered frames.
    for width, height in MODES:
        cap = open_device(index, backend)
        if cap is None:
            print(f"    mode {width}x{height}: reopen failed")
            continue
        try:
            probe_mode(cap, width, height)
        except Exception as exc:
            print(f"    mode {width}x{height} raised {type(exc).__name__}: {exc}")
        finally:
            cap.release()

    cap = open_device(index, backend)
    if cap is None:
        print("    controls: reopen failed")
        return
    try:
        probe_controls(cap)
    except Exception as exc:
        print(f"    controls raised {type(exc).__name__}: {exc}")
    finally:
        cap.release()


def main():
    print(f"OpenCV {cv2.__version__}")
    for index in DEVICE_INDICES:
        for backend_name, backend in BACKENDS:
            probe_device(index, backend_name, backend)


if __name__ == "__main__":
    main()
