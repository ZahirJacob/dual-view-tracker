# dual-view-tracker

Versión en español: [README.es.md](README.es.md)

Track people in 3D using two ordinary webcams pointed at the same room from different angles.

The plan: detect body keypoints in each camera, match the detections across views using the camera geometry, and triangulate them into 3D positions that can be tracked over time.

## Current status

Two-camera capture with timestamps, frame pairing and a flash-based synchronization test. No calibration or pose estimation yet.

## Requirements

- Windows 10/11 (cameras are read through Media Foundation or DirectShow)
- Python 3.12 or newer
- Two USB webcams, or one webcam plus a phone that serves a video stream

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

Find out what your cameras really support. The probe tries device indices 0 to 5 on both backends, at 640x480, 1280x720 and 1920x1080, and reports the actual resolution, frame rate and focus and exposure controls:

```powershell
python scripts/probe_cameras.py
```

Open both cameras and show them side by side:

```powershell
python -m dual_view_tracker.view --cam0 0 --cam1 1 --res0 1280x720 --res1 1920x1080
```

Those values are the defaults. Other options: `--fps` (30), `--backend msmf|dshow` (msmf), `--max-wait` seconds before pairing an unmatched frame (0.1), `--display-height` pixels (480), `--min-jump` smallest brightness rise, on a 0-255 scale, that counts as the flash (20).

`--cam0` and `--cam1` take a device index or a video stream URL. A phone running a camera app that serves a video stream over the network (DroidCam, for example) can be the second camera:

```powershell
python -m dual_view_tracker.view --cam0 0 --cam1 http://192.168.0.20:4747/video
```

For a stream, the phone decides the resolution and frame rate, so `--res1` is ignored for it.

On start, the viewer prints the requested and actual settings of each camera. The overlay shows each camera's real resolution and measured frame rate, and the time gap between the two frames currently paired, marked `settled` or `UNSETTLED`. Press `q` or Esc to quit.

### Synchronization test

Webcams cannot be synchronized in hardware, so the viewer measures how far apart the two cameras are in time. Point a phone flash so both cameras see it, press `f`, and fire the flash within three seconds. The viewer finds the frame where brightness jumps in each camera and reports the difference (camera 1 minus camera 0, so a positive value means camera 1 saw the flash later). If the rise is smaller than `--min-jump` on either camera, the trial is discarded; each trial prints the largest rise seen per camera, so a missed flash can be diagnosed. Repeat a few times; the mean and spread are shown on screen, printed on exit, and written to `output/flash_offsets.csv` if at least one trial succeeded.

## How it works

- `capture.py` reads each camera in its own thread and keeps the last half second of frames, each stamped with its arrival time. A stream URL is read through FFmpeg.
- `sync.py` pairs the newest frame of the reference camera with the closest frame of the other camera. It waits until the other camera has a frame at or after the reference frame, so no later frame can be a better match. This costs at most one frame period of latency. If the other camera stalls for longer than `--max-wait`, the pair is made anyway and marked unsettled.
- `flash.py` finds the flash onset as the largest brightness rise between consecutive frames.

## Tests

```powershell
pytest
```

The tests cover pairing, flash detection, the capture buffer and the viewer's trial logic, all with fake cameras.

## License

MIT
