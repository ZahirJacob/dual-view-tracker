from types import SimpleNamespace

import pytest

from dual_view_tracker import view
from dual_view_tracker.trials import TorchBatch, TrialResult
from dual_view_tracker.view import (
    batch_prompt,
    format_stats,
    parse_args,
    print_torch_summary,
    write_offsets_csv,
)


class IdleRemote:
    def toggle_torch(self):
        pass

    def toggle_exposure_lock(self):
        pass


def test_parse_args_defaults_and_resolutions():
    args = parse_args([])
    assert args.res0 == (1280, 720)
    assert args.res1 == (1920, 1080)
    assert args.remote is None
    assert args.trials == 20
    args = parse_args(["--res0", "640X480", "--cam1", "2"])
    assert args.res0 == (640, 480)
    assert args.cam1 == 2
    assert parse_args(["--cam1", "http://x:4747/video"]).cam1 == "http://x:4747/video"
    args = parse_args(["--remote", "http://x:4747", "--trials", "5"])
    assert args.remote == "http://x:4747"
    assert args.trials == 5
    with pytest.raises(SystemExit):
        parse_args(["--res0", "wide"])


@pytest.mark.parametrize("value", ["0", "-3", "2.5", "many"])
def test_trials_must_be_a_positive_whole_number(value, capsys):
    with pytest.raises(SystemExit):
        parse_args(["--trials", value])
    assert "--trials" in capsys.readouterr().err


def test_offsets_csv_header_states_sign_convention(tmp_path):
    path = tmp_path / "out" / "flash_offsets.csv"
    write_offsets_csv(path, [0.004, -0.0061])
    assert path.read_text().splitlines() == [
        "trial,offset_cam1_minus_cam0_seconds",
        "1,0.004000",
        "2,-0.006100",
    ]


def test_batch_prompt_shows_progress_and_running_mean():
    batch = TorchBatch([], IdleRemote(), trials=20, min_jump=20.0, now=0.0)
    assert batch_prompt(batch) == "TORCH BATCH trial 0/20  mean +0.0 ms"
    batch.results.append(TrialResult(0.004, (30.0, 30.0), (100.0, 100.0), (0, 0), False))
    batch.results.append(TrialResult(None, (30.0, 30.0), (0.0, 100.0), (0, 0), False))
    assert batch_prompt(batch) == "TORCH BATCH trial 2/20  mean +4.0 ms"


def test_torch_summary_excludes_slow_and_undetected_trials(capsys):
    results = [
        TrialResult(0.004, (30.0, 30.0), (100.0, 100.0), (0, 0), False),
        TrialResult(0.050, (25.0, 30.0), (100.0, 100.0), (0, 0), True),
        TrialResult(0.006, (30.0, 30.0), (100.0, 100.0), (0, 0), False),
        TrialResult(None, (30.0, 30.0), (0.0, 100.0), (0, 0), False),
        TrialResult(None, (20.0, 30.0), (0.0, 100.0), (0, 0), True),
    ]
    print_torch_summary(results)
    assert capsys.readouterr().out.strip() == (
        "torch offsets (cam1 minus cam0): count 2 valid (1 slow excluded, 2 not detected),"
        " mean +5.00 ms, std 1.00 ms, min +4.00 ms, max +6.00 ms"
    )
    assert format_stats([]) == "mean +0.00 ms, std 0.00 ms, min +0.00 ms, max +0.00 ms"


# Batch wiring in the main loop, run with fake cameras, a scripted keyboard
# and a clock that advances a tenth of a second per loop iteration.

TICK = 0.1


class FakeCameraStream:
    def __init__(self, config):
        self.config = config
        self.settings = {"width": 1, "height": 1, "fps": 30, "fourcc": "MJPG"}
        self.dropped_frames = 0
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def frames(self):
        return []

    def latest(self):
        return None

    def actual_settings(self):
        return dict(self.settings)

    def measured_fps(self):
        return 0.0


class ScriptedRemote:
    """Records toggles; optionally raises once a given number of commands went through."""

    def __init__(self, fail_after=None):
        self.calls = []
        self.torch = False
        self.locked = False
        self.fail_after = fail_after

    def _guard(self):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("PUT http://phone:4747/v1/camera/torch_toggle failed")

    def toggle_torch(self):
        self._guard()
        self.torch = not self.torch
        self.calls.append("torch")

    def toggle_exposure_lock(self):
        self._guard()
        self.locked = not self.locked
        self.calls.append("lock")


def run_main(monkeypatch, tmp_path, keys, argv, remote):
    """Run view.main with `keys` (one per loop iteration, 0 = no key), then q."""
    clock = {"now": 100.0}

    def perf_counter():
        clock["now"] += TICK
        return clock["now"]

    script = iter(list(keys) + [ord("q")])
    fake_cv2 = SimpleNamespace(waitKey=lambda _: next(script), destroyAllWindows=lambda: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(view, "time", SimpleNamespace(perf_counter=perf_counter))
    monkeypatch.setattr(view, "cv2", fake_cv2)
    monkeypatch.setattr(view, "CameraStream", FakeCameraStream)
    monkeypatch.setattr(view, "DroidCamRemote", lambda base_url: remote)
    view.main(argv)


def idle(seconds):
    return [0] * int(round(seconds / TICK))


def test_t_key_without_remote_prints_a_hint_and_starts_nothing(monkeypatch, tmp_path, capsys):
    run_main(monkeypatch, tmp_path, [ord("t")] + idle(1.0), [], ScriptedRemote())
    out = capsys.readouterr().out
    assert "torch batch needs --remote" in out
    assert "torch batch started" not in out
    assert not (tmp_path / "output" / "torch_offsets.csv").exists()


def test_completed_batch_prints_each_trial_once_and_writes_the_csv(monkeypatch, tmp_path, capsys):
    remote = ScriptedRemote()
    # One trial takes 1.5 + 1.0 + 2.0 + 1.0 s; idle well past that before quitting.
    run_main(
        monkeypatch, tmp_path, [ord("t")] + idle(8.0),
        ["--remote", "http://phone:4747", "--trials", "1"], remote,
    )
    out = capsys.readouterr().out
    assert out.count("torch trial 1:") == 1
    assert out.count("torch batch done") == 1
    assert "torch offsets (cam1 minus cam0): count 0 valid (0 slow excluded, 1 not detected)" in out
    assert remote.calls == ["torch", "lock", "torch", "torch", "torch", "lock"]
    assert not remote.torch and not remote.locked
    rows = (tmp_path / "output" / "torch_offsets.csv").read_text().splitlines()
    assert len(rows) == 2
    assert rows[1].startswith("1,,")


def test_quit_mid_batch_resets_the_phone_and_keeps_finished_trials(monkeypatch, tmp_path, capsys):
    remote = ScriptedRemote()
    # Trial 1 is evaluated 4.5 s after `t`; quit during the gap before trial 2.
    run_main(
        monkeypatch, tmp_path, [ord("t")] + idle(5.0),
        ["--remote", "http://phone:4747", "--trials", "5"], remote,
    )
    out = capsys.readouterr().out
    assert out.count("torch trial 1:") == 1
    assert "torch trial 2:" not in out
    assert "torch batch done" not in out
    assert not remote.torch and not remote.locked
    assert remote.calls == ["torch", "lock", "torch", "torch", "torch", "lock"]
    assert "count 0 valid (0 slow excluded, 1 not detected)" in out
    rows = (tmp_path / "output" / "torch_offsets.csv").read_text().splitlines()
    assert len(rows) == 2


def test_quit_while_lit_turns_the_torch_off(monkeypatch, tmp_path):
    remote = ScriptedRemote()
    run_main(
        monkeypatch, tmp_path, [ord("t")] + idle(3.5),
        ["--remote", "http://phone:4747", "--trials", "5"], remote,
    )
    assert remote.calls == ["torch", "lock", "torch", "torch", "torch", "lock"]
    assert not remote.torch and not remote.locked


def test_phone_reset_failure_on_quit_is_reported_not_raised(monkeypatch, tmp_path, capsys):
    remote = ScriptedRemote(fail_after=4)  # the abort's torch-off is the fifth command
    run_main(
        monkeypatch, tmp_path, [ord("t")] + idle(3.5),
        ["--remote", "http://phone:4747", "--trials", "5"], remote,
    )
    out = capsys.readouterr().out
    assert "torch batch: could not reset the phone: PUT http://phone:4747" in out
    assert "flash offsets (cam1 minus cam0)" in out


def test_t_key_is_ignored_during_a_flash_trial(monkeypatch, tmp_path, capsys):
    remote = ScriptedRemote()
    run_main(
        monkeypatch, tmp_path, [ord("f"), ord("t")] + idle(1.0),
        ["--remote", "http://phone:4747"], remote,
    )
    out = capsys.readouterr().out
    assert "flash trial armed" in out
    assert "torch batch started" not in out
    assert remote.calls == []


def test_phone_dropping_mid_batch_keeps_the_finished_trials(monkeypatch, tmp_path, capsys):
    # Trial 1 completes (5 commands); trial 2's torch-on is the 6th command and
    # fails, as does the unlock in the abort. The viewer keeps running until q.
    remote = ScriptedRemote(fail_after=5)
    run_main(
        monkeypatch, tmp_path, [ord("t")] + idle(8.0),
        ["--remote", "http://phone:4747", "--trials", "5"], remote,
    )
    out = capsys.readouterr().out
    assert out.count("torch trial 1:") == 1
    assert "torch batch stopped: PUT http://phone:4747" in out
    assert "could not reset the phone: PUT http://phone:4747" in out
    assert "exposure may still be locked" in out
    assert "torch batch done" not in out
    assert "count 0 valid (0 slow excluded, 1 not detected)" in out
    rows = (tmp_path / "output" / "torch_offsets.csv").read_text().splitlines()
    assert len(rows) == 2


def test_batch_that_cannot_start_leaves_the_viewer_running(monkeypatch, tmp_path, capsys):
    remote = ScriptedRemote(fail_after=0)
    run_main(
        monkeypatch, tmp_path, [ord("t")] + idle(1.0),
        ["--remote", "http://phone:4747", "--trials", "5"], remote,
    )
    out = capsys.readouterr().out
    assert "torch batch could not start: PUT http://phone:4747" in out
    assert "torch batch started" not in out
    assert "flash offsets (cam1 minus cam0)" in out
    assert not (tmp_path / "output" / "torch_offsets.csv").exists()


def test_trial_numbers_continue_across_batches(monkeypatch, tmp_path, capsys):
    remote = ScriptedRemote()
    one_batch = [ord("t")] + idle(6.0)
    run_main(
        monkeypatch, tmp_path, one_batch + one_batch,
        ["--remote", "http://phone:4747", "--trials", "1"], remote,
    )
    out = capsys.readouterr().out
    assert out.count("torch trial 1:") == 1
    assert out.count("torch trial 2:") == 1
    rows = (tmp_path / "output" / "torch_offsets.csv").read_text().splitlines()
    assert [row.split(",")[0] for row in rows[1:]] == ["1", "2"]
