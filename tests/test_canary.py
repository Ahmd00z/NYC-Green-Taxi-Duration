"""Tests for the canary helpers: traffic split rewriting and the rollback verdict."""

import shutil
from pathlib import Path

import pytest

from prodml import canary

CONF = Path(__file__).resolve().parents[1] / "deploy/nginx/nginx.conf"


def _line(dep: str, status: int, seconds: float) -> str:
    return f"t deployment={dep} status={status} upstream_time={seconds}"


@pytest.fixture
def conf(tmp_path) -> Path:
    path = tmp_path / "nginx.conf"
    shutil.copy(CONF, path)
    return path


@pytest.mark.parametrize("pct", [0, 10, 25, 50, 100])
def test_set_split_roundtrip(conf, pct):
    canary.set_split(conf, pct)
    assert canary.current_split(conf) == pct
    text = conf.read_text()
    # The rest of the file must be untouched and the markers must survive.
    assert text.count(canary.START) == 1 and text.count(canary.END) == 1
    assert "upstream stable" in text


def test_set_split_validates(conf, tmp_path):
    with pytest.raises(ValueError):
        canary.set_split(conf, 101)
    bad = tmp_path / "bad.conf"
    bad.write_text("no markers here")
    with pytest.raises(ValueError):
        canary.set_split(bad, 10)


def test_judge_healthy_canary():
    lines = [_line("canary", 200, 0.02)] * 20 + [_line("stable", 200, 0.02)] * 80
    v = canary.judge(lines)
    assert not v.bad and v.requests == 20


def test_judge_slow_canary_is_bad():
    lines = [_line("canary", 200, 0.9)] * 10
    v = canary.judge(lines)
    assert v.bad and "p95" in v.reason


def test_judge_errors_are_bad():
    lines = [_line("canary", 500, 0.01)] * 3 + [_line("canary", 200, 0.01)] * 7
    v = canary.judge(lines)
    assert v.bad and "errors" in v.reason


def test_judge_ignores_stable_and_low_traffic():
    assert not canary.judge([_line("stable", 500, 5)] * 50).bad
    assert not canary.judge([_line("canary", 500, 5)] * 2).bad  # too few requests


def test_watch_rolls_back_after_two_bad_intervals(conf, tmp_path, monkeypatch):
    log = tmp_path / "access.log"
    log.write_text("")
    canary.set_split(conf, 10)
    reloads = []
    monkeypatch.setattr(canary, "reload_nginx", lambda cmd: reloads.append(cmd))
    monkeypatch.setattr(canary.time, "sleep", lambda s: None)

    # Each "sleep" lets the next interval's traffic arrive.
    batches = iter([[_line("canary", 200, 0.8)] * 10] * 5)

    def fake_sleep(_):
        with log.open("a") as fh:
            fh.write("\n".join(next(batches)) + "\n")

    monkeypatch.setattr(canary.time, "sleep", fake_sleep)
    rolled = canary.watch(log, conf, ["reload"], interval_s=0, max_intervals=5)
    assert rolled and canary.current_split(conf) == 0 and reloads == [["reload"]]


def test_watch_single_spike_does_not_roll_back(conf, tmp_path, monkeypatch):
    log = tmp_path / "access.log"
    log.write_text("")
    canary.set_split(conf, 10)
    monkeypatch.setattr(canary, "reload_nginx", lambda cmd: None)
    batches = iter(
        [[_line("canary", 200, 0.8)] * 10, [_line("canary", 200, 0.01)] * 10] * 2
    )

    def fake_sleep(_):
        with log.open("a") as fh:
            fh.write("\n".join(next(batches)) + "\n")

    monkeypatch.setattr(canary.time, "sleep", fake_sleep)
    assert canary.watch(log, conf, ["reload"], interval_s=0, max_intervals=4) is False
    assert canary.current_split(conf) == 10


def test_cli_set(conf, capsys):
    assert canary.main(["set", "25", "--conf", str(conf)]) == 0
    assert canary.current_split(conf) == 25
