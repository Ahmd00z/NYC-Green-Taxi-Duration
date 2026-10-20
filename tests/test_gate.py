"""Model quality gate: the function CI uses to go red on a metric regression."""

import json

import pytest

from prodml.gate import check_regression, main, read_metric


@pytest.mark.parametrize(
    "candidate,baseline,ok",
    [
        (4.0, 4.0, True),  # identical
        (3.5, 4.0, True),  # better
        (4.19, 4.0, True),  # 4.75% worse: inside the 5% allowance
        (4.21, 4.0, False),  # 5.25% worse: blocked
        (9.0, 4.0, False),  # much worse
    ],
)
def test_check_regression(candidate, baseline, ok):
    passed, message = check_regression(candidate, baseline, 0.05)
    assert passed is ok
    assert ("REGRESSION" in message) is (not ok)


def test_no_baseline_passes_so_the_first_model_can_merge():
    passed, message = check_regression(4.0, None, 0.05)
    assert passed and "no baseline" in message


def _write(path, mae):
    path.write_text(json.dumps({"mae": mae, "rmse": mae * 1.4}))
    return str(path)


def test_main_exit_codes_green_and_red(tmp_path, capsys):
    base = _write(tmp_path / "base.json", 4.0)
    good = _write(tmp_path / "good.json", 3.9)
    bad = _write(tmp_path / "bad.json", 5.0)
    assert main(["--candidate", good, "--baseline-file", base]) == 0
    assert main(["--candidate", bad, "--baseline-file", base]) == 1
    assert "REGRESSION" in capsys.readouterr().out


def test_main_without_baseline_is_green(tmp_path):
    assert main(["--candidate", _write(tmp_path / "c.json", 4.0)]) == 0


def test_main_missing_candidate_file_is_red(tmp_path, capsys):
    assert main(["--candidate", str(tmp_path / "nope.json")]) == 1
    assert "not found" in capsys.readouterr().err


def test_main_respects_custom_tolerance(tmp_path):
    base = _write(tmp_path / "b.json", 4.0)
    cand = _write(tmp_path / "c.json", 4.3)  # +7.5%
    assert main(["--candidate", cand, "--baseline-file", base]) == 1
    args = ["--candidate", cand, "--baseline-file", base, "--max-regression", "0.10"]
    assert main(args) == 0


def test_read_metric_missing_file_returns_none(tmp_path):
    assert read_metric(tmp_path / "x.json", "mae") is None
