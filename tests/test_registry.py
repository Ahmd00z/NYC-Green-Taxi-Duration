"""Registry: None -> Staging -> Production, and the promotion rules."""

import json

import pytest
from mlflow.tracking import MlflowClient

from prodml import registry
from prodml.experiments import load_splits
from prodml.models import make_family
from prodml.tracking import log_family_run


@pytest.fixture
def runs(tracking_uri, synthetic_csv):
    """Two real runs on identical splits: (better, worse), ordered by validation MAE."""
    train, val = load_splits(synthetic_csv, 0.8)
    made = [
        log_family_run(
            make_family("linear"), train, val, run_name="a", data_path=synthetic_csv
        ),
        log_family_run(
            make_family("xgboost", {"n_estimators": 2, "max_depth": 1}),
            train,
            val,
            run_name="b",
            data_path=synthetic_csv,
        ),
    ]
    made.sort(key=lambda r: r.metrics["mae"])
    assert made[0].metrics["mae"] < made[1].metrics["mae"]
    return made[0], made[1]


def stage_of(version: int) -> str:
    return (
        MlflowClient()
        .get_model_version("ride-duration-predictor", str(version))
        .current_stage
    )


def test_first_candidate_wins_by_default_and_lands_in_staging(runs):
    better, _ = runs
    res = registry.promote_if_better(better.run_id)
    assert res.promoted and res.stage == "Staging" and res.baseline_value is None
    assert stage_of(res.version) == "Staging"


def test_a_worse_candidate_is_rejected_and_never_registered(runs):
    better, worse = runs
    registry.promote_to_production(registry.promote_if_better(better.run_id).version)
    res = registry.promote_if_better(worse.run_id)
    assert not res.promoted and "does not beat" in res.reason
    assert res.candidate_value > res.baseline_value
    assert len(registry.versions()) == 1  # a failed challenge leaves no trace


def test_a_better_candidate_beats_production(runs):
    better, worse = runs
    v_worse = registry.register_run(worse.run_id)
    registry.promote_to_production(int(v_worse.version))
    res = registry.promote_if_better(better.run_id)
    assert res.promoted and res.baseline_version == int(v_worse.version)
    assert stage_of(res.version) == "Staging"
    assert stage_of(int(v_worse.version)) == "Production"  # Production untouched


def test_margin_blocks_a_marginal_win(runs):
    better, worse = runs
    registry.promote_to_production(int(registry.register_run(worse.run_id).version))
    res = registry.promote_if_better(better.run_id, margin=0.99)  # must be 99% better
    assert not res.promoted


def test_production_promotion_archives_the_previous_production(runs):
    better, worse = runs
    v1 = int(registry.register_run(worse.run_id).version)
    v2 = int(registry.register_run(better.run_id).version)
    registry.promote_to_production(v1)
    registry.promote_to_production(v2)
    assert stage_of(v1) == "Archived" and stage_of(v2) == "Production"
    assert int(registry.version_in_stage("Production").version) == v2
    assert registry.production_metric("mae") == pytest.approx(better.metrics["mae"])


def test_registering_the_same_run_twice_reuses_the_version(runs):
    better, _ = runs
    first = registry.register_run(better.run_id)
    second = registry.register_run(better.run_id)
    assert first.version == second.version and len(registry.versions()) == 1


def test_unknown_stage_is_rejected(runs):
    registry.register_run(runs[0].run_id)
    with pytest.raises(ValueError, match="stage must be one of"):
        registry.set_stage(1, "Live")


def test_no_production_model_means_no_production_metric(tracking_uri):
    assert registry.version_in_stage("Production") is None
    assert registry.production_metric() is None


def test_cli_promote_show_and_github_output(runs, monkeypatch, tmp_path, capsys):
    better, _ = runs
    out_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))
    assert registry.main(["promote-if-better", "--run-id", better.run_id]) == 0
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["promoted"] is True
    assert "promoted=true" in out_file.read_text()
    assert "version=1" in out_file.read_text()

    assert registry.main(["show"]) == 0
    assert "Staging" in capsys.readouterr().out
    assert registry.main(["set-stage", "--version", "1", "--stage", "Production"]) == 0
    assert stage_of(1) == "Production"
    assert registry.main(["register", "--run-id", better.run_id]) == 0
    assert capsys.readouterr().out.strip() == "1"
    assert registry.main(["promote-production", "--version", "1"]) == 0


def test_cli_failed_challenge_exits_zero(runs, capsys):
    """A failed challenge is not a failed build: exit code 0, promoted=false."""
    better, worse = runs
    registry.promote_to_production(registry.promote_if_better(better.run_id).version)
    assert registry.main(["promote-if-better", "--run-id", worse.run_id]) == 0
    assert (
        json.loads(capsys.readouterr().out.strip().splitlines()[-1])["promoted"]
        is False
    )
