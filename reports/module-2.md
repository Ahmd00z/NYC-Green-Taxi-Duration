# Module 2 report: Tracking, registry, versioning and automation

> **Numbers caveat.** Every metric below that comes from the sandbox was produced on
> *synthetic* data (`scripts/make_synthetic_trips.py`), only to prove the plumbing works.
> Replace the comparison table with the results of YOUR August-2019 run before opening the PR.

## 1. MLflow experiment comparison

`python -m prodml.experiments` logs 15 runs on the same chronological split:

| Run | Family | What it is |
|---|---|---|
| linear | Linear | DictVectorizer + LinearRegression (the Module 1 baseline) |
| xgboost | XGBoost | native categorical pickup / dropoff zones + trip_distance |
| random-forest | Random forest | scikit-learn bagged trees on pickup zone, dropoff zone, trip_distance |
| sweep-001..010 | XGBoost | Optuna TPE trials, each a nested run under one parent |
| autolog | Linear | `mlflow.sklearn.autolog()` demo, to compare with the manual logging |

| Run | MAE (min) | RMSE | R2 | Train s | Size MB |
|---|---:|---:|---:|---:|---:|
| _fill from the MLflow UI: Compare runs_ | | | | | |

**Pick:** choose the lowest validation MAE, and mention the cost (training time, model size)
next to it. A model that is 1% better but 20x slower is a real trade-off, not an automatic win.

**Autolog vs manual.** Autolog captured the estimator's hyperparameters, a training score and
the fitted model with a signature without any code. It did NOT capture the git commit, the
data version, the author, my own validation metrics (it only knows the training data), the
residual plot, or `requirements.txt` pinned to the serving environment. That is why the
project logs manually (`tracking.log_family_run`) and keeps autolog only as a demo.

## 2. Reproducibility and lineage

```text
registry version  ->  MLflow run  ->  git_commit tag  ->  the exact code
                                  ->  data_version tag ->  md5 in data/*.dvc -> exact bytes in the DVC remote
                                  ->  params / metrics / artifacts / requirements.txt
```

Every run carries `git_commit` (with `-dirty` when the tree had uncommitted changes),
`data_version`, `author` and `framework`. Given a model in Production, one can check out that
commit, `dvc pull` that data version, `dvc repro`, and get the same MAE.

## 3. Registry and promotion policy

`None -> Staging -> Production`. The API loads `models:/ride-duration-predictor/Production`,
never a path, so changing the served model is a stage change, not a code change or a rebuild.

`promote_if_better` registers a candidate and moves it to **Staging** only if its MAE beats the
current Production MAE by the configured margin. A losing candidate is not even registered and
the workflow still ends green (a failed challenge is not a failed build).

**Why Production is never promoted automatically.** A metric on a validation split cannot see
a broken upstream feed, a business rule change or a bias a stakeholder cares about, so a human
must own the last step. Staging is cheap and reversible, Production serves real users, so the
asymmetry justifies one approval click (GitHub Environment `production` with required reviewers).

## 4. Continuous training: when should we retrain?

| Trigger | Implemented now? | Notes |
|---|---|---|
| Schedule (weekly cron) | Yes | `continuous-training.yml`, Monday 03:00 UTC |
| New labelled data | Partly | `repository_dispatch: new-data` / manual `workflow_dispatch` with `data_version` |
| Data drift | Hook only | `repository_dispatch: drift-detected`; the detector itself is Module 5 |
| Performance degradation | Hook only | `repository_dispatch: performance-degraded`; needs ground truth monitoring, Module 5 |

Flow: `dvc pull` -> validate data -> `dvc repro` -> `promote-if-better` -> (if promoted) build
image `ct-<run>` -> human approval -> Production.

## 5. CI gates

`ci.yml`: lint (ruff + black) -> tests with a 70% coverage gate -> **model gate** -> docker build.
The gate compares the committed `metrics/scores.json` with the one on `main` and exits 1 if
MAE regressed by more than 5%. It compares committed files because GitHub runners cannot reach
a local MLflow server. A passing first run has no baseline and passes.

## 6. Terraform state

`terraform.tfstate` records every attribute of every resource in plain text, passwords included,
so it must never be committed (it is in `.gitignore`). A remote backend (a cloud bucket with locking,
Terraform Cloud) stores it once in a shared, access-controlled and locked place.

## 7. Not verified by the author of this code

Docker build/run of the compose stack and the API image, `terraform validate/plan/apply`,
real GitHub Actions runs, and results on the real dataset. Run each and
paste the output in the PR.

## 8. PR checklist (Definition of Done)

- [ ] `docker compose up -d mlflow` brings the tracking server up; UI reachable on :5000
- [ ] 15 runs visible, each with params, metrics, plots, `requirements.txt` and tags
- [ ] Comparison table above filled with real numbers and a justified pick
- [ ] Model registered; Staging -> Production done; API serves the Production model
- [ ] Promoting another version changes predictions with no code change
- [ ] `dvc repro` second run skips every stage; `dvc metrics show` matches MLflow
- [ ] CI green; a deliberately bad model turns the gate red (screenshot)
- [ ] CT workflow ran at least once (manual dispatch); Production needs approval
- [ ] `terraform plan` pasted; destroy then apply rebuilds the stack
- [ ] No secret in git: only GitHub Secrets (Docker Hub), `.env` and `terraform.tfvars` ignored
