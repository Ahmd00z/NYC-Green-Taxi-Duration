# Module 4 report: monitoring and drift

All numbers come from a 2-core cloud sandbox with SYNTHETIC taxi data, so they show how
to read the tools, not how your real model behaves. Re-run the commands on your machine
and replace the tables.

## 1. Drift taxonomy

| Drift | What changes | Needs labels? | Tool here | First signal |
|---|---|---|---|---|
| Data (input) | distribution of distance, zones | no | PSI + KS (`drift.py`) | minutes to hours |
| Prediction (output) | distribution of predicted minutes | no | PSI on `prediction`, `prodml_prediction_minutes` | hours |
| Concept | same inputs, different true duration | **yes**, late | Page-Hinkley on the error (`concept.py`) | days |

Labels (the actual duration) arrive only after the trip ends, so the first two are what
we watch while we wait for the third.

## 2. What we measure continuously (metric contract)

| Metric | Type | Why |
|---|---|---|
| `prodml_http_requests_total{route,status}` | Counter | rate, error ratio |
| `prodml_http_request_duration_seconds{route}` | Histogram | p50/p95/p99 (an average hides the tail) |
| `prodml_inflight_requests` | Gauge | saturation |
| `prodml_predictions_total{model_version}` | Counter | traffic per model version |
| `prodml_prediction_minutes{model_version}` | Histogram | output distribution, no labels needed |
| `prodml_input_trip_distance_miles` | Histogram | cheap input-drift proxy |
| `prodml_model_info{version}` | Gauge | join any metric to the model version |
| `prodml_drift_psi{feature}` | Gauge | latest offline drift run, read from SQLite on scrape |

Label rule: only small, fixed sets (route template, status, model version). Zone ids, trip
ids and raw URLs are never labels (one time series per value kills Prometheus); that detail
lives in `data/serving_log.jsonl`. A test (`test_metric_labels_stay_bounded`) enforces it.

## 3. Drift detection, measured

Reference: 5,000 training rows. Current: 1,500 simulated trips. `--shift 1.5` makes trips
50% longer and sends 40% of pickups to one new zone.

| Feature | PSI (no shift) | PSI (shift 1.5) | KS p (shift 1.5) | Status after shift |
|---|---|---|---|---|
| trip_distance | 0.011 | 0.452 | 1e-78 | **drift** |
| prediction | 0.002 | 0.236 | 1e-41 | warn |
| PULocationID | 0.024 | 0.216 | n/a | warn |
| DOLocationID | 0.051 | 0.051 | n/a | ok |

Why PSI decides and KS only reports: with thousands of rows KS gives a tiny p-value for any
difference (demo 13: 1e-18 for a 20% shift whose PSI is only 0.075, "ok"). PSI measures the
size of the move, which is what an alert should threshold.

### Page-Hinkley tuning (a mistake worth keeping in the report)

The first defaults (`delta=0.05, threshold=20`) looked fine on a clean test stream but raised
a **false alarm on 47%** of 200 quiet, noisy streams. Measured sweep (noise sd 1.6 min,
error +25% +3 min after trip 800):

| delta | threshold | false alarms | median delay (trips) |
|---|---|---|---|
| 0.05 | 20 | 47% | n/a |
| 0.3 | 50 | 0% | 16 |
| 0.5 | 50 (**default now**) | 0% | 18 |
| 0.5 | 150 | 0% | 55 |

A unit test now runs 60 quiet streams and fails if more than 2 alarm.
Page-Hinkley saw the change in `demos/14` at trip 812 (it happened at 800), and PSI would
have seen nothing there because the inputs did not move.

## 4. Retrain gate

`drift -> retrain` directly is a trap: a broken feed looks like permanent drift. The gate
needs ALL of: drift on 2 days in a row, at least 1,000 new rows, no retrain in 7 days.
Every decision (including HOLD) is stored with its reasons in SQLite. It only decides; start
the training pipeline from it (for example trigger the Airflow DAG from Module 3).

## 5. Alerts

| Alert | Fires when | Severity |
|---|---|---|
| ApiDown | scrape fails 30 s | page |
| HighErrorRate | 5xx > 5% for 2 min (422 not counted) | page |
| HighLatencyP95 | /predict p95 > 500 ms for 5 min | page |
| NoTraffic | no predictions for 10 min | ticket |
| FeatureDrift | PSI > 0.25 for 1 min | ticket |
| DriftJobStale | no drift run for 2 days | ticket |

Each has `for:` (one blip never alerts), a severity and a runbook link (`docs/RUNBOOK.md`).

## 6. What was tested

| Item | How | Result |
|---|---|---|
| Drift maths, Page-Hinkley, store, gate, `/metrics`, serving log, CLI | 28 new pytest tests; full suite 151 passed, 88% coverage | passed |
| Alert rules | `promtool check rules` (real Prometheus 2.55.1 binary) | 6 rules valid |
| Scraping + alerts end to end | real Prometheus scraping the real API: target UP, `FeatureDrift` went to firing after `monitoring drift` on drifted traffic; `ApiDown` fired after the API was killed | worked |
| `docker-compose.monitoring.yml`, Grafana dashboard | YAML/JSON parse only | **not run** (no Docker daemon here) |
| Drift with Evidently, Loki logs, Alertmanager routing, Airflow DAG | not built (kept out to stay small) | n/a |

## 7. Deliberately left out

Evidently (PSI + KS cover the idea with no pinned dependency), Loki/Promtail (the serving
log plus the JSON application log is enough here), Alertmanager (rules fire in Prometheus;
add routing when there is someone to page), a drift Airflow DAG (schedule
`python -m prodml.monitoring drift` with cron, or copy the pattern of `batch_scoring.py`).