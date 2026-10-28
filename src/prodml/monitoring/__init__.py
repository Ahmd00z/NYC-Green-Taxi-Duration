"""Module 4: know when the model is no longer right.

metrics.py     live numbers for Prometheus (/metrics)
serving_log.py what the API was asked, one JSON line per trip
drift.py       PSI + KS: has the input (or the output) moved away from training?
concept.py     Page-Hinkley: has the ERROR started to grow? (needs real labels)
store.py       SQLite history of drift runs and retrain decisions
retrain.py     "should we retrain?" as pure, explainable gates
cli.py         python -m prodml.monitoring <freeze|simulate|drift|concept|gate|history>
"""
