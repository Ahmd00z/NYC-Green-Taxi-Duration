"""DEMO 14: concept drift (Page-Hinkley) and the retrain gate.

Part 1  Concept drift = the same inputs now lead to a different answer. The model's
        ERROR grows. You need real labels (actual trip durations), which arrive late.
        Page-Hinkley watches the error stream and alarms on a sustained rise.
Part 2  Retrain gate: "drift detected" alone must NOT trigger a retrain. We ask three
        questions (sustained? enough new data? cooled down?) and print the reasons.

Run:  uv run python demos/14_page_hinkley_and_gate.py

Real files: src/prodml/monitoring/concept.py and retrain.py (copied here in short form).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from common import make_trips, split, to_features
from sklearn.linear_model import LinearRegression


class PageHinkley:
    def __init__(self, delta=0.5, threshold=50.0, min_samples=30):
        self.delta, self.threshold, self.min_samples = delta, threshold, min_samples
        self.n, self.mean, self.total, self.lowest = 0, 0.0, 0.0, 0.0

    def update(self, value: float) -> bool:
        self.n += 1
        self.mean += (value - self.mean) / self.n  # running mean of the error
        self.total += value - self.mean - self.delta  # how far above "normal"
        self.lowest = min(self.lowest, self.total)
        return self.n >= self.min_samples and self.total - self.lowest > self.threshold


def part1() -> None:
    train, _ = split(make_trips(10_000, seed=0))
    model = LinearRegression().fit(to_features(train), train["duration"])

    # A live stream of 1500 trips. At trip 800 the world changes: traffic gets worse,
    # every trip takes 3 minutes + 25% longer, but the inputs look exactly the same.
    live = make_trips(1500, seed=5)
    actual = live["duration"].to_numpy().copy()
    actual[800:] = actual[800:] * 1.25 + 3
    predicted = model.predict(to_features(live))
    errors = np.abs(actual - predicted)

    detector = PageHinkley()
    alarm = next((i for i, e in enumerate(errors) if detector.update(e)), None)
    print(f"MAE before the change: {errors[:800].mean():.2f} min")
    print(f"MAE after  the change: {errors[800:].mean():.2f} min")
    print(f"Page-Hinkley alarm at trip {alarm} (the change happened at 800)")
    assert alarm is not None and 800 <= alarm < 1100
    print(
        "Note: input drift tests (PSI) would see NOTHING here: the inputs did not move."
    )


def decide(daily, new_rows, last_retrain, now):
    reasons, ok = [], True
    if len(daily) >= 2 and all(daily[-2:]):
        reasons.append("drift on the last 2 days: yes")
    else:
        ok = False
        reasons.append("not 2 drifted days in a row: not sustained")
    if new_rows >= 1000:
        reasons.append(f"{new_rows} new rows: enough")
    else:
        ok = False
        reasons.append(f"only {new_rows} new rows: not enough")
    if last_retrain is None or now - last_retrain >= timedelta(days=7):
        reasons.append("last retrain over 7 days ago: ok")
    else:
        ok = False
        reasons.append("retrained under 7 days ago: cooling down")
    return ok, reasons


def part2() -> None:
    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    cases = {
        "one noisy day": ([False, True], 5000, None),
        "real drift, 2 days": ([True, True], 5000, None),
        "drift, but retrained yesterday": ([True, True], 5000, now - timedelta(days=1)),
        "drift, but almost no new data": ([True, True], 40, None),
    }
    for name, (daily, rows, last) in cases.items():
        ok, reasons = decide(daily, rows, last, now)
        print(f"\n{name}: {'RETRAIN' if ok else 'HOLD'}")
        for reason in reasons:
            print("   -", reason)


if __name__ == "__main__":
    print("== Part 1: concept drift ==")
    part1()
    print("\n== Part 2: the retrain gate ==")
    part2()
