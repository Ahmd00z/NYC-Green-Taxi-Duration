"""Concept drift: the same inputs now lead to a different answer, so the ERROR grows.

You can only see this with real labels (here: the actual trip duration, known after
drop-off, so it arrives LATE). The detector below is Page-Hinkley, a classic streaming
test, in ~25 lines:

  keep the running mean of the error; add up how far each new value is above that mean
  (minus a small tolerance `delta`); remember the smallest that sum ever was;
  alarm when the sum climbs `threshold` above its smallest value.

A slow, steady rise in error triggers it; one noisy value does not.

Tuning matters. Measured on 200 simulated streams (noise sd 1.6 min, error +25% +3 min
after the change): delta=0.05, threshold=20 raised a FALSE alarm on 47% of quiet
streams; delta=0.5, threshold=50 on 0%, and still caught the real change in ~16
trips. Rule of thumb: delta about half the rise you care about, threshold about
50-100x delta, then replay your own error history to check the false-alarm rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PageHinkley:
    delta: float = 0.5  # tolerated wobble (minutes): ~half the rise you care about
    threshold: float = 50.0  # bigger = fewer false alarms, slower detection
    min_samples: int = 30  # do not alarm in the first few values

    def __post_init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.total = 0.0
        self.lowest = 0.0

    def update(self, value: float) -> bool:
        """Feed one error value. True means: the error level has shifted up."""
        self.n += 1
        self.mean += (value - self.mean) / self.n
        self.total += value - self.mean - self.delta
        self.lowest = min(self.lowest, self.total)
        return self.n >= self.min_samples and self.total - self.lowest > self.threshold


def first_alarm(errors: np.ndarray, **kwargs) -> int | None:
    """Index of the first alarm in a stream of errors, or None."""
    detector = PageHinkley(**kwargs)
    for i, value in enumerate(errors):
        if detector.update(float(value)):
            return i
    return None
