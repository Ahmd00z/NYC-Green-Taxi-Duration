"""Should we retrain? Three gates, each a pure function, each explainable.

The naive rule  `if drift: retrain()`  is dangerous: a broken upstream feed looks like
permanent drift and you retrain on garbage every day. So we ask three questions and
require ALL to be yes. Every decision lists the reasons, including for a "no".

  1. sustained  drift on `sustained_days` days IN A ROW (one noisy day is not drift)
  2. enough     at least `min_new_rows` new labelled rows to train on
  3. cooldown   at least `cooldown_days` since the last retrain (no retrain storms)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class Decision:
    retrain: bool
    reasons: list[str] = field(default_factory=list)

    def explain(self) -> str:
        head = "RETRAIN" if self.retrain else "HOLD"
        return head + "\n" + "\n".join(f"  - {r}" for r in self.reasons)


def is_sustained(daily: list[tuple[str, bool]], days: int = 2) -> bool:
    """The last `days` days are all drifted."""
    return len(daily) >= days and all(flag for _, flag in daily[-days:])


def cooldown_ok(
    last_retrain: datetime | None, now: datetime, cooldown_days: int = 7
) -> bool:
    return last_retrain is None or now - last_retrain >= timedelta(days=cooldown_days)


def decide(
    daily: list[tuple[str, bool]],
    new_rows: int,
    last_retrain: datetime | None,
    now: datetime,
    sustained_days: int = 2,
    min_new_rows: int = 1000,
    cooldown_days: int = 7,
) -> Decision:
    reasons: list[str] = []
    ok = True

    if is_sustained(daily, sustained_days):
        reasons.append(f"drift on the last {sustained_days} days: yes")
    else:
        ok = False
        reasons.append(f"no drift for {sustained_days} days in a row: not sustained")

    if new_rows >= min_new_rows:
        reasons.append(f"{new_rows} new rows (need {min_new_rows}): enough")
    else:
        ok = False
        reasons.append(f"only {new_rows} new rows (need {min_new_rows}): not enough")

    if cooldown_ok(last_retrain, now, cooldown_days):
        reasons.append(f"last retrain more than {cooldown_days} days ago: ok")
    else:
        ok = False
        reasons.append(f"retrained less than {cooldown_days} days ago: cooling down")

    return Decision(ok, reasons)
