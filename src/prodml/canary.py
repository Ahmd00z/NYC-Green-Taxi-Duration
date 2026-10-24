"""Mini Project 3, step 7: canary release helpers (traffic split + auto-rollback).

Three jobs, all plain Python so they can be unit-tested without nginx:

  1. set_split()   rewrite the "how much traffic goes to the canary" block in nginx.conf
  2. judge()       decide from the access log whether the canary is healthy
  3. watch()       loop: read new log lines every few seconds, roll back after
                   `bad_intervals_needed` bad intervals in a row

The nginx config marks the editable block with two comment markers:

    # CANARY_SPLIT_START
    split_clients "${request_id}" $deployment { 90% stable; * canary; }
    # CANARY_SPLIT_END

and writes one log line per request:

    ... deployment=canary status=200 upstream_time=0.012
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

START, END = "# CANARY_SPLIT_START", "# CANARY_SPLIT_END"
_LINE = re.compile(
    r"deployment=(?P<dep>\w+) status=(?P<status>\d+) upstream_time=(?P<t>[\d.]+|-)"
)


# ------------------------------------------------------------------ 1. the split
def split_block(canary_pct: int) -> str:
    """The nginx text for a given canary share (0-100)."""
    if not 0 <= canary_pct <= 100:
        raise ValueError("canary_pct must be between 0 and 100")
    if canary_pct == 0:
        body = "    * stable;"
    elif canary_pct == 100:
        body = "    * canary;"
    else:
        body = f"    {100 - canary_pct}% stable;\n    * canary;"
    return (
        f"{START}\n"
        f'  split_clients "${{request_id}}" $deployment {{\n{body}\n  }}\n'
        f"  {END}"
    )


def set_split(conf_path: str | Path, canary_pct: int) -> str:
    """Rewrite the marked block in the file and return the new text."""
    path = Path(conf_path)
    text = path.read_text()
    if START not in text or END not in text:
        raise ValueError(f"markers {START} / {END} not found in {path}")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    new = head + split_block(canary_pct) + tail
    path.write_text(new)
    return new


def current_split(conf_path: str | Path) -> int:
    """Read back the canary percentage from the file."""
    text = Path(conf_path).read_text()
    m = re.search(r"(\d+)%\s+stable", text)
    if m:
        return 100 - int(m.group(1))
    return 100 if re.search(r"\*\s+canary;", text) else 0


# ------------------------------------------------------------------ 2. the verdict
@dataclass
class Verdict:
    requests: int
    p95_ms: float
    error_rate: float
    bad: bool
    reason: str


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def judge(
    lines: list[str],
    p95_limit_ms: float = 300.0,
    error_limit: float = 0.05,
    min_requests: int = 5,
) -> Verdict:
    """Look only at CANARY requests. Too few requests = no opinion (not bad)."""
    times: list[float] = []
    errors = total = 0
    for line in lines:
        m = _LINE.search(line)
        if not m or m["dep"] != "canary":
            continue
        total += 1
        if int(m["status"]) >= 500:
            errors += 1
        if m["t"] != "-":
            times.append(float(m["t"]) * 1000)
    if total < min_requests:
        return Verdict(total, 0.0, 0.0, False, "not enough traffic")
    p95 = percentile(times, 0.95)
    rate = errors / total
    if p95 > p95_limit_ms:
        return Verdict(total, p95, rate, True, f"p95 {p95:.0f} ms > {p95_limit_ms:.0f}")
    if rate > error_limit:
        return Verdict(total, p95, rate, True, f"errors {rate:.1%} > {error_limit:.0%}")
    return Verdict(total, p95, rate, False, "ok")


# ------------------------------------------------------------------ 3. the watcher
def reload_nginx(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def watch(
    log_path: str | Path,
    conf_path: str | Path,
    reload_cmd: list[str],
    interval_s: float = 10.0,
    bad_intervals_needed: int = 2,
    max_intervals: int | None = None,
    **limits: float,
) -> bool:
    """Poll the log. Returns True if it rolled back, False if it stopped without.

    Two bad intervals IN A ROW are required, so one slow spike does not kill a good
    release. The rollback sets the canary share to 0% and reloads nginx.
    """
    log = Path(log_path)
    offset = log.stat().st_size if log.exists() else 0
    bad_in_row = 0
    n = 0
    while max_intervals is None or n < max_intervals:
        time.sleep(interval_s)
        n += 1
        with log.open() as fh:
            fh.seek(offset)
            new = fh.readlines()
            offset = fh.tell()
        v = judge(new, **limits)
        print(f"[watcher] {v.requests} canary req, p95={v.p95_ms:.0f} ms, {v.reason}")
        bad_in_row = bad_in_row + 1 if v.bad else 0
        if bad_in_row >= bad_intervals_needed:
            started = time.perf_counter()
            set_split(conf_path, 0)
            reload_nginx(reload_cmd)
            print(f"[watcher] ROLLBACK done in {time.perf_counter() - started:.2f} s")
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="canary helpers")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("set", help="set canary share, e.g. set 10")
    s.add_argument("pct", type=int)
    s.add_argument("--conf", default="deploy/nginx/nginx.conf")
    w = sub.add_parser("watch", help="auto-rollback watcher")
    w.add_argument("--log", default="deploy/nginx/logs/access.log")
    w.add_argument("--conf", default="deploy/nginx/nginx.conf")
    w.add_argument("--interval", type=float, default=10.0)
    w.add_argument("--p95-ms", type=float, default=300.0)
    w.add_argument("--error-rate", type=float, default=0.05)
    w.add_argument(
        "--reload-cmd",
        default="docker compose -f docker/docker-compose.canary.yml "
        "exec -T nginx nginx -s reload",
    )
    a = p.parse_args(argv)
    if a.cmd == "set":
        set_split(a.conf, a.pct)
        print(f"canary share is now {a.pct}%")
        return 0
    rolled = watch(
        a.log,
        a.conf,
        a.reload_cmd.split(),
        interval_s=a.interval,
        p95_limit_ms=a.p95_ms,
        error_limit=a.error_rate,
    )
    return 1 if rolled else 0


if __name__ == "__main__":
    sys.exit(main())
