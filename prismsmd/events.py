"""Append-only record of what happened to one run.

Each run directory keeps a line per event in ``run_events.jsonl`` and the newest event
in ``run_state.json``. :func:`count_consecutive_failures` reads the log to decide
whether a run is stuck and should be skipped rather than retried for ever.

An event carries only what this module can know by itself: when it happened, what
happened, and on which machine (:data:`FIXED_FIELDS`). Everything else is passed in by
the caller, which is the side that knows -- a job id and an array index under a batch
system, a return code or a seed elsewhere. Those fields are therefore present on some
lines and absent on others, so read them with ``.get()`` rather than by subscript.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
from pathlib import Path

__all__ = [
    "EVENTS_NAME",
    "FIXED_FIELDS",
    "MAX_CONSECUTIVE_FAILURES",
    "PROGRESS_STATUSES",
    "STATE_NAME",
    "count_consecutive_failures",
    "read_events",
    "record_event",
]

#: Append-only log of every event, one JSON object per line.
EVENTS_NAME = "run_events.jsonl"

#: The newest event, rewritten each time, so the current state can be read at a glance.
STATE_NAME = "run_state.json"

#: The fields every event carries. Anything else a caller passes is optional and may be
#: absent from a given line.
FIXED_FIELDS = ("timestamp", "status", "host", "prismsmd")

#: How many consecutive failures make a run stuck, so that it is skipped instead of
#: picked up again by the next job. A few attempts ride out a transient failure -- a
#: bad node, a filesystem hiccup -- while a deterministic failure stops at once.
MAX_CONSECUTIVE_FAILURES = 3

#: Statuses that mean the run actually got somewhere. Walking backwards through the
#: log, one of these ends a failure streak.
#:
#: Statuses that merely record entering a stage are deliberately absent: every failure
#: is preceded by one, so counting them as progress would pin the streak at 1 for ever.
PROGRESS_STATUSES = frozenset({
    "mdrun_end", "rebuild", "postprocess_end", "pmap_rebuild", "pbc_traj_removed",
    "build_done",
})


def record_event(workdir: str | Path, *, status: str, **fields) -> dict:
    """Append one event to the run's log and rewrite its state file.

    Args:
        workdir: the run directory, created if it does not exist.
        status: what happened, such as ``"mdrun_end"`` or ``"error"``. The values in
            :data:`PROGRESS_STATUSES` end a failure streak.
        **fields: anything else worth keeping, such as the return code, the seed,
            which stages had finished, or the batch job and array index the caller
            knows it is running under.

    Returns:
        The event as it was written, including its timestamp and host.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    from .provenance import stamp

    # Per line, not once per file: a run can be resumed days later under a different
    # build, and the log is the only place that difference is visible.
    event = {
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "host": platform.node(),
        "prismsmd": stamp(),
    }
    event.update(fields)
    with (workdir / EVENTS_NAME).open("a") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    (workdir / STATE_NAME).write_text(
        json.dumps(event, ensure_ascii=False, indent=2) + "\n"
    )
    return event


def read_events(workdir: str | Path) -> list[dict]:
    """Read the run's event log, oldest first.

    Returns an empty list when there is no log, and skips any line that is not JSON, so
    a truncated write does not make the whole history unreadable.
    """
    path = Path(workdir) / EVENTS_NAME
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def count_consecutive_failures(workdir: str | Path) -> int:
    """How many times in a row this run has failed without advancing a stage.

    Counts backwards from the newest event and stops at the first sign of progress, so
    a run that failed, was fixed, and later failed once more reads as 1 rather than
    accumulating for ever. Two things end a streak:

    * a status in :data:`PROGRESS_STATUSES`, which includes ``rebuild``, so discarding
      the inputs resets the count by construction;
    * an ``error`` that had finished more stages than the newest one did, since a run
      dying later than it used to is making progress.
    """
    path = Path(workdir) / EVENTS_NAME
    if not path.is_file():
        return 0
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return 0

    n = 0
    newest_finished: int | None = None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = ev.get("status")
        if status in PROGRESS_STATUSES:
            break
        if status != "error":
            # A record of entering a stage. Does not break the streak.
            continue
        finished = ev.get("finished_steps")
        n_finished = len(finished) if isinstance(finished, list) else None
        if n_finished is not None:
            if newest_finished is None:
                newest_finished = n_finished
            elif n_finished < newest_finished:
                # This failure died earlier than the newest one, so there was progress.
                break
        n += 1
    return n
