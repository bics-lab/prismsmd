"""How far one run has got, read from the files it leaves behind.

Completion of a stage is judged by its ``.gro``, which GROMACS writes only on reaching
``nsteps``. A stage cut short by the wall-clock limit leaves a ``.cpt`` and no ``.gro``,
and is resumed rather than restarted, so the same rule serves both the resume decision
and the progress report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import ProtocolConfig

__all__ = [
    "SYSTEM_FILES",
    "RunStatus",
    "last_step_in_log",
    "run_status",
]

#: The files a built system consists of. All three must be present before MD can start.
SYSTEM_FILES = ("input.top", "input.gro", "index.ndx")

#: Marks the step table in a GROMACS ``.log``; the step number is on the next line.
_STEP_HEADER = re.compile(r"^\s*Step\s+Time\s*$")


def last_step_in_log(path: str | Path, *, tail_bytes: int = 1 << 16) -> int | None:
    """The last step reached, read from a GROMACS ``.log``.

    Returns ``None`` when the log is missing or carries no step table, rather than
    guessing. Only the tail is read, because a long run's log is large.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > tail_bytes:
                fh.seek(size - tail_bytes)
                fh.readline()          # discard a line cut mid-way
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    last: int | None = None
    for i, line in enumerate(lines[:-1]):
        if _STEP_HEADER.match(line):
            fields = lines[i + 1].split()
            if fields:
                try:
                    last = int(fields[0])
                except ValueError:
                    continue
    return last


@dataclass
class RunStatus:
    """What one run directory currently holds.

    Attributes:
        workdir: the run directory.
        exists: whether it is there at all.
        system_built: whether every file in :data:`SYSTEM_FILES` is present.
        prepared: whether the run script and every step's mdp are present.
        finished_steps: the leading run of stages that have a ``.gro``.
        current_step: the stage to run next, or ``None`` when finished.
        current_step_ns: simulated time already covered in that stage.
        done: whether every stage has finished.
        remaining_ns: simulated time still to run.
        last_event: the newest entry of the run's event log, if any.
    """

    workdir: Path
    exists: bool
    system_built: bool
    prepared: bool
    finished_steps: tuple[str, ...] = ()
    current_step: str | None = None
    current_step_ns: float = 0.0
    done: bool = False
    remaining_ns: float = 0.0
    last_event: dict | None = None

    @property
    def failed(self) -> bool:
        """Whether the newest event records an error and the run is unfinished."""
        return bool(
            self.last_event and self.last_event.get("status") == "error"
        ) and not self.done

    @property
    def state(self) -> str:
        """One word for the run: done / failed / not_started / running."""
        if self.done:
            return "done"
        if self.failed:
            return "failed"
        if not self.exists or not self.system_built:
            return "not_started"
        return "running"

    def to_dict(self) -> dict:
        """Return the status as a plain dict."""
        return {
            "workdir": str(self.workdir),
            "state": self.state,
            "system_built": self.system_built,
            "prepared": self.prepared,
            "finished_steps": list(self.finished_steps),
            "n_finished_steps": len(self.finished_steps),
            "current_step": self.current_step,
            "current_step_ns": round(self.current_step_ns, 3),
            "done": self.done,
            "failed": self.failed,
            "remaining_ns": round(self.remaining_ns, 3),
            "last_event": self.last_event,
        }


def run_status(workdir: str | Path, protocol: ProtocolConfig) -> RunStatus:
    """Inspect one run directory against the protocol it should follow.

    Stages are counted from the start and stop at the first one without a ``.gro``, so
    a stray ``.gro`` from a later stage is not mistaken for progress.
    """
    wd = Path(workdir)
    exists = wd.is_dir()
    names = protocol.step_names
    system_built = exists and all((wd / f).is_file() for f in SYSTEM_FILES)
    prepared = exists and (wd / "mdrun.sh").is_file() and all(
        (wd / f"{n}.mdp").is_file() for n in names
    )

    finished: list[str] = []
    current: str | None = None
    for name in names:
        if exists and (wd / f"{name}.gro").is_file():
            finished.append(name)
            continue
        current = name
        break

    done = len(finished) == len(names)

    remaining = sum(protocol.step_ns(n) for n in names if n not in set(finished))
    current_ns = 0.0
    if current is not None and exists:
        steps_done = last_step_in_log(wd / f"{current}.log")
        if steps_done:
            current_ns = min(
                steps_done * protocol.dt / 1000.0, protocol.step_ns(current)
            )
            remaining = max(remaining - current_ns, 0.0)

    from .events import read_events

    events = read_events(wd)

    return RunStatus(
        workdir=wd,
        exists=exists,
        system_built=system_built,
        prepared=prepared,
        finished_steps=tuple(finished),
        current_step=None if done else current,
        current_step_ns=current_ns,
        done=done,
        remaining_ns=remaining,
        last_event=events[-1] if events else None,
    )
