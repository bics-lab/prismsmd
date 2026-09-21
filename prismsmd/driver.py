"""Advance one run as far as it can go, and record what happened.

:func:`advance_run` builds the system if it is not there, writes the MD inputs if they
are not there, and runs the generated ``mdrun.sh``. It is idempotent: what already
exists is not redone, so calling it again resumes rather than restarts. It does not
block until the run finishes -- a wall-clock budget makes ``mdrun.sh`` stop at a
checkpoint -- so a caller with a time limit calls it again in the next allocation.

Enumerating the runs, deciding where they live and looping over them belong to the
caller. This module takes one directory.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import ProbeConfig, ProtocolConfig, TargetConfig
from .events import MAX_CONSECUTIVE_FAILURES, count_consecutive_failures, record_event
from .runstate import RunStatus, run_status
from .seeds import (
    MAX_PLACEMENT_ATTEMPTS,
    placement_attempt,
    record_placement_attempt,
    run_seed,
    seed_for_attempt,
)

__all__ = ["StepResult", "advance_run", "prepare_run"]


@dataclass
class StepResult:
    """What one call to :func:`advance_run` did.

    Attributes:
        workdir: the run directory.
        action: ``done`` / ``built`` / ``prepared`` / ``ran`` / ``stuck`` / ``error``.
        messages: what to tell the user, in order.
        returncode: non-zero when the run could not be advanced.
        status: the run's state afterwards.
    """

    workdir: Path
    action: str
    messages: list[str] = field(default_factory=list)
    returncode: int = 0
    status: RunStatus | None = None

    def to_dict(self) -> dict:
        """Return the result as a plain dict."""
        return {
            "workdir": str(self.workdir),
            "action": self.action,
            "messages": list(self.messages),
            "returncode": self.returncode,
            "status": None if self.status is None else self.status.to_dict(),
        }


def prepare_run(
    workdir: str | Path,
    target: TargetConfig,
    probe: ProbeConfig,
    protocol: ProtocolConfig,
    *,
    seed: int,
    gmx: str | None = None,
    out_traj: str = "production.xtc",
    log=print,
) -> RunStatus:
    """Build the system and write the MD inputs, skipping whatever is already there.

    Args:
        workdir: the run directory, created if needed.
        target: the receptor.
        probe: the probe.
        protocol: the MD protocol.
        seed: the placement seed, from :func:`prismsmd.seeds.seed_for_attempt`.
        gmx: the ``gmx`` command for the solvation step.
        out_traj: name of the trajectory the last stage is linked to.
        log: called with one line per step taken.

    Returns:
        The run's state afterwards.
    """
    from .md.runner import write_mdp_files, write_mdrun_sh, write_postprocess_sh
    from .system.builder import build_system

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    st = run_status(wd, protocol)

    if not st.system_built:
        log(f"[driver] {wd}: building (seed {seed})")
        build_system(target, probe, protocol, wd, seed=seed, gmx=gmx)
        record_event(wd, status="build_done", seed=seed)
        st = run_status(wd, protocol)

    if not st.prepared:
        log(f"[driver] {wd}: writing MD inputs")
        write_mdp_files(protocol, wd, protocol.seed or 0)
        names = list(protocol.step_names)
        write_mdrun_sh(wd / "mdrun.sh", step_names=names, top="input.top",
                       gro="input.gro", out_traj=out_traj)
        write_postprocess_sh(wd / "postprocess.sh", tpr=f"{names[-1]}.tpr",
                             traj_in=out_traj,
                             traj_out=out_traj.replace(".xtc", "_pbcmol.xtc"))
        record_event(wd, status="prepared", steps=names)
        st = run_status(wd, protocol)

    return st


def pbc_trajectory(workdir: str | Path) -> Path:
    """The trajectory ``postprocess.sh`` writes, read out of the script itself.

    The name is decided when the script is written, so reading it back is the only way
    a caller cannot disagree with it.

    Raises:
        FileNotFoundError: when the run has no postprocess.sh.
        ValueError: when the script names no output.
    """
    script = Path(workdir) / "postprocess.sh"
    if not script.is_file():
        raise FileNotFoundError(
            f"{script} does not exist; prepare_run writes it before the MD runs."
        )
    for line in script.read_text().splitlines():
        if line.startswith("traj_out="):
            name = line.split("=", 1)[1].strip()
            if name:
                return Path(workdir) / name
    raise ValueError(f"{script} sets no traj_out; it was not written by prismsmd.")


def postprocess_run(
    workdir: str | Path,
    *,
    gmx: str | None = None,
    log=print,
) -> Path:
    """Run the periodic-boundary handling, and return the trajectory it produced.

    Idempotent: an existing output is returned untouched, so this can be called before
    every analysis without re-doing the work.

    Args:
        workdir: the run directory.
        gmx: the GROMACS command; the script's own default is used when omitted.
        log: called with one line per step taken.

    Returns:
        The path of the periodic-boundary-corrected trajectory.

    Raises:
        RuntimeError: when the script fails, with the tail of its output. The maps
            cannot be trusted without this step, so it is not something to skip past.
    """
    wd = Path(workdir)
    out = pbc_trajectory(wd)
    if out.is_file():
        log(f"[driver] {wd}: {out.name} is already there")
        return out
    env = dict(os.environ)
    if gmx:
        env["GMX"] = gmx
    log(f"[driver] {wd}: sh ./postprocess.sh")
    proc = subprocess.run(["sh", "./postprocess.sh"], cwd=str(wd), env=env,
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out.is_file():
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-8:])
        record_event(wd, status="error", stage="postprocess",
                     returncode=proc.returncode, message=tail[:400])
        raise RuntimeError(
            f"{wd}: postprocess.sh returned {proc.returncode} and"
            f" {'did not write ' + out.name if not out.is_file() else 'failed'}.\n{tail}"
        )
    record_event(wd, status="postprocess_end", trajectory=out.name,
                 bytes=out.stat().st_size)
    return out


def remove_pbc_trajectory(workdir: str | Path, *, log=print) -> bool:
    """Delete the periodic-boundary copy once the analysis has read it.

    It is the same size as the production trajectory and :func:`postprocess_run` makes
    it again from that, so keeping one per run doubles the campaign's disk for nothing.

    Returns:
        True when a file was deleted, False when there was nothing to delete.
    """
    wd = Path(workdir)
    out = pbc_trajectory(wd)
    if not out.is_file():
        return False
    size = out.stat().st_size
    out.unlink()
    record_event(wd, status="pbc_traj_removed", trajectory=out.name, bytes=size)
    log(f"[driver] {wd}: removed {out.name}")
    return True


def advance_run(
    workdir: str | Path,
    target: TargetConfig,
    probe: ProbeConfig,
    protocol: ProtocolConfig,
    *,
    run: int = 0,
    seed: int | None = None,
    ntomp: int | None = None,
    gpu_id: int | str | None = None,
    pin_offset: int | None = None,
    pin_stride: int = 1,
    seconds_remaining: float | None = None,
    tail_sec: int = 600,
    gmx: str | None = None,
    execute: bool = True,
    max_failures: int = MAX_CONSECUTIVE_FAILURES,
    log=print,
) -> StepResult:
    """Take one run as far as the time allows, recording each outcome.

    Args:
        workdir: the run directory.
        target, probe, protocol: what to run.
        run: the run's index within its ensemble, used to derive the seed.
        seed: the base seed; derived with :func:`prismsmd.seeds.run_seed` when omitted.
            The seed actually built with also depends on the placement attempt.
        ntomp: threads for ``mdrun``; left to GROMACS when omitted.
        gpu_id: device for ``mdrun``; left to the environment when omitted.
        pin_offset, pin_stride: thread pinning, computed by the caller since which
            CPUs are available depends on the machine and the allocation.
        seconds_remaining: wall-clock left. ``None`` means no limit, and the run goes
            to the end in one call.
        tail_sec: seconds held back for post-processing.
        gmx: the ``gmx`` command.
        execute: False stops after preparing, without running MD.
        max_failures: consecutive failures after which the run is left alone.
        log: called with one line per step taken.

    Returns:
        A :class:`StepResult`. A run that is already finished returns ``done`` without
        touching anything.
    """
    wd = Path(workdir)
    st = run_status(wd, protocol)
    if st.done:
        return StepResult(wd, "done", [f"{wd}: already finished"], 0, st)

    failures = count_consecutive_failures(wd)
    if failures >= max_failures:
        msg = (f"{wd}: {failures} failures in a row without advancing a stage;"
               " left alone rather than retried. Look at the last events before"
               " rerunning.")
        return StepResult(wd, "stuck", [msg], 1, st)

    base = run_seed(target.key, probe.cid, run) if seed is None else seed
    attempt = placement_attempt(wd)
    if attempt >= MAX_PLACEMENT_ATTEMPTS:
        msg = (f"{wd}: {attempt} placement attempts already made; the target and probe"
               " together may be the problem rather than the placement.")
        return StepResult(wd, "stuck", [msg], 1, st)

    from .system.clash import ProbeClashError

    try:
        st = prepare_run(wd, target, probe, protocol,
                         seed=seed_for_attempt(base, attempt), gmx=gmx, log=log)
    except ProbeClashError as exc:
        # The placement cannot be minimised. Record it and leave the next attempt for
        # the next call, so a caller with a queue moves on instead of looping here.
        record_placement_attempt(wd, attempt + 1, seed=seed_for_attempt(base, attempt),
                                 reason=str(exc)[:400])
        record_event(wd, status="error", stage="build", attempt=attempt,
                     error=type(exc).__name__, message=str(exc)[:400])
        msg = f"{wd}: placement rejected, attempt {attempt + 1} queued"
        log(f"[driver] {msg}")
        return StepResult(wd, "error", [msg, str(exc)], 1, run_status(wd, protocol))

    if not execute:
        return StepResult(wd, "prepared", [f"{wd}: inputs ready"], 0, st)

    env = dict(os.environ)
    env["JOB_START"] = str(int(time.time()))
    if seconds_remaining is not None:
        env["HRT_SEC"] = str(int(seconds_remaining))
        env["TAIL_SEC"] = str(int(tail_sec))
    if gpu_id is not None:
        env["GPU_ID"] = str(gpu_id)
    if pin_offset is not None:
        env["PIN_OFFSET"] = str(pin_offset)
        env["PIN_STRIDE"] = str(pin_stride)
    if gmx:
        env["GMX"] = gmx

    argv = ["sh", "./mdrun.sh"] + ([str(ntomp)] if ntomp else [])
    record_event(wd, status="mdrun_start", argv=argv,
                 finished_steps=list(st.finished_steps),
                 seconds_remaining=seconds_remaining, ntomp=ntomp, gpu_id=gpu_id,
                 pin_offset=pin_offset)
    log(f"[driver] {wd}: {' '.join(argv)}")
    proc = subprocess.run(argv, cwd=str(wd), env=env, check=False)
    after = run_status(wd, protocol)

    if proc.returncode != 0:
        # Record why, not only that: mdrun's own output goes to the job log, but the
        # stage it died in is what decides whether the run is stuck.
        record_event(wd, status="error", stage=after.current_step,
                     returncode=proc.returncode,
                     finished_steps=list(after.finished_steps))
        msg = (f"{wd}: mdrun.sh exited {proc.returncode} at"
               f" {after.current_step or 'an unknown stage'}")
        log(f"[driver] {msg}")
        return StepResult(wd, "error", [msg], proc.returncode, after)

    record_event(wd, status="mdrun_end", finished_steps=list(after.finished_steps),
                 done=after.done)
    msg = (f"{wd}: {len(after.finished_steps)}/{len(protocol.step_names)} stages,"
           f" {after.remaining_ns:.1f} ns left")
    log(f"[driver] {msg}")
    return StepResult(wd, "ran", [msg], 0, after)
