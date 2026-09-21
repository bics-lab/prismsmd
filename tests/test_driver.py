"""Invariants of advancing one run, without building or running anything real."""

import pytest

from prismsmd.config import MDStep, ProtocolConfig, TargetConfig
from prismsmd.driver import advance_run
from prismsmd.events import read_events, record_event
from prismsmd.runstate import SYSTEM_FILES
from prismsmd.seeds import PLACEMENT_RETRY_STRIDE, placement_attempt, run_seed

PROTOCOL = ProtocolConfig(
    dt=0.002,
    seed=42,
    steps=(
        MDStep(name="min1", type="minimization", nsteps=200),
        MDStep(name="pr", type="production", nsteps=500_000),
    ),
)
TARGET = TargetConfig(key="t1", name="T", pdb="/nonexistent.pdb")


class FakeProbe:
    """Enough of a ProbeConfig for the driver; nothing is actually built."""

    cid = "P1"
    residue_name = "P1"
    mol_file = None          # a build would stop here, which is what the tests want


def _ready(wd, *, prepared=True):
    """A directory that looks built and prepared, so no real work is attempted."""
    wd.mkdir(parents=True, exist_ok=True)
    for name in SYSTEM_FILES:
        (wd / name).write_text("")
    if prepared:
        (wd / "mdrun.sh").write_text("#!/bin/sh\nexit 0\n")
        for step in PROTOCOL.step_names:
            (wd / f"{step}.mdp").write_text("")
    return wd


def _finish(wd, *steps):
    for s in steps:
        (wd / f"{s}.gro").write_text("")


def test_a_finished_run_is_left_alone(tmp_path):
    wd = _ready(tmp_path / "run00")
    _finish(wd, *PROTOCOL.step_names)
    res = advance_run(wd, TARGET, FakeProbe(), PROTOCOL, log=lambda m: None)
    assert res.action == "done"
    assert res.returncode == 0
    assert read_events(wd) == []          # nothing was touched, so nothing was recorded


def test_a_stuck_run_is_not_retried_for_ever(tmp_path):
    """A run that dies on entry would otherwise be picked up by every later call."""
    wd = _ready(tmp_path / "run00")
    for _ in range(3):
        record_event(wd, status="error", finished_steps=[])
    res = advance_run(wd, TARGET, FakeProbe(), PROTOCOL, log=lambda m: None)
    assert res.action == "stuck"
    assert res.returncode == 1
    assert "failures in a row" in res.messages[0]


def test_progress_clears_the_way_for_another_attempt(tmp_path):
    wd = _ready(tmp_path / "run00")
    for _ in range(3):
        record_event(wd, status="error", finished_steps=[])
    record_event(wd, status="mdrun_end")          # something worked since
    res = advance_run(wd, TARGET, FakeProbe(), PROTOCOL, execute=False,
                      log=lambda m: None)
    assert res.action == "prepared"


def test_preparing_without_executing_runs_no_md(tmp_path):
    wd = _ready(tmp_path / "run00")
    res = advance_run(wd, TARGET, FakeProbe(), PROTOCOL, execute=False,
                      log=lambda m: None)
    assert res.action == "prepared"
    assert not (wd / "pr.gro").exists()


def test_running_records_a_start_and_an_end(tmp_path):
    wd = _ready(tmp_path / "run00")
    (wd / "mdrun.sh").write_text("#!/bin/sh\ntouch min1.gro pr.gro\n")
    res = advance_run(wd, TARGET, FakeProbe(), PROTOCOL, log=lambda m: None)
    assert res.action == "ran"
    assert [e["status"] for e in read_events(wd)] == ["mdrun_start", "mdrun_end"]
    assert res.status.done


def test_a_failure_is_recorded_with_the_stage_it_died_in(tmp_path):
    """The stage matters: dying later than last time is progress, not a stuck run."""
    wd = _ready(tmp_path / "run00")
    (wd / "mdrun.sh").write_text("#!/bin/sh\ntouch min1.gro\nexit 3\n")
    res = advance_run(wd, TARGET, FakeProbe(), PROTOCOL, log=lambda m: None)
    assert res.action == "error"
    assert res.returncode == 3
    last = read_events(wd)[-1]
    assert last["status"] == "error"
    assert last["stage"] == "pr"
    assert last["finished_steps"] == ["min1"]


def test_the_wall_clock_budget_reaches_the_script(tmp_path):
    wd = _ready(tmp_path / "run00")
    (wd / "mdrun.sh").write_text(
        '#!/bin/sh\necho "$HRT_SEC $TAIL_SEC $PIN_OFFSET $GPU_ID" > seen.txt\n'
        "touch min1.gro pr.gro\n"
    )
    advance_run(wd, TARGET, FakeProbe(), PROTOCOL, seconds_remaining=3600,
                tail_sec=300, pin_offset=8, gpu_id=1, log=lambda m: None)
    assert (wd / "seen.txt").read_text().split() == ["3600", "300", "8", "1"]


def test_no_budget_means_no_limit_is_imposed(tmp_path):
    """Without a limit the script keeps its own default rather than being cut short."""
    wd = _ready(tmp_path / "run00")
    (wd / "mdrun.sh").write_text(
        '#!/bin/sh\necho "[${HRT_SEC:-unset}]" > seen.txt\ntouch min1.gro pr.gro\n')
    advance_run(wd, TARGET, FakeProbe(), PROTOCOL, log=lambda m: None)
    assert (wd / "seen.txt").read_text().strip() == "[unset]"


def test_the_thread_count_is_passed_as_the_scripts_argument(tmp_path):
    wd = _ready(tmp_path / "run00")
    (wd / "mdrun.sh").write_text('#!/bin/sh\necho "$1" > seen.txt\ntouch min1.gro pr.gro\n')
    advance_run(wd, TARGET, FakeProbe(), PROTOCOL, ntomp=4, log=lambda m: None)
    assert (wd / "seen.txt").read_text().strip() == "4"


def test_the_seed_follows_the_placement_attempt(tmp_path):
    """A rebuild after a rejected placement must not reproduce the rejected system."""
    from prismsmd.seeds import record_placement_attempt, seed_for_attempt

    wd = _ready(tmp_path / "run00")
    base = run_seed(TARGET.key, "P1", 0)
    assert placement_attempt(wd) == 0
    record_placement_attempt(wd, 1, seed=base)
    assert placement_attempt(wd) == 1
    assert seed_for_attempt(base, 1) - base == PLACEMENT_RETRY_STRIDE


def test_too_many_placement_attempts_stop_the_run(tmp_path):
    from prismsmd.seeds import MAX_PLACEMENT_ATTEMPTS, record_placement_attempt

    wd = _ready(tmp_path / "run00")
    record_placement_attempt(wd, MAX_PLACEMENT_ATTEMPTS)
    res = advance_run(wd, TARGET, FakeProbe(), PROTOCOL, execute=False,
                      log=lambda m: None)
    assert res.action == "stuck"
    assert "placement attempts" in res.messages[0]


def test_an_unbuilt_directory_is_built_before_any_md_starts(tmp_path):
    """MD must never start on a directory whose system is missing."""
    wd = _ready(tmp_path / "run00")
    (wd / "mdrun.sh").write_text("#!/bin/sh\ntouch RAN\n")
    (wd / "input.top").unlink()
    with pytest.raises(ValueError, match="mol/sdf"):
        advance_run(wd, TARGET, FakeProbe(), PROTOCOL, log=lambda m: None)
    assert not (wd / "RAN").exists()
