"""Invariants of reading a run's progress off disk."""

from prismsmd.config import MDStep, ProtocolConfig
from prismsmd.events import record_event
from prismsmd.runstate import SYSTEM_FILES, last_step_in_log, run_status

PROTOCOL = ProtocolConfig(
    dt=0.002,
    steps=(
        MDStep(name="min1", type="minimization", nsteps=200),
        MDStep(name="heat", type="heating", nsteps=500_000),
        MDStep(name="equil1", type="equilibration", nsteps=500_000),
        MDStep(name="pr", type="production", nsteps=1_000_000),
    ),
)


def _built(tmp_path):
    for name in SYSTEM_FILES:
        (tmp_path / name).write_text("")
    (tmp_path / "mdrun.sh").write_text("")
    for name in PROTOCOL.step_names:
        (tmp_path / f"{name}.mdp").write_text("")
    return tmp_path


def _log(tmp_path, step, last_step):
    (tmp_path / f"{step}.log").write_text(
        f"           Step           Time\n      {last_step}       {last_step * 0.002}\n"
    )


def test_the_protocol_knows_its_own_length():
    assert PROTOCOL.step_ns("min1") == 0.0          # minimisation is not simulated time
    assert PROTOCOL.step_ns("heat") == 1.0
    assert PROTOCOL.total_ns == 4.0


def test_an_empty_directory_has_not_started(tmp_path):
    st = run_status(tmp_path / "nothing", PROTOCOL)
    assert st.state == "not_started"
    assert st.remaining_ns == PROTOCOL.total_ns


def test_a_built_but_unrun_directory_is_running(tmp_path):
    st = run_status(_built(tmp_path), PROTOCOL)
    assert st.system_built and st.prepared
    assert st.state == "running"
    assert st.current_step == "min1"
    assert st.finished_steps == ()


def test_a_stage_counts_as_finished_only_with_its_gro(tmp_path):
    """GROMACS writes the .gro on reaching nsteps, so it is the completion mark."""
    wd = _built(tmp_path)
    (wd / "min1.cpt").write_text("")       # cut short: checkpoint but no .gro
    _log(wd, "min1", 120)
    st = run_status(wd, PROTOCOL)
    assert st.finished_steps == ()
    assert st.current_step == "min1"


def test_progress_is_the_leading_run_of_finished_stages(tmp_path):
    """A stray .gro from a later stage must not be read as progress."""
    wd = _built(tmp_path)
    (wd / "min1.gro").write_text("")
    (wd / "pr.gro").write_text("")         # left over from somewhere
    st = run_status(wd, PROTOCOL)
    assert st.finished_steps == ("min1",)
    assert st.current_step == "heat"


def test_the_running_stage_reports_the_time_it_has_covered(tmp_path):
    wd = _built(tmp_path)
    (wd / "min1.gro").write_text("")
    _log(wd, "heat", 250_000)              # half of 1 ns
    st = run_status(wd, PROTOCOL)
    assert st.current_step_ns == 0.5
    assert st.remaining_ns == 3.5          # 1 + 1 + 2 total, less the 0.5 covered


def test_the_covered_time_never_exceeds_the_stage(tmp_path):
    """A resumed log can carry a step count past this stage's nsteps."""
    wd = _built(tmp_path)
    (wd / "min1.gro").write_text("")
    _log(wd, "heat", 900_000)
    assert run_status(wd, PROTOCOL).current_step_ns == 1.0


def test_a_finished_run_is_done(tmp_path):
    wd = _built(tmp_path)
    for name in PROTOCOL.step_names:
        (wd / f"{name}.gro").write_text("")
    st = run_status(wd, PROTOCOL)
    assert st.done and st.state == "done"
    assert st.current_step is None
    assert st.remaining_ns == 0.0


def test_an_error_event_marks_the_run_failed(tmp_path):
    wd = _built(tmp_path)
    record_event(wd, status="error", returncode=1)
    assert run_status(wd, PROTOCOL).state == "failed"


def test_a_finished_run_is_not_failed_by_an_old_error(tmp_path):
    wd = _built(tmp_path)
    record_event(wd, status="error", returncode=1)
    for name in PROTOCOL.step_names:
        (wd / f"{name}.gro").write_text("")
    assert run_status(wd, PROTOCOL).state == "done"


def test_a_missing_log_reads_as_no_progress_rather_than_zero_steps(tmp_path):
    assert last_step_in_log(tmp_path / "absent.log") is None


def test_the_last_step_wins_when_a_log_has_several_tables(tmp_path):
    (tmp_path / "x.log").write_text(
        "           Step           Time\n      100       0.2\n"
        "some output\n"
        "           Step           Time\n      900       1.8\n"
    )
    assert last_step_in_log(tmp_path / "x.log") == 900
