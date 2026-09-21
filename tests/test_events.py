"""Invariants of the run event log and the stuck-run counter."""

import json

from prismsmd.events import (
    EVENTS_NAME,
    FIXED_FIELDS,
    STATE_NAME,
    count_consecutive_failures,
    read_events,
    record_event,
)


def test_recording_creates_the_run_directory(tmp_path):
    target = tmp_path / "brd4" / "A04" / "run00"
    record_event(target, status="build_done")
    assert (target / EVENTS_NAME).is_file()


def test_the_log_is_append_only(tmp_path):
    for i in range(3):
        record_event(tmp_path, status="error", step=i)
    events = read_events(tmp_path)
    assert [e["step"] for e in events] == [0, 1, 2]


def test_state_holds_the_newest_event(tmp_path):
    record_event(tmp_path, status="mdrun_start")
    record_event(tmp_path, status="mdrun_end")
    state = json.loads((tmp_path / STATE_NAME).read_text())
    assert state["status"] == "mdrun_end"


def test_every_event_carries_the_fixed_fields(tmp_path):
    """Whatever else varies, these three are always there and never empty."""
    for fields in ({}, {"job_id": "1"}, {"returncode": 1, "seed": 7}):
        event = record_event(tmp_path, status="build_done", **fields)
        assert all(event.get(k) for k in FIXED_FIELDS)


def test_optional_fields_are_absent_rather_than_null(tmp_path):
    """A field nobody supplied is not recorded, so `.get()` is how to read one."""
    event = record_event(tmp_path, status="build_done")
    assert "job_id" not in event
    assert event.get("job_id") is None


def test_execution_context_is_supplied_by_the_caller(tmp_path):
    """The caller knows what batch system it is under; this module does not ask."""
    event = record_event(tmp_path, status="mdrun_start", job_id="12345", task_id=7)
    assert event["job_id"] == "12345"
    assert event["task_id"] == 7
    assert read_events(tmp_path)[0]["task_id"] == 7


def test_the_environment_is_never_consulted():
    """The interface is the argument list, so the module reads no variables at all.

    Stated over the source rather than over a list of names: a list only covers the
    schedulers someone thought of, and the next one to be added would pass.
    """
    import inspect

    from prismsmd import events

    source = inspect.getsource(events)
    for forbidden in ("os.environ", "getenv", "os.getenv"):
        assert forbidden not in source


def test_setting_variables_does_not_change_the_record(tmp_path, monkeypatch):
    """The behavioural half of the same invariant."""
    for i in range(4):
        monkeypatch.setenv(f"PRISMSMD_TEST_VAR_{i}", "should-be-ignored")
    event = record_event(tmp_path, status="build_done")
    assert "should-be-ignored" not in json.dumps(event)


def test_reading_an_absent_log_is_not_an_error(tmp_path):
    assert read_events(tmp_path) == []
    assert count_consecutive_failures(tmp_path) == 0


def test_a_truncated_line_does_not_hide_the_history(tmp_path):
    record_event(tmp_path, status="build_done")
    with (tmp_path / EVENTS_NAME).open("a") as fh:
        fh.write('{"status": "err\n')          # an interrupted write
    record_event(tmp_path, status="mdrun_end")
    assert [e["status"] for e in read_events(tmp_path)] == ["build_done", "mdrun_end"]


def test_failures_accumulate(tmp_path):
    for _ in range(3):
        record_event(tmp_path, status="error")
    assert count_consecutive_failures(tmp_path) == 3


def test_progress_ends_the_streak(tmp_path):
    record_event(tmp_path, status="error")
    record_event(tmp_path, status="error")
    record_event(tmp_path, status="mdrun_end")
    record_event(tmp_path, status="error")
    assert count_consecutive_failures(tmp_path) == 1


def test_a_rebuild_resets_the_count(tmp_path):
    """Discarding the inputs is an explicit statement that old failures do not apply."""
    for _ in range(5):
        record_event(tmp_path, status="error")
    record_event(tmp_path, status="rebuild")
    assert count_consecutive_failures(tmp_path) == 0


def test_entering_a_stage_does_not_break_the_streak(tmp_path):
    """Otherwise the streak would pin at 1, since a failure always follows a start."""
    for _ in range(3):
        record_event(tmp_path, status="mdrun_start")
        record_event(tmp_path, status="error")
    assert count_consecutive_failures(tmp_path) == 3


def test_dying_later_than_before_counts_as_progress(tmp_path):
    record_event(tmp_path, status="error", finished_steps=["min1"])
    record_event(tmp_path, status="error", finished_steps=["min1", "heat", "equil1"])
    assert count_consecutive_failures(tmp_path) == 1


def test_dying_at_the_same_stage_does_not(tmp_path):
    for _ in range(4):
        record_event(tmp_path, status="error", finished_steps=["min1", "heat"])
    assert count_consecutive_failures(tmp_path) == 4
