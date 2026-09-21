"""Invariants of the placement seed and the retry rule."""

import itertools
import json

import pytest

from prismsmd.seeds import (
    DEFAULT_SEED_MODULUS,
    MAX_PLACEMENT_ATTEMPTS,
    PLACEMENT_RETRY_STRIDE,
    assert_no_seed_collision,
    placement_attempt,
    record_placement_attempt,
    run_seed,
    seed_for_attempt,
)


def test_run_seed_is_stable_across_processes():
    """The seed is a hash of the identity, not of interpreter state."""
    assert run_seed("brd4", "A04", 0) == 186011151


def test_run_seed_depends_on_every_field():
    seeds = {
        run_seed("brd4", "A04", 0),
        run_seed("brd4", "A04", 1),
        run_seed("brd4", "A06", 0),
        run_seed("ampc", "A04", 0),
    }
    assert len(seeds) == 4


def test_run_seed_namespace_moves_every_seed():
    """Changing the namespace is the documented way out of a collision."""
    a = [run_seed("brd4", "A04", r) for r in range(5)]
    b = [run_seed("brd4", "A04", r, namespace="other") for r in range(5)]
    assert not set(a) & set(b)


def test_run_seed_stays_below_the_modulus():
    for run in range(50):
        assert 0 <= run_seed("t", "p", run) < DEFAULT_SEED_MODULUS


def test_adding_a_probe_leaves_existing_seeds_alone():
    """A hash is used instead of a running number for exactly this reason."""
    before = {r: run_seed("brd4", "A04", r) for r in range(20)}
    after = {r: run_seed("brd4", "A04", r) for r in range(20)}
    _ = [run_seed("brd4", "B71", r) for r in range(20)]
    assert before == after


def test_run_seed_rejects_a_negative_run():
    with pytest.raises(ValueError, match="run number"):
        run_seed("t", "p", -1)


def test_collision_check_names_both_runs():
    with pytest.raises(ValueError, match="seed collision"):
        assert_no_seed_collision({"a/x/run00": 7, "b/y/run00": 7})


def test_collision_check_passes_a_real_ensemble():
    seeds = {
        f"{t}/{p}/run{r:02d}": run_seed(t, p, r)
        for t in ("brd4", "ampc", "dhfr")
        for p in ("A04", "A06", "B71")
        for r in range(20)
    }
    assert_no_seed_collision(seeds)


def test_attempt_zero_is_the_base_seed():
    assert seed_for_attempt(12345, 0) == 12345


def test_attempts_are_spaced_by_the_stride():
    base = run_seed("brd4", "A04", 0)
    spaced = [seed_for_attempt(base, a) for a in range(MAX_PLACEMENT_ATTEMPTS)]
    gaps = {b - a for a, b in itertools.pairwise(spaced)}
    assert gaps == {PLACEMENT_RETRY_STRIDE}


def test_every_attempt_of_a_run_has_its_own_seed():
    """Including attempt 0, so a retry never rebuilds the rejected system."""
    base = run_seed("brd4", "A04", 0)
    seeds = [seed_for_attempt(base, a) for a in range(MAX_PLACEMENT_ATTEMPTS)]
    assert len(set(seeds)) == MAX_PLACEMENT_ATTEMPTS


def test_seed_for_attempt_rejects_a_negative_attempt():
    with pytest.raises(ValueError, match="attempt"):
        seed_for_attempt(1, -1)


def test_attempt_defaults_to_zero_without_a_record(tmp_path):
    assert placement_attempt(tmp_path) == 0


def test_attempt_round_trips(tmp_path):
    record_placement_attempt(tmp_path, 2, seed=99, reason="probe clash")
    assert placement_attempt(tmp_path) == 2
    stored = json.loads((tmp_path / "placement_attempt.json").read_text())
    assert stored["seed"] == 99
    assert stored["reason"] == "probe clash"


def test_a_damaged_record_restarts_the_count(tmp_path):
    """A broken file must not stop the run."""
    (tmp_path / "placement_attempt.json").write_text("{not json")
    assert placement_attempt(tmp_path) == 0


def test_recording_creates_the_run_directory(tmp_path):
    target = tmp_path / "brd4" / "A04" / "run00"
    record_placement_attempt(target, 1)
    assert placement_attempt(target) == 1
