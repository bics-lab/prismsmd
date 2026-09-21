"""Invariants of the build stamp written beside a result."""

import subprocess

import pytest

from prismsmd.provenance import DESCRIBE_ARGV, describe, stamp, version


def _git(stdout="", returncode=0):
    def run(argv, cwd):
        run.seen = (list(argv), cwd)
        return subprocess.CompletedProcess(list(argv), returncode, stdout, "")
    run.seen = None
    return run


def test_the_stamp_always_carries_the_version():
    assert stamp(run=_git("v0.1.0"))["version"] == version()


def test_a_clean_checkout_is_described_by_its_tag():
    assert stamp(run=_git("v0.1.0\n"))["source"] == "v0.1.0"


def test_commits_since_the_tag_are_visible():
    """Between releases the version alone cannot separate two builds."""
    assert stamp(run=_git("v0.1.0-3-g9327cc2\n"))["source"] == "v0.1.0-3-g9327cc2"


def test_a_modified_tree_says_so():
    """Running from an edited checkout is normal in research; hiding it is a lie."""
    assert stamp(run=_git("v0.1.0-3-g9327cc2-dirty\n"))["source"].endswith("-dirty")


def test_the_dirty_flag_is_actually_asked_for():
    """Without --dirty git reports an edited tree as if it were the commit."""
    assert "--dirty" in DESCRIBE_ARGV


def test_a_tagless_repository_still_reports_its_commit():
    assert stamp(run=_git("9327cc2\n"))["source"] == "9327cc2"


def test_no_checkout_means_no_source_rather_than_a_guess():
    """An installed copy has nothing to describe, and says nothing."""
    assert "source" not in stamp(run=_git("", returncode=128))


def test_git_missing_is_not_an_error():
    def run(argv, cwd):
        raise OSError("git not found")

    assert "source" not in stamp(run=run)


def test_git_taking_too_long_is_not_an_error():
    def run(argv, cwd):
        raise subprocess.TimeoutExpired(list(argv), 5.0)

    assert "source" not in stamp(run=run)


def test_the_source_asked_about_is_this_package_not_the_callers_directory():
    """Run from anywhere, the answer must be about the code that is running."""
    run = _git("v0.1.0")
    describe(run=run)
    _argv, cwd = run.seen
    assert (cwd / "provenance.py").is_file()


def test_the_stamp_reaches_the_map_manifest():
    import numpy as np

    from prismsmd.grid import GridSpec
    from prismsmd.maps import RunMaps

    spec = GridSpec(center=(0.0, 0.0, 0.0), dims=(2, 2, 2), spacing=1.0)
    maps = RunMaps(spec=spec, pmaps={}, gfes={}, bulk_densities={}, results={},
                   excluded_volume=0.0, valid_mask=np.ones(spec.dims, dtype=bool))
    assert maps.to_dict()["prismsmd"]["version"] == version()


def test_the_stamp_reaches_every_event(tmp_path):
    """A run resumed days later under another build shows it line by line."""
    from prismsmd.events import FIXED_FIELDS, read_events, record_event

    record_event(tmp_path, status="build_done")
    record_event(tmp_path, status="mdrun_end")
    events = read_events(tmp_path)
    assert len(events) == 2
    for event in events:
        assert event["prismsmd"]["version"] == version()
    assert "prismsmd" in FIXED_FIELDS


def test_a_result_without_a_stamp_is_the_thing_this_prevents():
    """Guards the wiring: a manifest that forgets the stamp cannot be traced back."""
    import inspect

    from prismsmd import events, maps

    assert "stamp()" in inspect.getsource(maps.RunMaps.to_dict)
    assert "stamp()" in inspect.getsource(events.record_event)


def test_the_stamp_names_the_package_it_describes():
    """Two packages' stamps end up side by side in one record."""
    assert stamp(run=_git("v0.1.0"))["package"] == "prismsmd"


@pytest.mark.parametrize("described", ["", "   ", "\n"])
def test_an_empty_description_is_no_description(described):
    assert "source" not in stamp(run=_git(described))
