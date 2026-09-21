"""Every check in the package is reached from somewhere.

A check that is defined but never called is the same as no check, and it fails
silently: the suite still passes, the docs still describe the guard, and the bad
input goes through. This sweep is here because two of them had gone dormant.
"""

import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parent.parent / "prismsmd"

#: Checks the package deliberately does not call, with the caller that must.
#: A name belongs here only when the package cannot see the thing being checked.
CALLED_BY_THE_CALLER = {
    # The package works one run at a time and never sees the whole ensemble, so the
    # code that decides which runs exist is the only place a collision is visible.
    "assert_no_seed_collision",
    # Its limit is an absolute occupied fraction, and what a correct run reaches
    # depends on how long it is: 0.15 ns of one run measures 0.48 %, 40 ns of twenty
    # measures 52 %, against a limit of 1 %. Only the caller knows its own sampling.
    "assert_density_in_valid_region",
}


def _defined_and_called():
    defined: dict[str, str] = {}
    called: set[str] = set()
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("assert_"):
                defined[node.name] = f"{path.name}:{node.lineno}"
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name:
                    called.add(name)
    return defined, called


def test_every_check_is_called_somewhere():
    defined, called = _defined_and_called()
    dormant = {
        name: where
        for name, where in defined.items()
        if name not in called and name not in CALLED_BY_THE_CALLER
    }
    assert not dormant, (
        "these checks are defined but never called, so they guard nothing: "
        + ", ".join(f"{n} ({w})" for n, w in sorted(dormant.items()))
        + ". Wire each one in, or list it in CALLED_BY_THE_CALLER with the reason."
    )


def test_the_sweep_finds_something_to_look_at():
    """A sweep over an empty set would pass for the wrong reason."""
    defined, _ = _defined_and_called()
    assert len(defined) >= 5


def test_the_exemptions_still_exist():
    """An exemption for a function that is gone hides the next dormant check."""
    defined, _ = _defined_and_called()
    assert set(defined) >= CALLED_BY_THE_CALLER


# --- the same sweep for the event vocabulary ---------------------------------
# A status listed as progress but never recorded is a silent hole: a run that only
# ever reaches that stage looks like it made no progress at all.

#: Statuses the minimal package deliberately never records, with the reason.
NOT_RECORDED_HERE = {
    # Rebuild policies ("throw away what is there and redo it") are the caller's:
    # this package only ever moves a run forward.
    "rebuild",
    "pmap_rebuild",
}


def _progress_statuses():
    from prismsmd.events import PROGRESS_STATUSES

    return set(PROGRESS_STATUSES)


def _recorded_statuses():
    import re

    recorded = set()
    for path in PKG.rglob("*.py"):
        recorded.update(re.findall(r'status="([a-z_]+)"', path.read_text()))
    return recorded


def test_every_progress_status_is_recorded_somewhere():
    missing = _progress_statuses() - _recorded_statuses() - NOT_RECORDED_HERE
    assert not missing, (
        "these statuses count as progress but nothing records them, so a run that"
        f" reaches only that stage looks stalled: {sorted(missing)}."
        " Record each one, or list it in NOT_RECORDED_HERE with the reason."
    )


def test_the_event_exemptions_still_exist():
    assert _progress_statuses() >= NOT_RECORDED_HERE
