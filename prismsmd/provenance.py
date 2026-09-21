"""Which build of this package produced a result.

A map or an event log outlives the checkout that made it, and the question asked of
it later is always the same: what code was this? The version alone does not answer it
during development, when every run between two releases reports the same number, nor
does it answer it at all when the tree was modified before running -- which is the
normal case in research.

:func:`stamp` therefore reports the version together with what ``git describe`` makes
of the working tree, so a result carries the tag, the distance from it, the commit and
**whether the tree was dirty**. A stamp that hides the last of those is worse than no
stamp: it reads as a promise the code cannot keep.

Outside a checkout -- an installed wheel, a copy without ``.git`` -- there is no source
to describe and the stamp is the version alone, which is then the truth.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

__all__ = ["DESCRIBE_ARGV", "describe", "stamp", "version"]

#: How the source is described. ``--dirty`` is what makes a modified tree visible;
#: ``--always`` keeps a repository without tags from reporting nothing at all.
DESCRIBE_ARGV = ("git", "describe", "--tags", "--always", "--dirty", "--abbrev=7")

#: Seconds to let git take. A stamp is never worth holding up the work that earns it.
DESCRIBE_TIMEOUT_SEC = 5.0

Runner = Callable[..., subprocess.CompletedProcess]


def _run(argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(argv), cwd=str(cwd), capture_output=True, text=True,
        timeout=DESCRIBE_TIMEOUT_SEC, check=False,
    )


def version() -> str:
    """The package version, from the installed metadata or the module."""
    from . import __version__

    return __version__


def describe(*, run: Runner = _run, root: Path | None = None) -> str | None:
    """What ``git describe`` says about the source this module was imported from.

    Args:
        run: how to invoke git; injectable so the behaviour can be exercised without
            a repository.
        root: where to ask. Defaults to the directory this package sits in, so the
            answer is about *this* copy rather than whatever the caller's cwd is.

    Returns:
        The description, or None when there is no repository to describe, git is not
        installed, or it takes too long. None is a fact -- there is no source here --
        and is reported as such rather than as an error.
    """
    where = root if root is not None else Path(__file__).resolve().parent
    try:
        proc = run(DESCRIBE_ARGV, where)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    return text or None


def stamp(*, run: Runner = _run, root: Path | None = None) -> dict:
    """The record to write beside a result.

    Returns:
        ``{"package": "prismsmd", "version": ..., "source": ...}``, where ``source`` is
        absent when there is no checkout to describe.
    """
    out = {"package": "prismsmd", "version": version()}
    found = describe(run=run, root=root)
    if found is not None:
        out["source"] = found
    return out
