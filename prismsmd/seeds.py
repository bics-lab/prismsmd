"""Deterministic placement seeds, and the retry rule when a placement is unusable.

:func:`run_seed` gives each run of an ensemble a seed derived from its identity, so the
same (target, probe, run) always rebuilds the same system. :func:`seed_for_attempt`
gives the seed of a retry, so a rebuild stays reproducible after
:class:`prismsmd.system.clash.ProbeClashError` forced a re-placement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

__all__ = [
    "DEFAULT_SEED_MODULUS",
    "DEFAULT_SEED_NAMESPACE",
    "MAX_PLACEMENT_ATTEMPTS",
    "PLACEMENT_RETRY_STRIDE",
    "PLACEMENT_STATE_NAME",
    "assert_no_seed_collision",
    "placement_attempt",
    "record_placement_attempt",
    "run_seed",
    "seed_for_attempt",
]

#: Default namespace mixed into :func:`run_seed`. Changing it moves every seed, which
#: is the way out of a collision.
DEFAULT_SEED_NAMESPACE = "prismsmd"

#: Default upper bound on a base seed, so a seed stays a readable size.
DEFAULT_SEED_MODULUS = 1 << 28

#: Stride between the seeds of successive placement attempts. A prime, so successive
#: attempts spread rather than falling into a pattern. It is smaller than
#: :data:`DEFAULT_SEED_MODULUS`, so a retry seed can in principle land on another run's
#: base seed; that is tolerated because :func:`assert_no_seed_collision` checks the base
#: seeds and a retry is rare.
PLACEMENT_RETRY_STRIDE = 1_000_003

#: How many times to re-place the probes before giving up on a run. A fresh placement
#: is independent of the one before it, so a second attempt nearly always works; more
#: than a few attempts means something systematic is wrong with the target.
MAX_PLACEMENT_ATTEMPTS = 3

#: Where the attempt counter lives, inside the run directory.
PLACEMENT_STATE_NAME = "placement_attempt.json"


def run_seed(
    target: str,
    probe: str,
    run: int,
    *,
    namespace: str = DEFAULT_SEED_NAMESPACE,
    modulus: int = DEFAULT_SEED_MODULUS,
) -> int:
    """Derive this run's placement seed from its identity.

    ``blake2b`` is used rather than the builtin ``hash()``, which varies per
    interpreter start with ``PYTHONHASHSEED``, so the same (target, probe, run) gives
    the same seed on any Python version or machine. A hash is used rather than a
    running number so that adding a target or a probe leaves every existing run's seed
    unchanged. :func:`assert_no_seed_collision` checks a whole ensemble for collisions.

    >>> run_seed("brd4", "A04", 0) == run_seed("brd4", "A04", 0)
    True
    >>> run_seed("brd4", "A04", 0) == run_seed("brd4", "A04", 1)
    False
    """
    if run < 0:
        raise ValueError(f"run number must be >= 0: {run}")
    if modulus < 1:
        raise ValueError(f"modulus must be positive: {modulus}")
    text = f"{namespace}|{target}|{probe}|run{run:02d}"
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulus


def assert_no_seed_collision(seeds: dict[str, int]) -> None:
    """Raise if two runs of an ensemble share a seed.

    Args:
        seeds: run key -> seed, as :func:`run_seed` produced them.

    Raises:
        ValueError: on a collision, naming both runs. Changing the namespace moves
            every seed and resolves it.
    """
    seen: dict[int, str] = {}
    for key, seed in seeds.items():
        if seed in seen:
            raise ValueError(
                f"seed collision: {seen[seed]} and {key} share seed {seed}."
                " Change the seed namespace to move every seed."
            )
        seen[seed] = key


def seed_for_attempt(base_seed: int, attempt: int) -> int:
    """The seed to build with, given the run's base seed and the attempt number.

    Deterministic in both arguments, so rebuilding attempt 2 gives back exactly the
    system attempt 2 produced.

    >>> seed_for_attempt(7, 0)
    7
    >>> seed_for_attempt(7, 2) - seed_for_attempt(7, 1) == PLACEMENT_RETRY_STRIDE
    True
    """
    if attempt < 0:
        raise ValueError(f"attempt must be >= 0: {attempt}")
    return base_seed + PLACEMENT_RETRY_STRIDE * attempt


def placement_attempt(workdir: str | Path) -> int:
    """Which placement attempt this run is on; 0 means the original seed.

    Returns 0 when no attempt has been recorded or the record cannot be read, so a
    damaged file restarts the count rather than stopping the run.
    """
    path = Path(workdir) / PLACEMENT_STATE_NAME
    if not path.is_file():
        return 0
    try:
        return int(json.loads(path.read_text()).get("attempt", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def record_placement_attempt(workdir: str | Path, attempt: int, **fields) -> Path:
    """Persist the attempt counter so a later job continues from it.

    Args:
        workdir: the run directory.
        attempt: the attempt number just tried.
        **fields: extra values to keep alongside it, such as the seed used and why the
            previous attempt was rejected.

    Returns:
        The path written.
    """
    if attempt < 0:
        raise ValueError(f"attempt must be >= 0: {attempt}")
    path = Path(workdir) / PLACEMENT_STATE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"attempt": attempt, **fields}, indent=2) + "\n")
    return path
