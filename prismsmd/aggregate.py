"""Voxel-wise aggregation of several grids.

:func:`combine_replicates` is the way from one probe atom's per-run PMAPs to a map that
can be converted to GFE. It returns the bulk number density to convert with, so the
conversion happens after the combination and never before: ``-RT ln`` is non-linear, so
averaging finished GFE grids is not the same thing and gives a different answer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .grid import Grid, GridSpec

__all__ = [
    "AGGREGATIONS",
    "RUN_AGGREGATIONS",
    "aggregate",
    "aggregate_runs",
    "combine_replicates",
]

#: Methods produced by :func:`aggregate_runs` by default.
RUN_AGGREGATIONS = ("ave", "max", "min")


def _stack(grids: Sequence[Grid]) -> np.ndarray:
    if not grids:
        raise ValueError("input is empty")
    base = grids[0].spec
    for g in grids[1:]:
        base.assert_compatible(g.spec)
    return np.stack([g.values for g in grids], axis=0)


AGGREGATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "ave": lambda a: np.mean(a, axis=0),
    "mean": lambda a: np.mean(a, axis=0),
    "max": lambda a: np.max(a, axis=0),
    "min": lambda a: np.min(a, axis=0),
    "median": lambda a: np.median(a, axis=0),
}


def aggregate(grids: Sequence[Grid], method: str = "ave") -> Grid:
    """Reduce several Grids to one, voxel-wise.

    method: ``ave`` / ``mean`` / ``max`` / ``min`` / ``median``.
    """
    if method not in AGGREGATIONS:
        raise ValueError(
            f"unknown aggregation {method!r}; available: {sorted(AGGREGATIONS)}"
        )
    stacked = _stack(grids)
    return Grid(spec=grids[0].spec, values=AGGREGATIONS[method](stacked))


def aggregate_runs(
    grids: Sequence[Grid],
    *,
    expected: GridSpec | None = None,
    methods: Sequence[str] = RUN_AGGREGATIONS,
    n_runs: int | None = None,
) -> dict[str, Grid]:
    """Apply several aggregation methods to one set of per-run grids.

    Parameters
    ----------
    expected:
        Grid every run must match. ``None`` uses the first run's grid.
    methods:
        Aggregation names, as keys of :data:`AGGREGATIONS`.
    n_runs:
        Required number of runs; a different count raises.

    Returns a ``{method: Grid}`` mapping. Raises if any run's grid differs from
    ``expected``.
    """
    from .pmap_cosolvkit import assert_grid_matches

    if not grids:
        raise ValueError("no input for cross-run aggregation")
    if n_runs is not None and len(grids) != n_runs:
        raise ValueError(
            f"only {len(grids)} runs present (expected {n_runs})."
            " Check that none were missed."
        )

    base = expected if expected is not None else grids[0].spec
    for i, g in enumerate(grids):
        assert_grid_matches(g, base, what=f"PMAP grid of run {i}")

    stacked = np.stack([g.values for g in grids], axis=0)
    out: dict[str, Grid] = {}
    for method in methods:
        if method not in AGGREGATIONS:
            raise ValueError(
                f"unknown aggregation {method!r}; available: {sorted(AGGREGATIONS)}"
            )
        out[method] = Grid(spec=base, values=AGGREGATIONS[method](stacked))
    return out


def combine_replicates(
    pmaps: Sequence[Grid],
    bulk_densities: Sequence[float],
    *,
    method: str = "ave",
    expected: GridSpec | None = None,
    n_runs: int | None = None,
) -> tuple[Grid, float]:
    """Combine one probe atom's PMAP over replicate runs, with the density to convert it.

    Args:
        pmaps: one probe atom's per-run PMAPs as **number densities** (per A^3), all on
            the same grid, as :meth:`prismsmd.pmap_cosolvkit.PMAPResult.number_density`
            returns them. Raw counts must not be passed: runs can differ in frame
            count, so combining counts weights the runs by length without saying so.
        bulk_densities: each run's measured bulk number density (per A^3), in the same
            order. Every run contributes one, because each has its own box volume and
            probe count.
        method: how to combine, as a key of :data:`AGGREGATIONS`.
        expected: grid every run must match. ``None`` uses the first run's.
        n_runs: required number of runs; a different count raises.

    Returns:
        ``(combined PMAP, bulk number density)``. Pass both to
        :func:`prismsmd.gfe.pmap_to_gfe`, which is the only supported route from
        replicates to GFE.

    Raises:
        ValueError: if the counts differ, a run is missing, a grid does not match, or a
            density is not positive. A missing density is never replaced by one derived
            from the nominal concentration: that silently changes what the map means.
    """
    if len(pmaps) != len(bulk_densities):
        raise ValueError(
            f"{len(pmaps)} PMAPs but {len(bulk_densities)} bulk densities."
            " Every run contributes exactly one of each."
        )
    if not bulk_densities:
        raise ValueError("no runs to combine")
    for i, density in enumerate(bulk_densities):
        if not density > 0:
            raise ValueError(
                f"run {i} has a bulk number density of {density!r}."
                " A run without a measured density cannot be combined; it is not"
                " substitutable by the nominal concentration."
            )
    combined = aggregate_runs(
        pmaps, expected=expected, methods=(method,), n_runs=n_runs
    )[method]
    return combined, float(np.mean(np.asarray(bulk_densities, dtype=float)))
