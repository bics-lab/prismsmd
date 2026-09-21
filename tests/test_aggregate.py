"""Invariants of combining replicate PMAPs."""

import numpy as np
import pytest

from prismsmd.aggregate import aggregate, combine_replicates
from prismsmd.gfe import GFEParams, pmap_to_gfe
from prismsmd.grid import Grid, GridSpec

SPEC = GridSpec(center=(0.0, 0.0, 0.0), dims=(4, 4, 4), spacing=1.0)


def _pmap(value):
    return Grid(spec=SPEC, values=np.full(SPEC.dims, float(value)))


def test_combining_averages_the_maps():
    combined, _ = combine_replicates([_pmap(1), _pmap(3)], [1e-3, 1e-3])
    assert np.allclose(combined.values, 2.0)


def test_the_density_is_the_mean_over_runs():
    """Each run has its own box volume and probe count, so each contributes one."""
    _, density = combine_replicates([_pmap(1)] * 3, [1.0e-3, 2.0e-3, 3.0e-3])
    assert density == pytest.approx(2.0e-3)


def test_max_is_available_too():
    combined, _ = combine_replicates([_pmap(1), _pmap(5)], [1e-3, 1e-3], method="max")
    assert np.allclose(combined.values, 5.0)


def test_a_missing_density_is_refused():
    with pytest.raises(ValueError, match="PMAPs but"):
        combine_replicates([_pmap(1), _pmap(2)], [1e-3])


def test_a_zero_density_is_refused_rather_than_substituted():
    """The nominal concentration is not a stand-in for a measurement."""
    with pytest.raises(ValueError, match="bulk number density"):
        combine_replicates([_pmap(1), _pmap(2)], [1e-3, 0.0])


def test_a_missing_run_is_refused():
    with pytest.raises(ValueError, match="expected 3"):
        combine_replicates([_pmap(1), _pmap(2)], [1e-3, 1e-3], n_runs=3)


def test_grids_must_match():
    other = Grid(
        spec=GridSpec(center=(9.0, 0.0, 0.0), dims=(4, 4, 4), spacing=1.0),
        values=np.ones((4, 4, 4)),
    )
    with pytest.raises(ValueError, match="origin"):
        combine_replicates([_pmap(1), other], [1e-3, 1e-3])


def test_combining_then_converting_differs_from_averaging_gfe():
    """The reason the API returns a PMAP: -RT ln is non-linear."""
    params = GFEParams(clip_max=None)
    runs = [_pmap(1e-3), _pmap(9e-3)]
    bulk = [2e-3, 2e-3]

    combined, density = combine_replicates(runs, bulk)
    correct = pmap_to_gfe(combined, params, clip=False, scale=False,
                          bulk_probability=density)

    per_run = [
        pmap_to_gfe(g, params, clip=False, scale=False, bulk_probability=d)
        for g, d in zip(runs, bulk, strict=True)
    ]
    wrong = aggregate(per_run, "ave")

    assert not np.allclose(correct.values, wrong.values)


def test_runs_of_different_length_combine_by_density_not_counts():
    """Densities are already per-frame, so a longer run does not count for more."""
    short = Grid(spec=SPEC, values=np.full(SPEC.dims, 100.0))     # counts, 100 frames
    long_ = Grid(spec=SPEC, values=np.full(SPEC.dims, 1000.0))    # counts, 1000 frames
    as_counts, _ = combine_replicates([short, long_], [1e-3, 1e-3])

    density = Grid(spec=SPEC, values=np.full(SPEC.dims, 1.0))     # both are 1.0 /A^3
    as_density, _ = combine_replicates([density, density], [1e-3, 1e-3])

    assert np.allclose(as_density.values, 1.0)
    assert not np.allclose(as_counts.values, 1.0)
