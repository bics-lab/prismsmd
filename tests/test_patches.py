"""Invariants of splitting a map into connected regions."""

import numpy as np
import pytest

from prismsmd.grid import Grid, GridSpec
from prismsmd.patches import find_patches

SPEC = GridSpec(center=(0.0, 0.0, 0.0), dims=(10, 10, 10), spacing=1.0)


def _grid(values):
    return Grid(spec=SPEC, values=np.asarray(values, dtype=float))


def _two_blobs():
    """Two regions with a gap between them, one twice the other.

    The smaller one sits at the lower indices deliberately: labels are numbered in
    scan order, so a test built the other way round would pass whether or not the
    result is sorted.
    """
    v = np.zeros(SPEC.dims)
    v[1:3, 1:3, 1:2] = -4.0        # 4 points, scanned first
    v[6:8, 6:8, 6:8] = -3.0        # 8 points, scanned second
    return _grid(v)


def test_separate_regions_are_found_separately():
    found = find_patches(_two_blobs())
    assert sorted(p.volume for p in found) == [4, 8]


def test_regions_come_back_largest_first():
    """Volume is the ranking that distinguishes a site from a passing contact.

    The larger region is the one scanned second, so this fails if the result is left
    in label order.
    """
    found = find_patches(_two_blobs())
    assert [p.volume for p in found.patches] == [8, 4]
    assert found.patches[0].label != 1


def test_the_deepest_is_not_always_the_largest():
    """The two orders disagree, which is the whole reason both are offered."""
    v = np.zeros(SPEC.dims)
    v[1:3, 1:3, 1:2] = -4.0        # 4 points, deeper, scanned first
    v[6:8, 6:8, 6:8] = -3.0        # 8 points, shallower, scanned second
    found = find_patches(_grid(v))
    assert found.patches[0].volume == 8
    assert found.by_value()[0].volume == 4


def test_touching_points_are_one_region_not_two():
    v = np.zeros(SPEC.dims)
    v[2:6, 3, 3] = -3.0
    assert [p.volume for p in find_patches(_grid(v)).patches] == [4]


def test_the_threshold_decides_what_a_region_is():
    v = np.zeros(SPEC.dims)
    v[1:3, 1:3, 1:3] = -4.0
    v[3, 2, 2] = -1.0              # joins only at a shallow threshold
    assert find_patches(_grid(v), threshold=-2.5).patches[0].volume == 8
    assert find_patches(_grid(v), threshold=-0.5).patches[0].volume == 9


def test_the_valid_region_keeps_two_sites_from_merging():
    """Without it, bulk solvent bridges regions that are not connected on the surface."""
    v = np.full(SPEC.dims, -3.0)
    mask = np.zeros(SPEC.dims, dtype=bool)
    mask[1:3, :, :] = True
    mask[6:8, :, :] = True
    assert len(find_patches(_grid(v))) == 1
    assert len(find_patches(_grid(v), mask)) == 2


def test_the_centre_is_in_the_grids_own_frame():
    v = np.zeros(SPEC.dims)
    v[0, 0, 0] = -4.0
    origin = tuple(c - (d - 1) * SPEC.spacing / 2
                   for c, d in zip(SPEC.center, SPEC.dims, strict=True))
    assert find_patches(_grid(v)).patches[0].centre == pytest.approx(origin)


def test_a_dropped_region_is_dropped_from_the_labels_too():
    """A label with no Patch would select voxels that no reported region owns."""
    found = find_patches(_two_blobs(), min_volume=5)
    assert [p.volume for p in found.patches] == [8]
    assert set(np.unique(found.labels)) == {0, found.patches[0].label}


def test_labels_select_exactly_the_reported_points():
    found = find_patches(_two_blobs())
    for patch in found:
        assert int((found.labels == patch.label).sum()) == patch.volume


def test_a_mask_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="does not match"):
        find_patches(_two_blobs(), np.ones((3, 3, 3), dtype=bool))


def test_a_map_with_nothing_below_the_threshold_gives_no_regions():
    assert len(find_patches(_grid(np.zeros(SPEC.dims)))) == 0
