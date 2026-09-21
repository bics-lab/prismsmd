"""Connected regions of a free-energy map.

A map's useful unit is rarely a single grid point: a site is a contiguous region, and
what distinguishes it from the rest of the surface is its extent as much as its depth.
Splitting a map into regions is therefore the step every ranking, report and figure
starts from, and doing it the same way each time is what makes two such outputs
comparable.

The threshold and the valid region are both arguments, because a region found at one
threshold is not the same object as a region found at another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .grid import Grid

__all__ = ["Patch", "PatchSet", "find_patches"]


@dataclass(frozen=True)
class Patch:
    """One connected region.

    Attributes:
        label: its number in :attr:`PatchSet.labels`, counting from 1.
        volume: grid points in the region. With 1 A spacing this is also A^3.
        value_min: the most negative value in it.
        centre: the region's centroid in the grid's own frame (A), unweighted.
    """

    label: int
    volume: int
    value_min: float
    centre: tuple[float, float, float]


@dataclass(frozen=True)
class PatchSet:
    """The regions of one map, with the label array they came from.

    ``labels`` is kept because selecting the samples or voxels of a region needs it;
    recomputing it elsewhere is how two pieces of code end up disagreeing about which
    points belong to which region.
    """

    patches: tuple[Patch, ...]
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.patches)

    def __iter__(self):
        return iter(self.patches)

    def by_value(self) -> tuple[Patch, ...]:
        """The regions, deepest first."""
        return tuple(sorted(self.patches, key=lambda p: p.value_min))


def find_patches(
    grid: Grid,
    valid_mask: np.ndarray | None = None,
    *,
    threshold: float = -2.5,
    min_volume: int = 1,
) -> PatchSet:
    """Split a map into the connected regions below a threshold.

    Args:
        grid: the map, normally GFE in kcal/mol.
        valid_mask: True where a grid point counts. Points outside it never join a
            region, which keeps bulk solvent from bridging two surface sites.
        threshold: a point joins a region when its value is at or below this.
        min_volume: regions smaller than this are dropped, and their label is cleared
            from :attr:`PatchSet.labels` so the two stay consistent.

    Returns:
        A :class:`PatchSet`, its regions ordered by volume, largest first.

    Raises:
        ValueError: when ``valid_mask`` does not match the grid.
    """
    from scipy import ndimage

    values = np.asarray(grid.values, dtype=float)
    if valid_mask is None:
        mask = np.ones(values.shape, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError(
                f"valid_mask shape {mask.shape} does not match the grid {values.shape}"
            )

    labels, n = ndimage.label((values <= threshold) & mask)
    spec = grid.spec
    origin = np.array(
        [c - (d - 1) * spec.spacing / 2
         for c, d in zip(spec.center, spec.dims, strict=True)]
    )
    found = []
    for label in range(1, n + 1):
        where = np.argwhere(labels == label)
        if len(where) < min_volume:
            labels[labels == label] = 0
            continue
        centre = origin + where.mean(axis=0) * spec.spacing
        found.append(Patch(
            label=label,
            volume=len(where),
            value_min=float(values[labels == label].min()),
            centre=tuple(round(float(c), 3) for c in centre),  # type: ignore[arg-type]
        ))
    found.sort(key=lambda p: p.volume, reverse=True)
    return PatchSet(patches=tuple(found), labels=labels)
