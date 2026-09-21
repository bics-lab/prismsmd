"""Stop CosolvKit from placing probe molecules across the box faces.

:func:`periodic_safe_placement` replaces ``check_coordinates_to_add`` for the duration
of a build, so a candidate is accepted only if every atom lies inside the real box, the
original clash test passes, and that test also passes for the candidate's periodic
images.
"""

from __future__ import annotations

import contextlib
import itertools

__all__ = ["PlacementPatchError", "periodic_safe_placement"]


class PlacementPatchError(RuntimeError):
    """Raised when the placement check cannot be patched."""


def _nm(quantity) -> float:
    """A length in nm, whether it arrives as an OpenMM Quantity or a bare float."""
    value_in_unit = getattr(quantity, "value_in_unit", None)
    if value_in_unit is None:
        return float(quantity)
    import openmm.unit as openmmunit

    return float(value_in_unit(openmmunit.nanometer))


def _bounds(system):
    """Return ``(lower, upper, box)`` as plain float triples, in nm."""
    lower = tuple(float(v) for v in system.lowerBound)
    upper = tuple(float(system.upperBound[i]) for i in range(3))
    return lower, upper, tuple(u - lo for lo, u in zip(lower, upper, strict=True))


def _image_shifts(coords, lower, upper, box, reach):
    """Return the periodic shifts worth testing for this candidate.

    Only axes where the molecule comes within ``reach`` of a face can produce an
    image overlap, so the interior of the box yields no shifts at all.
    """
    per_axis = []
    for axis in range(3):
        column = coords[:, axis]
        options = [0.0]
        if float(column.min()) - lower[axis] < reach:
            options.append(box[axis])          # wraps to the upper face
        if upper[axis] - float(column.max()) < reach:
            options.append(-box[axis])         # wraps to the lower face
        per_axis.append(options)
    shifts = [s for s in itertools.product(*per_axis) if any(s)]
    return shifts


@contextlib.contextmanager
def periodic_safe_placement(*, module=None, enabled: bool = True):
    """Make CosolvKit refuse placements that cross or wrap the box faces.

    Args:
        module: the module holding ``CosolventSystem``. Defaults to
            ``cosolvkit.cosolvent_system``; injectable for testing.
        enabled: ``False`` yields without patching, to reproduce the unpatched
            behaviour for comparison.

    Raises:
        PlacementPatchError: if ``check_coordinates_to_add`` is missing.
    """
    if not enabled:
        yield
        return

    import numpy as np

    if module is None:
        try:
            from cosolvkit import cosolvent_system as module  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise PlacementPatchError(
                "cosolvkit is not importable, so placement cannot be made "
                "periodic-safe"
            ) from exc

    cls = getattr(module, "CosolventSystem", None)
    original = getattr(cls, "check_coordinates_to_add", None)
    if original is None:
        raise PlacementPatchError(
            f"{module.__name__}.CosolventSystem has no check_coordinates_to_add; "
            "CosolvKit must have restructured its placement."
        )

    def patched(self, new_coords, cosolvent_kdtree, protein_kdtree):
        coords = np.atleast_2d(np.asarray(new_coords, dtype=float))
        lower, upper, box = _bounds(self)

        # [1] Every atom inside the real box, compared against the coordinates rather
        #     than the widths CosolvKit's own check uses.
        for axis in range(3):
            column = coords[:, axis]
            if float(column.min()) < lower[axis] or float(column.max()) > upper[axis]:
                return False

        # [2] The original test, unchanged.
        if not original(self, new_coords, cosolvent_kdtree, protein_kdtree):
            return False

        # [3] The same test for the periodic images, which [1]-[2] cannot see.
        reach = _nm(self.cosolvents_radius)
        prot_reach = _nm(self.protein_radius)
        for shift in _image_shifts(coords, lower, upper, box, max(reach, prot_reach)):
            shifted = coords + np.asarray(shift)
            if cosolvent_kdtree is not None and any(
                    cosolvent_kdtree.query_ball_point(shifted, reach)):
                return False
            if protein_kdtree is not None and any(
                    protein_kdtree.query_ball_point(shifted, prot_reach)):
                return False
        return True

    cls.check_coordinates_to_add = patched
    try:
        yield
    finally:
        cls.check_coordinates_to_add = original
