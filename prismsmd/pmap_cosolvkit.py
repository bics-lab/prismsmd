"""Turn the occupancy grid built by CosolvKit's ``Analysis`` into a PMAP.

The probabilities come from CosolvKit's ``Analysis``; the GFE conversion, clipping and
scaling are in :mod:`prismsmd.gfe`. ``Analysis.atomic_grid_free_energy()`` is not used:
it smooths by default and takes its bulk from the volume of water, so it is a different
definition from this pipeline's ``-RT ln(P/P_bulk)`` -> clip -> scale.

Two things about ``Analysis`` are worked around here.

The origin it writes into its dx is the lower edge of the first bin, while
gridDataFormats reads an ``origin=`` as the coordinate of ``values[0,0,0]``, so every
voxel comes out shifted low by half a voxel. :func:`read_cosolvkit_dx` corrects it and
:func:`assert_grid_matches` verifies the result.

``Analysis`` also cannot be given a grid: its ``_conclude`` derives the origin and
extent from the mean box size and the mean centre of geometry, which fluctuate under
NPT, so the grid would shift from run to run and no cross-run aggregation would be
possible. :func:`fixed_grid_analysis_class` subclasses ``Analysis`` and replaces only
``_conclude``, histogramming onto a grid derived deterministically from the reference
structure; frame iteration and position collection are reused from the library.

Superposition is a per-frame CA best-fit onto the reference structure, with the fit
group selected as "atom name CA, excluding the residue name CA (a calcium ion) and the
probe residue" (:func:`fit_selection_string`). The grid centre is the CA centroid of the
reference structure, so the reference need not be centred on the origin.

Periodic-boundary handling is done on the GROMACS side with ``gmx trjconv``, and a
trajectory that skipped it is detected from a physical quantity -- the largest
intramolecular distance of the probe molecules -- rather than from a marker file
(:func:`assert_probe_molecules_are_whole`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .align import kabsch
from .grid import Grid, GridSpec, read_dx, write_dx
from .volume import box_volume_from_dimensions

__all__ = [
    "PBC_CHECK_BOX_FRACTION",
    "PBC_MARKER_SUFFIX",
    "PMAP_CENTER_ROUND_DIGITS",
    "PROBE_INTRAMOLECULAR_LIMIT",
    "GridMismatchError",
    "PMAPResult",
    "TrajectoryNotAlignedError",
    "TrajectoryNotPreprocessedError",
    "TrajectoryNotUnwrappedError",
    "apply_valid_region",
    "assert_grid_matches",
    "assert_probe_molecules_are_whole",
    "cosolvkit_grid_spec",
    "cosolvkit_reported_origin",
    "count_outside_grid",
    "counts_to_gfe",
    "fit_selection_string",
    "fixed_grid_analysis_class",
    "fixed_grid_spec",
    "frame_deviation",
    "grid_center_from_reference",
    "grid_points_within",
    "histogram_on_fixed_grid",
    "histogram_to_number_density",
    "intramolecular_limit_from_sdf",
    "max_intramolecular_distance",
    "occupied_fraction_of_valid_region",
    "pbc_marker_path",
    "pmap_grid_spec_from_reference",
    "pocket_fit_selection",
    "probe_atom_indices_in_universe",
    "read_cosolvkit_dx",
    "read_pbc_marker",
    "reference_fit_positions",
    "reference_heavy_atom_coordinates",
    "run_analysis",
    "run_fixed_grid_analyses",
    "run_fixed_grid_analysis",
    "valid_region_mask",
    "window_histograms",
    "write_pbc_marker",
    "write_pmap_dx",
]


class GridMismatchError(ValueError):
    """Raised when the grid CosolvKit produced does not match the expected one."""


def cosolvkit_bin_counts(box_size: Sequence[float], gridsize: float) -> tuple[int, int, int]:
    """The histogram bin counts ``round(box_size / gridsize)`` that ``_conclude`` uses."""
    return tuple(int(round(float(b) / gridsize)) for b in box_size)  # type: ignore[return-value]


def cosolvkit_grid_spec(
    center: Sequence[float], box_size: Sequence[float], gridsize: float
) -> GridSpec:
    """Return the grid CosolvKit builds, with its grid points as bin centres."""
    dims = cosolvkit_bin_counts(box_size, gridsize)
    # The midpoint of the bin centres is the box centre, so GridSpec.origin lands
    # exactly on the first bin centre.
    return GridSpec(center=tuple(float(c) for c in center), dims=dims, spacing=gridsize)


def cosolvkit_reported_origin(
    center: Sequence[float], box_size: Sequence[float], gridsize: float
) -> tuple[float, float, float]:
    """The origin CosolvKit writes into the dx: the lower edge of the first bin,
    half a voxel below the bin centre."""
    return tuple(
        float(center[i]) - float(box_size[i]) / 2.0 for i in range(3)
    )  # type: ignore[return-value]


#: Correction (x gridsize) that turns CosolvKit's origin into the correct bin centre.
HALF_VOXEL = 0.5


def read_cosolvkit_dx(
    path: str | Path, *, correct_half_voxel: bool = True
) -> Grid:
    """Read a dx written by CosolvKit, correcting the half-voxel shift.

    Args:
        path: the dx file.
        correct_half_voxel: True adds ``+gridsize/2`` to the origin so grid points
            refer to bin centres; False returns CosolvKit's value as written.
    """
    grid = read_dx(path)
    if not correct_half_voxel:
        return grid
    spec = grid.spec
    shifted_center = tuple(c + HALF_VOXEL * spec.spacing for c in spec.center)
    return Grid(
        spec=GridSpec(center=shifted_center, dims=spec.dims, spacing=spec.spacing),
        values=grid.values,
    )


def assert_grid_matches(
    grid: Grid | GridSpec,
    expected: GridSpec,
    *,
    tol: float = 1e-6,
    what: str = "grid",
) -> None:
    """Raise :class:`GridMismatchError` unless the grid matches ``expected``."""
    spec = grid.spec if isinstance(grid, Grid) else grid
    problems: list[str] = []
    if spec.dims != expected.dims:
        problems.append(f"counts: {spec.dims} != {expected.dims}")
    if abs(spec.spacing - expected.spacing) > tol:
        problems.append(f"delta: {spec.spacing} != {expected.spacing}")
    for i, axis in enumerate("xyz"):
        d = spec.origin[i] - expected.origin[i]
        if abs(d) > tol:
            problems.append(
                f"origin[{axis}]: {spec.origin[i]:.6f} != {expected.origin[i]:.6f}"
                f" (difference {d:+.6f} = {d / expected.spacing:+.3f} voxel)"
            )
    if problems:
        raise GridMismatchError(
            f"the {what} does not match the expectation:\n  " + "\n  ".join(problems)
            + "\nA half-voxel shift means the CosolvKit origin problem."
            " Check that it went through read_cosolvkit_dx()."
        )


def histogram_to_number_density(
    counts: Grid, *, n_frames: int, spacing: float | None = None
) -> Grid:
    """Convert a raw count grid into a number density (per A^3).

    ``Analysis._histogram`` is ``np.histogramdd``'s output accumulated over all frames,
    so the normalisation happens here::

        rho(r) = count(r) / n_frames / voxel_volume

    The division by the voxel volume is explicit, which makes the result dimensionally
    a density and therefore comparable with the bulk number density
    (:class:`prismsmd.volume.BulkReference`) at any spacing.

    The count is not divided by the probe atom count: that would give a probability
    density per atom, whose dimensions no longer match ``rho_bulk``, and would add a
    constant offset of ``+RT ln(n_atoms)`` ahead of the GFE clip.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive: {n_frames}")
    sp = counts.spec.spacing if spacing is None else spacing
    voxel_volume = sp**3
    return counts.copy_with(counts.values / (n_frames * voxel_volume))


def run_analysis(
    topology: str | Path,
    trajectory: str | Path,
    selection: str,
    *,
    gridsize: float = 0.5,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
):
    """Run CosolvKit's ``Analysis`` and return ``(counts, n_frames, n_atoms)``.

    The grid is CosolvKit's own, with the half-voxel origin shift corrected.

    Raises:
        ImportError: when cosolvkit / MDAnalysis are not installed. Note that
            ``cosolvkit.analysis`` imports pymol unconditionally at module level.
    """
    try:
        import MDAnalysis as mda
        from cosolvkit.analysis import Analysis
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "importing cosolvkit.analysis failed.\n"
            "cosolvkit/analysis.py does `import pymol` at the top of the file, so it\n"
            "cannot be imported without pymol:\n"
            "  mamba install -c conda-forge pymol-open-source"
        ) from exc

    u = mda.Universe(str(topology), str(trajectory))
    ag = u.select_atoms(selection)
    if len(ag) == 0:
        raise ValueError(f"no atoms match the selection {selection!r}")

    ana = Analysis(ag, gridsize=gridsize)
    ana.run(start=start, stop=stop, step=step)

    hist = ana._histogram  # raw counts (the output of np.histogramdd)
    counts = np.asarray(hist.grid, dtype=float)
    origin = np.asarray(hist.origin, dtype=float)

    # CosolvKit's origin is the lower bin edge. Correct it to the bin centre.
    dims = tuple(int(n) for n in counts.shape)
    center = tuple(
        origin[i] + gridsize / 2.0 + (dims[i] - 1) * gridsize / 2.0 for i in range(3)
    )
    spec = GridSpec(center=center, dims=dims, spacing=gridsize)
    return Grid(spec=spec, values=counts), int(ana.n_frames), int(len(ag))


#: Number of digits the grid centre is rounded to, so that the same target with the
#: same config always gives an identical grid even if the lowest bits of the centroid
#: move with the numpy version or the summation order.
PMAP_CENTER_ROUND_DIGITS = 3


def fixed_grid_spec(
    center: Sequence[float],
    box_size: float | Sequence[float],
    spacing: float,
    *,
    round_digits: int | None = PMAP_CENTER_ROUND_DIGITS,
) -> GridSpec:
    """Build the fixed PMAP grid, with explicit origin, spacing and point count.

    Grid points are bin centres and ``dims[i] = round(box_size[i] / spacing)``. A scalar
    ``box_size`` means a cube.
    """
    if np.isscalar(box_size):
        box = (float(box_size),) * 3  # type: ignore[arg-type]
    else:
        box = tuple(float(b) for b in box_size)  # type: ignore[arg-type]
        if len(box) != 3:
            raise ValueError(f"box_size must be 3 elements or a scalar: {box_size!r}")
    if spacing <= 0:
        raise ValueError(f"spacing must be positive: {spacing!r}")
    for b in box:
        if b <= 0:
            raise ValueError(f"box_size must be positive: {box_size!r}")

    c = tuple(float(x) for x in center)
    if len(c) != 3:
        raise ValueError(f"center must have 3 elements: {center!r}")
    if round_digits is not None:
        c = tuple(round(x, round_digits) for x in c)
    return cosolvkit_grid_spec(c, box, spacing)


#: Residue names always excluded from the superposition and grid-centre selections.
#: ``"CA"`` is the residue name of a calcium ion, excluded so it is never confused with
#: the atom name CA. Water is in the same table so that the selection rule is identical
#: on the reference side and the system side.
FIT_EXCLUDE_RESNAMES: tuple[str, ...] = ("CA", "WAT", "HOH", "SOL")


def fit_selection_string(
    probe_resname: str | None = None,
    *,
    atom_name: str = "CA",
    exclude_resnames: Sequence[str] = FIT_EXCLUDE_RESNAMES,
) -> str:
    """The MDAnalysis selection for the fit group: atom name CA, excluding the calcium
    residue, water and the probe.

    >>> fit_selection_string("A04")
    'name CA and not resname CA and not resname HOH and not resname SOL and not resname WAT and not resname A04'
    """
    parts = [f"name {atom_name}"]
    parts += [f"not resname {r}" for r in sorted(set(exclude_resnames))]
    if (probe_resname
            and probe_resname.upper() not in {r.upper() for r in exclude_resnames}):
        parts.append(f"not resname {probe_resname}")
    return " and ".join(parts)


def pocket_fit_selection(
    reference_pdb: str | Path,
    centre,
    radius: float,
    probe_resname: str | None = None,
) -> tuple[str, list[int]]:
    """Select the CA atoms lining one site, for superposing on that site alone.

    A global CA fit leaves any hinge or lobe motion of the whole molecule in place,
    which smears the density at the site even though it has nothing to do with the site.
    Fitting on the site's own CA atoms instead lowers the pocket RMSF.

    Args:
        reference_pdb: the reference structure the positions are taken from.
        centre: the centre of the site (3 coordinates).
        radius: how far from ``centre`` a CA still counts as lining the site.
        probe_resname: the probe residue name, excluded from the CA set.

    Returns:
        ``(selection, indices)``: the selection string for all CA atoms, unchanged, and
        the positions within that selection that belong to the site. Positions rather
        than residue numbers, because the reference and ``input.gro`` need not number
        residues the same way -- GROMACS renumbers from 1 when it writes a ``.gro``.
        The positions are decided once on the reference structure; choosing them per
        frame would let the very motion being removed move the set. They are returned
        as a pair so that neither side can be narrowed without the other.
    """
    import numpy as np

    from .pdbio import read_pdb

    base = fit_selection_string(probe_resname)
    centre = np.asarray(centre, dtype=float)
    if centre.shape != (3,):
        raise ValueError(f"centre has to be 3 numbers, got {centre.shape}")
    if radius <= 0:
        raise ValueError(f"radius has to be positive, got {radius}")

    # The reference CA set is built by the same rule as reference_fit_positions; two
    # separate rules would drift apart silently.
    frame = read_pdb(reference_pdb)
    mask = frame.atom_name == "CA"
    excluded = list(FIT_EXCLUDE_RESNAMES)
    if probe_resname:
        excluded.append(probe_resname)
    for rn in excluded:
        mask &= frame.res_name != rn
    coords = np.asarray(frame.coord[mask], dtype=float)
    if not len(coords):
        raise ValueError(f"{reference_pdb}: not a single CA could be selected")

    d = np.linalg.norm(coords - centre, axis=1)
    indices = [int(i) for i in np.flatnonzero(d <= radius)]
    if len(indices) < 4:
        raise ValueError(
            f"only {len(indices)} CA within {radius} A of {tuple(centre)};"
            " a rigid fit needs at least 4. Widen the radius or check the centre."
        )
    return base, indices


def grid_center_from_reference(
    reference_pdb: str | Path,
    *,
    selection: str = "CA",
    exclude_resnames: Sequence[str] = FIT_EXCLUDE_RESNAMES,
    probe_resname: str | None = None,
    round_digits: int | None = PMAP_CENTER_ROUND_DIGITS,
) -> tuple[float, float, float]:
    """Determine the grid centre deterministically from the reference structure.

    No coordinates are moved: the reference structure's CA centroid becomes the grid
    centre, which presupposes that the trajectory is fitted onto the reference.
    :func:`run_fixed_grid_analysis` checks that.

    Args:
        reference_pdb: the reference structure.
        selection: atom name, ``"CA"`` or ``"*"`` for all atoms.
        exclude_resnames: residue names to exclude; see :data:`FIT_EXCLUDE_RESNAMES`.
        probe_resname: the probe residue name, also excluded.
        round_digits: see :data:`PMAP_CENTER_ROUND_DIGITS`.
    """
    from .pdbio import read_pdb

    frame = read_pdb(reference_pdb)
    mask = np.ones(len(frame), dtype=bool)
    excluded = list(exclude_resnames)
    if probe_resname:
        excluded.append(probe_resname)
    for rn in excluded:
        mask &= frame.res_name != rn
    if selection != "*":
        mask &= frame.atom_name == selection
    if not mask.any():
        raise ValueError(
            f"{reference_pdb}: no atoms match the selection (atom_name={selection!r})"
        )
    center = frame.coord[mask].mean(axis=0)
    if round_digits is not None:
        center = np.round(center, round_digits)
    return (float(center[0]), float(center[1]), float(center[2]))


def pmap_grid_spec_from_reference(
    reference_pdb: str | Path,
    *,
    box_size: float | Sequence[float] = 80.0,
    spacing: float = 1.0,
    selection: str = "CA",
    probe_resname: str | None = None,
    round_digits: int | None = PMAP_CENTER_ROUND_DIGITS,
) -> GridSpec:
    """Determine the fixed grid from the reference structure and the config.

    The same reference and config always return an identical :class:`GridSpec`.
    """
    center = grid_center_from_reference(
        reference_pdb,
        selection=selection,
        probe_resname=probe_resname,
        round_digits=round_digits,
    )
    return fixed_grid_spec(center, box_size, spacing, round_digits=round_digits)


def histogram_on_fixed_grid(positions: np.ndarray, spec: GridSpec) -> Grid:
    """Take ``np.histogramdd`` on the fixed grid.

    ``spec.edges()`` are boundaries built from the bin centres, so the values returned
    are the occupancy counts at ``spec``'s grid points. Positions falling outside the
    grid are discarded; :func:`count_outside_grid` counts them.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    hist, _ = np.histogramdd(pos, bins=spec.edges())
    return Grid(spec=spec, values=hist)


def window_histograms(
    positions: np.ndarray,
    extra_positions: dict,
    spec: GridSpec,
    windows: Sequence[tuple[int, int]],
) -> tuple[dict, dict, dict]:
    """Histogram several frame intervals of one already-collected pass.

    ``positions`` is ``(n_frames, n_atoms, 3)`` for the primary atom and each value of
    ``extra_positions`` is one ``(n_atoms, 3)`` array per frame. Windows are half-open
    ``[start, stop)`` trajectory frame numbers; the primary atom is keyed by ``None``,
    because its atom index is only named on the way out.

    This exists so that sweeping how much of the beginning to discard costs one
    trajectory read rather than one per interval. A window reaching past the end is
    clipped, and the frame count actually used is returned.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3:
        raise ValueError(
            f"positions has to be (n_frames, n_atoms, 3), got {positions.shape}"
        )
    n_collected = int(positions.shape[0])
    counts: dict = {}
    outside: dict = {}
    n_frames: dict = {}
    for a, b in windows:
        a, b = int(a), int(b)
        if a < 0 or b <= a:
            raise ValueError(
                f"the window ({a}, {b}) is not a half-open [start, stop)"
                " interval with 0 <= start < stop"
            )
        hi = min(b, n_collected)
        if hi <= a:
            raise ValueError(
                f"the window ({a}, {b}) contains no frames:"
                f" the trajectory has {n_collected}"
            )
        n_frames[(a, b)] = hi - a
        sub = positions[a:hi].reshape(-1, 3)
        per_index = {None: histogram_on_fixed_grid(sub, spec)}
        per_outside = {None: count_outside_grid(sub, spec)}
        for key, frames in extra_positions.items():
            pos = (
                np.concatenate(frames[a:hi], axis=0)
                if frames
                else np.empty((0, 3), dtype=float)
            )
            per_index[key] = histogram_on_fixed_grid(pos, spec)
            per_outside[key] = count_outside_grid(pos, spec)
        counts[(a, b)] = per_index
        outside[(a, b)] = per_outside
    return counts, outside, n_frames


def count_outside_grid(positions: np.ndarray, spec: GridSpec) -> int:
    """Number of positions that fell outside the grid, for the provenance record."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    edges = spec.edges()
    inside = np.ones(len(pos), dtype=bool)
    for i in range(3):
        inside &= (pos[:, i] >= edges[i][0]) & (pos[:, i] <= edges[i][-1])
    return int((~inside).sum())


class TrajectoryNotAlignedError(ValueError):
    """Raised when the trajectory could not be fitted onto the reference structure.

    The fixed grid is defined in the reference structure's frame, so without the
    superposition the spatial relationship between grid and system is broken. It fires
    on the residual RMSD after fitting, whose usual causes are a trajectory whose
    protein is split across the periodic boundary, a fit group whose atom order does
    not correspond to the reference, or a reference that is a different molecule.
    """


class TrajectoryNotUnwrappedError(ValueError):
    """Molecules are split across the periodic boundary.

    Decided from a physical quantity -- the largest intramolecular atom distance of the
    probe molecules (:func:`assert_probe_molecules_are_whole`) -- and not from the
    marker file, which is provenance only: a marker does not survive copying or
    renaming, can be carried across on its own, and is no evidence of a physical fact.
    """


class ProbeDensityOffCentreError(ValueError):
    """The probe density sits far off the grid centre, i.e. the box was not centred.

    ``gmx trjconv -pbc mol`` keeps molecules whole but leaves the protein wherever it
    drifted inside the box, so after the CA fit the probes appear piled on one side of
    it. The molecules are whole and the protein fits, so only the position of the
    density shows it. The fix is ``-pbc mol -center`` with a protein centring group.
    """


#: Offset of the probe density centroid from the grid centre (A) that is worth a
#: warning. Not a rejection criterion: a probe that dwells on one face of the protein
#: legitimately produces an offset as large as a genuinely un-centred trajectory does,
#: so the two cannot be separated by a threshold. Rejection is left to
#: :func:`assert_probe_molecules_are_whole` (the cause) and
#: :func:`assert_density_in_valid_region` (the consequence).
WARN_DENSITY_CENTROID_OFFSET_A = 8.0

#: Deprecated alias, kept so existing callers do not break; it no longer rejects.
MAX_DENSITY_CENTROID_OFFSET_A = WARN_DENSITY_CENTROID_OFFSET_A

#: Smallest acceptable fraction of the valid region that carries any probe density.
#: An un-centred trajectory puts the valid region in bulk solvent, where it holds none
#: at all; a healthy run puts density on of order 10% of it.
MIN_VALID_REGION_OCCUPIED_FRACTION = 0.01


def density_centroid(result) -> tuple[float, float, float]:
    """Occupancy-weighted centroid of a PMAP, in the grid's own frame."""
    import numpy as np

    grid = result.grid if hasattr(result, "grid") else result
    values = np.asarray(grid.values, dtype=float)
    weights = np.where(values > 0, values, 0.0)
    total = weights.sum()
    spec = grid.spec
    if total <= 0:
        raise ValueError("the PMAP is empty; there is no density to locate")
    origin = np.array(
        [c - (n - 1) * spec.spacing / 2
         for c, n in zip(spec.center, spec.dims, strict=True)]
    )
    axes = [origin[k] + np.arange(spec.dims[k]) * spec.spacing for k in range(3)]
    out = []
    for k in range(3):
        marginal = weights.sum(axis=tuple(i for i in range(3) if i != k))
        out.append(float((marginal * axes[k]).sum() / total))
    return tuple(out)  # type: ignore[return-value]


def assert_density_is_centred(
    result,
    *,
    max_offset_a: float = WARN_DENSITY_CENTROID_OFFSET_A,
    what: str = "PMAP",
    log=print,
) -> float:
    """Report how far the density centroid sits from the grid centre.

    Warns; does not raise. See :data:`WARN_DENSITY_CENTROID_OFFSET_A`.

    Returns:
        The offset in Angstrom, so callers can record it either way.
    """
    import numpy as np

    grid = result.grid if hasattr(result, "grid") else result
    centroid = density_centroid(result)
    offset = float(np.linalg.norm(np.array(centroid) - np.array(grid.spec.center)))
    if offset > max_offset_a:
        log(
            f"[pmap] {what}: the probe density centroid is {offset:.1f} A from the "
            f"grid centre (warning threshold {max_offset_a:.1f}). "
            f"centroid {tuple(round(v, 2) for v in centroid)} vs centre "
            f"{tuple(round(v, 2) for v in grid.spec.center)}. This is a warning: a "
            "probe that dwells on one face of the protein does this legitimately."
        )
    return offset


def occupied_fraction_of_valid_region(result, valid_mask, *, detail: bool = False):
    """Fraction of the valid region that carries any probe density.

    Split out from :func:`assert_density_in_valid_region` so a caller can measure
    without rejecting, which the windows of a frame sweep need: a short window
    legitimately covers less of the region.
    """
    import numpy as np

    grid = result.grid if hasattr(result, "grid") else result
    values = np.asarray(grid.values, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError(
            f"valid_mask shape {mask.shape} does not match the grid {values.shape}"
        )
    n_valid = int(mask.sum())
    if n_valid == 0:
        raise ValueError("the valid region is empty; nothing to check")
    occupied = int((mask & (values > 0)).sum())
    fraction = occupied / n_valid
    return (fraction, occupied, n_valid) if detail else fraction


def assert_density_in_valid_region(
    result,
    valid_mask,
    *,
    min_fraction: float = MIN_VALID_REGION_OCCUPIED_FRACTION,
    what: str = "PMAP",
) -> float:
    """Require that probe density actually reaches the protein.

    Args:
        result: the PMAP (or an object with ``.grid``).
        valid_mask: boolean array, True where the grid point is within the valid
            distance of the reference structure.
        min_fraction: see :data:`MIN_VALID_REGION_OCCUPIED_FRACTION`.
        what: label for the message.

    Returns:
        The occupied fraction, so callers can log it when it passes.

    Raises:
        ProbeDensityOffCentreError: when the valid region is essentially empty.
    """
    fraction, occupied, n_valid = occupied_fraction_of_valid_region(
        result, valid_mask, detail=True
    )
    if fraction < min_fraction:
        raise ProbeDensityOffCentreError(
            f"{what}: only {occupied} of {n_valid} valid grid points "
            f"({100 * fraction:.3f} %) carry any probe density, below the limit of "
            f"{100 * min_fraction:.1f} %. The probe never reached the protein, which "
            "is what happens when the trajectory is not re-imaged around it -- "
            "gmx trjconv needs '-pbc mol -center'."
        )
    return fraction


#: Old name, from when the marker file decided this. Kept as an alias.
TrajectoryNotPreprocessedError = TrajectoryNotUnwrappedError


#: Upper bound on the largest intramolecular atom distance of a probe molecule (A);
#: above it, the molecule counts as split. Small cosolvent probes measure a few A
#: across (benzene's para H-H, at 4.97 A, is the widest of the usual set), so this
#: leaves a factor of two of headroom, while a molecule split without ``-pbc mol`` ends
#: up a box edge apart. :func:`intramolecular_limit_from_sdf` derives the bound from a
#: specific molecule when a probe is much larger.
PROBE_INTRAMOLECULAR_LIMIT = 10.0

#: The coefficient in the requirement "bound < shortest box edge x this". A split
#: molecule's atoms end up at least (shortest edge - molecular extent) apart, so a
#: larger bound would miss splits and the test would be meaningless.
PBC_CHECK_BOX_FRACTION = 0.4


def intramolecular_limit_from_sdf(
    sdf: str | Path, *, margin_factor: float = 2.0
) -> float:
    """Measure the largest intramolecular distance of an sdf conformer and multiply it
    by ``margin_factor`` to make the bound."""
    lines = Path(sdf).read_text().splitlines()
    if len(lines) < 4:
        raise ValueError(f"{sdf}: too short to be an sdf")
    n_atoms = int(lines[3][0:3])
    coords = np.array(
        [
            [float(lines[4 + i][0:10]), float(lines[4 + i][10:20]),
             float(lines[4 + i][20:30])]
            for i in range(n_atoms)
        ],
        dtype=float,
    )
    return float(max_intramolecular_distance(coords[None, :, :]) * margin_factor)


def max_intramolecular_distance(positions: np.ndarray) -> float:
    """Largest intramolecular atom distance (A) from ``(n_molecules, n_atoms, 3)``.

    Takes all atom-pair distances per molecule and returns the maximum of those maxima.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"must be (n_molecules, n_atoms, 3): {pos.shape}")
    if pos.shape[1] < 2:
        return 0.0
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    return float(np.sqrt((diff**2).sum(axis=-1)).max())


def assert_probe_molecules_are_whole(
    positions: np.ndarray,
    *,
    limit: float = PROBE_INTRAMOLECULAR_LIMIT,
    box_edges: Sequence[float] | None = None,
    frame: int | None = None,
) -> float:
    """Confirm from the coordinates that probe molecules are not split across the
    periodic boundary.

    Args:
        positions: probe coordinates as ``(n_molecules, n_atoms, 3)``.
        limit: upper bound on the largest intramolecular atom distance (A).
        box_edges: the three box edge lengths (A). Passing them cross-checks that the
            bound is small enough relative to the box (:data:`PBC_CHECK_BOX_FRACTION`).
        frame: frame number to include in the error message.

    Returns:
        The observed largest intramolecular atom distance (A).

    Raises:
        TrajectoryNotUnwrappedError: when the bound is exceeded.
        ValueError: when the bound is too large relative to the box for the test to
            mean anything.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive: {limit!r}")
    if box_edges is not None:
        shortest = min(float(b) for b in box_edges[:3])
        if shortest > 0 and limit > shortest * PBC_CHECK_BOX_FRACTION:
            raise ValueError(
                f"the intramolecular distance bound {limit:.1f} A is too large relative to the"
                f" shortest box edge {shortest:.1f} A (require bound < shortest edge x {PBC_CHECK_BOX_FRACTION}).\n"
                "With this setting a split across the periodic boundary would be missed, so the test is meaningless."
            )

    worst = max_intramolecular_distance(positions)
    if worst > limit:
        where = "" if frame is None else f" (frame {frame})"
        raise TrajectoryNotUnwrappedError(
            f"the largest intramolecular atom distance of the probe molecules is {worst:.1f} A,"
            f" exceeding the bound of {limit:.1f} A{where}.\n"
            "The molecule is split across the periodic boundary, so the PBC handling"
            " was not done, and a fixed-grid PMAP presupposes coordinates in which"
            " molecules are whole.\n"
            "How to fix it:\n"
            "  gmx trjconv -s <production>.tpr -f <traj>.xtc -o <traj>_pbcmol.xtc"
            " -n index.ndx -pbc mol\n"
            "  (the postprocess.sh prismsmd.md.runner writes runs exactly this)\n"
            "Then pass the -pbc mol trajectory to PMAP.\n"
            "Raise intramolecular_limit only when the bound is genuinely too small for"
            " the size of the molecule."
        )
    return worst


#: Suffix of the provenance file the preprocessing script leaves behind. It records
#: which command was applied and is never used for decisions; see
#: :class:`TrajectoryNotUnwrappedError`.
PBC_MARKER_SUFFIX = ".pbcmol.json"


def pbc_marker_path(trajectory: str | Path) -> Path:
    """Path of ``<trajectory>.pbcmol.json``."""
    return Path(str(trajectory) + PBC_MARKER_SUFFIX)


def write_pbc_marker(trajectory: str | Path, command: str) -> Path:
    """Record which command did the PBC handling. Provenance only."""
    import json

    path = pbc_marker_path(trajectory)
    path.write_text(
        json.dumps(
            {"pbc": "mol", "command": command, "trajectory": str(trajectory)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return path


def read_pbc_marker(trajectory: str | Path) -> dict:
    """Read the provenance file, returning an empty dict if it is absent or unreadable.

    Never raises, and never stops an analysis; only the physical test does that.
    """
    import json

    path = pbc_marker_path(trajectory)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def reference_fit_positions(
    reference_pdb: str | Path,
    *,
    selection: str = "CA",
    exclude_resnames: Sequence[str] = FIT_EXCLUDE_RESNAMES,
    probe_resname: str | None = None,
    indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Extract the reference coordinates (N, 3) used for fitting.

    The same selection rule as :func:`grid_center_from_reference`, so
    :func:`fit_selection_string` on the MDAnalysis side gives a matching atom order.
    ``indices`` narrows the set by position within the selection, never by residue
    number, which the reference and the system need not share.
    """
    from .pdbio import read_pdb

    frame = read_pdb(reference_pdb)
    mask = frame.atom_name == selection if selection != "*" else np.ones(len(frame), bool)
    excluded = list(exclude_resnames)
    if probe_resname:
        excluded.append(probe_resname)
    for rn in excluded:
        mask &= frame.res_name != rn
    if not mask.any():
        raise ValueError(f"{reference_pdb}: no atoms match the selection {selection!r}")
    out = np.asarray(frame.coord[mask], dtype=float)
    if indices is not None:
        idx = np.asarray(sorted({int(i) for i in indices}))
        if idx.size == 0:
            raise ValueError(f"{reference_pdb}: indices is empty")
        if idx.max() >= len(out) or idx.min() < 0:
            raise ValueError(
                f"{reference_pdb}: index {idx.max()} is out of range for {len(out)} CA atoms")
        out = out[idx]
    return out


def frame_deviation(positions: np.ndarray, reference: np.ndarray) -> float:
    """RMSD (A) without superposition.

    This quantity judges whether the coordinates are already fitted, so it deliberately
    does not least-squares superpose first.
    """
    pos = np.asarray(positions, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if pos.shape != ref.shape:
        raise ValueError(f"atom counts differ: {pos.shape} vs {ref.shape}")
    return float(np.sqrt(np.mean(np.sum((pos - ref) ** 2, axis=1))))


@dataclass
class PMAPResult:
    """The occupancy histogram for one run and one probe atom.

    Attributes:
        counts: raw counts on the fixed grid, accumulated over all frames.
        n_frames: frames iterated.
        n_atoms: probe atoms selected.
        outside: positions that fell outside the grid. Nonzero is not an anomaly; the
            grid only covers a box around the target.
        max_fit_deviation: maximum residual RMSD (A) after the CA best-fit, or ``None``
            when no fit was done.
        mean_box_volume: mean box volume (A^3) over the analysed frames, used as the
            denominator of the bulk reference density.
        max_intramolecular_distance: the observed largest intramolecular atom distance
            of the probes (A), or ``None`` when the PBC-split test was not run.
    """

    counts: Grid
    n_frames: int
    n_atoms: int
    outside: int = 0
    max_fit_deviation: float | None = None
    mean_box_volume: float | None = None
    max_intramolecular_distance: float | None = None

    def number_density(self) -> Grid:
        """Convert the counts to a number density (per A^3)."""
        return histogram_to_number_density(self.counts, n_frames=self.n_frames)

    def bulk_reference(self, excluded_volume: float):
        """The bulk reference ``n_atoms / (mean_box_volume - excluded_volume)``
        (:class:`prismsmd.volume.BulkReference`)."""
        from .volume import BulkReference

        if self.mean_box_volume is None:
            raise ValueError(
                "no box volume available, so the bulk reference cannot be built."
                " Check that the trajectory carries box information (ts.dimensions)."
            )
        return BulkReference(
            n_atoms=self.n_atoms,
            box_volume=self.mean_box_volume,
            excluded_volume=float(excluded_volume),
        )

    def to_dict(self) -> dict:
        """Return the result as a plain dict."""
        return {
            "grid": self.counts.spec.to_dict(),
            "n_frames": self.n_frames,
            "n_atoms": self.n_atoms,
            "outside": self.outside,
            "total_counts": float(self.counts.values.sum()),
            "max_fit_deviation": self.max_fit_deviation,
            "mean_box_volume": self.mean_box_volume,
            "max_intramolecular_distance": self.max_intramolecular_distance,
        }


def fixed_grid_analysis_class():
    """Return a fixed-grid subclass of ``cosolvkit.analysis.Analysis``.

    The class definition is inside a function because importing ``cosolvkit`` entails
    importing pymol. Only ``_conclude`` and the front of ``_single_frame`` are
    replaced: the PBC-split test and the CA best-fit happen before the library's own
    collection, and the histogram is taken on the fixed :class:`GridSpec` passed to the
    constructor instead of one derived from the mean box size. ``_histogram`` and
    ``_density`` are built with ``gridData.Grid(..., edges=...)``, which makes gridData
    treat the origin as a bin centre and avoids the half-voxel shift.
    """
    from cosolvkit.analysis import Analysis, _grid_density
    from gridData import Grid as GDGrid

    class FixedGridAnalysis(Analysis):  # type: ignore[misc,valid-type]
        """An :class:`Analysis` that takes the occupancy histogram on a fixed grid."""

        def __init__(
            self,
            atomgroup,
            spec: GridSpec,
            *,
            fit_group=None,
            fit_reference: np.ndarray | None = None,
            fit_tolerance: float = 10.0,
            align: bool = True,
            probe_molecule_group=None,
            n_atoms_per_probe: int | None = None,
            intramolecular_limit: float = PROBE_INTRAMOLECULAR_LIMIT,
            pbc_check_stride: int = 1,
            extra_groups: dict | None = None,
            windows: Sequence[tuple[int, int]] | None = None,
            **kwargs,
        ):
            super().__init__(atomgroup, gridsize=spec.spacing, **kwargs)
            self.spec = spec
            # Frame intervals to histogram in the same pass, on top of the full-length
            # map. Indices are trajectory frame numbers, half-open [start, stop), which
            # is why run() must not be given a start/step that shifts the
            # correspondence (checked in _prepare).
            self._windows = tuple(
                (int(a), int(b)) for a, b in (windows or ())
            )
            for a, b in self._windows:
                if a < 0 or b <= a:
                    raise ValueError(
                        f"the window ({a}, {b}) is not a half-open [start, stop)"
                        " interval with 0 <= start < stop"
                    )
            self.window_counts: dict[tuple[int, int], dict[int | None, Grid]] = {}
            self.window_outside: dict[tuple[int, int], dict[int | None, int]] = {}
            self.window_frames: dict[tuple[int, int], int] = {}
            # Additional probe atoms counted in the same pass, as
            # ``{atom_index: AtomGroup}`` excluding the primary atomgroup. Reading the
            # frames is the expensive part; collecting one more position array per
            # frame is not.
            self._extra_groups = dict(extra_groups or {})
            self._extra_positions: dict[int, list] = {}
            self.extra_counts: dict[int, Grid] = {}
            self.extra_outside: dict[int, int] = {}
            # Held here rather than relying on cosolvkit's internal attribute names.
            self._universe = atomgroup.universe
            self._fit_group = fit_group
            self._fit_reference = (
                None if fit_reference is None else np.asarray(fit_reference, float)
            )
            self._fit_tolerance = float(fit_tolerance)
            # Superposition is on by default, matching run_fixed_grid_analysis, so
            # using the class directly never silently runs without it.
            self._align = bool(align)
            if self._align and (fit_group is None or fit_reference is None):
                raise ValueError(
                    "align=True requires fit_group and fit_reference."
                    " To count already-superposed coordinates, pass align=False."
                )
            self._probe_molecule_group = probe_molecule_group
            self._n_atoms_per_probe = (
                None if n_atoms_per_probe is None else int(n_atoms_per_probe)
            )
            self._intramolecular_limit = float(intramolecular_limit)
            self._pbc_check_stride = max(1, int(pbc_check_stride))
            self._checked_frames = 0
            self.max_intramolecular_distance: float | None = None
            self.fit_deviations: list[float] = []
            self.box_volumes: list[float] = []
            self.counts: Grid | None = None
            self.outside = 0

        def _prepare(self):
            super()._prepare()
            if self._windows:
                # A window is a trajectory frame number, so the pass has to start at
                # frame 0 and take every frame. Silently sliding the correspondence
                # would put the density of the wrong time range in the output.
                start = getattr(self, "start", None)
                step = getattr(self, "step", None)
                if start not in (None, 0) or step not in (None, 1):
                    raise ValueError(
                        f"windows are trajectory frame numbers, so run(start={start},"
                        f" step={step}) would not correspond to them."
                        " Pass start=0/None and step=1/None, or drop the windows."
                    )
            self.fit_deviations = []
            self.box_volumes = []
            self.max_intramolecular_distance = None
            self._checked_frames = 0
            self._extra_positions = {k: [] for k in self._extra_groups}

        def _probe_molecule_positions(self):
            """Reshape the probe coordinates into ``(n_molecules, n_atoms, 3)``."""
            pos = np.asarray(self._probe_molecule_group.positions, dtype=float)
            n_per = self._n_atoms_per_probe
            if n_per is None or n_per <= 0 or len(pos) % n_per != 0:
                raise ValueError(
                    f"the probe atom count {len(pos)} is not divisible by {n_per} atoms"
                    " per molecule. Either the atom selection or n_atoms_per_probe is wrong."
                )
            return pos.reshape(-1, n_per, 3)

        def _single_frame(self):
            # (a) Box volume, the denominator of the bulk reference density. It
            #     fluctuates under NPT, so every frame is collected.
            dims = getattr(self._universe, "dimensions", None)
            if dims is not None and float(dims[0]) > 0:
                self.box_volumes.append(box_volume_from_dimensions(dims))

            # (b) PBC-split test. A rigid transform does not change intramolecular
            #     distances, so measuring before or after the fit is equivalent.
            if self._probe_molecule_group is not None and (
                self._checked_frames % self._pbc_check_stride == 0
            ):
                worst = assert_probe_molecules_are_whole(
                    self._probe_molecule_positions(),
                    limit=self._intramolecular_limit,
                    box_edges=None if dims is None else dims[:3],
                    frame=getattr(self, "_frame_index", None),
                )
                self.max_intramolecular_distance = max(
                    worst, self.max_intramolecular_distance or 0.0
                )
            if self._probe_molecule_group is not None:
                self._checked_frames += 1

            # (c) CA best-fit onto the reference, by the Kabsch method, applied to the
            #     whole system. It has to happen before super()._single_frame(), or
            #     unfitted coordinates get collected.
            if self._align:
                sup = kabsch(
                    np.asarray(self._fit_group.positions, dtype=float),
                    self._fit_reference,
                )
                all_atoms = self._universe.atoms
                all_atoms.positions = sup.apply(all_atoms.positions)

            super()._single_frame()          # the library's collection, unchanged

            # (d) Additional atoms, collected in the same pass. After the fit, like the
            #     library's own collection, or these coordinates would be in a
            #     different frame of reference from the primary atom's.
            for key, group in self._extra_groups.items():
                self._extra_positions[key].append(
                    np.array(group.positions, dtype=float)
                )

            if self._fit_group is not None and self._fit_reference is not None:
                # Residual RMSD after the fit. With align=True this tests whether it is
                # the same protein, not the quality of the superposition.
                self.fit_deviations.append(
                    frame_deviation(self._fit_group.positions, self._fit_reference)
                )

        def _conclude(self):
            self._positions = np.array(self._positions, dtype=float)
            # Kept as reference values; not used for the grid.
            self._box_size = np.mean(self._dimensions, axis=0)
            self._center = np.mean(self._centers, axis=0)

            if self.fit_deviations:
                worst = max(self.fit_deviations)
                if worst > self._fit_tolerance:
                    raise TrajectoryNotAlignedError(
                        f"the maximum RMSD against the reference structure is {worst:.2f} A,"
                        f" exceeding the tolerance of {self._fit_tolerance:.2f} A"
                        f" (CA best-fit {'applied' if self._align else 'not applied'}).\n"
                        "The fixed grid is defined in the reference structure's frame, so if the"
                        " superposition does not hold, the spatial relationship between grid and"
                        " system is broken.\n"
                        "Possible causes:\n"
                        "  * gmx trjconv -pbc mol was not applied and the protein is"
                        "split across the periodic boundary\n"
                        "  * the atom order of the fit group does not correspond to the reference\n"
                        "  * the reference and the system's protein are different molecules"
                    )

            positions = self._get_positions()
            self.outside = count_outside_grid(positions, self.spec)
            self.counts = histogram_on_fixed_grid(positions, self.spec)

            # The additional atoms go on the same fixed grid, so per-atom minima and
            # cross-run aggregation can be taken across them without further checks.
            for key, frames in self._extra_positions.items():
                pos = (
                    np.concatenate(frames, axis=0)
                    if frames
                    else np.empty((0, 3), dtype=float)
                )
                self.extra_outside[key] = count_outside_grid(pos, self.spec)
                self.extra_counts[key] = histogram_on_fixed_grid(pos, self.spec)

            # Per-window histograms, from the positions already in memory.
            if self._windows:
                (
                    self.window_counts,
                    self.window_outside,
                    self.window_frames,
                ) = window_histograms(
                    self._positions, self._extra_positions, self.spec, self._windows
                )

            hist = self.counts.values
            edges = self.spec.edges()
            # Passing edges= makes gridData's origin a bin centre, so the half-voxel
            # shift of the native origin= route does not occur.
            self._histogram = GDGrid(hist, edges=edges)
            self._density = GDGrid(_grid_density(hist), edges=edges)

        def results_by_index(self, primary: int) -> dict:
            """``{atom_index: PMAPResult}`` for the primary atom and every extra one.

            Every entry shares the frame count, the fit deviation, the box volume and
            the intramolecular distance, all of which came from one pass; only the
            counts, the atom count and the outside count differ.
            """
            base = self.result()
            out = {int(primary): base}
            for key, counts in self.extra_counts.items():
                out[int(key)] = PMAPResult(
                    counts=counts,
                    n_frames=base.n_frames,
                    n_atoms=int(self._extra_groups[key].n_atoms),
                    outside=self.extra_outside[key],
                    max_fit_deviation=base.max_fit_deviation,
                    mean_box_volume=base.mean_box_volume,
                    max_intramolecular_distance=base.max_intramolecular_distance,
                )
            return out

        def results_by_window(self, primary: int) -> dict:
            """``{(start, stop): {atom_index: PMAPResult}}`` for every window.

            The fit deviation, the box volume and the intramolecular distance are
            whole-pass quantities and are carried over unchanged.
            """
            base = self.result()
            out: dict[tuple[int, int], dict[int, PMAPResult]] = {}
            for window, per_index in self.window_counts.items():
                n_frames = self.window_frames[window]
                entries: dict[int, PMAPResult] = {}
                for key, counts in per_index.items():
                    index = int(primary) if key is None else int(key)
                    n_atoms = (
                        int(self._n_atoms) if key is None
                        else int(self._extra_groups[key].n_atoms)
                    )
                    entries[index] = PMAPResult(
                        counts=counts,
                        n_frames=int(n_frames),
                        n_atoms=n_atoms,
                        outside=self.window_outside[window][key],
                        max_fit_deviation=base.max_fit_deviation,
                        mean_box_volume=base.mean_box_volume,
                        max_intramolecular_distance=base.max_intramolecular_distance,
                    )
                out[window] = entries
            return out

        def result(self) -> PMAPResult:
            """The :class:`PMAPResult` of the primary atom group."""
            if self.counts is None:
                raise RuntimeError("result() cannot be used before calling run()")
            return PMAPResult(
                counts=self.counts,
                n_frames=int(self.n_frames),
                n_atoms=int(self._n_atoms),
                outside=self.outside,
                max_fit_deviation=(
                    max(self.fit_deviations) if self.fit_deviations else None
                ),
                mean_box_volume=(
                    float(np.mean(self.box_volumes)) if self.box_volumes else None
                ),
                max_intramolecular_distance=self.max_intramolecular_distance,
            )

    return FixedGridAnalysis


def probe_atom_indices_in_universe(
    universe,
    resname: str,
    atom_index: int,
    *,
    order=None,
) -> list[int]:
    """Return the global index of the ``atom_index``-th atom of each probe residue.

    CosolvKit/OpenFF rebuild atom names, so the 0-based index within the molecule is the
    primary information and names are not used for selection.

    Args:
        universe: the MDAnalysis Universe.
        resname: the probe residue name.
        atom_index: the 0-based index within the molecule.
        order: :class:`prismsmd.probeorder.ProbeAtomOrder`. Passing it cross-checks the
            selected atom's name and element against the table, and raises if the index
            table and OpenFF's naming disagree.
    """
    ag = universe.select_atoms(f"resname {resname}")
    if len(ag) == 0:
        raise ValueError(f"no atoms with resname {resname}")

    picked: list[int] = []
    for res in ag.residues:
        atoms = res.atoms
        if atom_index >= len(atoms):
            raise ValueError(
                f"a residue with resname {resname} has only {len(atoms)} atoms, so"
                f" index {atom_index} cannot be taken"
            )
        picked.append(int(atoms[atom_index].index))

    if order is not None:
        expected_name = order.openff_names[atom_index]
        expected_element = order.elements[atom_index]
        for idx in picked:
            atom = universe.atoms[idx]
            if str(atom.name) != str(expected_name):
                raise ValueError(
                    f"the atom name at index {atom_index} of probe {resname} is"
                    f" {atom.name!r}, differing from the predicted OpenFF name {expected_name!r}.\n"
                    "The index table (ProbeAtomOrder) and the system's atom order may"
                    "disagree."
                )
            element = _element_of(atom)
            if element is not None and element != expected_element:
                raise ValueError(
                    f"the element at index {atom_index} of probe {resname} is {element!r},"
                    f" differing from the expected {expected_element!r}"
                )
    return picked


def _element_of(atom) -> str | None:
    """Infer the element symbol of an MDAnalysis atom, or ``None`` if unavailable.

    The name is tried first. A ``.gro`` carries no element column, so MDAnalysis
    guesses ``type`` from the name and that guess silently drops the second letter of
    some two-letter elements (``Br1x`` comes out as boron). The OpenFF name is
    ``{element}{serial}x`` by construction and the caller has already checked it, so it
    is both the more reliable and the already-validated source.
    """
    from .probeprep import element_from_atom_name

    name = str(getattr(atom, "name", "") or "")
    if name:
        try:
            return element_from_atom_name(name)
        except ValueError:
            pass
    for attr in ("element", "type"):
        value = getattr(atom, attr, None)
        if not value:
            continue
        text = str(value).strip()
        if len(text) >= 2 and text[:2].capitalize() in ("Cl", "Br", "Si"):
            return text[:2].capitalize()
        if text[0].isalpha():
            return text[0].upper()
    return None


def run_fixed_grid_analyses(
    topology: str | Path,
    trajectory: str | Path,
    spec: GridSpec,
    *,
    resname: str,
    atom_indices: Sequence[int],
    order=None,
    reference_pdb: str | Path | None = None,
    fit_selection: str | None = None,
    fit_indices: Sequence[int] | None = None,
    fit_tolerance: float = 10.0,
    align: bool = True,
    check_probe_molecules: bool = True,
    intramolecular_limit: float = PROBE_INTRAMOLECULAR_LIMIT,
    pbc_check_stride: int = 1,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    windows: Sequence[tuple[int, int]] | None = None,
    verbose: bool = False,
) -> dict[int, PMAPResult] | tuple[dict[int, PMAPResult], dict]:
    """Take the occupancy histograms of several probe atoms in one pass.

    Reading the frames dominates the cost, so the atoms are collected together and the
    histograms split at the end. Every :class:`PMAPResult` returned is on the same fixed
    grid and shares the frame count, fit deviation, box volume and intramolecular
    distance; only the counts, the atom count and the outside count differ.

    The arguments are those of :func:`run_fixed_grid_analysis`, with ``atom_index``
    replaced by ``atom_indices`` and ``windows`` added.

    Returns:
        ``{atom_index: PMAPResult}``, or, when ``windows`` is given, that dict paired
        with ``{(start, stop): {atom_index: PMAPResult}}``.
    """
    try:
        import MDAnalysis as mda
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("MDAnalysis is missing") from exc
    try:
        cls = fixed_grid_analysis_class()
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "importing cosolvkit.analysis failed.\n"
            "cosolvkit/analysis.py does `import pymol` at the top of the file, so it\n"
            "cannot be imported without pymol:\n"
            "  mamba install -c conda-forge pymol-open-source"
        ) from exc

    u = mda.Universe(str(topology), str(trajectory))
    wanted = [int(i) for i in atom_indices]
    if not wanted:
        raise ValueError("atom_indices is empty")
    if len(set(wanted)) != len(wanted):
        raise ValueError(f"atom_indices contains duplicates: {wanted}")
    groups = {}
    for want in wanted:
        picked = probe_atom_indices_in_universe(u, resname, want, order=order)
        if not picked:
            # Analysis.__init__ calls sys.exit(1) on zero atoms rather than raising, so
            # stop here first.
            raise ValueError(
                f"no atom could be selected for resname {resname} index {want}"
            )
        groups[want] = u.atoms[picked]
    primary, *rest = wanted
    ag = groups[primary]

    fit_group = None
    fit_reference = None
    if reference_pdb is not None:
        if fit_selection is None:
            fit_selection = fit_selection_string(resname)
        # Match the counts on the full CA set first, then narrow by position. The
        # other order cannot tell a bad narrowing apart from a protein mismatch.
        fit_reference = reference_fit_positions(reference_pdb, probe_resname=resname)
        fit_group = u.select_atoms(fit_selection)
        if len(fit_group) != len(fit_reference):
            raise TrajectoryNotAlignedError(
                f"the fit group atom counts do not match: system {len(fit_group)} vs"
                f" reference {len(fit_reference)} (selection={fit_selection!r}).\n"
                "Check that the reference and the system's protein are the same."
            )
        if fit_indices is not None:
            # The full CA sets already match, so narrowing both by the same positions
            # keeps them aligned.
            idx = sorted({int(i) for i in fit_indices})
            if not idx or idx[-1] >= len(fit_reference) or idx[0] < 0:
                raise ValueError(
                    f"fit_indices is out of range for {len(fit_reference)} CA atoms: "
                    f"{idx[:3]}...{idx[-1:] if idx else ''}")
            fit_group = fit_group[idx]
            fit_reference = fit_reference[idx]
    elif align:
        raise ValueError(
            "align=True requires reference_pdb (the CA best-fit onto the reference)."
            " For an already fitted trajectory, use align=False."
        )

    # The PBC-split test needs all probe atoms: a group of one atom per molecule cannot
    # measure an intramolecular distance. The atoms per molecule come from the residue.
    probe_group = None
    n_atoms_per_probe = None
    if check_probe_molecules:
        probe_group = u.select_atoms(f"resname {resname}")
        counts_per_residue = {len(res.atoms) for res in probe_group.residues}
        if len(counts_per_residue) != 1:
            raise ValueError(
                f"the residues with resname {resname} do not all have the same atom count:"
                f" {sorted(counts_per_residue)}. They should be the same molecule, so the system is broken."
            )
        n_atoms_per_probe = counts_per_residue.pop()

    ana = cls(
        ag,
        spec,
        fit_group=fit_group,
        fit_reference=fit_reference,
        fit_tolerance=fit_tolerance,
        align=align,
        probe_molecule_group=probe_group,
        n_atoms_per_probe=n_atoms_per_probe,
        intramolecular_limit=intramolecular_limit,
        pbc_check_stride=pbc_check_stride,
        extra_groups={k: groups[k] for k in rest},
        windows=windows,
        verbose=verbose,
    )
    ana.run(start=start, stop=stop, step=step)
    results = ana.results_by_index(primary)
    # Every grid produced matches the fixed grid that was passed in.
    for index, res in results.items():
        assert_grid_matches(
            res.counts, spec, what=f"PMAP fixed grid (index {index})"
        )
    if not windows:
        return results
    # With windows the return value gains a second element rather than changing shape,
    # so existing callers keep working unchanged.
    by_window = ana.results_by_window(primary)
    for window, entries in by_window.items():
        for index, res in entries.items():
            assert_grid_matches(
                res.counts, spec,
                what=f"PMAP fixed grid (index {index}, frames {window[0]}-{window[1]})",
            )
    return results, by_window


def run_fixed_grid_analysis(
    topology: str | Path,
    trajectory: str | Path,
    spec: GridSpec,
    *,
    resname: str,
    atom_index: int,
    order=None,
    reference_pdb: str | Path | None = None,
    fit_selection: str | None = None,
    fit_indices: Sequence[int] | None = None,
    fit_tolerance: float = 10.0,
    align: bool = True,
    check_probe_molecules: bool = True,
    intramolecular_limit: float = PROBE_INTRAMOLECULAR_LIMIT,
    pbc_check_stride: int = 1,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    verbose: bool = False,
) -> PMAPResult:
    """Take the occupancy histogram of one probe atom on a fixed grid.

    Args:
        topology: the system topology.
        trajectory: the trajectory to analyse.
        spec: the fixed grid, built by :func:`pmap_grid_spec_from_reference`.
        resname: the probe residue name.
        atom_index: the 0-based index of the atom within the molecule. Atom names are
            not used.
        order: :class:`prismsmd.probeorder.ProbeAtomOrder`, to cross-check the atom
            name and element.
        reference_pdb: reference structure for the superposition; required when
            ``align=True``.
        fit_selection: MDAnalysis selection for the fit group. ``None`` lets
            :func:`fit_selection_string` build it.
        fit_indices: positions within the fit group to narrow it to, as
            :func:`pocket_fit_selection` returns.
        fit_tolerance: maximum residual RMSD (A) after fitting.
        align: True does a per-frame CA best-fit onto the reference; False assumes the
            trajectory is already fitted and only checks the residual RMSD.
        check_probe_molecules: measure the largest intramolecular distance of the probe
            molecules per frame and detect splits across the periodic boundary.
        intramolecular_limit: the bound for that test (A).
        pbc_check_stride: frame interval for the split test; 1 means every frame.
        start, stop, step: the frame range passed to ``run()``.
        verbose: passed through to ``Analysis``.

    Raises:
        ImportError: when cosolvkit / MDAnalysis are not installed.
        TrajectoryNotUnwrappedError: when probe molecules are split across the
            periodic boundary.
        TrajectoryNotAlignedError: when the residual RMSD after fitting exceeds
            ``fit_tolerance``.
    """
    try:
        import MDAnalysis as mda
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("MDAnalysis is missing") from exc
    try:
        cls = fixed_grid_analysis_class()
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "importing cosolvkit.analysis failed.\n"
            "cosolvkit/analysis.py does `import pymol` at the top of the file, so it\n"
            "cannot be imported without pymol:\n"
            "  mamba install -c conda-forge pymol-open-source"
        ) from exc

    u = mda.Universe(str(topology), str(trajectory))
    indices = probe_atom_indices_in_universe(u, resname, atom_index, order=order)
    if not indices:
        # Analysis.__init__ calls sys.exit(1) on zero atoms rather than raising, so
        # stop here first.
        raise ValueError(f"no atom could be selected for resname {resname} index {atom_index}")
    ag = u.atoms[indices]

    fit_group = None
    fit_reference = None
    if reference_pdb is not None:
        if fit_selection is None:
            fit_selection = fit_selection_string(resname)
        # Match the counts on the full CA set first, then narrow by position.
        fit_reference = reference_fit_positions(reference_pdb, probe_resname=resname)
        fit_group = u.select_atoms(fit_selection)
        if len(fit_group) != len(fit_reference):
            raise TrajectoryNotAlignedError(
                f"the fit group atom counts do not match: system {len(fit_group)} vs"
                f" reference {len(fit_reference)} (selection={fit_selection!r}).\n"
                "Check that the reference and the system's protein are the same."
            )
        if fit_indices is not None:
            idx = sorted({int(i) for i in fit_indices})
            if not idx or idx[-1] >= len(fit_reference) or idx[0] < 0:
                raise ValueError(
                    f"fit_indices is out of range for {len(fit_reference)} CA atoms: "
                    f"{idx[:3]}...{idx[-1:] if idx else ''}")
            fit_group = fit_group[idx]
            fit_reference = fit_reference[idx]
    elif align:
        raise ValueError(
            "align=True requires reference_pdb (the CA best-fit onto the reference)."
            " For an already fitted trajectory, use align=False."
        )

    # The PBC-split test needs all probe atoms; the atoms per molecule come from the
    # residue.
    probe_group = None
    n_atoms_per_probe = None
    if check_probe_molecules:
        probe_group = u.select_atoms(f"resname {resname}")
        counts_per_residue = {len(res.atoms) for res in probe_group.residues}
        if len(counts_per_residue) != 1:
            raise ValueError(
                f"the residues with resname {resname} do not all have the same atom count:"
                f" {sorted(counts_per_residue)}. They should be the same molecule, so the system is broken."
            )
        n_atoms_per_probe = counts_per_residue.pop()

    ana = cls(
        ag,
        spec,
        fit_group=fit_group,
        fit_reference=fit_reference,
        fit_tolerance=fit_tolerance,
        align=align,
        probe_molecule_group=probe_group,
        n_atoms_per_probe=n_atoms_per_probe,
        intramolecular_limit=intramolecular_limit,
        pbc_check_stride=pbc_check_stride,
        verbose=verbose,
    )
    ana.run(start=start, stop=stop, step=step)
    result = ana.result()
    # The resulting grid matches the fixed grid that was passed in.
    assert_grid_matches(result.counts, spec, what="PMAP fixed grid")
    return result


def reference_heavy_atom_coordinates(reference_pdb: str | Path) -> np.ndarray:
    """Coordinates (N, 3) of the atoms in the ``ATOM`` records of the reference
    structure, which excludes hetero atoms and water."""
    from .pdbio import parse_pdb_string

    text = Path(reference_pdb).read_text()
    atom_lines = "".join(
        line + "\n" for line in text.splitlines() if line.startswith("ATOM")
    )
    frames = parse_pdb_string(atom_lines)
    if not frames or len(frames[0]) == 0:
        raise ValueError(f"{reference_pdb}: no ATOM records")
    return np.asarray(frames[0].coord, dtype=float)


def valid_region_mask(
    spec: GridSpec, reference_pdb: str | Path, cutoff: float = 5.0
) -> np.ndarray:
    """A mask that is True only for grid points within ``cutoff`` A of the reference
    structure -- the valid region the PMAP is restricted to."""
    coords = reference_heavy_atom_coordinates(reference_pdb)
    return grid_points_within(spec, coords, cutoff)


def grid_points_within(
    spec: GridSpec, coordinates: np.ndarray, cutoff: float
) -> np.ndarray:
    """Boolean array of whether each grid point lies within ``cutoff`` of any coordinate."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must be (N, 3)")

    axes = [spec.axis_coordinates(i) for i in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([m.ravel() for m in mesh], axis=-1)

    tree = cKDTree(coordinates)
    dist, _ = tree.query(points, k=1)
    return (dist < cutoff).reshape(spec.dims)


def apply_valid_region(
    grid: Grid, mask: np.ndarray, *, outside_value: float = 0.0
) -> Grid:
    """Fill outside the mask with ``outside_value``.

    Zero rather than a sentinel value, so that no reader has to decide whether a value
    is real. On the GFE path it makes no difference: any value at or below zero is
    replaced by ``floor_probability`` before the logarithm.
    """
    if mask.shape != grid.values.shape:
        raise ValueError(
            f"mask shape {mask.shape} differs from the grid {grid.values.shape}"
        )
    values = np.where(mask, grid.values, outside_value)
    return grid.copy_with(values)


def counts_to_gfe(
    counts: Grid,
    *,
    n_frames: int,
    params=None,
    clip: bool = True,
    scale: bool = False,
    reference: Grid | None = None,
    bulk_probability: float | None = None,
) -> Grid:
    """Raw count grid -> number density -> GFE::

        rho(r)   = count(r) / n_frames / spacing^3        [per A^3]
        rho_bulk = n_atoms / (V_box - V_excluded)         [per A^3]
        GFE(r)   = -RT ln( rho(r) / rho_bulk )

    ``rho_bulk`` is passed as ``bulk_probability``, and
    :meth:`PMAPResult.bulk_reference` supplies it::

        bulk = result.bulk_reference(excluded_volume(reference_pdb))
        gfe  = counts_to_gfe(result.counts, n_frames=result.n_frames,
                             bulk_probability=bulk.number_density)

    Only when it is ``None`` is it computed from ``params.molar`` instead.

    A voxel with zero occupancy is replaced by ``params.floor_probability`` rather than
    becoming ``+inf``, so it converges on the least favourable clipped value and the
    subsequent averaging, minimum and linear scaling never meet NaN or inf.
    """
    from .gfe import GFEParams, pmap_to_gfe

    if params is None:
        params = GFEParams()
    rho = histogram_to_number_density(counts, n_frames=n_frames)
    return pmap_to_gfe(
        rho,
        params,
        reference=reference,
        clip=clip,
        scale=scale,
        bulk_probability=bulk_probability,
    )


def write_pmap_dx(path: str | Path, grid: Grid) -> Path:
    """Write a PMAP/GFE grid to dx.

    Writing and reading back go through :func:`prismsmd.grid.write_dx` /
    :func:`prismsmd.grid.read_dx`, whose origin is a bin centre. The half-voxel
    correction of :func:`read_cosolvkit_dx` applies only to a dx written by CosolvKit;
    the two must not be mixed.
    """
    return write_dx(path, grid)
