"""One run's trajectory to its probe density and free-energy maps.

:func:`run_maps` is the whole procedure in one call: fix the grid from the reference
structure, histogram every mapped atom of the probe in a single pass, and convert each
one to GFE against the bulk density that run measured. Going through it rather than
assembling the steps by hand is what keeps the order right -- the density is combined
before the logarithm, never after -- and what keeps the reference structure, the
topology and the trajectory from being mixed up.

:func:`combine_runs` does the same for an ensemble: it folds the runs' densities
together and converts once, so the non-linearity is met only after the combination.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProbeConfig
from .gfe import GFEParams, pmap_to_gfe
from .grid import Grid, GridSpec

__all__ = ["RunMaps", "atom_labels", "combine_runs", "run_maps", "write_maps"]


def atom_labels(probe: ProbeConfig) -> dict[int, str]:
    """Map each mapped atom index to the name to call it in a file.

    Falls back to the index when the probe carries no atom order, so a map is never
    written under a name that does not identify it.
    """
    order = probe.atom_order
    names = getattr(order, "mol2_names", None) if order is not None else None
    out = {}
    for i in probe.mapped_atom_indices:
        out[i] = str(names[i]) if names and i < len(names) else f"atom{i}"
    return out


@dataclass
class RunMaps:
    """The maps of one run, keyed by the probe's atom index.

    Attributes:
        spec: the grid every map is on.
        pmaps: probe number density (per A^3).
        gfes: free energy (kcal/mol), on the same grid.
        bulk_densities: the bulk number density each map was converted against.
        results: the raw :class:`prismsmd.pmap_cosolvkit.PMAPResult` per atom.
        excluded_volume: the reference structure's excluded volume (A^3).
        valid_mask: True where a grid point is near the reference structure.
        labels: atom index -> name, as :func:`atom_labels` gives them.
    """

    spec: GridSpec
    pmaps: dict[int, Grid]
    gfes: dict[int, Grid]
    bulk_densities: dict[int, float]
    results: dict[int, Any]
    excluded_volume: float
    valid_mask: np.ndarray
    labels: dict[int, str] = field(default_factory=dict)

    def _result_dict(self, index: int) -> dict:
        """The per-atom provenance, whether it came from one run or several."""
        got = self.results[index]
        if isinstance(got, list):
            return {"runs": [r.to_dict() for r in got], "n_runs": len(got)}
        return got.to_dict()

    def to_dict(self) -> dict:
        """Return the provenance of these maps as a plain dict."""
        from .provenance import stamp

        return {
            "prismsmd": stamp(),
            "grid": self.spec.to_dict(),
            "excluded_volume": self.excluded_volume,
            "valid_points": int(self.valid_mask.sum()),
            "atoms": {
                self.labels.get(i, str(i)): {
                    "bulk_number_density": self.bulk_densities[i],
                    **self._result_dict(i),
                }
                for i in sorted(self.pmaps)
            },
        }


def run_maps(
    topology: str | Path,
    trajectory: str | Path,
    reference_pdb: str | Path,
    probe: ProbeConfig,
    *,
    box_size: float = 80.0,
    spacing: float = 1.0,
    center_selection: str = "CA",
    valid_distance: float = 5.0,
    gfe: GFEParams | None = None,
    clip: bool = True,
    align: bool = True,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    log=print,
) -> RunMaps:
    """Turn one run's trajectory into the probe's density and free-energy maps.

    Every mapped atom of the probe is histogrammed in one pass over the trajectory, so
    the cost is one read however many atoms are mapped.

    Args:
        topology: the run's topology, as MDAnalysis reads it. Not the reference
            structure and not ``input.top``, which MDAnalysis reads as Amber.
        trajectory: the trajectory, after the periodic-boundary handling.
        reference_pdb: the structure the grid and the fit are defined against.
        probe: the probe, which supplies the residue name and the mapped atoms.
        box_size: edge of the grid (A).
        spacing: grid spacing (A).
        center_selection: atom name whose centroid fixes the grid centre.
        valid_distance: how far from the reference a grid point still counts (A).
        gfe: conversion parameters; the defaults are used when omitted.
        clip: clip the GFE from above, and fill outside the valid region with the
            same bound so the two agree.
        align: fit each frame onto the reference before histogramming.
        start, stop, step: the frames to use.
        log: called with one line per stage.

    Returns:
        :class:`RunMaps`.

    Raises:
        ValueError: when the probe has no mapped atoms.
    """
    from .pmap_cosolvkit import (
        apply_valid_region,
        assert_density_is_centred,
        pmap_grid_spec_from_reference,
        run_fixed_grid_analyses,
        valid_region_mask,
    )
    from .volume import excluded_volume

    indices = probe.mapped_atom_indices
    if not indices:
        raise ValueError(
            f"probe {probe.cid} has no mapped atoms, so there is nothing to map."
            " Set map_atoms in probes.yaml."
        )
    params = gfe if gfe is not None else GFEParams()
    labels = atom_labels(probe)

    spec = pmap_grid_spec_from_reference(
        reference_pdb, box_size=box_size, spacing=spacing,
        selection=center_selection, probe_resname=probe.residue_name,
    )
    log(f"[maps] grid {spec.dims} at {spec.spacing} A, centre {spec.center}")

    results = run_fixed_grid_analyses(
        topology, trajectory, spec,
        resname=probe.residue_name, atom_indices=indices, order=probe.atom_order,
        reference_pdb=reference_pdb, align=align,
        start=start, stop=stop, step=step,
    )

    excl = excluded_volume(reference_pdb)
    mask = valid_region_mask(spec, reference_pdb, cutoff=valid_distance)
    log(f"[maps] excluded volume {excl:.0f} A^3,"
        f" valid region {int(mask.sum())} / {mask.size} points")

    outside = params.clip_max if (clip and params.clip_max is not None) else 0.0
    pmaps: dict[int, Grid] = {}
    gfes: dict[int, Grid] = {}
    bulks: dict[int, float] = {}
    for index in indices:
        res = results[index]
        bulk = res.bulk_reference(excl).number_density
        density = res.number_density()
        energy = pmap_to_gfe(density, params, clip=clip, scale=False,
                             bulk_probability=bulk)
        # Where the density sits relative to the reference. A trajectory the protein
        # was not re-imaged around is invisible in the maps themselves -- the molecules
        # are whole and the fit succeeds -- so the position is the only witness.
        #
        # assert_density_in_valid_region is deliberately NOT called here. Its threshold
        # is an absolute occupied fraction, which a short trajectory cannot reach
        # however correct it is: 0.15 ns of one run measured 0.48 % against a limit of
        # 1 %, while 40 ns of twenty runs measured 52 %. Rejecting on it would throw
        # out correct short runs, and displacing a correct map by the 13 A the bug is
        # known to produce *raises* the fraction rather than lowering it, so there is
        # no evidence it catches the failure either. It stays available for a caller
        # that knows its own sampling.
        offset = assert_density_is_centred(
            density, what=f"{probe.cid} {labels[index]}", log=log)

        pmaps[index] = density
        gfes[index] = apply_valid_region(energy, mask, outside_value=outside)
        bulks[index] = bulk
        inside = gfes[index].values[mask]
        log(f"[maps] {labels[index]}: bulk {bulk:.4e} /A^3,"
            f" GFE min {inside.min():+.2f} kcal/mol,"
            f" {int((inside < 0).sum())} favourable points,"
            f" centroid offset {offset:.1f} A")

    return RunMaps(spec=spec, pmaps=pmaps, gfes=gfes, bulk_densities=bulks,
                   results=results, excluded_volume=excl, valid_mask=mask,
                   labels=labels)


def write_maps(
    maps: RunMaps,
    outdir: str | Path,
    probe: ProbeConfig,
    *,
    pmap: bool = True,
    gfe: bool = True,
) -> list[Path]:
    """Write the maps as OpenDX, one file per atom, and the provenance beside them.

    File names carry the probe and the atom, so maps of different atoms cannot
    overwrite one another.
    """
    import json

    from .pmap_cosolvkit import write_pmap_dx

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in sorted(maps.pmaps):
        label = maps.labels.get(index, str(index))
        if pmap:
            written.append(write_pmap_dx(
                out / f"PMAP_{probe.cid}_{label}.dx", maps.pmaps[index]))
        if gfe:
            written.append(write_pmap_dx(
                out / f"GFE_{probe.cid}_{label}.dx", maps.gfes[index]))
    manifest = out / f"maps_{probe.cid}.json"
    manifest.write_text(json.dumps(maps.to_dict(), indent=2) + "\n")
    written.append(manifest)
    return written


def combine_runs(
    runs: Sequence[RunMaps],
    *,
    method: str = "ave",
    gfe: GFEParams | None = None,
    clip: bool = True,
    n_runs: int | None = None,
) -> RunMaps:
    """Fold the maps of several runs of one probe into one set of maps.

    The densities are combined and converted afterwards, which is the only correct
    order: ``-RT ln`` is non-linear, so averaging the runs' finished GFE grids gives a
    different and wrong answer.

    Args:
        runs: the runs' maps, all of the same probe on the same grid.
        method: how to combine, as a key of :data:`prismsmd.aggregate.AGGREGATIONS`.
        gfe: conversion parameters; the defaults are used when omitted.
        clip: clip the result from above, and fill outside the valid region to match.
        n_runs: required number of runs; a different count raises.

    Returns:
        A :class:`RunMaps` holding the combined density, the GFE converted from it, and
        the bulk density used. ``results`` carries the per-run results of each atom.

    Raises:
        ValueError: when the runs are empty, map different atoms, or sit on grids that
            do not match.
    """
    from .aggregate import combine_replicates
    from .pmap_cosolvkit import apply_valid_region

    if not runs:
        raise ValueError("no runs to combine")
    first = runs[0]
    atoms = sorted(first.pmaps)
    for i, r in enumerate(runs[1:], start=1):
        if sorted(r.pmaps) != atoms:
            raise ValueError(
                f"run {i} maps atoms {sorted(r.pmaps)}, run 0 maps {atoms}."
                " Combining different atoms would mix unrelated maps."
            )
        first.spec.assert_compatible(r.spec)

    params = gfe if gfe is not None else GFEParams()
    outside = params.clip_max if (clip and params.clip_max is not None) else 0.0

    pmaps: dict[int, Grid] = {}
    gfes: dict[int, Grid] = {}
    bulks: dict[int, float] = {}
    for index in atoms:
        density, bulk = combine_replicates(
            [r.pmaps[index] for r in runs],
            [r.bulk_densities[index] for r in runs],
            method=method, expected=first.spec, n_runs=n_runs,
        )
        energy = pmap_to_gfe(density, params, clip=clip, scale=False,
                             bulk_probability=bulk)
        pmaps[index] = density
        gfes[index] = apply_valid_region(energy, first.valid_mask,
                                         outside_value=outside)
        bulks[index] = bulk

    return RunMaps(
        spec=first.spec, pmaps=pmaps, gfes=gfes, bulk_densities=bulks,
        results={i: [r.results[i] for r in runs] for i in atoms},
        excluded_volume=first.excluded_volume, valid_mask=first.valid_mask,
        labels=dict(first.labels),
    )
