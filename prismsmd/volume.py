"""Excluded volume and box volume: the denominator of the GFE bulk reference density,

    rho_bulk = (probe atoms selected for that map) / (box volume - excluded volume)

whose numerator is the atom count reported by
:class:`prismsmd.pmap_cosolvkit.PMAPResult`.

The excluded volume here is defined with the VdW radius plus a carbon radius, which is
not CosolvKit's ``calculate_mol_volume`` (a 1.5 A offset, used to decide probe counts
at build time). The two are not interchangeable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pdbio import PDB_COLUMNS, atom_coord

__all__ = [
    "ATOMIC_RADII",
    "DEFAULT_ATOMIC_RADIUS",
    "SOLVENT_RADIUS_ELEMENT",
    "VOLUME_EXCLUDED_RESNAMES",
    "BulkReference",
    "atoms_for_excluded_volume",
    "box_volume_from_dimensions",
    "estimate_volume",
    "excluded_volume",
    "mean_box_volume",
]


#: VdW radii in A, matching BioPython's ``Bio.PDB.SASA.ATOMIC_RADII``
#: (Bondi, J. Phys. Chem. 68 (1964) 441). Keys are uppercase element symbols.
ATOMIC_RADII: dict[str, float] = {
    "H": 1.200,
    "HE": 1.400,
    "C": 1.700,
    "N": 1.550,
    "O": 1.520,
    "F": 1.470,
    "NA": 2.270,
    "MG": 1.730,
    "P": 1.800,
    "S": 1.800,
    "CL": 1.750,
    "K": 2.750,
    "CA": 2.310,
    "NI": 1.630,
    "CU": 1.400,
    "ZN": 1.390,
    "SE": 1.900,
    "BR": 1.850,
    "CD": 1.580,
    "I": 1.980,
    "HG": 1.550,
}

#: Radius used for elements absent from :data:`ATOMIC_RADII`.
DEFAULT_ATOMIC_RADIUS = 2.0

#: The solvent VdW radius is represented by a single carbon.
SOLVENT_RADIUS_ELEMENT = "C"

#: Residue names left out of the excluded volume.
VOLUME_EXCLUDED_RESNAMES: tuple[str, ...] = ("HOH", "WAT")


def _radius_of(element: str) -> float:
    return ATOMIC_RADII.get(element.strip().upper(), DEFAULT_ATOMIC_RADIUS)


def estimate_volume(
    points: np.ndarray, radii: np.ndarray, granularity: int = 10
) -> float:
    """Estimate the volume (A^3) of the union of spheres of differing radii.

    Counts the grid points inside any sphere. The pitch is
    ``min((max - min) / granularity, 1)`` per axis, i.e. 1 A at protein scale, and the
    grid spans ``arange(min - rmax, max + rmax + 1, pitch)``.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if len(points) == 0:
        raise ValueError("no atoms at all")
    if len(points) != len(radii):
        raise ValueError(f"point and radius counts differ: {len(points)} vs {len(radii)}")
    if np.any(radii <= 0):
        raise ValueError("radii must be positive")

    lo = points.min(axis=0)
    hi = points.max(axis=0)
    rmax = float(radii.max())
    pitch = np.minimum((hi - lo) / granularity, 1.0)
    if np.any(pitch <= 0):
        raise ValueError(
            "the pitch became <= 0 (all atoms are coplanar or coincident)"
        )

    axes = [np.arange(lo[i] - rmax, hi[i] + rmax + 1, pitch[i]) for i in range(3)]
    shape = tuple(len(a) for a in axes)
    occupied = np.zeros(shape, dtype=bool)

    for (px, py, pz), r in zip(points, radii, strict=True):
        r2 = r * r
        sl = []
        for i, c in enumerate((px, py, pz)):
            a = axes[i]
            sl.append(
                (
                    int(np.searchsorted(a, c - r, side="left")),
                    int(np.searchsorted(a, c + r, side="right")),
                )
            )
        if any(b <= a for a, b in sl):
            continue
        dx = axes[0][sl[0][0]:sl[0][1]] - px
        dy = axes[1][sl[1][0]:sl[1][1]] - py
        dz = axes[2][sl[2][0]:sl[2][1]] - pz
        d2 = dx[:, None, None] ** 2 + dy[None, :, None] ** 2 + dz[None, None, :] ** 2
        occupied[
            sl[0][0]:sl[0][1], sl[1][0]:sl[1][1], sl[2][0]:sl[2][1]
        ] |= d2 <= r2

    return float(occupied.sum()) * float(pitch[0] * pitch[1] * pitch[2])


def _element_from_pdb_line(line: str) -> str:
    """Take the element symbol from a PDB line.

    Columns 77-78 are the official element field; if it is empty, the element is
    inferred from the leading letters of the atom name.
    """
    element = line[slice(*PDB_COLUMNS["element"])].strip().upper()
    if element:
        return element
    name = line[slice(*PDB_COLUMNS["atom_name"])].strip().upper()
    two = name[:2]
    if two in ATOMIC_RADII and not name[:1].isdigit():
        return two
    for ch in name:
        if ch.isalpha():
            return ch
    return ""


def atoms_for_excluded_volume(
    pdb: str | Path,
    *,
    exclude_resnames: Sequence[str] = VOLUME_EXCLUDED_RESNAMES,
) -> tuple[np.ndarray, list[str]]:
    """Return the coordinates (N, 3) and element symbols used for the excluded volume.

    Every ``ATOM`` and ``HETATM`` whose residue name is not in ``exclude_resnames``,
    hydrogens included.
    """
    excluded = {r.strip().upper() for r in exclude_resnames}
    coords: list[tuple[float, float, float]] = []
    elements: list[str] = []
    for line in Path(pdb).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[slice(*PDB_COLUMNS["res_name"])].strip().upper() in excluded:
            continue
        coords.append(atom_coord(line))
        elements.append(_element_from_pdb_line(line))
    if not coords:
        raise ValueError(f"{pdb}: no atoms usable for the excluded volume")
    return np.array(coords, dtype=float), elements


def excluded_volume(
    pdb: str | Path,
    *,
    exclude_resnames: Sequence[str] = VOLUME_EXCLUDED_RESNAMES,
    granularity: int = 10,
) -> float:
    """Excluded volume of the reference structure (A^3).

    Each atom is a sphere of its VdW radius plus the carbon VdW radius, and the volume
    of their union -- the volume no solvent centre can enter -- is counted.
    """
    coords, elements = atoms_for_excluded_volume(
        pdb, exclude_resnames=exclude_resnames
    )
    radii = np.array([_radius_of(e) for e in elements], dtype=float)
    radii = radii + ATOMIC_RADII[SOLVENT_RADIUS_ELEMENT]
    return estimate_volume(coords, radii, granularity=granularity)


def box_volume_from_dimensions(dimensions: Sequence[float]) -> float:
    """Box volume (A^3) from MDAnalysis ``ts.dimensions``.

    ``dimensions`` is ``[a, b, c, alpha, beta, gamma]`` (lengths in A, angles in
    degrees); with only 3 elements the cell is taken as orthorhombic. The triclinic
    form is needed because GROMACS octahedral and dodecahedral cells are expressed
    as triclinic.
    """
    d = [float(x) for x in dimensions]
    if len(d) == 3:
        a, b, c = d
        angles = (90.0, 90.0, 90.0)
    elif len(d) == 6:
        a, b, c = d[:3]
        angles = (d[3], d[4], d[5])
    else:
        raise ValueError(f"dimensions must have 3 or 6 elements: {dimensions!r}")
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(f"box edge length is not positive: {dimensions!r}")

    ca, cb, cg = (math.cos(math.radians(x)) for x in angles)
    factor = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
    if factor <= 0:
        raise ValueError(f"invalid box angles: {angles}")
    return a * b * c * math.sqrt(factor)


def mean_box_volume(dimensions_per_frame: Sequence[Sequence[float]]) -> float:
    """Mean box volume (A^3) over the frames the PMAP was taken from."""
    vols = [box_volume_from_dimensions(d) for d in dimensions_per_frame]
    if not vols:
        raise ValueError("no frames at all")
    return float(np.mean(vols))


@dataclass(frozen=True)
class BulkReference:
    """The bulk reference density and its breakdown, for the provenance record.

    Attributes:
        n_atoms: total number of atoms the map's atom selection matched, over every
            probe molecule in the system.
        box_volume: mean box volume (A^3) over the analysed frames.
        excluded_volume: excluded volume of the target (A^3).
    """

    n_atoms: int
    box_volume: float
    excluded_volume: float

    @property
    def free_volume(self) -> float:
        """Volume available to probes (A^3) = box volume - excluded volume."""
        v = self.box_volume - self.excluded_volume
        if v <= 0:
            raise ValueError(
                f"box volume {self.box_volume:.1f} A^3 is not greater than the excluded "
                f"volume {self.excluded_volume:.1f} A^3, leaving no free volume"
            )
        return v

    @property
    def number_density(self) -> float:
        """Bulk number density (per A^3)."""
        if self.n_atoms <= 0:
            raise ValueError(f"atom count must be positive: {self.n_atoms}")
        return self.n_atoms / self.free_volume

    @property
    def molar(self) -> float:
        """The density above expressed in mol/L, to compare with the nominal value."""
        from scipy import constants as C

        return self.number_density / (C.N_A * C.angstrom**3 / C.liter)

    def to_dict(self) -> dict:
        """Return the breakdown as a plain dict."""
        return {
            "n_atoms": self.n_atoms,
            "box_volume": self.box_volume,
            "excluded_volume": self.excluded_volume,
            "free_volume": self.free_volume,
            "number_density": self.number_density,
            "effective_molar": self.molar,
        }
