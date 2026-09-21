"""Virtual repulsion between probes.

CosolvKit's ``add_repulsive_forces()`` builds an OpenMM force that never reaches the
saved topology, so the repulsion is written into the GROMACS files here instead:

* one massless, chargeless virtual site ``VIS`` at each probe's centre of mass;
* a ``VIS`` entry in ``[ atomtypes ]``, with LJ given only to the VIS-VIS pair through
  ``[ nonbond_params ]``, so nothing else interacts with it.

Units follow GROMACS ``[ nonbond_params ]``: nm and kJ/mol. ``func 1`` there is the
sigma/epsilon form, so the Rmin the protocol specifies is converted here rather than
being stated as a sigma in the config.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from scipy import constants as C

__all__ = [
    "TWO_POW_ONE_SIXTH",
    "VIS_ATOMTYPE_BLOCK",
    "GroAtom",
    "add_virtual_site_to_gro",
    "add_virtual_site_to_top",
    "epsilon_kcal_to_kj",
    "rmin_angstrom_to_sigma_nm",
    "sigma_nm_to_rmin_angstrom",
]

#: Ratio between LJ Rmin and sigma: the minimum of
#: ``U(r) = 4 eps ((sigma/r)^12 - (sigma/r)^6)`` lies at ``r = 2^(1/6) sigma``.
TWO_POW_ONE_SIXTH = 2.0 ** (1.0 / 6.0)


def rmin_angstrom_to_sigma_nm(rmin_angstrom: float) -> float:
    """Convert LJ Rmin (A) to sigma (nm), as GROMACS nonbond_params wants it.

    >>> round(rmin_angstrom_to_sigma_nm(20.0), 7)
    1.7817974
    """
    if rmin_angstrom <= 0:
        raise ValueError(f"Rmin must be positive: {rmin_angstrom!r}")
    return rmin_angstrom / TWO_POW_ONE_SIXTH / 10.0


def epsilon_kcal_to_kj(epsilon_kcal: float) -> float:
    """Convert LJ epsilon from kcal/mol to kJ/mol.

    >>> round(epsilon_kcal_to_kj(1.0e-6), 12)
    4.184e-06
    """
    if epsilon_kcal < 0:
        raise ValueError(f"epsilon must be >= 0: {epsilon_kcal!r}")
    return epsilon_kcal * C.calorie


def sigma_nm_to_rmin_angstrom(sigma_nm: float) -> float:
    """Convert sigma (nm) back to LJ Rmin (A), for cross-checking.

    >>> round(sigma_nm_to_rmin_angstrom(1.7817974), 6)
    20.0
    """
    return sigma_nm * 10.0 * TWO_POW_ONE_SIXTH


VIS_ATOMTYPE_BLOCK = """
[ atomtypes ]
{name:<8s} {name:<8s}  0.00000  0.00000   V     0.00000e+00   0.00000e+00 ; virtual interaction site

[ nonbond_params ]
; i j func sigma epsilon
{name:<5s} {name:<5s}  1  {sigma:1.6e}   {epsilon:1.6e}
"""


def add_virtual_site_to_top(
    top_string: str,
    probe_resnames: Sequence[str],
    *,
    rmin_angstrom: float = 20.0,
    epsilon_kcal: float = 1.0e-6,
    site_name: str = "VIS",
) -> str:
    """Add the probe virtual sites and the VIS-VIS LJ to a topology string.

    The VIS definition goes in right after the ``[ atomtypes ]`` section ends, and the
    VIS atom plus ``[ virtual_sitesn ]`` are appended at the end of each probe's
    ``[ atoms ]``.

    Args:
        top_string: the topology to modify.
        probe_resnames: the moleculetype names that get a virtual site.
        rmin_angstrom: LJ Rmin (A), converted to sigma (nm) on write.
        epsilon_kcal: LJ epsilon (kcal/mol), converted to kJ/mol on write.
        site_name: name of the virtual site atom and atomtype.
    """
    probe_resnames = list(probe_resnames)
    sigma_nm = rmin_angstrom_to_sigma_nm(rmin_angstrom)
    epsilon_kj = epsilon_kcal_to_kj(epsilon_kcal)
    out: list[str] = []
    current_section: str | None = None
    current_molecule: str | None = None
    natoms = 0
    atomtypes_seen = False

    for raw in top_string.split("\n"):
        line = raw.split(";")[0].rstrip()
        if line.lstrip().startswith("["):
            prev_section = current_section
            if prev_section == "atomtypes":
                out.append(
                    VIS_ATOMTYPE_BLOCK.format(
                        name=site_name, sigma=sigma_nm, epsilon=epsilon_kj
                    )
                )
                atomtypes_seen = True
            elif prev_section == "atoms" and current_molecule in probe_resnames:
                out.append(
                    _virtual_site_records(natoms, current_molecule, site_name)
                )
            if prev_section == "atoms":
                natoms = 0
            current_section = line[line.find("[") + 1 : line.find("]")].strip()
            if current_section == "moleculetype":
                current_molecule = None
        elif current_section == "atoms" and line.strip():
            natoms += 1
        elif (
            current_section == "moleculetype"
            and current_molecule is None
            and line.strip()
        ):
            current_molecule = line.split()[0].strip()

        out.append(line)

    # When the file ends with the probe's [ atoms ]
    if current_section == "atoms" and current_molecule in probe_resnames:
        out.append(_virtual_site_records(natoms, current_molecule, site_name))

    if not atomtypes_seen:
        raise ValueError(
            "no [ atomtypes ] section found."
            " Check the GROMACS output format of CosolvKit/parmed."
        )
    return "\n".join(out)


def _virtual_site_records(natoms: int, molecule: str, site_name: str) -> str:
    idx = natoms + 1
    constructing = " ".join(str(i) for i in range(1, natoms + 1))
    return (
        f"{idx:6d} {site_name:>8s}      1    {molecule}    {site_name}"
        f"  {idx:6d} 0.00000000   0.000000\n"
        f"\n[ virtual_sitesn ]\n"
        f"{idx:6d}   2  {constructing}\n"
    )


_ATOMIC_MASS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "P": 30.974, "S": 32.06, "CL": 35.45, "BR": 79.904, "I": 126.904,
    "NA": 22.990, "MG": 24.305, "K": 39.098, "CA": 40.078, "ZN": 65.38,
}


def _mass_of(atom_name: str) -> float:
    name = atom_name.strip().upper()
    for length in (2, 1):
        if name[:length] in _ATOMIC_MASS:
            return _ATOMIC_MASS[name[:length]]
    return 12.011  # unknown elements are treated as carbon


@dataclass
class GroAtom:
    """One atom record of a ``.gro`` file. Coordinates are in nm."""

    resi: int
    resn: str
    name: str
    index: int
    x: float
    y: float
    z: float

    @classmethod
    def parse(cls, line: str) -> GroAtom:
        """Parse one fixed-column ``.gro`` atom line."""
        return cls(
            resi=int(line[0:5]),
            resn=line[5:10].strip(),
            name=line[10:15].strip(),
            index=int(line[15:20]),
            x=float(line[20:28]),
            y=float(line[28:36]),
            z=float(line[36:44]),
        )

    def format(self, index: int) -> str:
        """Render the atom as a ``.gro`` line with the given 1-based atom index."""
        return (
            f"{self.resi % 100000:5d}{self.resn:<5s}{self.name:>5s}"
            f"{index % 100000:5d}{self.x:8.3f}{self.y:8.3f}{self.z:8.3f}"
        )


def add_virtual_site_to_gro(
    gro_string: str,
    probe_resname: str,
    *,
    site_name: str = "VIS",
) -> str:
    """Add one virtual-site atom at the centre of mass of each probe molecule."""
    lines = gro_string.split("\n")
    if len(lines) < 3:
        raise ValueError("gro is too short")
    title = lines[0]
    natoms = int(lines[1].strip())
    atom_lines = lines[2 : 2 + natoms]
    box_line = lines[2 + natoms] if len(lines) > 2 + natoms else ""

    atoms = [GroAtom.parse(ln) for ln in atom_lines]

    # Group by probe residue (preserving order of appearance)
    groups: dict[int, list[GroAtom]] = {}
    for a in atoms:
        if a.resn == probe_resname:
            groups.setdefault(a.resi, []).append(a)
    if not groups:
        raise ValueError(f"probe residue {probe_resname!r} is not in the gro")

    virtual: dict[int, GroAtom] = {}
    for resi, mol in groups.items():
        total = 0.0
        cx = cy = cz = 0.0
        for a in mol:
            m = _mass_of(a.name)
            total += m
            cx += m * a.x
            cy += m * a.y
            cz += m * a.z
        virtual[resi] = GroAtom(
            resi=resi, resn=probe_resname, name=site_name, index=0,
            x=cx / total, y=cy / total, z=cz / total,
        )

    # Insert right after the last atom of each probe molecule
    last_index_of: dict[int, int] = {}
    for i, a in enumerate(atoms):
        if a.resn == probe_resname:
            last_index_of[a.resi] = i

    out_atoms: list[GroAtom] = []
    for i, a in enumerate(atoms):
        out_atoms.append(a)
        for resi, last_i in last_index_of.items():
            if last_i == i:
                out_atoms.append(virtual[resi])

    body = "\n".join(a.format(i + 1) for i, a in enumerate(out_atoms))
    return f"{title}\n{len(out_atoms):5d}\n{body}\n{box_line}\n"
