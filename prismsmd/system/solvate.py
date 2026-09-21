"""Fill in the water a freshly built system is missing, using ``gmx solvate``.

The system CosolvKit's ``build()`` produces is short of water, at a density around
0.81 g/cm^3, and NPT then shrinks the box and raises the probe concentration above its
nominal value. It cannot be topped up on the OpenMM side: calling
``Modeller.addSolvent`` again adds nothing, and ``numAdded`` changes the box.

``gmx solvate`` adds the water without touching the box, but it appends its solvent box
under its own residue name (``SOL``, with ``OW / HW1 / HW2``), which is not the water
moleculetype of the ``.top``. This module therefore remaps the added residues to that
moleculetype -- renaming, and reordering the atoms if necessary -- and rewrites the
appended ``[ molecules ]`` line. The moleculetype and its atom order are read from the
``.top``, never hard-coded, and elements are determined from the masses rather than the
spelling of the names.

Only :func:`fill_water_with_gmx` starts an external process; the remapping and the
density and box checks are in the pure function :func:`relabel_added_water`. The command
can be swapped via the ``gmx=`` argument or the ``MSMD_GMX`` environment variable.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from scipy import constants as C

__all__ = [
    "BOX_TOLERANCE_NM",
    "DEFAULT_SOLVENT_GRO",
    "DENSITY_RELATIVE_TOLERANCE",
    "WATER_DENSITY_G_CM3",
    "BoxChangedError",
    "SystemDensityError",
    "TopMoleculeType",
    "WaterFillError",
    "WaterFillResult",
    "WaterNameMappingError",
    "density_g_cm3",
    "expected_density_g_cm3",
    "fill_water_with_gmx",
    "find_water_moleculetype",
    "gro_atom_count",
    "gro_box_nm",
    "gro_box_volume_a3",
    "molar_from_count",
    "parse_molecules_section",
    "parse_moleculetypes",
    "relabel_added_water",
    "resolve_gmx_command",
    "total_mass_amu",
]


#: Density of liquid water (g/cm^3) at 298.15 K / 1 atm (CRC Handbook, 0.99705).
WATER_DENSITY_G_CM3 = 0.997

#: Relative tolerance on the density, around the density the composition implies
#: (:func:`expected_density_g_cm3`) rather than around pure water. A heavy cosolvent
#: legitimately shifts the composition density by ~10%, so a fixed reference would
#: couple the check to how heavy the probe is. The reference itself is good to ~1%,
#: while the shortfall this check exists to catch is of order -20%.
DENSITY_RELATIVE_TOLERANCE = 0.05

#: Partial specific volume of a folded protein (cm^3/g). Source: Squire & Himmel,
#: Arch. Biochem. Biophys. 196, 165 (1979), the mean over 30 proteins. The
#: small-molecule estimator below must not be used on a protein, which is packed far
#: more densely than the sum of its groups.
PROTEIN_SPECIFIC_VOLUME_CM3_PER_G = 0.73

#: A moleculetype with at least this many atoms is treated as a macromolecule and given
#: the partial specific volume above.
MACROMOLECULE_ATOM_COUNT = 200

#: Girolami's volume-increment method for the molar volume of a small molecule:
#: ``V (cm^3/mol) = 5 * sum(increment of each atom)``, the increment set by the atom's
#: row of the periodic table. Source: G. S. Girolami, *J. Chem. Educ.* **71**, 962
#: (1994). Water is deliberately not estimated this way -- its hydrogen bonding puts
#: Girolami ~10% off -- and uses :data:`WATER_DENSITY_G_CM3` instead.
_GIROLAMI_SCALE_CM3_PER_MOL = 5.0
_GIROLAMI_INCREMENT_BY_PERIOD = {1: 1.0, 2: 2.0, 3: 4.0, 4: 5.0, 5: 7.5, 6: 9.0}

#: Standard atomic weights (amu) and periodic-table row for every element that can
#: appear in these systems, used to turn the masses written in the ``.top`` into
#: Girolami increments. Source: IUPAC 2021 standard atomic weights.
_ELEMENT_MASS_PERIOD: dict[str, tuple[float, int]] = {
    "H": (1.008, 1),
    "C": (12.011, 2),
    "N": (14.007, 2),
    "O": (15.999, 2),
    "F": (18.998, 2),
    "Na": (22.990, 3),
    "Mg": (24.305, 3),
    "P": (30.974, 3),
    "S": (32.060, 3),
    "Cl": (35.450, 3),
    "K": (39.098, 4),
    "Ca": (40.078, 4),
    "Fe": (55.845, 4),
    "Zn": (65.380, 4),
    "Br": (79.904, 4),
    "I": (126.904, 5),
}

#: Below this mass an atom is a massless virtual site (TIP4P's EP, a lone pair), which
#: occupies no volume and is skipped rather than matched to an element.
_VIRTUAL_SITE_MASS_AMU = 0.5

#: Tolerance for judging box vectors equal (nm). A ``.gro`` box is written as
#: ``%10.5f``, so this allows one rounding digit on the representable precision.
BOX_TOLERANCE_NM = 2.0e-5

#: The solvent box passed to ``gmx solvate -cs``: 216 water molecules, shipped with
#: GROMACS.
DEFAULT_SOLVENT_GRO = "spc216.gro"

#: Standard atomic weights (amu) for element determination on the water side.
#: Source: IUPAC 2021 standard atomic weights.
_WATER_ELEMENT_MASSES = {"H": 1.008, "O": 15.999}

#: Tolerance (amu) when determining an element from a mass. Wide enough to absorb
#: isotopic substitution and force-field rounding, never wide enough to confuse H with
#: O. A ``.top`` with hydrogen mass repartitioning is rejected here.
_MASS_TOLERANCE_AMU = 0.6

#: Unit conversions, so no bare numeric literals appear below.
_GRAM_PER_AMU = 1.0 / C.N_A
_CM3_PER_A3 = (C.angstrom / C.centi) ** 3
#: nm^3 -> A^3
_A3_PER_NM3 = (C.nano / C.angstrom) ** 3
#: A^3 per litre
_A3_PER_LITER = C.liter / C.angstrom**3


class WaterFillError(RuntimeError):
    """Raised when the water-filling stage is inconsistent."""


class BoxChangedError(WaterFillError):
    """Raised when filling changed the box vectors."""


class SystemDensityError(WaterFillError):
    """Raised when the density after filling is outside the tolerance around the target."""


class WaterNameMappingError(WaterFillError):
    """Raised when the added water cannot be remapped to the ``.top`` water moleculetype."""


@dataclass(frozen=True)
class TopMoleculeType:
    """One ``[ moleculetype ]`` of a ``.top``.

    Attributes:
        name: the moleculetype name, as referenced from ``[ molecules ]``.
        atom_names: atom names in ``[ atoms ]`` order, which the ``.gro`` must follow.
        masses: masses in the same order (amu).
    """

    name: str
    atom_names: tuple[str, ...]
    masses: tuple[float, ...]

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the moleculetype."""
        return len(self.atom_names)

    def elements(self) -> tuple[str, ...]:
        """Element symbols determined from the masses, so the spelling of the names
        (``O`` / ``OW`` / ``OH2``) does not matter. Only H and O are resolved."""
        out: list[str] = []
        for name, mass in zip(self.atom_names, self.masses, strict=True):
            out.append(_element_from_mass(mass, name=name, molecule=self.name))
        return tuple(out)


def _element_from_mass(mass: float, *, name: str, molecule: str) -> str:
    for element, standard in _WATER_ELEMENT_MASSES.items():
        if abs(mass - standard) <= _MASS_TOLERANCE_AMU:
            return element
    raise WaterNameMappingError(
        f"cannot determine the element of atom {name!r} of moleculetype {molecule!r}"
        f" from its mass of {mass} amu (supported: "
        + " / ".join(f"{e} {m}" for e, m in _WATER_ELEMENT_MASSES.items())
        + f" +- {_MASS_TOLERANCE_AMU} amu).\n"
        "water other than a 3-point model (TIP3P/SPC) - e.g. the virtual site of TIP4P -"
        " or a .top with hydrogen mass repartitioning (HMR) cannot be handled by this stage."
    )


def _strip_comment(line: str) -> str:
    return line.split(";", 1)[0].rstrip()


def _section_name(line: str) -> str | None:
    s = line.strip()
    if not (s.startswith("[") and "]" in s):
        return None
    return s[s.find("[") + 1 : s.find("]")].strip().lower()


def _check_preprocessor(line: str, section: str | None) -> None:
    """Raise on ``#include``: what cannot be seen must not be analysed."""
    if line.split()[0].lower() == "#include":
        raise WaterFillError(
            f"the .top contains {line.strip()!r}. The water moleculetype can only be read"
            " from a self-contained .top with #include expanded."
            f" (in section [{section}])."
        )


def parse_moleculetypes(top_string: str) -> dict[str, TopMoleculeType]:
    """Return every ``[ moleculetype ]`` of a ``.top`` as name -> :class:`TopMoleculeType`.

    ``#include`` raises. ``#ifdef`` and friends are ignored in themselves -- GROMACS
    water normally puts its bonds in and out with ``#ifdef FLEXIBLE`` after
    ``[ atoms ]`` -- but an atom table that continues across a conditional raises,
    since neither its masses nor its atom order would be unique.
    """
    out: dict[str, TopMoleculeType] = {}
    section: str | None = None
    current: str | None = None
    names: list[str] = []
    masses: list[float] = []
    conditional = False

    def flush() -> None:
        if current is not None and names:
            if current in out:
                raise WaterFillError(
                    f"the .top defines moleculetype {current!r} twice"
                )
            out[current] = TopMoleculeType(
                name=current, atom_names=tuple(names), masses=tuple(masses)
            )

    for raw in top_string.split("\n"):
        sec = _section_name(raw)
        if sec is not None:
            if sec == "moleculetype":
                flush()
                current = None
                names, masses = [], []
            section = sec
            conditional = False
            continue
        line = _strip_comment(raw)
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            _check_preprocessor(line, section)
            conditional = True
            continue
        if section == "moleculetype" and current is None:
            current = line.split()[0]
        elif section == "atoms" and current is not None:
            if conditional:
                raise WaterFillError(
                    f"the [ atoms ] of moleculetype {current!r} continues inside a"
                    " conditional (#ifdef etc). The atom order and masses are not"
                    " unique, so the water cannot be remapped."
                )
            fields = line.split()
            # GROMACS [ atoms ]:
            #   nr type resnr residue atom cgnr charge [mass] ...
            if len(fields) < 8:
                raise WaterFillError(
                    f"an [ atoms ] line of moleculetype {current!r} has no mass column:"
                    f" {line.strip()!r}\n"
                    " the density cannot be computed from a .top with masses omitted"
                    " (GROMACS fills them from the atomtype, but this stage cannot)."
                )
            names.append(fields[4])
            masses.append(float(fields[7]))
    flush()
    if not out:
        raise WaterFillError(
            "the .top contains no [ moleculetype ] at all."
            " Check whether this really is parmed's GROMACS output."
        )
    return out


def parse_molecules_section(top_string: str) -> list[tuple[str, int]]:
    """Return ``[ molecules ]`` as ``(moleculetype name, count)`` in order of appearance.

    The order has to match the ``.gro`` atom order, so repeated names are not merged.
    """
    out: list[tuple[str, int]] = []
    section: str | None = None
    conditional = False
    for raw in top_string.split("\n"):
        sec = _section_name(raw)
        if sec is not None:
            section = sec
            conditional = False
            continue
        if section != "molecules":
            continue
        line = _strip_comment(raw)
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            _check_preprocessor(line, section)
            conditional = True
            continue
        if conditional:
            raise WaterFillError(
                "[ molecules ] continues inside a conditional (#ifdef etc)."
                " The molecule sequence is not unique, so the appended water cannot be handled."
            )
        fields = line.split()
        if len(fields) < 2:
            raise WaterFillError(f"unreadable [ molecules ] line: {line.strip()!r}")
        out.append((fields[0], int(fields[1])))
    if not out:
        raise WaterFillError("the .top has no [ molecules ]")
    return out


def total_mass_amu(top_string: str) -> float:
    """Total mass of the system (amu), from the ``.top`` alone.

    Uses the force-field masses written in the file rather than an element table, so
    the density stays self-consistent with the force field.
    """
    types = parse_moleculetypes(top_string)
    total = 0.0
    for name, count in parse_molecules_section(top_string):
        mt = types.get(name)
        if mt is None:
            raise WaterFillError(
                f"no [ moleculetype ] corresponds to {name!r} in [ molecules ]."
                " This is the state grompp dies on with a Fatal error (for instance"
                " the SOL appended by gmx solvate never having been remapped)."
            )
        total += count * sum(mt.masses)
    return total


def density_g_cm3(mass_amu: float, volume_a3: float) -> float:
    """Density (g/cm^3) from a mass (amu) and a volume (A^3)."""
    if volume_a3 <= 0:
        raise ValueError(f"volume is not positive: {volume_a3!r}")
    if mass_amu <= 0:
        raise ValueError(f"mass is not positive: {mass_amu!r}")
    return (mass_amu * _GRAM_PER_AMU) / (volume_a3 * _CM3_PER_A3)


def _element_of(mass_amu: float) -> str | None:
    """Element symbol for a mass written in the ``.top``; ``None`` for a virtual site."""
    if mass_amu < _VIRTUAL_SITE_MASS_AMU:
        return None
    best = min(_ELEMENT_MASS_PERIOD,
               key=lambda el: abs(_ELEMENT_MASS_PERIOD[el][0] - mass_amu))
    if abs(_ELEMENT_MASS_PERIOD[best][0] - mass_amu) > _MASS_TOLERANCE_AMU:
        raise WaterFillError(
            f"no element matches the mass {mass_amu} amu in the .top"
            f" (nearest is {best}, {_ELEMENT_MASS_PERIOD[best][0]} amu)."
            " The expected density cannot be computed from the composition."
            " A .top with HMR lands here, and so does an element this pipeline has"
            " not seen before -- in that case add it to _ELEMENT_MASS_PERIOD."
        )
    return best


def _is_water(mt: TopMoleculeType) -> bool:
    """Is this moleculetype a water model? Decided by composition, not by name, since
    both ``SOL`` and ``HOH`` appear in the same ``.top`` mid-pipeline."""
    elements = [_element_of(m) for m in mt.masses]
    return sorted(e for e in elements if e is not None) == ["H", "H", "O"]


def _molecule_volume_a3(mt: TopMoleculeType) -> float:
    """Volume one molecule of this moleculetype occupies (A^3)."""
    mass_g = sum(mt.masses) * _GRAM_PER_AMU
    if _is_water(mt):
        return mass_g / WATER_DENSITY_G_CM3 / _CM3_PER_A3
    if len(mt.masses) >= MACROMOLECULE_ATOM_COUNT:
        return mass_g * PROTEIN_SPECIFIC_VOLUME_CM3_PER_G / _CM3_PER_A3
    increments = 0.0
    for m in mt.masses:
        el = _element_of(m)
        if el is None:
            continue
        increments += _GIROLAMI_INCREMENT_BY_PERIOD[_ELEMENT_MASS_PERIOD[el][1]]
    molar_volume_cm3 = _GIROLAMI_SCALE_CM3_PER_MOL * increments
    return molar_volume_cm3 / C.N_A / _CM3_PER_A3


def expected_density_g_cm3(top_string: str) -> float:
    """The density this composition should have (g/cm^3), by volume additivity.

    Every ``[ molecules ]`` entry contributes its own volume -- water from the measured
    density of water, a macromolecule from the protein partial specific volume, and
    anything else from the Girolami estimate -- and the total mass is divided by the
    total volume. A reference accurate to ~1%, not a measurement.
    """
    types = parse_moleculetypes(top_string)
    volume = 0.0
    for name, count in parse_molecules_section(top_string):
        mt = types.get(name)
        if mt is None:
            raise WaterFillError(
                f"no [ moleculetype ] corresponds to {name!r} in [ molecules ]."
            )
        volume += count * _molecule_volume_a3(mt)
    if volume <= 0:
        raise WaterFillError("the composition implies a volume of zero")
    return density_g_cm3(total_mass_amu(top_string), volume)


def molar_from_count(count: int, volume_a3: float) -> float:
    """Molar concentration (mol/L) from a count and a volume (A^3)."""
    if volume_a3 <= 0:
        raise ValueError(f"volume is not positive: {volume_a3!r}")
    return count / C.N_A / (volume_a3 / _A3_PER_LITER)


def _gro_lines(gro_string: str) -> tuple[str, int, list[str], str]:
    """Split a ``.gro`` into ``(title, natoms, atom lines, box line)``, verbatim."""
    lines = gro_string.split("\n")
    if len(lines) < 3:
        raise WaterFillError("the .gro is too short")
    try:
        natoms = int(lines[1].strip())
    except ValueError as exc:
        raise WaterFillError(f"line 2 of the .gro is not an atom count: {lines[1]!r}") from exc
    if len(lines) < 2 + natoms + 1:
        raise WaterFillError(
            f"the .gro has too few lines (it states {natoms} atoms but"
            f" actually has {max(len(lines) - 3, 0)})"
        )
    return lines[0], natoms, lines[2 : 2 + natoms], lines[2 + natoms]


def gro_atom_count(gro_string: str) -> int:
    """Atom count declared on line 2 of a ``.gro``."""
    return _gro_lines(gro_string)[1]


def gro_box_nm(gro_string: str) -> tuple[float, ...]:
    """Return the box vectors (nm) at the end of a ``.gro`` verbatim (3 or 9 elements)."""
    box_line = _gro_lines(gro_string)[3]
    fields = box_line.split()
    if len(fields) not in (3, 9):
        raise WaterFillError(f"unreadable box line in the .gro: {box_line!r}")
    return tuple(float(v) for v in fields)


def gro_box_volume_a3(gro_string: str) -> float:
    """Box volume (A^3).

    A GROMACS box is normalised to lower triangular form, so even a 9-element
    triclinic cell has the volume of the product of its diagonal.
    """
    box = gro_box_nm(gro_string)
    return box[0] * box[1] * box[2] * _A3_PER_NM3


def _gro_residues(atom_lines: Sequence[str]) -> list[tuple[int, str, list[int]]]:
    """Return ``(residue number, residue name, line indices)`` in order of appearance.

    Residue numbers wrap at 99999, so a molecule is a run of consecutive identical
    (residue number, residue name).
    """
    out: list[tuple[int, str, list[int]]] = []
    prev: tuple[int, str] | None = None
    for i, line in enumerate(atom_lines):
        try:
            resi = int(line[0:5])
        except ValueError as exc:
            raise WaterFillError(f"unreadable atom line in the .gro: {line!r}") from exc
        resn = line[5:10].strip()
        if (resi, resn) != prev:
            out.append((resi, resn, []))
            prev = (resi, resn)
        out[-1][2].append(i)
    return out


def _water_element_from_atom_name(name: str) -> str:
    """Take the element from a water atom name (``OW`` -> ``O``, ``HW1`` -> ``H``).

    A ``.gro`` carries no masses, so this is the only way to determine elements on the
    solvent-box side. Only 3-point water is handled, so a leading letter other than H
    or O raises.
    """
    letters = [ch for ch in name.strip() if ch.isalpha()]
    if not letters:
        raise WaterNameMappingError(f"cannot determine the element from water atom name: {name!r}")
    element = letters[0].upper()
    if element not in _WATER_ELEMENT_MASSES:
        raise WaterNameMappingError(
            f"the element of added water atom name {name!r} is neither H nor O."
            " This stage only handles 3-point water (spc216's OW/HW1/HW2 etc)."
        )
    return element


def find_water_moleculetype(top_string: str) -> str:
    """Decide the water moleculetype by reading the ``.top``.

    Among the moleculetypes appearing in ``[ molecules ]``, the one with 3 atoms whose
    masses are O, H, H in any order. Decided from content; no name is hard-coded.

    Raises:
        WaterNameMappingError: when there is no candidate, or several differently-named
            ones, leaving the choice undecidable.
    """
    types = parse_moleculetypes(top_string)
    used = [name for name, _count in parse_molecules_section(top_string)]
    candidates: list[str] = []
    rejected: list[str] = []
    for name in dict.fromkeys(used):          # order of appearance, deduplicated
        mt = types.get(name)
        if mt is None or mt.n_atoms != 3:
            continue
        try:
            elements = mt.elements()
        except WaterNameMappingError as exc:
            # Never silently discard a 3-atom molecule whose elements are unreadable:
            # missing a "looks like water but is not" case would make the density
            # check meaningless.
            rejected.append(f"{name}: {exc}")
            continue
        if sorted(elements) == ["H", "H", "O"]:
            candidates.append(name)
    if not candidates:
        detail = (
            "\n  moleculetypes with 3 atoms whose elements could not be determined:\n    "
            + "\n    ".join(rejected)
            if rejected
            else ""
        )
        raise WaterNameMappingError(
            "the .top contains no water moleculetype (3 atoms with masses O,H,H)."
            " Either the system has no water, or it uses a 4-or-more-point water model." + detail
        )
    if len(candidates) > 1:
        raise WaterNameMappingError(
            f"the .top has several water moleculetypes: {candidates}."
            " Which one the added water belongs in cannot be decided, so state it"
            " explicitly with water_moleculetype=."
        )
    return candidates[0]


@dataclass(frozen=True)
class WaterFillResult:
    """Result of the filling and remapping.

    Attributes:
        gro: the ``.gro`` text after remapping.
        top: the ``.top`` text after remapping.
        n_added_molecules: number of water molecules added.
        n_added_atoms: number of atoms added.
        added_resname: residue name on the solvent-box side.
        water_moleculetype: name of the moleculetype the water was remapped to.
        atom_name_map: ``(solvent-box atom name, remapped atom name)``, in remapped order.
        reorder: position after remapping -> position on the solvent-box side.
            Identity is ``(0, 1, 2)``.
        mass_amu: mass of the system after filling.
        volume_a3: box volume after filling.
        density_g_cm3: density after filling.
    """

    gro: str
    top: str
    n_added_molecules: int
    n_added_atoms: int
    added_resname: str
    water_moleculetype: str
    atom_name_map: tuple[tuple[str, str], ...]
    reorder: tuple[int, ...]
    mass_amu: float
    volume_a3: float
    density_g_cm3: float


def relabel_added_water(
    gro_before: str,
    gro_after: str,
    top_before: str,
    top_after: str,
    *,
    water_moleculetype: str | None = None,
    target_density_g_cm3: float | None = None,
    density_relative_tolerance: float = DENSITY_RELATIVE_TOLERANCE,
    box_tolerance_nm: float = BOX_TOLERANCE_NM,
) -> WaterFillResult:
    """Remap the water ``gmx solvate`` added onto the ``.top`` water moleculetype.

    What was added is decided by the before/after difference, never by assuming a
    residue name: in the ``.gro`` the first ``n_before`` atoms are unchanged and the
    addition is appended, and in the ``.top`` the original ``[ molecules ]`` sequence is
    a prefix with the append at the end. Either assumption failing raises.

    The append stays a separate ``[ molecules ]`` line rather than being added into an
    existing water line: the ``[ molecules ]`` order has to match the ``.gro`` atom
    order, and the added water sits at the end of the ``.gro``. Merging it would assert
    a different order, which grompp accepts as long as the atom counts match and which
    then reads the wrong molecule's coordinates.

    Two further things are verified: that the box vectors are unchanged
    (:class:`BoxChangedError`), and that the density is near the target
    (:class:`SystemDensityError`, see :data:`DENSITY_RELATIVE_TOLERANCE`).

    Args:
        gro_before: the ``.gro`` before filling.
        gro_after: the ``.gro`` ``gmx solvate`` wrote.
        top_before: the ``.top`` before filling.
        top_after: the ``.top`` ``gmx solvate`` rewrote.
        water_moleculetype: the moleculetype to remap onto; ``None`` finds it with
            :func:`find_water_moleculetype`.
        target_density_g_cm3: the density to check against; ``None`` uses
            :func:`expected_density_g_cm3`.
        density_relative_tolerance: relative tolerance around that target.
        box_tolerance_nm: tolerance for judging the box unchanged.
    """
    _title_b, n_before, _atoms_b, _box_b = _gro_lines(gro_before)
    title, n_after, atom_lines, box_line = _gro_lines(gro_after)

    # --- The box is unchanged ---
    box_b = gro_box_nm(gro_before)
    box_a = gro_box_nm(gro_after)
    if len(box_b) != len(box_a) or any(
        abs(x - y) > box_tolerance_nm for x, y in zip(box_b, box_a, strict=False)
    ):
        raise BoxChangedError(
            "filling changed the box vectors, breaking the intent of the padding.\n"
            f"  before: {box_b}\n  after:  {box_a}\n"
            f"  tolerance: {box_tolerance_nm} nm\n"
            "gmx solvate does not change the box, so check that no -box or similar"
            " argument slipped in."
        )

    if n_after < n_before:
        raise WaterFillError(
            f"the atom count after filling, {n_after}, is below the {n_before} before."
            " A stage that is supposed to add water has lost atoms."
        )

    # --- Extract what was appended to the .top ---
    molecules_before = parse_molecules_section(top_before)
    molecules_after = parse_molecules_section(top_after)
    if molecules_after[: len(molecules_before)] != molecules_before:
        raise WaterFillError(
            "the existing part of [ molecules ] has been rewritten. gmx solvate should only"
            " append at the end, so the assumption is broken.\n"
            f"  before: {molecules_before}\n"
            f"  after:  {molecules_after[: len(molecules_before)]}"
        )
    added = molecules_after[len(molecules_before) :]

    wmt = water_moleculetype or find_water_moleculetype(top_before)
    types = parse_moleculetypes(top_before)
    if wmt not in types:
        raise WaterNameMappingError(
            f"the water moleculetype {wmt!r} is not in the .top."
            f" Defined: {sorted(types)}"
        )
    target = types[wmt]

    if not added:
        # Nothing went in. There is nothing to remap, but the density is still checked,
        # because proceeding while short shrinks the box.
        result = WaterFillResult(
            gro=gro_after,
            top=top_after,
            n_added_molecules=0,
            n_added_atoms=0,
            added_resname="",
            water_moleculetype=wmt,
            atom_name_map=(),
            reorder=(),
            mass_amu=total_mass_amu(top_after),
            volume_a3=gro_box_volume_a3(gro_after),
            density_g_cm3=0.0,
        )
        return _finish(result, target_density_g_cm3, density_relative_tolerance)

    added_names = {name for name, _ in added}
    if len(added_names) != 1:
        raise WaterFillError(
            f"the append to [ molecules ] is not a single kind: {added}."
            " Check that gmx solvate was not asked to insert several solvents."
        )
    added_resname = added[0][0]
    n_added_molecules = sum(count for _name, count in added)
    n_added_atoms = n_after - n_before
    if n_added_atoms != n_added_molecules * target.n_atoms:
        raise WaterFillError(
            f"the {n_added_atoms} atoms gained in the .gro do not match the .top append of"
            f" {n_added_molecules} molecules x {target.n_atoms} atoms"
            f" = {n_added_molecules * target.n_atoms}."
            " Check that the .gro and .top come from the same filling run."
        )

    # --- Cut the .gro addition into residues ---
    added_lines = atom_lines[n_before:]
    residues = _gro_residues(added_lines)
    if len(residues) != n_added_molecules:
        raise WaterFillError(
            f"the .gro addition is {len(residues)} residues but the .top append is"
            f" {n_added_molecules} molecules. The counts do not match."
        )

    source_names = [added_lines[i][10:15].strip() for i in residues[0][2]]
    reorder = _atom_order_map(source_names, target)
    name_map = tuple(
        (source_names[reorder[k]], target.atom_names[k])
        for k in range(target.n_atoms)
    )

    # --- Rewrite the added lines ---
    new_lines = list(atom_lines[:n_before])
    serial = n_before
    for resi, resn, idxs in residues:
        if resn != added_resname:
            raise WaterFillError(
                f"the residue name of the .gro addition, {resn!r}, differs from the .top"
                f" append, {added_resname!r}"
            )
        if len(idxs) != target.n_atoms:
            raise WaterFillError(
                f"added residue {resi} in the .gro has {len(idxs)} atoms."
                f" Water must have {target.n_atoms}."
            )
        names_here = [added_lines[i][10:15].strip() for i in idxs]
        if names_here != source_names:
            raise WaterFillError(
                f"the atom names of the .gro addition differ between residues:"
                f" {source_names} vs {names_here} (residue {resi})."
            )
        for k in range(target.n_atoms):
            src = added_lines[idxs[reorder[k]]]
            serial += 1
            # Column 21 onwards (coordinates and velocities) is carried over verbatim.
            new_lines.append(
                f"{resi % 100000:5d}{wmt:<5s}{target.atom_names[k]:>5s}"
                f"{serial % 100000:5d}{src[20:]}"
            )

    new_gro = "\n".join([title, f"{n_after:5d}", *new_lines, box_line]) + "\n"
    new_top = _rename_added_molecules(top_after, len(molecules_before), wmt)

    result = WaterFillResult(
        gro=new_gro,
        top=new_top,
        n_added_molecules=n_added_molecules,
        n_added_atoms=n_added_atoms,
        added_resname=added_resname,
        water_moleculetype=wmt,
        atom_name_map=name_map,
        reorder=tuple(reorder),
        mass_amu=total_mass_amu(new_top),
        volume_a3=gro_box_volume_a3(new_gro),
        density_g_cm3=0.0,
    )
    return _finish(result, target_density_g_cm3, density_relative_tolerance)


def _finish(
    result: WaterFillResult, target: float | None, tolerance: float
) -> WaterFillResult:
    """Compute and verify the density, returning a filled-in :class:`WaterFillResult`.

    ``target`` of ``None`` means the density this composition should have
    (:func:`expected_density_g_cm3`).
    """
    density = density_g_cm3(result.mass_amu, result.volume_a3)
    result = dataclasses.replace(result, density_g_cm3=density)
    if target is None:
        target = expected_density_g_cm3(result.top)
        source = "expected from the composition"
    else:
        source = "given"
    lo = target * (1.0 - tolerance)
    hi = target * (1.0 + tolerance)
    if not (lo <= density <= hi):
        raise SystemDensityError(
            f"the density after filling, {density:.4f} g/cm^3, is outside the tolerance"
            f" [{lo:.4f}, {hi:.4f}] ({source}: {target:.4f}"
            f" +- {tolerance * 100:.0f}%).\n"
            f"  mass {result.mass_amu:.1f} amu / volume {result.volume_a3:.0f} A^3"
            f" / water added {result.n_added_molecules} molecules\n"
            "  If it is too low, water is still missing, and NPT will shrink the box"
            " and move the probe concentration off its nominal value. Check the log"
            " for how many molecules gmx solvate actually inserted."
        )
    return result


def _atom_order_map(source_names: Sequence[str], target: TopMoleculeType) -> list[int]:
    """Match the solvent-box atoms onto the ``.top`` water atom order.

    ``reorder[k]`` is the solvent-box position corresponding to the k-th atom of the
    ``.top``. Atoms of matching elements are assigned in order of appearance; a
    differing order gives a non-identity ``reorder`` and the coordinates are reordered.

    Raises:
        WaterNameMappingError: when the atom counts or elemental makeup differ.
    """
    if len(source_names) != target.n_atoms:
        raise WaterNameMappingError(
            f"the solvent-box water has {len(source_names)} atoms ({list(source_names)}) but"
            f" {target.name} in the .top has {target.n_atoms}"
            f" ({list(target.atom_names)}). They cannot be matched."
        )
    source_elements = [_water_element_from_atom_name(n) for n in source_names]
    target_elements = list(target.elements())
    if sorted(source_elements) != sorted(target_elements):
        raise WaterNameMappingError(
            f"different elemental makeup: solvent box {source_elements} vs"
            f" {target.name} in the .top {target_elements}. They cannot be matched."
        )
    used = [False] * len(source_elements)
    reorder: list[int] = []
    for element in target_elements:
        for i, (e, taken) in enumerate(zip(source_elements, used, strict=True)):
            if not taken and e == element:
                used[i] = True
                reorder.append(i)
                break
        else:  # pragma: no cover - excluded by the multiset match above
            raise WaterNameMappingError(
                f"no solvent-box atom corresponds to element {element}"
            )
    return reorder


def _rename_added_molecules(
    top_string: str, n_original_entries: int, new_name: str
) -> str:
    """Rename the ``[ molecules ]`` lines from ``n_original_entries`` onwards.

    The addition stays its own line rather than being merged into an existing one of
    the same name; see :func:`relabel_added_water`.
    """
    out: list[str] = []
    section: str | None = None
    seen = 0
    for raw in top_string.split("\n"):
        sec = _section_name(raw)
        if sec is not None:
            section = sec
            out.append(raw)
            continue
        stripped = _strip_comment(raw).strip()
        if (
            section != "molecules"
            or not stripped
            or stripped.startswith("#")      # preprocessor lines are not counted
        ):
            out.append(raw)
            continue
        fields = stripped.split()
        if seen >= n_original_entries:
            out.append(f"{new_name:<15s} {fields[1]}")
        else:
            out.append(raw)
        seen += 1
    if seen <= n_original_entries:
        raise WaterFillError(
            f"no append found in [ molecules ] ({seen} lines,"
            f" {n_original_entries} before filling)"
        )
    return "\n".join(out)


def resolve_gmx_command(gmx: str | Sequence[str] | None = None) -> list[str]:
    """Decide the ``gmx`` command.

    Priority: the ``gmx`` argument, then the ``MSMD_GMX`` environment variable, then
    ``"gmx"``. The string is split shell-style, so a wrapper command works too.
    """
    if gmx is None:
        gmx = os.environ.get("MSMD_GMX") or "gmx"
    argv = shlex.split(gmx) if isinstance(gmx, str) else [str(x) for x in gmx]
    if not argv:
        raise ValueError("the gmx command is empty")
    if shutil.which(argv[0]) is None and not Path(argv[0]).exists():
        raise FileNotFoundError(
            f"the GROMACS executable {argv[0]!r} cannot be found.\n"
            "System building uses gmx solvate to fix the water shortfall. Do one of:\n"
            "  * put GROMACS on PATH\n"
            "  * set the MSMD_GMX environment variable to the command\n"
            "  * pass it explicitly as build_system(..., gmx=[...])"
        )
    return argv


def fill_water_with_gmx(
    gro_path: str | Path,
    top_path: str | Path,
    *,
    out_gro: str | Path,
    out_top: str | Path,
    gmx: str | Sequence[str] | None = None,
    solvent_gro: str = DEFAULT_SOLVENT_GRO,
    water_moleculetype: str | None = None,
    target_density_g_cm3: float | None = None,
    density_relative_tolerance: float = DENSITY_RELATIVE_TOLERANCE,
    extra_args: Sequence[str] = (),
) -> WaterFillResult:
    """Add water with ``gmx solvate``, remap it onto the ``.top`` water, and save.

    ``gro_path`` and ``top_path`` are not modified; the result goes to ``out_gro`` and
    ``out_top``.

    Args:
        gro_path: the ``.gro`` to fill.
        top_path: its ``.top``.
        out_gro: where the filled ``.gro`` is written.
        out_top: where the filled ``.top`` is written.
        gmx: the ``gmx`` command, as in :func:`resolve_gmx_command`.
        solvent_gro: the solvent box passed to ``-cs``.
        water_moleculetype: passed to :func:`relabel_added_water`.
        target_density_g_cm3: passed to :func:`relabel_added_water`.
        density_relative_tolerance: passed to :func:`relabel_added_water`.
        extra_args: further arguments appended to the ``gmx solvate`` command line.

    Raises:
        FileNotFoundError: when ``gmx`` cannot be found.
        subprocess.CalledProcessError: when ``gmx solvate`` fails.
        WaterFillError: when the remapping, box or density are inconsistent.
    """
    argv = resolve_gmx_command(gmx)
    gro_path = Path(gro_path)
    top_path = Path(top_path)
    out_gro = Path(out_gro)
    out_top = Path(out_top)

    gro_before = gro_path.read_text()
    top_before = top_path.read_text()

    # gmx solvate -p rewrites the .top it is given in place, so copy it to the output
    # side first and leave the pre-filling .top intact.
    out_top.write_text(top_before)

    # When gmx solvate updates the .top it creates temp.topXXXXXX in the current
    # directory and renames it to the -p path, which fails with
    #   "System I/O error: Failed to rename temp.topXXXXXX to <destination>"
    # whenever cwd differs from the -p directory. So the output directory becomes the
    # cwd and the file names are passed relative to it.
    workdir = out_top.parent
    workdir.mkdir(parents=True, exist_ok=True)

    def _arg(path: Path) -> str:
        """Relative if it can be based on workdir, otherwise left absolute."""
        try:
            return str(path.resolve().relative_to(workdir.resolve()))
        except ValueError:
            return str(path.resolve())

    cmd = [
        *argv,
        "solvate",
        "-cp", _arg(gro_path),
        "-cs", solvent_gro,
        "-o", _arg(out_gro),
        "-p", _arg(out_top),
        *extra_args,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(workdir), check=False)
    if proc.returncode != 0:
        # CalledProcessError's default text does not include stderr, so the tail is put
        # into the exception message and the log alone explains the cause.
        tail = "\n".join(
            (proc.stderr or "").strip().splitlines()[-25:]
        ) or "(stderr was empty)"
        out_tail = "\n".join(
            (proc.stdout or "").strip().splitlines()[-10:]
        ) or "(stdout was empty)"
        raise subprocess.CalledProcessError(
            proc.returncode, cmd,
            output=proc.stdout, stderr=proc.stderr,
        ) from RuntimeError(
            f"gmx solvate failed with rc={proc.returncode}.\n"
            f"--- stderr (last 25 lines) ---\n{tail}\n"
            f"--- stdout (last 10 lines) ---\n{out_tail}"
        )

    result = relabel_added_water(
        gro_before,
        out_gro.read_text(),
        top_before,
        out_top.read_text(),
        water_moleculetype=water_moleculetype,
        target_density_g_cm3=target_density_g_cm3,
        density_relative_tolerance=density_relative_tolerance,
    )
    out_gro.write_text(result.gro)
    out_top.write_text(result.top)
    return result
