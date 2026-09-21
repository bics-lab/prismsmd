"""Reading mol2 files and mapping GAFF atom types to element symbols."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Mol2Molecule",
    "UnsupportedElementError",
    "gaff_type_to_element",
    "mol2_with_element_types",
    "parse_mol2",
    "read_mol2",
]


class UnsupportedElementError(ValueError):
    """Raised when an atom type cannot be reduced to an element symbol."""


#: GAFF/GAFF2 types whose two characters are the element symbol itself.
_GAFF_TWO_LETTER = {"cl": "Cl", "br": "Br", "si": "Si"}

#: First character of a GAFF type -> element symbol.
_GAFF_FIRST_LETTER = {
    "c": "C",
    "h": "H",
    "n": "N",
    "o": "O",
    "s": "S",
    "p": "P",
    "f": "F",
    "i": "I",
}


def gaff_type_to_element(atom_type: str) -> str:
    """Reduce an atom type to an element symbol.

    An all-lowercase type is read as GAFF: the first character gives the element,
    except for the two-character types ``cl`` / ``br`` / ``si``. A type containing
    an uppercase letter is read as SYBYL (``C.3``, ``N.ar``, ``Na``). Raises
    :class:`UnsupportedElementError` if neither applies.

    >>> gaff_type_to_element("c3")
    'C'
    >>> gaff_type_to_element("ca")     # aromatic carbon
    'C'
    >>> gaff_type_to_element("na")     # aromatic nitrogen
    'N'
    >>> gaff_type_to_element("cl")
    'Cl'
    >>> gaff_type_to_element("N.ar")
    'N'
    """
    t = (atom_type or "").strip()
    if not t:
        raise ValueError("empty atom type")

    if t != t.lower():
        symbol = t.split(".")[0]
        if len(symbol) > 2:
            raise UnsupportedElementError(f"atom type cannot be reduced to an element: {t!r}")
        return symbol.capitalize()

    if t in _GAFF_TWO_LETTER:
        return _GAFF_TWO_LETTER[t]
    if t[0] in _GAFF_FIRST_LETTER:
        return _GAFF_FIRST_LETTER[t[0]]
    raise UnsupportedElementError(f"unknown GAFF atom type: {t!r}")


@dataclass
class Mol2Molecule:
    """Contents of a mol2 file, in the file's own atom order.

    Attributes
    ----------
    bonds:
        0-based adjacency list.
    raw_text:
        The original file contents.
    """

    name: str
    atom_names: list[str]
    gaff_types: list[str]
    elements: list[str]
    coords: list[tuple[float, float, float]]
    charges: list[float]
    bonds: list[list[int]] = field(default_factory=list)
    raw_text: str = ""

    def __len__(self) -> int:
        return len(self.atom_names)

    def heavy_atom_indices(self) -> list[int]:
        """Indices of the non-hydrogen atoms."""
        return [i for i, e in enumerate(self.elements) if e != "H"]


_SECTION = re.compile(r"^@<TRIPOS>(\S+)")


def read_mol2(path: str | Path) -> Mol2Molecule:
    """Read a mol2 file."""
    text = Path(path).read_text()
    return parse_mol2(text)


def parse_mol2(text: str) -> Mol2Molecule:
    """Parse mol2 text into a :class:`Mol2Molecule`. Raises if it has no atoms."""
    section = None
    name = "UNL"
    names: list[str] = []
    gaff: list[str] = []
    coords: list[tuple[float, float, float]] = []
    charges: list[float] = []
    bond_pairs: list[tuple[int, int]] = []
    mol_line = 0

    for line in text.splitlines():
        m = _SECTION.match(line.strip())
        if m:
            section = m.group(1).upper()
            mol_line = 0
            continue
        if not line.strip():
            continue
        if section == "MOLECULE":
            if mol_line == 0:
                name = line.strip()
            mol_line += 1
        elif section == "ATOM":
            f = line.split()
            if len(f) < 6:
                raise ValueError(f"mol2 ATOM line is too short: {line!r}")
            names.append(f[1])
            coords.append((float(f[2]), float(f[3]), float(f[4])))
            gaff.append(f[5])
            charges.append(float(f[8]) if len(f) > 8 else 0.0)
        elif section == "BOND":
            f = line.split()
            if len(f) < 4:
                continue
            bond_pairs.append((int(f[1]) - 1, int(f[2]) - 1))

    if not names:
        raise ValueError("the mol2 has no atoms")

    adjacency: list[list[int]] = [[] for _ in names]
    for i, j in bond_pairs:
        if not (0 <= i < len(names) and 0 <= j < len(names)):
            raise ValueError(f"mol2 BOND out of range: {i + 1}-{j + 1}")
        adjacency[i].append(j)
        adjacency[j].append(i)

    return Mol2Molecule(
        name=name,
        atom_names=names,
        gaff_types=gaff,
        elements=[gaff_type_to_element(t) for t in gaff],
        coords=coords,
        charges=charges,
        bonds=adjacency,
        raw_text=text,
    )


def mol2_with_element_types(mol: Mol2Molecule) -> str:
    """Return the mol2 text with the atom-type column replaced by element symbols.

    Atom names, coordinates, charges and bonds are unchanged. RDKit cannot read
    GAFF types, but can read the result.
    """
    out: list[str] = []
    section = None
    idx = 0
    for line in mol.raw_text.splitlines(keepends=True):
        m = _SECTION.match(line.strip())
        if m:
            section = m.group(1).upper()
            out.append(line)
            continue
        if section == "ATOM" and line.strip():
            f = line.split()
            f[5] = mol.elements[idx]
            idx += 1
            out.append(
                "%7s %-8s %10s %10s %10s %-6s %4s %-6s %10s\n"  # noqa: UP031
                % tuple(f[:9] + [""] * (9 - len(f)))
            )
            continue
        out.append(line)
    return "".join(out)
