"""Atom order of a probe molecule, resolved against its SMILES."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ProbeAtomOrder", "resolve_probe_atom_order"]


@dataclass(frozen=True)
class ProbeAtomOrder:
    """Atom order of one probe molecule, and the identity of each atom.

    Attributes
    ----------
    probe:
        Probe identifier.
    smiles:
        SMILES that defines the order.
    elements:
        Element symbols, in order.
    mol2_names:
        mol2 atom names, in the same order.
    openff_names:
        Atom names OpenFF assigns, in the same order.
    mol2_order:
        ``mol2_order[k]`` is the 0-based mol2 index of the atom at position k.
    bonds:
        Bonds as index pairs, in the same ordering.
    """

    probe: str
    smiles: str
    elements: tuple[str, ...]
    mol2_names: tuple[str, ...]
    openff_names: tuple[str, ...]
    mol2_order: tuple[int, ...]
    bonds: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        n = len(self.elements)
        for name, seq in (
            ("mol2_names", self.mol2_names),
            ("openff_names", self.openff_names),
            ("mol2_order", self.mol2_order),
        ):
            if len(seq) != n:
                raise ValueError(
                    f"probe {self.probe}: the length of {name}, {len(seq)},"
                    f" differs from the atom count {n}"
                )

    def heavy_indices(self) -> list[int]:
        """Indices of the non-hydrogen atoms, ascending."""
        return [i for i, e in enumerate(self.elements) if e.upper() != "H"]

    def index_of(self, mol2_name: str) -> int:
        """Index of the atom with this mol2 name. Raises KeyError if absent."""
        try:
            return self.mol2_names.index(mol2_name)
        except ValueError:
            raise KeyError(
                f"probe {self.probe}: atom {mol2_name!r} is not in the mol2"
                f" (names: {list(self.mol2_names)})"
            ) from None

    def to_dict(self) -> dict:
        """Return the contents as plain JSON-serialisable types."""
        return {
            "probe": self.probe,
            "smiles": self.smiles,
            "elements": list(self.elements),
            "mol2_names": list(self.mol2_names),
            "openff_names": list(self.openff_names),
            "mol2_order": list(self.mol2_order),
            "bonds": [list(b) for b in self.bonds],
        }


def _bonds_in_new_order(
    adjacency: Sequence[Sequence[int]], order: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    """Re-index a mol2 adjacency list into (i, j) pairs in the new order, i < j."""
    inv = [-1] * len(order)
    for new_index, old_index in enumerate(order):
        inv[old_index] = new_index
    pairs = set()
    for old_i, neighbours in enumerate(adjacency):
        for old_j in neighbours:
            i, j = inv[old_i], inv[old_j]
            pairs.add((min(i, j), max(i, j)))
    return tuple(sorted(pairs))


def resolve_probe_atom_order(
    mol2_path: str | Path, smiles: str, *, probe_id: str = ""
) -> ProbeAtomOrder:
    """Build a :class:`ProbeAtomOrder` from a mol2 file and a SMILES.

    The SMILES defines the order; the mol2 is mapped onto it by graph
    isomorphism. Raises if no mapping exists or the element sequences differ.
    """
    from .mol2 import read_mol2
    from .probeprep import (
        map_to_smiles_order,
        mol_from_mol2,
        openff_atom_names,
        smiles_reference_elements,
    )

    label = probe_id or Path(mol2_path).stem
    mol = read_mol2(mol2_path)
    rdmol = mol_from_mol2(mol)
    order = map_to_smiles_order(rdmol, smiles, label=label)

    elements = [mol.elements[i] for i in order]
    reference = smiles_reference_elements(smiles)
    if elements != reference:
        raise ValueError(
            f"probe {label}: the reordered element sequence {elements} does not match"
            f" the element sequence of Molecule.from_smiles, {reference}"
        )
    return ProbeAtomOrder(
        probe=label,
        smiles=smiles,
        elements=tuple(elements),
        mol2_names=tuple(mol.atom_names[i] for i in order),
        openff_names=tuple(openff_atom_names(elements)),
        mol2_order=tuple(order),
        bonds=_bonds_in_new_order(mol.bonds, order),
    )
