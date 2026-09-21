"""mol2 -> sdf conversion for CosolvKit, and validation of that conversion.

CosolvKit's ``CosolventMolecule`` does not accept mol2; it reads mol/sdf with
``Chem.MolFromMolFile()`` and then takes ``mol.GetConformer().GetPositions()``, so
coordinates are mandatory. The conversion path is

    mol2 -> (column 6 replaced by element symbols) -> RDKit -> reorder to SMILES order -> sdf

**The sdf must be written in the same atom order as ``Molecule.from_smiles(smiles)``**,
because CosolvKit builds the topology from the SMILES and takes the positions from the
sdf. A disagreement puts an oxygen's coordinates on a carbon, raising no exception.
Nothing guarantees a mol2 follows that order, so this module maps mol2 atoms onto SMILES
atoms by graph isomorphism and writes the sdf in that order, with the coordinates
following. :func:`prismsmd.system.builder.verify_probe_atoms_in_gro` re-checks it on the
generated ``.gro``.

That the mol2 and the sdf are the same molecule is confirmed by the atom count, the
elemental composition, the bond count, the coordinates (tracking the reordering), the
canonical SMILES surviving a write-and-reread, the sdf element sequence matching
``Molecule.from_smiles``, the absence of radical electrons, and -- when
``reference_smiles`` is given -- the formal charge.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from scipy import constants as C

from .mol2 import Mol2Molecule, mol2_with_element_types, read_mol2

__all__ = [
    "NM_TO_ANGSTROM",
    "AtomOrderError",
    "ProbeConversionError",
    "ProbeConversionReport",
    "canonical_smiles",
    "element_from_atom_name",
    "map_to_smiles_order",
    "mol_from_mol2",
    "openff_atom_names",
    "prepare_probe_sdf",
    "read_sdf_atoms",
    "smiles_reference_elements",
    "smiles_reference_mol",
]

#: nm -> A. ``.gro`` is in nm while bond lengths are quoted in A.
NM_TO_ANGSTROM = C.nano / C.angstrom


class ProbeConversionError(ValueError):
    """Raised when validation of the mol2 -> sdf conversion fails."""


class AtomOrderError(ProbeConversionError):
    """Raised when mol2 atoms could not be matched to the SMILES atoms.

    Proceeding silently would assign coordinates to the wrong atoms.
    """


@dataclass
class ProbeConversionReport:
    """Record of the conversion, kept in the provenance."""

    cid: str
    mol2_path: str
    sdf_path: str
    n_atoms: int
    formula: dict[str, int]
    n_bonds: int
    formal_charge: int
    n_radical_electrons: int
    smiles: str
    #: Whether the canonical SMILES survives a write-and-reread through sdf.
    smiles_roundtrip_ok: bool
    #: Whether the sdf atom order matches the mol2 (judged by element sequence).
    #: False is not an anomaly; it states that reordering was needed.
    atom_order_preserved: bool
    #: Whether coordinates are preserved (compared while tracking the reordering).
    coords_preserved: bool
    #: The SMILES the reordering was based on. ``None`` means no reordering.
    order_reference_smiles: str | None = None
    #: ``mol2_order[k]`` = the 0-based mol2 index of the atom at sdf position k.
    mol2_order: list[int] = field(default_factory=list)
    #: Element sequence in sdf atom order.
    elements: list[str] = field(default_factory=list)
    #: mol2 atom names arranged in sdf atom order, as labels.
    mol2_names: list[str] = field(default_factory=list)
    #: The names OpenFF Toolkit assigns, ``f"{element}{serial}x"``. Derived, not primary.
    openff_names: list[str] = field(default_factory=list)
    #: Whether the sdf element sequence matches that of ``Molecule.from_smiles``.
    smiles_order_ok: bool | None = None
    #: Whether it matches the config's smiles, when one was supplied.
    matches_reference_smiles: bool | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the report as a plain dict."""
        return {
            "cid": self.cid,
            "mol2": self.mol2_path,
            "sdf": self.sdf_path,
            "n_atoms": self.n_atoms,
            "formula": self.formula,
            "n_bonds": self.n_bonds,
            "formal_charge": self.formal_charge,
            "n_radical_electrons": self.n_radical_electrons,
            "smiles": self.smiles,
            "smiles_roundtrip_ok": self.smiles_roundtrip_ok,
            "atom_order_preserved": self.atom_order_preserved,
            "coords_preserved": self.coords_preserved,
            "order_reference_smiles": self.order_reference_smiles,
            "mol2_order": self.mol2_order,
            "elements": self.elements,
            "mol2_names": self.mol2_names,
            "openff_names": self.openff_names,
            "smiles_order_ok": self.smiles_order_ok,
            "matches_reference_smiles": self.matches_reference_smiles,
            "warnings": self.warnings,
        }


def _require_rdkit():
    try:
        from rdkit import Chem, RDLogger
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "rdkit is required:\n"
            "  mamba install -c conda-forge rdkit"
        ) from exc
    RDLogger.DisableLog("rdApp.warning")
    return Chem


def mol_from_mol2(mol: Mol2Molecule):
    """Build an RDKit Mol from a :class:`Mol2Molecule`.

    Column 6 (the GAFF type) is replaced with element symbols before reading. Atom
    names, coordinates, charges and bonds are untouched.
    """
    Chem = _require_rdkit()
    block = mol2_with_element_types(mol)
    rdmol = Chem.MolFromMol2Block(block, removeHs=False, sanitize=True)
    if rdmol is None:
        raise ProbeConversionError(
            f"{mol.name}: RDKit could not read the mol2.\n"
            f"Check the result of the GAFF type -> element symbol conversion: "
            f"{sorted(set(zip(mol.gaff_types, mol.elements, strict=False)))}"
        )
    return rdmol


def canonical_smiles(rdmol, *, keep_hydrogens: bool = False) -> str:
    """Canonical SMILES of a Mol. Hydrogens are dropped by default."""
    Chem = _require_rdkit()
    m = rdmol if keep_hydrogens else Chem.RemoveHs(Chem.Mol(rdmol))
    return Chem.MolToSmiles(m)


def _formula(rdmol) -> dict[str, int]:
    return dict(sorted(Counter(a.GetSymbol() for a in rdmol.GetAtoms()).items()))


def openff_atom_names(elements: Sequence[str]) -> list[str]:
    """Reproduce the atom names OpenFF Toolkit reassigns.

    ``_ensure_unique_atom_names("residues")`` renames the atoms of a molecule built
    from ``Molecule.from_smiles`` as element symbol + per-element serial + ``"x"``, so
    the mol2 names are not preserved. These names are derived values; the
    implementation works on indices.

    >>> openff_atom_names(["C", "C", "O", "H", "H"])
    ['C1x', 'C2x', 'O1x', 'H1x', 'H2x']
    """
    counter: dict[str, int] = {}
    out: list[str] = []
    for el in elements:
        counter[el] = counter.get(el, 0) + 1
        out.append(f"{el}{counter[el]}x")
    return out


def smiles_reference_mol(smiles: str):
    """Return an RDKit Mol in the same atom order as ``Molecule.from_smiles(smiles)``.

    openff-toolkit's ``RDKitToolkitWrapper.from_smiles`` goes ``MolFromSmiles`` ->
    ``AddHs`` -> ``from_rdkit`` and preserves that order. Where openff-toolkit can be
    imported, the element sequence is cross-checked against the real thing.

    Raises:
        AtomOrderError: if the smiles cannot be read, or openff-toolkit disagrees.
    """
    Chem = _require_rdkit()
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise AtomOrderError(f"RDKit cannot read the smiles: {smiles!r}")
    m = Chem.AddHs(m)
    elements = [a.GetSymbol() for a in m.GetAtoms()]
    _crosscheck_openff_order(smiles, elements)
    return m


def smiles_reference_elements(smiles: str) -> list[str]:
    """Element sequence of ``Molecule.from_smiles(smiles)``."""
    return [a.GetSymbol() for a in smiles_reference_mol(smiles).GetAtoms()]


def _crosscheck_openff_order(smiles: str, elements: Sequence[str]) -> None:
    """Cross-check against openff-toolkit's atom order, if it is installed."""
    try:
        from openff.toolkit.topology import Molecule  # type: ignore
    except Exception:  # noqa: BLE001  # pragma: no cover - optional dependency
        return
    try:
        off = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise AtomOrderError(
            f"openff-toolkit cannot read the smiles {smiles!r}: {exc}"
        ) from exc
    off_elements = [a.symbol for a in off.atoms]
    if off_elements != list(elements):
        raise AtomOrderError(
            f"the atom-order assumption is broken: smiles {smiles!r}\n"
            f"  RDKit AddHs(MolFromSmiles(...)) : {list(elements)}\n"
            f"  openff Molecule.from_smiles(...): {off_elements}\n"
            "CosolvKit builds the topology from Molecule.from_smiles, so the sdf"
            " must follow openff's ordering."
            " Fix the implementation of smiles_reference_mol() to match openff's."
        )


def map_to_smiles_order(rdmol, smiles: str, *, label: str = "") -> list[int]:
    """Return the permutation that reorders a mol2-derived Mol into SMILES order.

    ``new_order[k]`` is the ``rdmol`` atom index that goes to new position k, ready to
    pass to ``Chem.RenumberAtoms(rdmol, new_order)``. The correspondence is decided by
    graph isomorphism (a bijection including hydrogens, via ``GetSubstructMatch``),
    never by guessing from names or order. For a symmetric molecule several
    isomorphisms exist; any of them preserves the graph, and ``GetSubstructMatch`` is
    deterministic, so the choice is reproducible.

    Raises:
        AtomOrderError: if no isomorphism exists.
    """
    # Raise here if rdkit is missing (the return value is unused)
    _require_rdkit()
    what = f"probe {label}: " if label else ""
    ref = smiles_reference_mol(smiles)

    n = rdmol.GetNumAtoms()
    if ref.GetNumAtoms() != n:
        raise AtomOrderError(
            f"{what}atom counts differ: mol2 {n} / smiles {smiles!r} "
            f"{ref.GetNumAtoms()}. Not the same molecule."
        )

    # Take the bijection with ref as target and the mol2-derived Mol as query.
    # match[i] = the ref atom index corresponding to mol2 atom i
    match = ref.GetSubstructMatch(rdmol, useChirality=False)
    if len(match) != n or sorted(match) != list(range(n)):
        raise AtomOrderError(
            f"{what}no graph isomorphism between the mol2 and smiles {smiles!r}"
            f" (GetSubstructMatch = {tuple(match)}).\n"
            f"  mol2 element sequence  : {[a.GetSymbol() for a in rdmol.GetAtoms()]}\n"
            f"  smiles element sequence: {[a.GetSymbol() for a in ref.GetAtoms()]}\n"
            "Bond orders, formal charges or hydrogen counts disagree somewhere."
            " Stopping is safer than assigning coordinates to the wrong atoms."
        )

    new_order = [-1] * n
    for mol2_index, ref_index in enumerate(match):
        new_order[ref_index] = mol2_index

    _assert_order_is_isomorphism(rdmol, ref, new_order, what=what, smiles=smiles)
    return new_order


def _assert_order_is_isomorphism(rdmol, ref, new_order, *, what: str, smiles: str) -> None:
    """Verify independently that the reordering is a graph-preserving bijection,
    by cross-checking the element sequence, the formal charges and the bond set."""
    n = rdmol.GetNumAtoms()
    problems: list[str] = []

    src = list(rdmol.GetAtoms())
    dst = list(ref.GetAtoms())
    for k, i in enumerate(new_order):
        if src[i].GetSymbol() != dst[k].GetSymbol():
            problems.append(
                f"position {k}: mol2 atom {i} is {src[i].GetSymbol()} but"
                f" the smiles side is {dst[k].GetSymbol()}"
            )
        if src[i].GetFormalCharge() != dst[k].GetFormalCharge():
            problems.append(
                f"position {k}: formal charges differ "
                f"({src[i].GetFormalCharge():+d} vs {dst[k].GetFormalCharge():+d})"
            )

    inv = [-1] * n
    for k, i in enumerate(new_order):
        inv[i] = k
    src_bonds = {
        frozenset((inv[b.GetBeginAtomIdx()], inv[b.GetEndAtomIdx()]))
        for b in rdmol.GetBonds()
    }
    dst_bonds = {
        frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in ref.GetBonds()
    }
    if src_bonds != dst_bonds:
        problems.append(
            f"the bond sets do not match (mol2 only {sorted(map(sorted, src_bonds - dst_bonds))},"
            f" smiles only {sorted(map(sorted, dst_bonds - src_bonds))})"
        )

    if problems:
        raise AtomOrderError(
            f"{what}invalid mol2 -> smiles {smiles!r} atom correspondence:\n  "
            + "\n  ".join(problems)
        )


def read_sdf_atoms(path: str | Path) -> tuple[list[str], list[tuple[int, int]]]:
    """Return ``(element sequence, bond index pairs)`` from an sdf."""
    Chem = _require_rdkit()
    mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=False)
    if mol is None:
        raise ProbeConversionError(f"RDKit cannot read the sdf: {path}")
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
    ]
    return elements, bonds


#: Extract the element symbol from an atom name such as ``C1x`` / ``Cl2x`` / ``O1``.
#: ``{1,2}?`` is non-greedy, so ``Cl1x`` first fails as ``C`` + ``l1x`` and backs
#: off to ``Cl`` + ``1x`` (``C1x`` succeeds with one character).
_ATOM_NAME_ELEMENT = re.compile(r"^([A-Za-z]{1,2}?)(\d+)")


def element_from_atom_name(name: str) -> str:
    """Extract the element symbol from the prefix of an atom name.

    The names OpenFF assigns are ``f"{element}{serial}x"``, so the leading letters are
    the element symbol itself (``C1x`` -> ``C``, ``Cl1x`` -> ``Cl``).
    """
    n = (name or "").strip()
    m = _ATOM_NAME_ELEMENT.match(n)
    if m:
        return m.group(1).capitalize()
    letters = "".join(ch for ch in n if ch.isalpha())
    if not letters:
        raise ValueError(f"cannot determine the element from the atom name: {name!r}")
    # Names without a serial (monatomic ions etc). Take up to 2 characters as the element.
    return letters[:2].capitalize() if len(letters) >= 2 else letters.capitalize()


def prepare_probe_sdf(
    mol2_path: str | Path,
    out_path: str | Path,
    *,
    cid: str | None = None,
    reference_smiles: str | None = None,
    order_smiles: str | None = None,
    strict: bool = True,
) -> ProbeConversionReport:
    """Build the sdf for CosolvKit from a mol2 and verify it is the same molecule.

    The sdf is written in the same atom order as ``Molecule.from_smiles(order_smiles)``,
    with the coordinates following the reordering.

    Args:
        mol2_path: the probe's mol2.
        out_path: where to write the sdf.
        cid: label used in messages and in the report; defaults to the mol2's name.
        reference_smiles: the expected structure, compared as canonical SMILES.
            ``None`` skips the comparison, which is what lets a bond-order mix-up
            through.
        order_smiles: the SMILES that defines the atom order; defaults to
            ``reference_smiles``. Only when both are ``None`` is the file written in
            mol2 order without reordering, which must not be handed to CosolvKit.
        strict: True raises as soon as validation fails; False accumulates into
            ``warnings`` and continues.

    Returns:
        :class:`ProbeConversionReport`.
    """
    Chem = _require_rdkit()
    import numpy as np

    mol2_path = Path(mol2_path)
    out_path = Path(out_path)
    mol = read_mol2(mol2_path)
    label = cid or mol.name

    rdmol = mol_from_mol2(mol)
    problems: list[str] = []

    # --- Atom count, elemental composition, bond count ---
    if rdmol.GetNumAtoms() != len(mol):
        problems.append(
            f"atom count changed: mol2 {len(mol)} -> RDKit {rdmol.GetNumAtoms()}"
        )
    mol2_formula = dict(sorted(Counter(mol.elements).items()))
    rd_formula = _formula(rdmol)
    if mol2_formula != rd_formula:
        problems.append(f"elemental composition changed: {mol2_formula} -> {rd_formula}")
    n_bonds_mol2 = sum(len(b) for b in mol.bonds) // 2
    if rdmol.GetNumBonds() != n_bonds_mol2:
        problems.append(
            f"bond count changed: mol2 {n_bonds_mol2} -> RDKit {rdmol.GetNumBonds()}"
        )

    # --- Did RDKit keep the mol2 atom order? If not, the reordering has no basis. ---
    rd_elements = [a.GetSymbol() for a in rdmol.GetAtoms()]
    if rd_elements != mol.elements:
        problems.append(
            f"RDKit changed the mol2 atom order: {mol.elements} -> {rd_elements}"
        )

    # --- Coordinates ---
    coords_preserved = False
    if rdmol.GetNumConformers() == 0:
        problems.append(
            "the RDKit Mol has no conformer. CosolvKit calls "
            "mol.GetConformer().GetPositions(), so coordinates are mandatory."
        )
    else:
        rd_coords = rdmol.GetConformer().GetPositions()
        if rd_coords.shape[0] == len(mol):
            coords_preserved = bool(
                np.allclose(rd_coords, np.asarray(mol.coords), atol=1e-3)
            )
        if not coords_preserved:
            problems.append("coordinates are not preserved")

    # --- Charge and radicals ---
    formal_charge = Chem.rdmolops.GetFormalCharge(rdmol)
    n_radical = sum(a.GetNumRadicalElectrons() for a in rdmol.GetAtoms())
    if n_radical:
        problems.append(
            f"radical electrons: {n_radical}. Most likely the GAFF mol2 writes"
            f" a delocalised bond as a single bond and RDKit misreads the valence"
            f" (carboxylate etc). Fix the bond orders, or do not use this probe."
        )

    # --- Align the atom order with Molecule.from_smiles ---
    # CosolvKit builds the topology from smiles and the coordinates from the sdf, so a
    # different order puts coordinates on the wrong atoms, with no exception raised.
    order_ref = order_smiles or reference_smiles
    out_mol = rdmol
    mol2_order = list(range(rdmol.GetNumAtoms()))
    smiles_order_ok: bool | None = None
    if order_ref:
        try:
            mol2_order = map_to_smiles_order(rdmol, order_ref, label=label)
            out_mol = Chem.RenumberAtoms(rdmol, mol2_order)
            smiles_order_ok = [a.GetSymbol() for a in out_mol.GetAtoms()] == (
                smiles_reference_elements(order_ref)
            )
            if not smiles_order_ok:
                problems.append(
                    "the reordered element sequence does not match that of Molecule.from_smiles"
                )
        except AtomOrderError as exc:
            smiles_order_ok = False
            problems.append(str(exc))
            out_mol = rdmol
            mol2_order = list(range(rdmol.GetNumAtoms()))

    sdf_elements = [a.GetSymbol() for a in out_mol.GetAtoms()]
    mol2_names_ordered = [mol.atom_names[i] for i in mol2_order]
    openff_names = openff_atom_names(sdf_elements)
    # For the record: False states that reordering was needed, not that anything failed.
    atom_order_preserved = mol2_order == list(range(rdmol.GetNumAtoms()))

    # Verify independently that the reordering carried the coordinates along, rather
    # than assuming RenumberAtoms carries the conformer.
    if out_mol.GetNumConformers():
        reordered = out_mol.GetConformer().GetPositions()
        expected_coords = np.asarray(mol.coords, dtype=float)[mol2_order]
        if not np.allclose(reordered, expected_coords, atol=1e-3):
            problems.append(
                "the reordered coordinates do not match a permutation of the mol2 coordinates"
                " (coordinates did not follow the atoms)"
            )

    # --- Write the sdf ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_path))
    try:
        out_mol.SetProp("_Name", label)
        writer.write(out_mol)
    finally:
        writer.close()

    # --- Read it back and compare ---
    reread = Chem.MolFromMolFile(str(out_path), removeHs=False, sanitize=True)
    smiles = canonical_smiles(rdmol)
    smiles_roundtrip_ok = False
    if reread is None:
        problems.append("RDKit could not read back the sdf we wrote")
    else:
        smiles_roundtrip_ok = canonical_smiles(reread) == smiles
        if not smiles_roundtrip_ok:
            problems.append(
                f"the SMILES changed through the sdf: {smiles} -> {canonical_smiles(reread)}"
            )
        if [a.GetSymbol() for a in reread.GetAtoms()] != sdf_elements:
            problems.append("the atom order changes when the sdf is read back")
        if reread.GetNumConformers() == 0:
            problems.append("the sdf contains no coordinates")

    # --- Compare with the expected structure ---
    matches_reference: bool | None = None
    if reference_smiles:
        ref = Chem.MolFromSmiles(reference_smiles)
        if ref is None:
            problems.append(f"RDKit cannot read the config smiles: {reference_smiles!r}")
        else:
            expected = Chem.MolToSmiles(ref)
            matches_reference = expected == smiles
            if not matches_reference:
                problems.append(
                    f"does not match the config smiles:\n"
                    f"    config    : {reference_smiles!r} -> {expected}\n"
                    f"    from mol2 : {smiles}\n"
                    f"  formal charge {formal_charge:+d} / radicals {n_radical}."
                    f" Either the mol2 bond orders or the config smiles is wrong."
                )

    report = ProbeConversionReport(
        cid=label,
        mol2_path=str(mol2_path),
        sdf_path=str(out_path),
        n_atoms=rdmol.GetNumAtoms(),
        formula=rd_formula,
        n_bonds=rdmol.GetNumBonds(),
        formal_charge=formal_charge,
        n_radical_electrons=n_radical,
        smiles=smiles,
        smiles_roundtrip_ok=smiles_roundtrip_ok,
        atom_order_preserved=atom_order_preserved,
        coords_preserved=coords_preserved,
        order_reference_smiles=order_ref,
        mol2_order=mol2_order,
        elements=sdf_elements,
        mol2_names=mol2_names_ordered,
        openff_names=openff_names,
        smiles_order_ok=smiles_order_ok,
        matches_reference_smiles=matches_reference,
        warnings=problems,
    )

    if problems and strict:
        out_path.unlink(missing_ok=True)  # do not leave a broken artefact behind
        raise ProbeConversionError(
            f"probe {label}: validation of mol2 -> sdf failed\n  "
            + "\n  ".join(problems)
        )
    return report
