"""Put reference (RESP) probe charges into the GROMACS topology.

CosolvKit parametrises from ``Molecule.from_smiles`` and GAFF assigns AM1-BCC charges,
so the RESP charges in the probe's mol2 are never read. This module substitutes them
into the written ``.top``, keeping GAFF's types and bonded terms: only the probe's
charge column changes, which is asserted here.

Atoms are matched between the two files by their bond-graph environment, since position,
atom name and atom type all fail to correspond. Chemically equivalent atoms (the three
methyl hydrogens) share a key; a split within such a group raises rather than picking
one silently.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ChargeComparison",
    "ChargeSubstitution",
    "TopAtom",
    "apply_reference_charges",
    "compare_with_reference",
    "read_mol2_charges",
    "read_top_probe_atoms",
]

#: GAFF/mol2 atom type prefix -> element. Two letters first (``cl`` before ``c``).
GAFF_ELEMENT = {"cl": "Cl", "br": "Br", "c": "C", "n": "N", "o": "O", "h": "H",
                "s": "S", "p": "P", "f": "F", "i": "I"}

#: How close a topology charge has to be to the reference to count as already
#: substituted. Above the artefacts of 6-decimal rounding and of the neutralisation
#: below, and far below any physically meaningful difference.
MATCH_TOLERANCE_E = 1e-4

#: How close two charges have to be to count as "the same" inside an
#: equivalence group (methyl hydrogens). Looser than mol2's 4 decimals.
GROUP_TOLERANCE_E = 1e-3

#: How much residual total charge is allowed before the substitution is refused.
TOTAL_TOLERANCE_E = 1e-3

#: How far the written file's total charge may sit from an integer. The file holds
#: 6 decimals, so a correctly quantised molecule sums to an integer exactly.
TOTAL_INTEGRAL_TOLERANCE_E = 1e-7


@dataclass(frozen=True)
class TopAtom:
    """One row of a moleculetype's ``[ atoms ]``."""

    line: int
    name: str
    element: str
    charge: float
    mass: float

    @property
    def is_virtual_site(self) -> bool:
        """True for the VIS repulsion centre: massless and bonded to nothing."""
        return self.mass == 0.0


@dataclass(frozen=True)
class ChargeComparison:
    """What the topology's charges are, next to the reference mol2's."""

    resname: str
    n_atoms: int
    max_abs_delta: float
    #: (top atom name, top charge, reference charge)
    rows: tuple[tuple[str, float, float], ...]
    #: Atoms skipped because they do not exist in the mol2 (the VIS site).
    skipped: tuple[str, ...]
    #: Sum of every charge of the moleculetype as the file stores it, the virtual
    #: site included -- this is what grompp adds up.
    total_charge: float

    @property
    def total_is_integral(self) -> bool:
        """True when the molecule's charges sum to an integer in the file.

        Not implied by :attr:`matches_reference`: every atom can sit within tolerance
        of the reference while the sum is off, because a 6-decimal file cannot hold the
        mol2's charges exactly.
        """
        return abs(self.total_charge - round(self.total_charge)) <= TOTAL_INTEGRAL_TOLERANCE_E

    @property
    def matches_reference(self) -> bool:
        """True when the topology already carries the reference charges,
        to within :data:`MATCH_TOLERANCE_E`."""
        return self.max_abs_delta <= MATCH_TOLERANCE_E


@dataclass(frozen=True)
class ChargeSubstitution:
    """What :func:`apply_reference_charges` did, for the log and for the gate."""

    resname: str
    n_atoms: int
    max_abs_delta: float
    total_before: float
    total_after: float
    per_atom: tuple[tuple[str, float, float], ...]  # (top atom name, old, new)
    #: Residual spread over the atoms to keep the total exactly where it was
    #: (see :data:`TOTAL_TOLERANCE_E`).
    neutralised_by: float


def element_from_gaff(atom_type: str) -> str:
    """Map a GAFF/mol2 atom type to its element symbol."""
    t = atom_type.lower()
    return GAFF_ELEMENT.get(t[:2], GAFF_ELEMENT.get(t[:1], "?"))


def element_from_name(name: str) -> str:
    """Map an OpenFF atom name to its element: ``C2x`` -> ``C``."""
    head = "".join(c for c in name if c.isalpha())
    if head.endswith("x") and len(head) > 1:
        head = head[:-1]
    two = head[:2].capitalize()
    return two if two in GAFF_ELEMENT.values() else head[:1].upper()


def read_mol2_charges(text: str) -> tuple[list[tuple[str, str, float]], list[tuple[int, int]]]:
    """Read a mol2 and return ``([(name, element, charge), ...], bonds)``.

    Bond indices are 0-based.
    """
    lines = text.splitlines()
    if "@<TRIPOS>ATOM" not in lines:
        raise ValueError("the mol2 has no @<TRIPOS>ATOM section")
    atoms: list[tuple[str, str, float]] = []
    for ln in lines[lines.index("@<TRIPOS>ATOM") + 1:]:
        if ln.startswith("@"):
            break
        f = ln.split()
        if len(f) < 9:
            raise ValueError(f"mol2 ATOM line is too short: {ln!r}")
        atoms.append((f[1], element_from_gaff(f[5]), float(f[8])))
    bonds: list[tuple[int, int]] = []
    if "@<TRIPOS>BOND" in lines:
        for ln in lines[lines.index("@<TRIPOS>BOND") + 1:]:
            if ln.startswith("@"):
                break
            f = ln.split()
            if len(f) >= 3:
                bonds.append((int(f[1]) - 1, int(f[2]) - 1))
    if not atoms or not bonds:
        raise ValueError(f"cannot read the mol2 ({len(atoms)} atoms / {len(bonds)} bonds)")
    return atoms, bonds


def read_top_probe_atoms(
    text: str, resname: str
) -> tuple[list[TopAtom], list[tuple[int, int]]]:
    """Read one moleculetype's ``[ atoms ]`` and ``[ bonds ]`` from a topology.

    The receiving list is closed when a different moleculetype's section starts, so
    the next molecule's rows are not appended to this one's.
    """
    name: str | None = None
    mode: str | None = None
    atoms: list | None = None
    bonds: list | None = None
    found_atoms: list | None = None
    found_bonds: list | None = None
    for i, raw in enumerate(text.splitlines()):
        s = raw.split(";")[0].strip()
        if s.startswith("["):
            section = s.strip("[] ").strip()
            mode = {"moleculetype": "mt", "atoms": "at", "bonds": "bd"}.get(section)
            if mode == "at":
                atoms = [] if name == resname else None
                if atoms is not None:
                    found_atoms = atoms
            elif mode == "bd":
                bonds = [] if name == resname else None
                if bonds is not None:
                    found_bonds = bonds
            continue
        if not s:
            continue
        if mode == "mt":
            name = s.split()[0]
        elif mode == "at" and atoms is not None:
            f = s.split()
            if len(f) < 7:
                raise ValueError(f"{resname}: [ atoms ] line is too short: {raw!r}")
            atoms.append(TopAtom(line=i, name=f[4],
                                 element=element_from_name(f[4]),
                                 charge=float(f[6]), mass=float(f[7])))
        elif mode == "bd" and bonds is not None:
            f = s.split()
            if len(f) >= 2:
                bonds.append((int(f[0]) - 1, int(f[1]) - 1))
    if found_atoms is None:
        raise ValueError(f"the topology has no moleculetype {resname}")
    return found_atoms, (found_bonds or [])


def _adjacency(n: int, bonds) -> dict[int, set[int]]:
    d: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j in bonds:
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"bond atom index out of range: {(i, j)} (of {n} atoms)")
        d[i].add(j)
        d[j].add(i)
    return d


def _refine(elements: list[str], adj: dict[int, set[int]]) -> list[str]:
    """Colour-refinement (1-WL) labels: two atoms share a label iff no walk
    distinguishes them.

    Refinement runs to a fixed point rather than a fixed number of shells, which is
    what separates atoms whose difference only shows up several bonds away (pyridine's
    para and meta hydrogens need three rounds).

    The labels are hashes of the neighbourhood, not per-molecule serial numbers, so
    they can be compared between the mol2 graph and the top graph.
    """
    labels = list(elements)
    n = len(elements)
    for _ in range(n):
        nxt = [
            hashlib.sha1(
                (labels[i] + "(" + ",".join(sorted(labels[j] for j in adj[i])) + ")")
                .encode()
            ).hexdigest()[:16]
            for i in range(n)
        ]
        if _partition(nxt) == _partition(labels):
            return labels
        labels = nxt
    return labels


def _partition(labels: list[str]) -> set[frozenset[int]]:
    """The grouping the labels induce, with the label values themselves discarded."""
    by: dict[str, set[int]] = {}
    for i, lab in enumerate(labels):
        by.setdefault(lab, set()).add(i)
    return {frozenset(v) for v in by.values()}


def _grouped(elements: list[str], adj: dict[int, set[int]]) -> dict[str, list[int]]:
    labels = _refine(elements, adj)
    groups: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)
    return groups


def _one_value(values: list[float], what: str) -> float:
    if max(values) - min(values) > GROUP_TOLERANCE_E:
        raise ValueError(
            f"the charges of an equivalence group are split in the {what}: {values}"
        )
    return values[0]


def apply_reference_charges(
    top_text: str, resname: str, mol2_text: str
) -> tuple[str, ChargeSubstitution]:
    """Return the topology with ``resname``'s charges taken from the mol2.

    Only the charge field of that moleculetype's ``[ atoms ]`` rows changes; the check
    at the end asserts every other line is untouched.

    Raises:
        ValueError: atom counts differ, the bond-graph environments do not match, an
            equivalence group has split charges, or the total charge moves.
    """
    m_atoms, m_bonds = read_mol2_charges(mol2_text)
    t_atoms, t_bonds = read_top_probe_atoms(top_text, resname)
    if not t_bonds:
        raise ValueError(
            f"{resname}: the topology has no [ bonds ], so the graphs cannot be matched"
        )
    if len(m_atoms) != len(t_atoms):
        raise ValueError(
            f"{resname}: atom counts differ (mol2 {len(m_atoms)} / top {len(t_atoms)})."
            " Check whether the topology already carries the virtual site"
        )
    m_elements = [e for _, e, _ in m_atoms]
    t_elements = [a.element for a in t_atoms]
    m_groups = _grouped(m_elements, _adjacency(len(m_atoms), m_bonds))
    t_groups = _grouped(t_elements, _adjacency(len(t_atoms), t_bonds))
    if set(m_groups) != set(t_groups) or any(
        len(m_groups[k]) != len(t_groups[k]) for k in m_groups
    ):
        raise ValueError(
            f"{resname}: the bond-graph environments do not match\n"
            f"  mol2 {sorted((k, len(v)) for k, v in m_groups.items())}\n"
            f"  top  {sorted((k, len(v)) for k, v in t_groups.items())}"
        )

    # Decide every target value first and write once: writing twice would make the
    # second pass's "old" value disagree with the file.
    target_by_line: dict[int, tuple[str, float, float]] = {}
    max_delta = 0.0
    for key, m_idx in m_groups.items():
        target = _one_value([m_atoms[i][2] for i in m_idx], "mol2")
        _one_value([t_atoms[i].charge for i in t_groups[key]], "top")
        for i in t_groups[key]:
            a = t_atoms[i]
            target_by_line[a.line] = (a.name, a.charge, target)
            max_delta = max(max_delta, abs(target - a.charge))

    before = sum(a.charge for a in t_atoms)
    raw_after = sum(new for _, _, new in target_by_line.values())
    if abs(raw_after - before) > TOTAL_TOLERANCE_E:
        raise ValueError(
            f"{resname}: the total charge moves by {raw_after - before:+.6f}"
            f" ({before:+.6f} -> {raw_after:+.6f}), which changes how many"
            " neutralising ions are needed"
        )
    # Handing every member of an equivalence group the representative value shifts the
    # total by the differences in the group's last digits. Spread that back evenly
    # rather than leaving it to PME's neutralising background.
    shift = (before - raw_after) / len(t_atoms)

    # Quantise before killing the residual: the ``.top`` writes charges with 6
    # decimals, so matching the total in unquantised floats still leaves a residual in
    # the written values, and grompp sums it over every probe copy in the box. So
    # quantise to 1e-6, then put what is left on a single-atom group, which leaves
    # equivalent atoms equal.
    quantum = 1e-6
    # The target is the total rounded to 6 decimals, not the original float sum: the
    # original itself carries quantisation error, so a float target can never be met.
    target_total = round(before, 6)
    lines_order = sorted(target_by_line)
    quantised = {ln: round(target_by_line[ln][2] + shift, 6) for ln in lines_order}
    residual = target_total - sum(quantised.values())
    if abs(residual) > len(t_atoms) * quantum:
        raise ValueError(
            f"{resname}: the 6-decimal quantisation residual is too large ({residual:+.3e})"
        )
    # Put it on the singleton group (an atom with no equivalent partners) whose
    # charge has the largest magnitude.
    singleton_lines = [t_atoms[i].line for key, idx in t_groups.items()
                       if len(idx) == 1 for i in idx]
    if round(residual, 6) == 0.0:
        # No residual, so no sink is needed. This branch also covers molecules whose
        # atoms are all equivalent (benzene), which have no singleton group at all.
        pass
    elif singleton_lines:
        sink = max(singleton_lines, key=lambda ln: abs(quantised[ln]))
        quantised[sink] = round(quantised[sink] + residual, 6)
    else:
        # A molecule with equivalence groups only, so there is no sink: spread the
        # residual one quantum at a time over the largest group, breaking equivalence
        # by 1e-6 e per atom to make the written total match exactly.
        big = max(t_groups.values(), key=len)
        spread = [t_atoms[i].line for i in big]
        n_q = int(round(abs(residual) / quantum))
        if n_q > len(spread):
            raise ValueError(
                f"{resname}: a residual of {residual:+.3e} does not fit into the"
                f" largest group of {len(spread)} atoms at 1e-6 each ({n_q} quanta)"
            )
        step = quantum if residual > 0 else -quantum
        for ln in spread[:n_q]:
            quantised[ln] = round(quantised[ln] + step, 6)

    lines = top_text.splitlines(keepends=True)
    per_atom: list[tuple[str, float, float]] = []
    running = 0.0
    for line_no in lines_order:          # file order = atom order within the molecule
        atom_name, old_q, _ = target_by_line[line_no]
        new = quantised[line_no]
        running += new
        # parmed writes "; qtot <running total>" on each row. Replacing only the
        # charge would leave that note stale, so it is updated alongside.
        lines[line_no] = _rewrite_qtot(
            _replace_charge(lines[line_no], old_q, new), running)
        per_atom.append((atom_name, old_q, new))

    after = sum(new for _, _, new in per_atom)
    new_text = "".join(lines)
    _assert_only_charges_changed(top_text, new_text, set(target_by_line))
    # Verify the total by reading back what was written, not the in-memory floats:
    # the rounding residual is only visible in the file.
    written = sum(a.charge for a in read_top_probe_atoms(new_text, resname)[0])
    if abs(written - target_total) > 1e-9:
        raise ValueError(
            f"{resname}: the written total does not match the target "
            f"(aimed at {target_total:+.9f}, wrote {written:+.9f}); the 6-decimal "
            "rounding was not absorbed"
        )
    return new_text, ChargeSubstitution(
        resname=resname, n_atoms=len(t_atoms), max_abs_delta=max_delta,
        total_before=before, total_after=after, per_atom=tuple(per_atom),
        neutralised_by=shift,
    )


def _replace_charge(line: str, old: float, new: float) -> str:
    """Rewrite the 7th field, keeping the column width where it fits."""
    head, sep, tail = line.rstrip("\n"), "", ""
    if line.endswith("\n"):
        sep, tail = "", "\n"
    fields = head.split()
    if len(fields) < 7:
        raise ValueError(f"[ atoms ] line is too short: {line!r}")
    token = fields[6]
    if abs(float(token) - old) > 1e-9:
        raise ValueError(f"the charge is not in the expected field: {token} != {old}")
    start = head.index(token, sum(len(f) for f in fields[:6]))
    text = f"{new:.6f}"
    return head[:start] + text.rjust(len(token)) + head[start + len(token):] + sep + tail


def _rewrite_qtot(line: str, running: float) -> str:
    """Update a trailing ``; qtot <number>`` comment, if the line has one."""
    head, sep, comment = line.partition(";")
    if not sep or "qtot" not in comment:
        return line
    tail = "\n" if comment.endswith("\n") else ""
    parts = comment.split()
    try:
        i = parts.index("qtot")
        float(parts[i + 1])
    except (ValueError, IndexError):
        return line
    parts[i + 1] = f"{running:.6f}"
    return f"{head};{' ' if comment.startswith(' ') else ''}{' '.join(parts)}{tail}"


def _assert_only_charges_changed(before: str, after: str, allowed: set[int]) -> None:
    b, a = before.splitlines(), after.splitlines()
    if len(b) != len(a):
        raise ValueError(f"the line count changed, {len(b)} -> {len(a)}")
    changed = {i for i, (x, y) in enumerate(zip(b, a, strict=True)) if x != y}
    if not changed <= allowed:
        raise ValueError(
            f"lines other than charges changed: {sorted(changed - allowed)[:5]}"
        )


def apply_reference_charges_from_dir(
    top_text: str, resname: str, mol2_dir: Path
) -> tuple[str, ChargeSubstitution]:
    """:func:`apply_reference_charges` with the mol2 looked up as ``<resname>.mol2``."""
    path = Path(mol2_dir) / f"{resname}.mol2"
    if not path.is_file():
        raise ValueError(f"no reference-charge mol2 at {path}")
    return apply_reference_charges(top_text, resname, path.read_text())


def compare_with_reference(
    top_text: str, resname: str, mol2_text: str
) -> ChargeComparison:
    """How far the topology's probe charges are from the reference mol2's.

    The read-only counterpart of :func:`apply_reference_charges`. The VIS repulsion
    centre is skipped, having no counterpart in the mol2.
    """
    m_atoms, m_bonds = read_mol2_charges(mol2_text)
    t_all, t_bonds = read_top_probe_atoms(top_text, resname)
    skipped = tuple(a.name for a in t_all if a.is_virtual_site)
    keep = [i for i, a in enumerate(t_all) if not a.is_virtual_site]
    t_atoms = [t_all[i] for i in keep]
    if len(t_atoms) != len(m_atoms):
        raise ValueError(
            f"{resname}: atom counts differ (mol2 {len(m_atoms)} / top {len(t_atoms)}"
            f", after dropping {len(skipped)} virtual site(s))"
        )
    if not t_bonds:
        raise ValueError(f"{resname}: the topology has no [ bonds ]")
    # The virtual site was dropped, so renumber the bonds.
    remap = {old: new for new, old in enumerate(keep)}
    bonds = [(remap[i], remap[j]) for i, j in t_bonds if i in remap and j in remap]
    m_groups = _grouped([e for _, e, _ in m_atoms], _adjacency(len(m_atoms), m_bonds))
    t_groups = _grouped([a.element for a in t_atoms], _adjacency(len(t_atoms), bonds))
    if set(m_groups) != set(t_groups) or any(
        len(m_groups[k]) != len(t_groups[k]) for k in m_groups
    ):
        raise ValueError(f"{resname}: the bond-graph environments do not match")
    rows: list[tuple[str, float, float]] = []
    worst = 0.0
    for key, m_idx in m_groups.items():
        ref = _one_value([m_atoms[i][2] for i in m_idx], "mol2")
        for i in t_groups[key]:
            rows.append((t_atoms[i].name, t_atoms[i].charge, ref))
            worst = max(worst, abs(t_atoms[i].charge - ref))
    return ChargeComparison(resname=resname, n_atoms=len(t_atoms),
                            max_abs_delta=worst, rows=tuple(rows), skipped=skipped,
                            total_charge=sum(a.charge for a in t_all))
