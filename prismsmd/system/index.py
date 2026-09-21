"""Generation of the GROMACS index file (``.ndx``).

Writes ``Non-Water`` and ``Water`` as an exact partition of every atom, which is
what the mdp templates use for temperature coupling.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

__all__ = [
    "ION_RESNAMES",
    "WATER_RESNAMES",
    "index_groups",
    "iter_gro_atoms",
    "render_index_ndx",
    "write_index_ndx",
]

#: Residue names treated as water.
WATER_RESNAMES = frozenset({"HOH", "WAT", "SOL", "TIP3", "T3P"})

#: Monatomic ions. Grouped with ``Non-Water``.
ION_RESNAMES = frozenset(
    {"NA", "CL", "NA+", "CL-", "K", "K+", "MG", "MG2", "ZN", "CA2", "BR", "LI", "RB", "CS"}
)

#: Atom numbers per line.
_PER_LINE = 15


def iter_gro_atoms(gro_string: str):
    """Yield ``(resi, resname, atom_name)`` for each atom line of a ``.gro``.

    The atom-number column is not read; it wraps at 99999.
    """
    lines = gro_string.split("\n")
    if len(lines) < 3:
        raise ValueError("gro is too short")
    natoms = int(lines[1].strip())
    for line in lines[2 : 2 + natoms]:
        yield line[0:5], line[5:10].strip(), line[10:15].strip()


def index_groups(
    gro_string: str,
    probe_resnames: Sequence[str] = (),
) -> dict[str, list[int]]:
    """Build index groups from a ``.gro``. Values are 1-based atom numbers.

    ==============  ===============================================
    ``System``      every atom
    ``Water``       residues in :data:`WATER_RESNAMES`
    ``Non-Water``   everything else (protein + ions + probes)
    ``Ion``         residues in :data:`ION_RESNAMES` (omitted if empty)
    ``Protein``     anything that is not water, ion or probe
    ``<resname>``   one per probe residue, including virtual sites
    ==============  ===============================================

    Raises if the gro has no atoms, no protein atom, or a named probe residue
    that does not occur.
    """
    probes = {p.upper() for p in probe_resnames}
    system: list[int] = []
    water: list[int] = []
    non_water: list[int] = []
    ions: list[int] = []
    protein: list[int] = []
    per_probe: dict[str, list[int]] = {p: [] for p in probes}

    for i, (_resi, resn, _name) in enumerate(iter_gro_atoms(gro_string), start=1):
        key = resn.upper()
        system.append(i)
        if key in WATER_RESNAMES:
            water.append(i)
            continue
        non_water.append(i)
        if key in probes:
            per_probe[key].append(i)
        elif key in ION_RESNAMES:
            ions.append(i)
        else:
            protein.append(i)

    if not system:
        raise ValueError("the gro has no atoms")
    if not protein:
        raise ValueError(
            "no protein atom found at all."
            " Check the residue-name classification (WATER_RESNAMES / ION_RESNAMES)."
        )
    for name, ids in per_probe.items():
        if not ids:
            raise ValueError(f"probe residue {name!r} is not in the gro")

    groups: dict[str, list[int]] = {
        "System": system,
        "Non-Water": non_water,
        "Water": water,
        "Protein": protein,
    }
    if ions:
        groups["Ion"] = ions
    for name in sorted(per_probe):
        groups[name] = per_probe[name]
    return groups


def render_index_ndx(gro_string: str, probe_resnames: Sequence[str] = ()) -> str:
    """Return the contents of the ``.ndx`` as a string."""
    return _format_groups(index_groups(gro_string, probe_resnames))


def _format_groups(groups: dict[str, Iterable[int]]) -> str:
    out: list[str] = []
    for name, ids in groups.items():
        ids = list(ids)
        out.append(f"[ {name} ]")
        for start in range(0, len(ids), _PER_LINE):
            out.append(" ".join(f"{i:d}" for i in ids[start : start + _PER_LINE]))
    return "\n".join(out) + "\n"


def write_index_ndx(
    path: str | Path, gro_string: str, probe_resnames: Sequence[str] = ()
) -> Path:
    """Write the ``.ndx`` to ``path`` and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_index_ndx(gro_string, probe_resnames))
    return path
