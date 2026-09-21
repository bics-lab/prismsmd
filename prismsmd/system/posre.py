"""Staged position restraints embedded into the GROMACS topology.

The GROMACS topology CosolvKit / parmed writes contains no ``#ifdef POSRES...``
block, so one is generated here.

Restraint strength is specified in kcal/mol/A^2 and converted to the GROMACS unit
kJ/mol/nm^2 (:data:`POSRES_UNIT_FACTOR`). The number in the define name is that
strength multiplied by :data:`DEFINE_NAME_SCALE`, so ``POSRES1000`` means
10 kcal/mol/A^2, and the ``define:`` of every mdp step must follow the same convention:
GROMACS silently ignores an undefined ``-DPOSRESxxx`` rather than erroring.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import jinja2
from scipy import constants as C

__all__ = [
    "DEFINE_NAME_SCALE",
    "POSRES_UNIT_FACTOR",
    "define_name",
    "embed_position_restraints",
    "embed_position_restraints_multi",
    "force_constant_kj_nm2",
    "render_position_restraints",
]

#: Conversion factor kcal/mol/A^2 -> kJ/mol/nm^2 (= 418.4): ``C.calorie`` is cal -> J
#: and ``(C.angstrom / C.nano) ** 2`` is the factor of 100 for A^-2 -> nm^-2.
POSRES_UNIT_FACTOR = C.calorie / (C.angstrom / C.nano) ** 2

#: Number in the define name = strength (kcal/mol/A^2) x this.
DEFINE_NAME_SCALE = 100


def define_name(prefix: str, strength_kcal: float) -> str:
    """Build a define name from a strength (kcal/mol/A^2).

    >>> define_name("POSRES", 10)
    'POSRES1000'
    >>> define_name("POSRES", 0.01)
    'POSRES1'
    >>> define_name("CAPOSRES", 0)
    'CAPOSRES0'
    """
    scaled = strength_kcal * DEFINE_NAME_SCALE
    n = round(scaled)
    if abs(scaled - n) > 1e-9:
        raise ValueError(
            f"strength {strength_kcal} kcal/mol/A^2 cannot be a define name "
            f"(must be an integer after x{DEFINE_NAME_SCALE})"
        )
    return f"{prefix}{n}"


def force_constant_kj_nm2(strength_kcal: float) -> float:
    """Convert kcal/mol/A^2 to kJ/mol/nm^2.

    >>> round(force_constant_kj_nm2(10), 6)
    4184.0
    """
    if strength_kcal < 0:
        raise ValueError(f"strength must be >= 0: {strength_kcal}")
    return strength_kcal * POSRES_UNIT_FACTOR


_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(
        str(Path(__file__).parent.parent / "md" / "templates")
    ),
    keep_trailing_newline=True,
    undefined=jinja2.StrictUndefined,
)


def render_position_restraints(
    atom_ids: Sequence[int],
    prefix: str,
    strength_kcal: float,
) -> str:
    """Build one ``#ifdef <prefix><n> ... #endif`` block.

    ``strength_kcal`` is in kcal/mol/A^2; the force constant written out is in
    kJ/mol/nm^2.
    """
    return _ENV.get_template("position_restraints.j2").render(
        define_name=define_name(prefix, strength_kcal),
        atom_id_list=list(atom_ids),
        force_constant=force_constant_kj_nm2(strength_kcal),
    )


def embed_position_restraints_multi(
    top_string: str,
    ids_per_molecule: Sequence[Sequence[int]],
    *,
    prefix: str,
    strengths_kcal: Iterable[float],
) -> str:
    """Insert restraint blocks into several protein moleculetypes.

    Args:
        ids_per_molecule: one list of local, 1-based atom numbers per protein
            moleculetype, in the order the moleculetypes appear in the topology.
            An empty list leaves that moleculetype without a block.
        prefix: define-name prefix, as in :func:`define_name`.
        strengths_kcal: restraint strengths in kcal/mol/A^2. They are the same for
            every moleculetype, so one ``-DPOSRES1000`` restrains all the chains.
    """
    out = top_string
    for index, ids in enumerate(ids_per_molecule):
        if not ids:
            continue
        out = embed_position_restraints(
            out,
            ids,
            prefix=prefix,
            strengths_kcal=strengths_kcal,
            molecule_index=index + 1,
        )
    return out


def embed_position_restraints(
    top_string: str,
    atom_ids: Sequence[int],
    *,
    prefix: str,
    strengths_kcal: Iterable[float],
    molecule_index: int = 1,
) -> str:
    """Insert the blocks just before the given ``[ moleculetype ]`` of a topology string.

    Args:
        atom_ids: 1-based atom numbers within that molecule.
        strengths_kcal: restraint strengths in kcal/mol/A^2, one block each.
        molecule_index: which ``[ moleculetype ]`` to insert before, counted from 0.
            The default 1 puts the blocks at the end of the first molecule, i.e. the
            protein definition.
    """
    blocks = [render_position_restraints(atom_ids, prefix, s) for s in strengths_kcal]

    out: list[str] = []
    seen = 0
    inserted = False
    for line in top_string.split("\n"):
        if line.lstrip().startswith("["):
            section = line[line.find("[") + 1 : line.find("]")].strip()
            if section == "moleculetype":
                if seen == molecule_index and not inserted:
                    out.extend(blocks)
                    inserted = True
                seen += 1
        out.append(line)

    if not inserted:
        raise ValueError(
            f"fewer than {molecule_index + 1} [ moleculetype ] sections found "
            f"(found {seen}). Check the structure of the topology."
        )
    return "\n".join(out)
