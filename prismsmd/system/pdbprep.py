"""PDB tidying applied before the structure is handed to PDBFixer."""
from __future__ import annotations

from collections.abc import Iterable

from ..pdbio import PDB_COLUMNS

#: Residue names treated as water.
WATER_RESNAMES = frozenset({"HOH", "WAT", "SOL", "TIP3", "T3P", "DOD"})

#: Candidate chain IDs to move water into. Pick one the protein does not use.
_CHAIN_CANDIDATES = "WXYZwxyz0123456789"


class ChainSeparationError(ValueError):
    """No chain ID could be reserved for the water."""


def _is_water(line: str) -> bool:
    return line[slice(*PDB_COLUMNS["res_name"])].strip().upper() in WATER_RESNAMES


def used_chain_ids(pdb_text: str) -> set[str]:
    """Return every chain ID that appears on an ATOM/HETATM record."""
    return {
        ln[21]
        for ln in pdb_text.split("\n")
        if ln.startswith(("ATOM", "HETATM")) and len(ln) > 21
    }


def separate_water_chain(pdb_text: str, *, candidates: Iterable[str] = _CHAIN_CANDIDATES) -> tuple[str, str | None]:
    """Move water to an unused chain ID.

    Returns ``(tidied PDB text, chain ID used)``, or ``(original text, None)`` if
    there is no water or it already sits in a chain of its own.

    Raises:
        ChainSeparationError: if no chain ID is available.
    """
    lines = pdb_text.split("\n")
    water_chains = {
        ln[21] for ln in lines
        if ln.startswith(("ATOM", "HETATM")) and len(ln) > 21 and _is_water(ln)
    }
    if not water_chains:
        return pdb_text, None

    protein_chains = {
        ln[21] for ln in lines
        if ln.startswith(("ATOM", "HETATM")) and len(ln) > 21 and not _is_water(ln)
    }
    if not (water_chains & protein_chains):
        return pdb_text, None

    used = used_chain_ids(pdb_text)
    new_id = next((c for c in candidates if c not in used), None)
    if new_id is None:
        raise ChainSeparationError(
            f"no free chain ID to move water into (in use: {sorted(used)})."
            " Continuing would leave PDBFixer without the C-terminal OXT, and"
            " OpenMM's createSystem would stop with"
            " 'No template found ... missing 1 C atom'."
        )

    out = []
    for ln in lines:
        if ln.startswith(("ATOM", "HETATM")) and len(ln) > 21 and _is_water(ln):
            out.append(ln[:21] + new_id + ln[22:])
        else:
            out.append(ln)
    return "\n".join(out), new_id


#: Amber-style residue names that encode a protonation state, mapped to the
#: standard PDB name plus the OpenMM ``addHydrogens`` variant that reproduces it.
PROTONATION_VARIANTS: dict[str, tuple[str, str]] = {
    "HID": ("HIS", "HID"),      # neutral His, proton on ND1
    "HIE": ("HIS", "HIE"),      # neutral His, proton on NE2
    "HIP": ("HIS", "HIP"),      # protonated His, +1
    "LYN": ("LYS", "LYN"),      # neutral Lys
    "GLH": ("GLU", "GLH"),      # neutral Glu
    "ASH": ("ASP", "ASH"),      # neutral Asp
    "CYX": ("CYS", "CYX"),      # disulfide-bonded Cys
}


def normalise_protonation_names(pdb_text: str) -> tuple[str, dict[tuple[str, str], str]]:
    """Rewrite Amber protonation names to standard ones, remembering the states.

    Args:
        pdb_text: the receptor PDB.

    Returns:
        ``(text, variants)``. ``variants`` maps ``(chain_id, residue_id)`` -- the
        residue id kept as the raw 5-character field so insertion codes survive --
        to the OpenMM variant name.
    """
    out: list[str] = []
    variants: dict[tuple[str, str], str] = {}
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            resname = line[17:20].strip()
            entry = PROTONATION_VARIANTS.get(resname)
            if entry is not None:
                standard, variant = entry
                variants[(line[21], line[22:27])] = variant
                line = line[:17] + f"{standard:>3s}" + line[20:]
        out.append(line)
    return "\n".join(out) + "\n", variants


def variants_for_topology(topology, variants: dict[tuple[str, str], str]) -> list:
    """Build the ``variants`` list ``Modeller.addHydrogens`` expects.

    One entry per residue in topology order, ``None`` meaning "use the default".
    Matching is by ``(chain id, residue id)``.

    Raises:
        ValueError: if a recorded state has no residue to attach to, which means
            the residue was dropped during preparation.
    """
    result = []
    seen: set[tuple[str, str]] = set()
    for residue in topology.residues():
        key = (residue.chain.id, f"{residue.id:>5s}" if isinstance(residue.id, str)
               else f"{residue.id:>5d}")
        variant = variants.get(key)
        if variant is None:
            # Try the stripped form too: PDBFixer may reformat the id field.
            for (chain, resid), value in variants.items():
                if chain == residue.chain.id and resid.strip() == str(residue.id).strip():
                    variant, key = value, (chain, resid)
                    break
        if variant is not None:
            seen.add(key)
        result.append(variant)

    missing = {k: v for k, v in variants.items() if k not in seen}
    if missing:
        raise ValueError(
            f"{len(missing)} protonation state(s) have no residue in the prepared "
            f"topology: {sorted((c, r.strip(), v) for (c, r), v in missing.items())}. "
            "A residue carrying an explicit protonation state was dropped during "
            "preparation. Do not proceed."
        )
    return result
